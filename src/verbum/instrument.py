"""Instrumented forward pass — record attention patterns per head.

Level-1 mechanistic interpretability. Raw PyTorch hooks on HuggingFace
models — no framework dependencies (TransformerLens, nnsight). Simpler
to understand, simpler to release.

Usage::

    from verbum.instrument import load_model, record_attention

    model, tokenizer = load_model("Qwen/Qwen3-4B")
    patterns = record_attention(model, tokenizer, ["The dog runs."])
    # patterns["The dog runs."].shape == (n_layers, n_heads, seq_len, seq_len)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import structlog
import torch

__all__ = [
    "AttentionCapture",
    "LAMBDA_INDICATORS",
    "LayerAblationResult",
    "ModelInfo",
    "ablate_heads",
    "ablate_layers",
    "capture_bos_residuals",
    "head_selectivity",
    "load_model",
    "patch_bos_generate",
    "record_attention",
    "zero_heads_generate",
]

_LOG = structlog.get_logger(__name__)


# ─────────────────────────── architecture helpers ─────────────────────


def _get_layers(model: Any) -> Any:
    """Return the list of transformer layers, handling multiple architectures.

    Supports:
    - ``model.model.layers`` — Qwen2, Phi3, LLaMA, Mistral, etc.
    - ``model.gpt_neox.layers`` — GPTNeoX (Pythia, GPT-NeoX-20B)
    - ``model.transformer.h`` — GPT-2, GPT-J, GPT-Neo
    """
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return model.gpt_neox.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    msg = (
        f"Cannot find transformer layers in {type(model).__name__}. "
        "Supported: model.model.layers, model.gpt_neox.layers, model.transformer.h"
    )
    raise AttributeError(msg)


def _get_self_attn(layer: Any) -> Any:
    """Return the self-attention module from a transformer layer.

    Supports:
    - ``layer.self_attn`` — Qwen2, Phi3, LLaMA, Mistral
    - ``layer.attention`` — GPTNeoX (Pythia)
    - ``layer.attn`` — GPT-2, GPT-J
    """
    if hasattr(layer, "self_attn"):
        return layer.self_attn
    if hasattr(layer, "attention"):
        return layer.attention
    if hasattr(layer, "attn"):
        return layer.attn
    msg = (
        f"Cannot find attention module in {type(layer).__name__}. "
        "Supported: layer.self_attn, layer.attention, layer.attn"
    )
    raise AttributeError(msg)


# ─────────────────────────── model loading ────────────────────────────


@dataclass(frozen=True)
class ModelInfo:
    """Metadata about the loaded model."""

    name: str
    n_layers: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    hidden_size: int
    device: str


def load_model(
    model_name: str = "Qwen/Qwen3-4B",
    *,
    device: str | None = None,
    dtype: torch.dtype = torch.float16,
) -> tuple[Any, Any, ModelInfo]:
    """Load a HuggingFace causal LM with attention output enabled.

    Returns (model, tokenizer, info).

    The model is set to eval mode with ``output_attentions=True`` in its
    config so that forward passes return per-layer attention weights.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if device is None:
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

    _LOG.info("instrument.loading", model=model_name, device=device, dtype=str(dtype))

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=dtype,
        device_map=device,
        attn_implementation="eager",  # need full attention matrices, not flash
    )
    model.eval()
    model.config.output_attentions = True

    config = model.config
    info = ModelInfo(
        name=model_name,
        n_layers=config.num_hidden_layers,
        n_heads=config.num_attention_heads,
        n_kv_heads=getattr(config, "num_key_value_heads", config.num_attention_heads),
        head_dim=config.hidden_size // config.num_attention_heads,
        hidden_size=config.hidden_size,
        device=device,
    )

    _LOG.info(
        "instrument.loaded",
        n_layers=info.n_layers,
        n_heads=info.n_heads,
        n_kv_heads=info.n_kv_heads,
        head_dim=info.head_dim,
        total_heads=info.n_layers * info.n_heads,
    )
    return model, tokenizer, info


# ─────────────────────────── attention recording ──────────────────────


@dataclass
class AttentionCapture:
    """Captured attention patterns from a single forward pass.

    ``patterns`` has shape ``(n_layers, n_heads, seq_len, seq_len)`` —
    the full attention weight matrix for every head at every layer.
    """

    prompt: str
    n_tokens: int
    token_strs: list[str]
    patterns: np.ndarray  # (n_layers, n_heads, seq_len, seq_len)


def record_attention(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    *,
    max_new_tokens: int = 1,
) -> dict[str, AttentionCapture]:
    """Run prompts through the model and capture attention patterns.

    We generate only ``max_new_tokens`` (default 1) — we care about the
    attention patterns on the input, not about generation quality. The
    single forward pass over the prompt tokens gives us the full
    attention matrix.

    Returns a dict mapping prompt → AttentionCapture.
    """
    results: dict[str, AttentionCapture] = {}

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        n_tokens = inputs["input_ids"].shape[1]
        token_ids = inputs["input_ids"][0].tolist()
        token_strs = [tokenizer.decode([tid]) for tid in token_ids]

        _LOG.info(
            "instrument.forward",
            prompt=prompt[:60],
            n_tokens=n_tokens,
        )

        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True)

        # outputs.attentions is a tuple of (n_layers,) tensors
        # each tensor shape: (batch=1, n_heads, seq_len, seq_len)
        attn_tuple = outputs.attentions
        n_layers = len(attn_tuple)

        # Stack into (n_layers, n_heads, seq_len, seq_len)
        patterns = np.stack(
            [layer_attn[0].cpu().float().numpy() for layer_attn in attn_tuple],
            axis=0,
        )

        results[prompt] = AttentionCapture(
            prompt=prompt,
            n_tokens=n_tokens,
            token_strs=token_strs,
            patterns=patterns,
        )

        _LOG.info(
            "instrument.captured",
            prompt=prompt[:60],
            shape=patterns.shape,
        )

    return results


# ─────────────────────────── selectivity ──────────────────────────────


def head_selectivity(
    condition: AttentionCapture,
    baseline: AttentionCapture,
) -> np.ndarray:
    """Compute per-head selectivity between a condition and baseline.

    Returns array of shape ``(n_layers, n_heads)`` where each value is
    the mean L2 distance between the condition's attention pattern and
    the baseline's attention pattern for that head.

    Since prompts may differ in length, we compare over the minimum
    shared prefix length (both start with the gate, so the first N
    tokens overlap).
    """
    min_seq = min(condition.patterns.shape[2], baseline.patterns.shape[2])

    # Trim to shared length: (n_layers, n_heads, min_seq, min_seq)
    c = condition.patterns[:, :, :min_seq, :min_seq]
    b = baseline.patterns[:, :, :min_seq, :min_seq]

    # L2 distance per head, averaged over sequence positions
    # shape: (n_layers, n_heads)
    diff = c - b
    per_head = np.sqrt(np.mean(diff**2, axis=(-2, -1)))
    return per_head


# ─────────────────────────── activation patching ──────────────────────

LAMBDA_INDICATORS = ["λ", "∀", "∃", "→", "∧", "∨", "¬", "ι"]


def _detect_lambda(text: str) -> bool:
    """Heuristic: does this text contain lambda-calculus-like content?"""
    return "λ" in text or sum(text.count(s) for s in LAMBDA_INDICATORS) >= 3


def _generate(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int = 30,
) -> str:
    """Generate text from a prompt. Returns only the new tokens."""
    # Temporarily disable output_attentions for generation (not needed,
    # and some architectures change their output format when it's on).
    prev_attn = model.config.output_attentions
    model.config.output_attentions = False
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        n_prompt = inputs["input_ids"].shape[1]
        with torch.no_grad():
            # Some models ship generation_config with sampling params
            # (e.g. Qwen3 has top_k/temperature/top_p) which conflict
            # with greedy decoding. Clear them if present.
            gen_cfg = model.generation_config
            if getattr(gen_cfg, "temperature", None) is not None:
                gen_cfg.temperature = None
            if getattr(gen_cfg, "top_p", None) is not None:
                gen_cfg.top_p = None
            if getattr(gen_cfg, "top_k", None) is not None:
                gen_cfg.top_k = None
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        new_ids = output_ids[0, n_prompt:]
        return tokenizer.decode(new_ids, skip_special_tokens=True)
    finally:
        model.config.output_attentions = prev_attn


@dataclass
class LayerAblationResult:
    """Result of ablating one layer (or one head) during generation."""

    layer: int
    head: int | None  # None = whole layer ablated
    generation: str
    has_lambda: bool
    lambda_count: int


def ablate_layers(
    model: Any,
    tokenizer: Any,
    prompt: str,
    info: ModelInfo,
    *,
    max_new_tokens: int = 30,
) -> tuple[str, list[LayerAblationResult]]:
    """Skip-ablate each layer and check if compilation survives.

    For each layer L, we register hooks that replace the layer's output
    with its input — effectively skipping it. Then we generate and check
    whether the output still contains lambda indicators.

    The "skip" ablation is cleaner than zeroing (which destroys the
    residual stream) — it removes the layer's contribution while
    preserving the residual.

    Returns (baseline_text, list_of_results).
    """
    # Baseline: generate without any ablation
    baseline = _generate(model, tokenizer, prompt, max_new_tokens)
    _LOG.info(
        "ablation.baseline", text=baseline[:100], has_lambda=_detect_lambda(baseline)
    )

    results: list[LayerAblationResult] = []

    # Access the transformer layers
    layers = _get_layers(model)

    for layer_idx in range(info.n_layers):
        captured_input: dict[str, Any] = {}

        def pre_hook(module: Any, args: Any, *, _cap: dict = captured_input) -> None:
            _cap["hidden"] = args[0].clone()

        def post_hook(
            module: Any, args: Any, output: Any, *, _cap: dict = captured_input
        ) -> Any:
            # Replace hidden states with input (skip layer).
            # Output may be a Tensor or a tuple depending on model config.
            if isinstance(output, tuple):
                return (_cap["hidden"],) + output[1:]
            return _cap["hidden"]

        h_pre = layers[layer_idx].register_forward_pre_hook(pre_hook)
        h_post = layers[layer_idx].register_forward_hook(post_hook)

        try:
            gen = _generate(model, tokenizer, prompt, max_new_tokens)
        finally:
            h_pre.remove()
            h_post.remove()

        has_l = _detect_lambda(gen)
        l_count = sum(gen.count(s) for s in LAMBDA_INDICATORS)

        results.append(
            LayerAblationResult(
                layer=layer_idx,
                head=None,
                generation=gen,
                has_lambda=has_l,
                lambda_count=l_count,
            )
        )

        status = "✓ survives" if has_l else "✗ BREAKS"
        _LOG.info(
            "ablation.layer",
            layer=layer_idx,
            status=status,
            lambda_count=l_count,
            gen=gen[:80],
        )

    return baseline, results


# ─────────────────────────── multi-head zeroing ───────────────────────


def zero_heads_generate(
    model: Any,
    tokenizer: Any,
    prompt: str,
    info: ModelInfo,
    heads: list[tuple[int, int]],
    *,
    max_new_tokens: int = 30,
) -> tuple[str, bool, int]:
    """Zero-ablate multiple heads simultaneously and generate.

    ``heads`` is a list of ``(layer, head)`` tuples to zero out.
    Returns ``(generation, has_lambda, lambda_count)``.
    """
    layers_module = _get_layers(model)
    head_dim = info.head_dim
    hooks = []

    try:
        for layer_idx, head_idx in heads:
            start = head_idx * head_dim
            end = start + head_dim

            def attn_hook(
                module: Any,
                args: Any,
                output: Any,
                *,
                _s: int = start,
                _e: int = end,
            ) -> Any:
                patched = output[0].clone()
                patched[:, :, _s:_e] = 0.0
                return (patched, *output[1:])

            attn = _get_self_attn(layers_module[layer_idx])
            h = attn.register_forward_hook(attn_hook)
            hooks.append(h)

        gen = _generate(model, tokenizer, prompt, max_new_tokens)
    finally:
        for h in hooks:
            h.remove()

    has_l = _detect_lambda(gen)
    l_count = sum(gen.count(s) for s in LAMBDA_INDICATORS)
    return gen, has_l, l_count


# ─────────────────────────── BOS residual patching ────────────────────


def capture_bos_residuals(
    model: Any,
    tokenizer: Any,
    prompt: str,
    info: ModelInfo,
) -> list[torch.Tensor]:
    """Forward a prompt and capture the residual stream at position 0.

    Returns a list of tensors, one per layer, each of shape
    ``(hidden_size,)`` — the hidden state at position 0 after each
    transformer layer.
    """
    layers_module = _get_layers(model)
    bos_residuals: list[torch.Tensor] = []
    hook_handles = []

    def make_hook(storage: list[torch.Tensor]) -> Any:
        def hook_fn(module: Any, args: Any, output: Any) -> None:
            # output is (hidden_states, ...) or just hidden_states
            hidden = output[0] if isinstance(output, tuple) else output
            storage.append(hidden[0, 0, :].detach().clone())

        return hook_fn

    try:
        for layer in layers_module:
            h = layer.register_forward_hook(make_hook(bos_residuals))
            hook_handles.append(h)

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        prev_attn = model.config.output_attentions
        model.config.output_attentions = False
        try:
            with torch.no_grad():
                model(**inputs)
        finally:
            model.config.output_attentions = prev_attn
    finally:
        for h in hook_handles:
            h.remove()

    return bos_residuals


def patch_bos_generate(
    model: Any,
    tokenizer: Any,
    prompt: str,
    info: ModelInfo,
    patch_layer: int,
    patch_value: torch.Tensor,
    *,
    max_new_tokens: int = 30,
) -> tuple[str, bool, int]:
    """Patch the BOS residual at a specific layer and generate.

    Hooks ``patch_layer`` to replace the hidden state at position 0
    with ``patch_value`` (captured from a different prompt). This tests
    whether that layer's contribution to the BOS composition register
    is necessary for compilation.

    Returns ``(generation, has_lambda, lambda_count)``.
    """
    layers_module = _get_layers(model)

    def bos_patch_hook(
        module: Any,
        args: Any,
        output: Any,
        *,
        _val: torch.Tensor = patch_value,
    ) -> Any:
        hidden = output[0] if isinstance(output, tuple) else output
        patched = hidden.clone()
        patched[0, 0, :] = _val
        if isinstance(output, tuple):
            return (patched, *output[1:])
        return patched

    h = layers_module[patch_layer].register_forward_hook(bos_patch_hook)
    try:
        gen = _generate(model, tokenizer, prompt, max_new_tokens)
    finally:
        h.remove()

    has_l = _detect_lambda(gen)
    l_count = sum(gen.count(s) for s in LAMBDA_INDICATORS)
    return gen, has_l, l_count


def ablate_heads(
    model: Any,
    tokenizer: Any,
    prompt: str,
    info: ModelInfo,
    *,
    target_layers: list[int] | None = None,
    max_new_tokens: int = 30,
) -> tuple[str, list[LayerAblationResult]]:
    """Zero-ablate individual attention heads within specified layers.

    For each head in each target layer, we hook the attention output
    projection to zero out that head's contribution, then generate and
    check whether compilation survives.

    If ``target_layers`` is None, all layers are tested (expensive:
    n_layers × n_heads forward passes).

    Returns (baseline_text, list_of_results).
    """
    if target_layers is None:
        target_layers = list(range(info.n_layers))

    baseline = _generate(model, tokenizer, prompt, max_new_tokens)

    results: list[LayerAblationResult] = []
    layers = _get_layers(model)
    head_dim = info.head_dim

    for layer_idx in target_layers:
        for head_idx in range(info.n_heads):
            # Hook the attention output to zero out this head's slice
            start = head_idx * head_dim
            end = start + head_dim

            def attn_hook(
                module: Any,
                args: Any,
                output: Any,
                *,
                _s: int = start,
                _e: int = end,
            ) -> Any:
                # output is (attn_output, attn_weights, past_kv)
                # attn_output shape: (batch, seq_len, hidden_size)
                patched = output[0].clone()
                patched[:, :, _s:_e] = 0.0
                return (patched,) + output[1:]

            h = _get_self_attn(layers[layer_idx]).register_forward_hook(attn_hook)

            try:
                gen = _generate(model, tokenizer, prompt, max_new_tokens)
            finally:
                h.remove()

            has_l = _detect_lambda(gen)
            l_count = sum(gen.count(s) for s in LAMBDA_INDICATORS)

            results.append(
                LayerAblationResult(
                    layer=layer_idx,
                    head=head_idx,
                    generation=gen,
                    has_lambda=has_l,
                    lambda_count=l_count,
                )
            )

        _LOG.info(
            "ablation.heads",
            layer=layer_idx,
            broken=[
                r.head for r in results if r.layer == layer_idx and not r.has_lambda
            ],
        )

    return baseline, results
