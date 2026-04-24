"""TernaryLinear — ternary routing that learns through flip accumulation.

The ternary weights {-1, 0, +1} define routing topology. They evolve
during training through a lightweight accumulate-and-flip mechanism:

  1. Forward: ternary matmul via custom Metal kernel (add/sub only)
  2. Backward: STE computes gradient for ternary weights
  3. Gradient routes to a flip accumulator (not to the optimizer)
  4. Periodically: weights whose accumulator exceeds threshold FLIP
     one step (-1→0, 0→+1, +1→0, etc.) and the accumulator resets

Per-channel gamma provides continuous fine-tuning on top of the
discrete ternary routing. Gamma is trained normally with Adam.

Memory per ternary weight:
  Training:  1 byte (int8) + 4 bytes (fp32 accumulator) = 5 bytes
  Inference: 0.25 bytes (packed 2-bit)

License: MIT
"""

from __future__ import annotations

import math
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from verbum.v6.kernels import ternary_matmul, ternary_matmul_t


# ══════════════════════════════════════════════════════════════════════
# Ternary initialization
# ══════════════════════════════════════════════════════════════════════


def _ternary_init(out_features: int, in_features: int) -> tuple[mx.array, mx.array]:
    """Initialize ternary weights from Kaiming normal → quantize.

    Returns:
        w_q:   (out_features, in_features) int8 ternary {-1, 0, +1}
        gamma: (out_features,) float32 per-channel scale
    """
    # Kaiming normal: std = sqrt(2 / in_features)
    std = math.sqrt(2.0 / in_features)
    w_init = mx.random.normal((out_features, in_features)) * std

    # Per-channel absmean quantization
    gamma = mx.abs(w_init).mean(axis=-1)
    w_scaled = w_init / (mx.expand_dims(gamma, axis=-1) + 1e-8)
    w_q = mx.clip(mx.round(w_scaled), -1, 1).astype(mx.int8)

    return w_q, gamma


# ══════════════════════════════════════════════════════════════════════
# Ternary forward with custom VJP
# ══════════════════════════════════════════════════════════════════════


@mx.custom_function
def _ternary_linear_fwd(x: mx.array, w: mx.array, gamma: mx.array) -> mx.array:
    """Forward: y = ternary_matmul(x, w) * gamma

    Custom Metal kernel does add/sub only — no fp32 multiplies
    in the matmul. Gamma scaling is a cheap pointwise multiply.
    """
    y_pre = ternary_matmul(x, w)
    return y_pre * gamma


@_ternary_linear_fwd.vjp
def _ternary_linear_vjp(primals, cotangent, output):
    """Backward: STE for ternary weights, ternary matmul for grad_x.

    ∂L/∂x:     ternary_matmul_t(grad_out * gamma, w)  — also add/sub on Metal
    ∂L/∂w:     (grad_out * gamma).T @ x                — dense matmul → flip accumulator
    ∂L/∂gamma: sum(grad_out * y_pre, reduce_dims)      — per-channel
    """
    x, w, gamma = primals
    grad_out = cotangent

    # Scale grad_out by gamma once (used for both grad_x and grad_w)
    grad_scaled = grad_out * gamma

    # ∂L/∂x — ternary matmul backward (also add/sub on Metal)
    grad_x = ternary_matmul_t(grad_scaled, w)

    # ∂L/∂w — dense matmul for flip accumulator
    # Reshape to 2D for matmul: (*, N) x (*, K) → (N, K)
    gs_2d = grad_scaled.reshape(-1, grad_scaled.shape[-1])
    x_2d = x.reshape(-1, x.shape[-1])
    grad_w = gs_2d.T @ x_2d

    # ∂L/∂gamma — per-channel: recompute y_pre (cheaper than saving)
    y_pre = ternary_matmul(x, w)
    # Sum over all dims except last (output features)
    reduce_axes = tuple(range(grad_out.ndim - 1))
    grad_gamma = (grad_out * y_pre).sum(axis=reduce_axes)

    return grad_x, grad_w, grad_gamma


# ══════════════════════════════════════════════════════════════════════
# TernaryLinear — nn.Module with flip accumulation
# ══════════════════════════════════════════════════════════════════════


class TernaryLinear(nn.Module):
    """Linear layer with learnable ternary routing via flip accumulation.

    Forward: y = ternary_matmul(RMSNorm(x), W_int8) * gamma

    The ternary weights evolve through discrete flips, not continuous
    gradient descent. Each flip moves one step: -1→0, 0→±1, ±1→0.
    The accumulator captures gradient pressure; the threshold controls
    how much evidence is needed before committing to a flip.

    Args:
        in_features:  input dimension
        out_features: output dimension
        pre_norm:     if True, apply RMSNorm before projection
    """

    def __init__(self, in_features: int, out_features: int, pre_norm: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.pre_norm = pre_norm

        if pre_norm:
            self.norm = nn.RMSNorm(in_features)

        # Initialize: Kaiming → quantize → int8 weight + gamma
        w_q, gamma = _ternary_init(out_features, in_features)
        self.ternary_weight = w_q
        self.gamma = gamma

        # Flip accumulator — tracks gradient pressure per weight
        # Not a parameter (not trained by optimizer), but needs to persist.
        # Int8 with saturation at ±127: each micro-batch votes ±1, so
        # |accum| ≤ N_votes. Saturating at 127 means 127+ consecutive
        # votes in one direction = overwhelming consensus. Cuts training
        # memory from 5 bytes/weight (int8 + fp32) to 2 bytes/weight.
        self._flip_accum = mx.zeros(w_q.shape, dtype=mx.int8)

    def __call__(self, x: mx.array) -> mx.array:
        if self.pre_norm:
            x = self.norm(x)
        return _ternary_linear_fwd(x, self.ternary_weight, self.gamma)

    def ternary_stats(self) -> dict[str, float]:
        """Report ternary weight and gamma statistics."""
        w = self.ternary_weight
        total = w.size
        return {
            "sparsity": (w == 0).sum().item() / total,
            "pos_frac": (w == 1).sum().item() / total,
            "neg_frac": (w == -1).sum().item() / total,
            "gamma_mean": self.gamma.mean().item(),
            "gamma_std": mx.sqrt(mx.var(self.gamma)).item(),
            "accum_mean": mx.abs(self._flip_accum.astype(mx.float32)).mean().item(),
            "accum_max": mx.abs(self._flip_accum.astype(mx.float32)).max().item(),
        }


# ══════════════════════════════════════════════════════════════════════
# TernaryFFN — ternary feed-forward network
# ══════════════════════════════════════════════════════════════════════


class TernaryFFN(nn.Module):
    """Feed-forward network with ternary routing.

    RMSNorm → TernaryLinear(up) → GELU → TernaryLinear(down) + residual
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.up = TernaryLinear(d_model, d_ff, pre_norm=True)
        self.down = TernaryLinear(d_ff, d_model, pre_norm=False)
        self.dropout = nn.Dropout(dropout)

    def __call__(self, x: mx.array) -> mx.array:
        return x + self.dropout(self.down(nn.gelu(self.up(x))))


# ══════════════════════════════════════════════════════════════════════
# Flip accumulation utilities
# ══════════════════════════════════════════════════════════════════════


def zero_ternary_grads(model: nn.Module, grads: dict) -> dict:
    """Zero out ternary_weight gradients in the grad pytree.

    Ternary weight gradients feed the flip accumulator (sign-based),
    not the optimizer. Including them in clip_grad_norm poisons the
    continuous parameter updates: a single large ternary gradient
    dominates the total norm, clipping continuous params to near-zero.

    Call this AFTER accumulate_flips and BEFORE clip_grad_norm.
    """
    # Collect paths to ternary weight parameters
    ternary_paths: set[str] = set()
    for path, module in _walk_ternary_modules(model):
        ternary_paths.add(f"{path}.ternary_weight" if path else "ternary_weight")

    def _zero(path_prefix: str, tree):
        if isinstance(tree, dict):
            return {
                k: _zero(f"{path_prefix}.{k}" if path_prefix else k, v)
                for k, v in tree.items()
            }
        elif isinstance(tree, list):
            return [
                _zero(f"{path_prefix}.{i}" if path_prefix else str(i), v)
                for i, v in enumerate(tree)
            ]
        elif isinstance(tree, mx.array) and path_prefix in ternary_paths:
            return mx.zeros_like(tree)
        return tree

    return _zero("", grads)


def restore_ternary(model: nn.Module) -> None:
    """Re-cast any ternary weights back to int8 after optimizer update.

    The optimizer may cast int8 weights to float during its update step.
    This restores them to int8 (rounding to nearest integer, clamping to
    {-1, 0, +1}). Call after every optimizer.update().
    """
    def _walk(mod):
        if isinstance(mod, TernaryLinear):
            if mod.ternary_weight.dtype != mx.int8:
                mod.ternary_weight = mx.clip(
                    mx.round(mod.ternary_weight), -1, 1
                ).astype(mx.int8)
        if isinstance(mod, nn.Module):
            for name, child in mod.children().items():
                if isinstance(child, nn.Module):
                    _walk(child)
                elif isinstance(child, list):
                    for item in child:
                        if isinstance(item, nn.Module):
                            _walk(item)
    _walk(model)


def _walk_ternary_modules(model: nn.Module):
    """Yield (path, module) for all TernaryLinear modules in model."""
    for path, module in model.named_modules():
        if isinstance(module, TernaryLinear):
            yield path, module


def split_ternary_grads(
    grads: dict[str, Any],
    model: nn.Module,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split gradient pytree into ternary weight grads and continuous grads.

    Walks the model to identify which parameters are ternary weights
    (int8, in TernaryLinear modules). Their gradients route to the
    flip accumulator. All other gradients route to the optimizer.

    Args:
        grads: gradient pytree from mx.value_and_grad
        model: the model (to identify ternary vs continuous params)

    Returns:
        (ternary_grads, continuous_grads) — two pytrees with the same
        structure as grads, but with None for excluded parameters.
    """
    # Collect paths to ternary_weight parameters
    ternary_paths: set[str] = set()
    for path, module in _walk_ternary_modules(model):
        ternary_paths.add(f"{path}.ternary_weight" if path else "ternary_weight")

    def _split(path_prefix: str, grad_tree):
        if isinstance(grad_tree, dict):
            ternary = {}
            continuous = {}
            for key, val in grad_tree.items():
                child_path = f"{path_prefix}.{key}" if path_prefix else key
                t, c = _split(child_path, val)
                ternary[key] = t
                continuous[key] = c
            return ternary, continuous
        elif isinstance(grad_tree, list):
            ternary = []
            continuous = []
            for i, val in enumerate(grad_tree):
                child_path = f"{path_prefix}.{i}" if path_prefix else str(i)
                t, c = _split(child_path, val)
                ternary.append(t)
                continuous.append(c)
            return ternary, continuous
        else:
            # Leaf — check if this path is a ternary weight
            if path_prefix in ternary_paths:
                return grad_tree, None
            else:
                return None, grad_tree

    return _split("", grads)


def accumulate_flips(model: nn.Module, ternary_grads: dict[str, Any]) -> None:
    """Accumulate gradient direction votes for ternary weight flips.

    Uses sign(grad) rather than raw gradient magnitude. Each call
    adds +1 or -1 per weight, so after N calls |accum| ≤ N. This
    makes the accumulator scale-invariant and the threshold meaningful
    in units of "directional consensus across micro-batches."

    Call after loss backward, per micro-batch.

    Args:
        model: the model containing TernaryLinear modules
        ternary_grads: gradient pytree (full or ternary-only)
    """
    def _extract_grad(tree, path_parts):
        """Navigate the grad pytree to find the gradient at a given path."""
        node = tree
        for part in path_parts:
            if isinstance(node, dict):
                node = node.get(part)
            elif isinstance(node, list):
                node = node[int(part)]
            else:
                return None
            if node is None:
                return None
        return node

    accums = []
    for path, module in _walk_ternary_modules(model):
        parts = path.split(".") if path else []
        parts.append("ternary_weight")
        grad = _extract_grad(ternary_grads, parts)
        if grad is not None:
            # NaN guard: don't poison the accumulator with NaN gradients
            if mx.any(mx.isnan(grad)).item():
                continue
            # Sign-based accumulation: direction only, not magnitude.
            # Each micro-batch casts a vote (+1 or -1) per weight.
            # Int8 with saturating clip at ±127: 127+ consecutive votes
            # in one direction = overwhelming consensus. Beyond that,
            # additional votes don't add information.
            # Memory: 2 bytes/weight (int8 weight + int8 accum) vs 5.
            vote = mx.sign(grad).astype(mx.int8)
            module._flip_accum = mx.clip(
                module._flip_accum.astype(mx.int16) + vote.astype(mx.int16),
                -127, 127,
            ).astype(mx.int8)
            accums.append(module._flip_accum)

    # Materialize accumulators to prevent lazy graph buildup.
    # Without this, each call chains another addition node — after
    # 100 steps × 4 micro-batches × 147 modules the graph leaks GBs.
    if accums:
        mx.eval(*accums)


def compute_flip_threshold(model: nn.Module, target_pct: float) -> float:
    """Compute threshold to flip approximately target_pct of ternary weights.

    Uses the percentile of accumulator absolute values so that exactly
    target_pct fraction of weights exceed the threshold. This decouples
    the flip decision from accumulator scale.

    Args:
        model: the model containing TernaryLinear modules
        target_pct: fraction of weights to flip (e.g. 0.005 = 0.5%)

    Returns:
        Threshold value. Returns float('inf') if no valid accumulators.
    """
    import numpy as np
    chunks = []
    for _, module in _walk_ternary_modules(model):
        mx.eval(module._flip_accum)
        # Int8 accumulators can't be NaN — skip the guard
        chunks.append(mx.abs(module._flip_accum).astype(mx.int16).reshape(-1))
    if not chunks:
        return float("inf")
    all_abs = mx.concatenate(chunks)
    # Convert to numpy for percentile (mx doesn't have percentile)
    all_np = np.array(all_abs)
    pct = 100.0 * (1.0 - target_pct)
    return float(np.percentile(all_np, pct))


def normalize_shared_grads(model: nn.Module, grads: dict, n_passes: int = 5) -> dict:
    """Divide gradients of shared-across-passes modules by n_passes.

    The VSM runs 5 passes through the same shared weights (prep,
    stride_stack, consolidate, mod_projs, s4). Each pass contributes
    a gradient computed from a DIFFERENT ∂L/∂x magnitude (pass 0 sees
    accumulated gradient from all downstream; pass 4 sees only direct
    output gradient). Their sum oscillates wildly between steps.

    Dividing by n_passes turns this volatile sum into a stable average.
    This is the key fix for gradient norm instability — it lets Adam's
    running statistics (v_t) converge instead of chasing a moving target.

    Only affects continuous parameters (gamma, norm weights).
    Ternary weights are already zeroed by zero_ternary_grads.

    Shared:     prep, stride_stack, consolidate, mod_projs, s4
    Not shared: s3_passes (per-pass), meta_s3, meta_s4, embeds, norms
    """
    shared_prefixes = {"prep", "stride_stack", "consolidate", "mod_projs", "s4"}
    scale = 1.0 / n_passes

    def _scale(path: str, tree):
        if isinstance(tree, dict):
            return {k: _scale(f"{path}.{k}" if path else k, v)
                    for k, v in tree.items()}
        elif isinstance(tree, list):
            return [_scale(f"{path}.{i}" if path else str(i), v)
                    for i, v in enumerate(tree)]
        elif isinstance(tree, mx.array):
            top_key = path.split(".")[0] if path else ""
            if top_key in shared_prefixes:
                return tree * scale
            return tree
        return tree

    return _scale("", grads)


def apply_flips(model: nn.Module, threshold: int = 50, max_flip_pct: float = 0.001) -> int:
    """Flip ternary weights where accumulated consensus exceeds threshold.

    Like synaptic plasticity: each weight flips only when IT has
    accumulated enough directional evidence. But capped: at most
    max_flip_pct of total ternary weights can flip per call, to prevent
    catastrophic mass mutation when early-training gradients are globally
    coherent (every weight agrees because the model knows nothing).

    When more weights cross the threshold than the cap allows, only the
    strongest consensus (highest |accum|) flip. This preserves the
    synaptic metaphor: strongest evidence goes first.

    Each flip moves one step in the gradient direction:
      -1 + positive pressure → 0
       0 + positive pressure → +1
      +1 + negative pressure → 0
       0 + negative pressure → -1

    Args:
        model: the model containing TernaryLinear modules
        threshold: minimum |accumulator| to trigger a flip (vote units)
        max_flip_pct: maximum fraction of ternary weights to flip per call
                      (0.001 = 0.1% = ~35K of 35M weights)

    Returns:
        Total number of weights flipped across all modules.
    """
    # Step 1: collect all accumulators that exceed threshold
    candidates = []  # [(module, accum_abs_flat)]
    total_ternary = 0
    for _, module in _walk_ternary_modules(model):
        total_ternary += module.ternary_weight.size
        accum_abs = mx.abs(module._flip_accum.astype(mx.int16))
        candidates.append((module, accum_abs))

    max_flips = int(total_ternary * max_flip_pct)

    # Step 2: find effective threshold (raise above base if too many qualify)
    # Count qualifying per threshold using cheap per-module sums (no big concat).
    def _count_above(t):
        return sum((a > t).sum().item() for _, a in candidates)

    n_qualifying = _count_above(threshold)
    effective_threshold = threshold

    if n_qualifying > max_flips and max_flips > 0:
        # Too many qualify — binary search for threshold that caps at max_flips.
        # Range: [threshold, 127] (int8 accum saturates at 127).
        lo, hi = threshold, 127
        while lo < hi:
            mid = (lo + hi) // 2
            if _count_above(mid) > max_flips:
                lo = mid + 1
            else:
                hi = mid
        effective_threshold = lo

    # Step 3: apply flips with effective threshold
    total_flipped = 0
    mutated = []

    for module, accum_abs in candidates:
        mask = accum_abs > int(effective_threshold)
        n_flipped = mask.sum().item()

        if n_flipped > 0:
            direction = mx.sign(module._flip_accum.astype(mx.int16)).astype(mx.int8)
            current = module.ternary_weight.astype(mx.int16)
            new_vals = mx.clip(current + direction.astype(mx.int16), -1, 1).astype(mx.int8)

            module.ternary_weight = mx.where(mask, new_vals, module.ternary_weight)
            module._flip_accum = mx.where(mask, mx.zeros_like(module._flip_accum), module._flip_accum)

            mutated.extend([module.ternary_weight, module._flip_accum])
            total_flipped += int(n_flipped)

    if mutated:
        mx.eval(*mutated)

    return total_flipped


# ══════════════════════════════════════════════════════════════════════
# Per-group flip functions (VSM-modulated)
# ══════════════════════════════════════════════════════════════════════


def _classify_group(path: str) -> str:
    """Map a TernaryLinear module path to its VSM group.

    Order matters: check longer/more-specific prefixes first to avoid
    'meta_s3' matching 's3' before 'meta'.
    """
    # Check meta first (meta_s3, meta_s4 are control, not S3/S4 operations)
    if path.startswith("meta_s3") or path.startswith("meta_s4") or path.startswith("meta."):
        return "meta"
    for gk in ["prep", "stride_stack", "consolidate", "mod_projs", "s4.", "s3_"]:
        if gk in path:
            return gk.rstrip("._")
    return "other"


def apply_flips_per_group(
    model: nn.Module,
    group_targets: dict[str, float],
) -> dict[str, int]:
    """Apply flips with per-group adaptive thresholds.

    Instead of one global threshold, each VSM group gets its own
    flip target percentage. The threshold is computed per-group
    from the accumulator distribution within that group.

    Args:
        model: the model containing TernaryLinear modules
        group_targets: {group_name: target_pct} from VSM signal modulation

    Returns:
        {group_name: n_flipped} — number of weights flipped per group
    """
    import numpy as np

    # Step 1: collect modules by group
    groups: dict[str, list[tuple[str, TernaryLinear]]] = {}
    for path, module in _walk_ternary_modules(model):
        group = _classify_group(path)
        groups.setdefault(group, []).append((path, module))

    # Step 2: compute per-group thresholds and apply
    group_flipped: dict[str, int] = {}
    mutated = []

    for group, modules in groups.items():
        target_pct = group_targets.get(group, 0.005)

        # Collect accumulators for this group (int8 — no NaN possible)
        chunks = []
        for _, mod in modules:
            mx.eval(mod._flip_accum)
            chunks.append(mx.abs(mod._flip_accum.astype(mx.int16)).reshape(-1))

        if not chunks:
            group_flipped[group] = 0
            continue

        # Compute group-specific threshold
        all_abs = mx.concatenate(chunks)
        all_np = np.array(all_abs)
        pct = 100.0 * (1.0 - target_pct)
        threshold = float(np.percentile(all_np, pct))

        # Apply flips for this group
        n_flipped = 0
        for _, mod in modules:
            accum_abs = mx.abs(mod._flip_accum.astype(mx.int16)).astype(mx.int8)
            mask = accum_abs > int(threshold)
            n = mask.sum().item()
            if n > 0:
                direction = mx.sign(mod._flip_accum.astype(mx.int16)).astype(mx.int8)
                current = mod.ternary_weight.astype(mx.int16)
                new_vals = mx.clip(current + direction.astype(mx.int16), -1, 1).astype(mx.int8)
                mod.ternary_weight = mx.where(mask, new_vals, mod.ternary_weight)
                mod._flip_accum = mx.where(mask, mx.zeros_like(mod._flip_accum), mod._flip_accum)
                mutated.extend([mod.ternary_weight, mod._flip_accum])
                n_flipped += int(n)

        group_flipped[group] = n_flipped

    if mutated:
        mx.eval(*mutated)

    return group_flipped
