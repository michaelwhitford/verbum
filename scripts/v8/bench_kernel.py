#!/usr/bin/env python3
"""Benchmark TernaryLinear (mx.quantized_matmul, 2-bit AMX) at v8 dimensions.

Measures throughput of the quantized_matmul path — both forward and
backward through x — at:
  - d_model=1024, d_ff=4096  (v8 target dimensions)
  - Various batch sizes (1, 8, 32, 128, 512)

Usage:
    cd ~/src/verbum
    uv run python scripts/v8/bench_kernel.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import mlx.core as mx
from ternary import TernaryLinear, pack_ternary_mlx


def bench_one(name: str, fn, warmup: int = 5, iters: int = 50):
    """Benchmark a callable, return timing statistics dict."""
    # Warmup
    for _ in range(warmup):
        result = fn()
        mx.eval(result)

    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        result = fn()
        mx.eval(result)
        times.append((time.perf_counter() - t0) * 1000)

    times.sort()
    median = times[len(times) // 2]
    mean = sum(times) / len(times)
    p10 = times[len(times) // 10]
    p90 = times[9 * len(times) // 10]
    return {"name": name, "median_ms": median, "mean_ms": mean, "p10_ms": p10, "p90_ms": p90}


def _make_layer(N: int, K: int) -> TernaryLinear:
    """Build a pre_norm=False TernaryLinear(K, N) and eval it."""
    layer = TernaryLinear(in_features=K, out_features=N, pre_norm=False)
    mx.eval(layer.weight, layer.gamma)
    return layer


def main():
    print("=" * 70)
    print("  TernaryLinear Benchmark — mx.quantized_matmul (2-bit AMX)")
    print("  v8 target dimensions: d_model=1024, d_ff=4096")
    print("=" * 70)

    d_model = 1024
    d_ff = 4096
    batch_sizes = [1, 8, 32, 128, 512]

    # Build weight layers once
    attn_layer  = _make_layer(N=d_model, K=d_model)   # 1024 → 1024
    ffn_up_layer  = _make_layer(N=d_ff,   K=d_model)  # 1024 → 4096
    ffn_dn_layer  = _make_layer(N=d_model, K=d_ff)    # 4096 → 1024

    print(f"\nWeight shapes (uint32 packed, 16 values/element):")
    print(f"  Attention : weight={attn_layer.weight.shape}   gamma={attn_layer.gamma.shape}")
    print(f"  FFN up    : weight={ffn_up_layer.weight.shape}   gamma={ffn_up_layer.gamma.shape}")
    print(f"  FFN down  : weight={ffn_dn_layer.weight.shape}  gamma={ffn_dn_layer.gamma.shape}")

    all_results = []

    for M in batch_sizes:
        print(f"\n{'─'*70}")
        seq_desc = f"batch=1 × seq={M}" if M <= 512 else f"M={M}"
        print(f"  M={M} positions  ({seq_desc})")
        print(f"{'─'*70}")

        x_dm  = mx.random.normal((M, d_model))  # inputs to attn / ffn_up
        x_dff = mx.random.normal((M, d_ff))     # input to ffn_down
        mx.eval(x_dm, x_dff)

        # ── Forward: y = quantized_matmul(x, W, scales, biases) ──────────────

        r = bench_one(
            f"fwd attn  M={M:4d} (1024→1024)",
            lambda: attn_layer(x_dm),
        )
        ops = 2 * M * d_model * d_model
        r["gops"] = ops / (r["median_ms"] * 1e6)
        all_results.append(r)
        print(f"  FWD attn  (1024→1024): {r['median_ms']:7.3f} ms  {r['gops']:8.1f} GOP/s")

        r = bench_one(
            f"fwd ffn↑  M={M:4d} (1024→4096)",
            lambda: ffn_up_layer(x_dm),
        )
        ops = 2 * M * d_ff * d_model
        r["gops"] = ops / (r["median_ms"] * 1e6)
        all_results.append(r)
        print(f"  FWD ffn↑  (1024→4096): {r['median_ms']:7.3f} ms  {r['gops']:8.1f} GOP/s")

        r = bench_one(
            f"fwd ffn↓  M={M:4d} (4096→1024)",
            lambda: ffn_dn_layer(x_dff),
        )
        ops = 2 * M * d_model * d_ff
        r["gops"] = ops / (r["median_ms"] * 1e6)
        all_results.append(r)
        print(f"  FWD ffn↓  (4096→1024): {r['median_ms']:7.3f} ms  {r['gops']:8.1f} GOP/s")

        # ── Forward+Backward through x (grad_x via autograd) ─────────────────

        def fwd_bwd_attn():
            g = mx.grad(lambda x: attn_layer(x).sum())(x_dm)
            return g

        r = bench_one(f"fwd+bwd attn  M={M:4d} (1024→1024)", fwd_bwd_attn)
        ops = 2 * 2 * M * d_model * d_model  # fwd + bwd approx 2× fwd
        r["gops"] = ops / (r["median_ms"] * 1e6)
        all_results.append(r)
        print(f"  FWD+BWD attn (1024→1024): {r['median_ms']:7.3f} ms  {r['gops']:8.1f} GOP/s")

        def fwd_bwd_ffn_up():
            g = mx.grad(lambda x: ffn_up_layer(x).sum())(x_dm)
            return g

        r = bench_one(f"fwd+bwd ffn↑  M={M:4d} (1024→4096)", fwd_bwd_ffn_up)
        ops = 2 * 2 * M * d_ff * d_model
        r["gops"] = ops / (r["median_ms"] * 1e6)
        all_results.append(r)
        print(f"  FWD+BWD ffn↑ (1024→4096): {r['median_ms']:7.3f} ms  {r['gops']:8.1f} GOP/s")

    # Summary table
    print(f"\n{'='*70}")
    print(f"  Summary")
    print(f"{'='*70}")
    print(f"  {'Name':<42} {'Median':>8} {'P10':>8} {'P90':>8} {'GOP/s':>8}")
    for r in all_results:
        print(
            f"  {r['name']:<42} {r['median_ms']:7.3f}ms "
            f"{r['p10_ms']:7.3f}ms {r['p90_ms']:7.3f}ms {r['gops']:7.1f}"
        )


if __name__ == "__main__":
    main()
