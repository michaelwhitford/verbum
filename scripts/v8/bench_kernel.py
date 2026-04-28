#!/usr/bin/env python3
"""Benchmark ternary matmul kernels at v8 target dimensions.

Measures throughput of packed forward and transpose kernels at:
  - d_model=1024, d_ff=4096 (v8 target dimensions)
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
from ternary import (
    pack_ternary,
    ternary_matmul_packed,
    ternary_matmul_t_packed,
)


def bench_one(name: str, fn, warmup: int = 5, iters: int = 50):
    """Benchmark a callable, return median time in ms."""
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


def main():
    print("=" * 70)
    print("  Ternary Kernel Benchmark — v8 target dimensions")
    print("=" * 70)

    # v8 dimensions
    configs = [
        # (label, M, N, K) — M=batch*seq positions
        # Attention projections: d_model → d_model (1024 → 1024)
        # FFN projections: d_model → d_ff (1024 → 4096) and d_ff → d_model (4096 → 1024)
    ]

    batch_sizes = [1, 8, 32, 128, 512]
    d_model = 1024
    d_ff = 4096

    # Generate weight matrices (ternary, packed)
    w_attn_int8 = mx.random.randint(-1, 2, (d_model, d_model)).astype(mx.int8)  # (N=1024, K=1024)
    w_ffn_up_int8 = mx.random.randint(-1, 2, (d_ff, d_model)).astype(mx.int8)   # (N=4096, K=1024)
    w_ffn_down_int8 = mx.random.randint(-1, 2, (d_model, d_ff)).astype(mx.int8) # (N=1024, K=4096)

    w_attn = pack_ternary(w_attn_int8)       # (1024, 256)
    w_ffn_up = pack_ternary(w_ffn_up_int8)   # (4096, 256)
    w_ffn_down = pack_ternary(w_ffn_down_int8)  # (1024, 1024)
    mx.eval(w_attn, w_ffn_up, w_ffn_down)

    print(f"\nWeight shapes:")
    print(f"  Attention:  {w_attn_int8.shape} → packed {w_attn.shape}")
    print(f"  FFN up:     {w_ffn_up_int8.shape} → packed {w_ffn_up.shape}")
    print(f"  FFN down:   {w_ffn_down_int8.shape} → packed {w_ffn_down.shape}")

    all_results = []

    for M in batch_sizes:
        print(f"\n{'─'*70}")
        print(f"  M={M} positions (e.g., batch={M // 512 if M >= 512 else 1} × seq={min(M, 512)})")
        print(f"{'─'*70}")

        x_attn = mx.random.normal((M, d_model))    # for attention proj
        x_ffn_up = mx.random.normal((M, d_model))   # for FFN gate/up
        x_ffn_down = mx.random.normal((M, d_ff))     # for FFN down
        mx.eval(x_attn, x_ffn_up, x_ffn_down)

        # Forward: x @ W.T
        r = bench_one(
            f"fwd attn  M={M} N={d_model} K={d_model}",
            lambda: ternary_matmul_packed(x_attn, w_attn, d_model),
        )
        ops = 2 * M * d_model * d_model  # multiply-add equivalents
        r["gops"] = ops / (r["median_ms"] * 1e6)
        all_results.append(r)
        print(f"  FWD attn  (1024→1024): {r['median_ms']:7.2f} ms  {r['gops']:7.1f} GOP/s")

        r = bench_one(
            f"fwd ffn_up M={M} N={d_ff} K={d_model}",
            lambda: ternary_matmul_packed(x_ffn_up, w_ffn_up, d_model),
        )
        ops = 2 * M * d_ff * d_model
        r["gops"] = ops / (r["median_ms"] * 1e6)
        all_results.append(r)
        print(f"  FWD ffn↑ (1024→4096): {r['median_ms']:7.2f} ms  {r['gops']:7.1f} GOP/s")

        r = bench_one(
            f"fwd ffn_dn M={M} N={d_model} K={d_ff}",
            lambda: ternary_matmul_packed(x_ffn_down, w_ffn_down, d_ff),
        )
        ops = 2 * M * d_model * d_ff
        r["gops"] = ops / (r["median_ms"] * 1e6)
        all_results.append(r)
        print(f"  FWD ffn↓ (4096→1024): {r['median_ms']:7.2f} ms  {r['gops']:7.1f} GOP/s")

        # Transpose: x @ W (for backward)
        r = bench_one(
            f"bwd attn  M={M} N={d_model} K={d_model}",
            lambda: ternary_matmul_t_packed(x_attn, w_attn, d_model),
        )
        ops = 2 * M * d_model * d_model
        r["gops"] = ops / (r["median_ms"] * 1e6)
        all_results.append(r)
        print(f"  BWD attn  (1024→1024): {r['median_ms']:7.2f} ms  {r['gops']:7.1f} GOP/s")

        r = bench_one(
            f"bwd ffn_up M={M} N={d_ff} K={d_model}",
            lambda: ternary_matmul_t_packed(
                mx.random.normal((M, d_ff)), w_ffn_up, d_model
            ),
        )
        ops = 2 * M * d_model * d_ff
        r["gops"] = ops / (r["median_ms"] * 1e6)
        all_results.append(r)
        print(f"  BWD ffn↑ (4096→1024): {r['median_ms']:7.2f} ms  {r['gops']:7.1f} GOP/s")

    # Summary
    print(f"\n{'='*70}")
    print(f"  Summary")
    print(f"{'='*70}")
    print(f"  {'Name':<40} {'Median':>8} {'P10':>8} {'P90':>8} {'GOP/s':>8}")
    for r in all_results:
        print(f"  {r['name']:<40} {r['median_ms']:7.2f}ms {r['p10_ms']:7.2f}ms {r['p90_ms']:7.2f}ms {r['gops']:7.1f}")


if __name__ == "__main__":
    main()
