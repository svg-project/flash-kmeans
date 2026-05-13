"""Compact bench harness for the no-xsq efficiency-gain comparison.

Runs the production-path (heuristic) Triton + CuteDSL assign kernels
over a small shape grid and writes a JSON with per-shape latencies.
The companion script in the pre-change worktree
(/tmp/flash-kmeans-old/scripts/bench_eff_gain.py) reads the same JSON
schema, so the two can be diffed cleanly to produce the
``old vs new`` table.

Usage::

    # in this worktree (new, no-xsq):
    CUDA_VISIBLE_DEVICES=4 python scripts/bench_eff_gain.py --out new.json
    # in the old worktree (with x_sq):
    CUDA_VISIBLE_DEVICES=4 python scripts/bench_eff_gain.py --out old.json
"""
from __future__ import annotations

import argparse
import gc
import json
import time

import torch


SHAPES = [
    # (N, D, K) — same as bench_heavy_assign.py headline set, plus a couple
    # of lighter shapes so we can also see the launch-overhead regime.
    (10_000_000, 64,  20_000),
    (10_000_000, 128, 20_000),
    (10_000_000, 256, 20_000),
    (5_000_000,  256, 20_000),
    (2_500_000,  512, 20_000),
    (1_048_576,  128, 1024),
    (1_048_576,  256, 4096),
    (65_536,     128, 4096),
    (65_536,     256, 65_536),
]


def _bench_us(fn, repeats: int = 20, warmup: int = 5) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(repeats):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / repeats * 1000  # us


def _bench_one(N: int, D: int, K: int, dtype: torch.dtype):
    from flash_kmeans.assign_euclid_triton import euclid_assign_triton
    from flash_kmeans.cutedsl_impl import cutedsl_assign_euclid

    g = torch.Generator(device="cuda")
    g.manual_seed(0)
    x = torch.randn(1, N, D, device="cuda", dtype=dtype, generator=g)
    cents = torch.randn(1, K, D, device="cuda", dtype=dtype, generator=g)
    c_sq = (cents.float() ** 2).sum(-1).view(K).contiguous()
    out = torch.empty((1, N), device="cuda", dtype=torch.int32)

    # The presence of x_sq in the API differs between the two worktrees.
    # We sniff it dynamically so the same script runs in both.
    import inspect
    sig_tri = inspect.signature(euclid_assign_triton).parameters
    sig_cute = inspect.signature(cutedsl_assign_euclid).parameters
    needs_xsq = "x_sq" in sig_tri  # both APIs are in sync per worktree

    if needs_xsq:
        x_sq = (x.float() ** 2).sum(-1)

        def tri_call():
            euclid_assign_triton(x, cents, x_sq, out=out, c_sq=c_sq.view(1, K))

        def cute_call():
            cutedsl_assign_euclid(x, cents, x_sq, out=out, c_sq=c_sq)
    else:
        def tri_call():
            euclid_assign_triton(x, cents, out=out, c_sq=c_sq.view(1, K))

        def cute_call():
            cutedsl_assign_euclid(x, cents, out=out, c_sq=c_sq)

    # Warm CuteDSL once — its first call compiles.
    cute_call()

    t_tri = _bench_us(tri_call)
    t_cute = _bench_us(cute_call)

    del x, cents, c_sq, out
    if needs_xsq:
        del x_sq
    gc.collect()
    torch.cuda.empty_cache()

    return t_tri, t_cute, needs_xsq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--dtype", choices=["fp16", "bf16"], default="bf16")
    args = ap.parse_args()

    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]

    results = []
    for N, D, K in SHAPES:
        print(f"=== N={N:,} D={D} K={K:,} dtype={args.dtype} ===", flush=True)
        t_tri, t_cute, needs_xsq = _bench_one(N, D, K, dtype)
        flops = 2.0 * N * K * D
        results.append({
            "N": N, "D": D, "K": K, "dtype": args.dtype,
            "triton_us": t_tri,
            "cutedsl_us": t_cute,
            "tflops_tri": flops / (t_tri * 1e-6) / 1e12,
            "tflops_cute": flops / (t_cute * 1e-6) / 1e12,
            "needs_xsq": needs_xsq,
        })
        print(f"  Triton:  {t_tri:>10.1f} us ({results[-1]['tflops_tri']:>5.1f} TFLOPs)",
              flush=True)
        print(f"  CuteDSL: {t_cute:>10.1f} us ({results[-1]['tflops_cute']:>5.1f} TFLOPs)",
              flush=True)

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {len(results)} shapes -> {args.out}")


if __name__ == "__main__":
    main()
