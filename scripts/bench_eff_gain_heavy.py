"""Heavy-shape efficiency-gain bench (large N × large K).

Same JSON schema as ``bench_eff_gain.py`` but exercises the regime
this codebase actually cares about: N up to 10M crossed with K up to
200K. Each cell stresses a different bottleneck:

 * Big-K (K≥65K) at moderate D: amortises the K loop overhead;
   for the old kernel this also forced N=BM rows to keep more
   per-thread argmin state alive, so the gain from dropping x_sq
   (free regs / less HBM) should be biggest here.
 * Heavy N (N=20M): probes the X-read bandwidth ceiling — the
   old kernel's x_sq HBM load is 2N·4B = ~160 MB for N=20M, which
   the new kernel skips entirely.
 * Wide-D (D=256/512, K=65K): compute-bound, expect Δ ~ in-register
   epilogue savings only.

Run as::

    CUDA_VISIBLE_DEVICES=4 python scripts/bench_eff_gain_heavy.py --out new_heavy.json
"""
from __future__ import annotations

import argparse
import gc
import inspect
import json

import torch


SHAPES = [
    # Heavy K ----------------------------------------------------------------
    (10_000_000,  64,  65_536),
    (10_000_000, 128,  65_536),
    (10_000_000, 256,  65_536),
    ( 5_000_000, 128,  65_536),
    ( 5_000_000, 256,  65_536),

    # Very-large K (codebook learning regime) --------------------------------
    ( 5_000_000,  64, 200_000),
    ( 2_500_000, 128, 200_000),
    ( 1_048_576, 128, 200_000),
    ( 1_048_576, 256, 200_000),

    # Heavy N at moderate K --------------------------------------------------
    (20_000_000,  64,  20_000),
    (20_000_000, 128,  20_000),

    # Wide-D × heavy K -------------------------------------------------------
    ( 1_048_576, 512,  65_536),
    ( 2_500_000, 512,  20_000),
]


def _bench_us(fn, repeats: int = 10, warmup: int = 3) -> float:
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

    needs_xsq = "x_sq" in inspect.signature(euclid_assign_triton).parameters
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

    cute_call()  # warm (autotune + JIT)

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
        try:
            t_tri, t_cute, needs_xsq = _bench_one(N, D, K, dtype)
        except Exception as exc:
            print(f"  SKIP: {exc!r}", flush=True)
            continue
        flops = 2.0 * N * K * D
        results.append({
            "N": N, "D": D, "K": K, "dtype": args.dtype,
            "triton_us": t_tri,
            "cutedsl_us": t_cute,
            "tflops_tri": flops / (t_tri * 1e-6) / 1e12,
            "tflops_cute": flops / (t_cute * 1e-6) / 1e12,
            "needs_xsq": needs_xsq,
        })
        print(f"  Triton:  {t_tri:>12.1f} us ({results[-1]['tflops_tri']:>5.1f} TFLOPs)",
              flush=True)
        print(f"  CuteDSL: {t_cute:>12.1f} us ({results[-1]['tflops_cute']:>5.1f} TFLOPs)",
              flush=True)
        # Incremental save (resilient to OOM later in grid).
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {len(results)} shapes -> {args.out}")


if __name__ == "__main__":
    main()
