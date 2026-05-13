"""Heavy-workload assign benchmark: N up to 10M, K up to 20K.

Reports wall-clock latency, achieved TFLOPs (vs Hopper bf16 WGMMA peak
~989 TFLOPs/s) and achieved HBM BW (vs H200 peak ~4.8 TB/s) for both
the Triton K-streaming kernel and the new CuteDSL fused TMA+WGMMA
kernel. Sweeps a few (BM, BN) configs through the explicit
``cute_tile=`` knob so we can see whether the static heuristic is
picking the right one for these very-large-K shapes.

Usage:
    python examples/bench_heavy_assign.py
    python examples/bench_heavy_assign.py --shapes 10000000,128,20000
"""
from __future__ import annotations

import argparse
import gc
import time
from typing import Optional

import torch


# Hopper SM90 fp16/bf16 WGMMA peak (per H200 datasheet).
WGMMA_PEAK_TFLOPS = 989.0
HBM_PEAK_TBPS = 4.8


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


def _flops_for_assign(N: int, D: int, K: int) -> float:
    """Cross-term GEMM dominates: 2*N*K*D FMAs.

    The dist + argmin epilogue is O(N*K) FLOPs which is negligible
    against the GEMM term once D >= 16. Reporting only the GEMM term
    is the standard convention (matches what cuBLAS/cutlass GEMM
    benchmarks report).
    """
    return 2.0 * N * K * D


def _hbm_bytes_for_assign(N: int, D: int, K: int, dtype_bytes: int) -> float:
    """HBM bytes the kernel is *forced* to read/write at minimum.

    Optimal: each X element read once across all CTAs, each centroid
    element read once per CTA (or fewer with multicast / L2 caching),
    each output id written once.

    Lower-bound (assumes K*D fits in L2 so centroids are read once per
    SM run, not per CTA): N*D*dtype + K*D*dtype + N*4 (out).
    Upper-bound (no L2 caching, every CTA re-reads C): same X read +
    (N/BM)*K*D*dtype centroids.

    Returns the lower bound — this is what you'd get if the
    centroids fully fit in L2, which is the typical regime for
    flash-kmeans (K*D up to a few MB).
    """
    return N * D * dtype_bytes + K * D * dtype_bytes + N * 4


def _allocate(N: int, D: int, K: int, dtype: torch.dtype, seed: int = 0):
    g = torch.Generator(device="cuda")
    g.manual_seed(seed)
    x = torch.randn(1, N, D, device="cuda", dtype=dtype, generator=g)
    cents = torch.randn(1, K, D, device="cuda", dtype=dtype, generator=g)
    c_sq = (cents.float() ** 2).sum(-1).view(K).contiguous()
    out = torch.empty((1, N), device="cuda", dtype=torch.int32)
    return x, cents, c_sq, out


def _try_cute_tile(
    cute_tile: Optional[tuple],
    x, cents, c_sq, out,
    repeats: int,
):
    from flash_kmeans.cutedsl_impl import cutedsl_assign_euclid

    def fn():
        cutedsl_assign_euclid(x, cents, out=out, c_sq=c_sq, cute_tile=cute_tile)

    # Sanity: warmup also surfaces compile / SMEM errors as exceptions.
    try:
        fn()
    except Exception as exc:
        return None, str(exc)
    return _bench_us(fn, repeats=repeats), None


def run_one(
    N: int, D: int, K: int, dtype: torch.dtype, repeats: int, dtype_bytes: int,
    cute_tiles: list,
):
    print(f"\n=== N={N:,} D={D} K={K:,} dtype={str(dtype).split('.')[-1]} ===")
    x, cents, c_sq, out = _allocate(N, D, K, dtype)
    flops = _flops_for_assign(N, D, K)
    hbm_lo = _hbm_bytes_for_assign(N, D, K, dtype_bytes)

    print(f"  Theoretical work: {flops/1e12:.2f} TFLOPs / call, "
          f"~{hbm_lo/1e9:.2f} GB HBM (lower bound, K*D in L2)")

    # ---- Triton ----------------------------------------------------------
    from flash_kmeans.assign_euclid_triton import euclid_assign_triton
    t_tri_us = _bench_us(
        lambda: euclid_assign_triton(x, cents, out=out, c_sq=c_sq.view(1, K)),
        repeats=repeats,
    )
    tri_tflops = flops / (t_tri_us * 1e-6) / 1e12
    tri_bw = hbm_lo / (t_tri_us * 1e-6) / 1e12
    print(f"  Triton:                         {t_tri_us:>8.0f} us | "
          f"{tri_tflops:>5.1f} TFLOPs ({tri_tflops/WGMMA_PEAK_TFLOPS*100:>4.1f}% peak) | "
          f"{tri_bw:>4.2f} TB/s ({tri_bw/HBM_PEAK_TBPS*100:>4.1f}% HBM)")

    # ---- CuteDSL: heuristic + explicit tile sweep ----------------------------
    # heuristic
    from flash_kmeans.cutedsl_impl import cutedsl_assign_euclid
    cutedsl_assign_euclid(x, cents, out=out, c_sq=c_sq)  # warm
    t_cute_h_us = _bench_us(
        lambda: cutedsl_assign_euclid(x, cents, out=out, c_sq=c_sq),
        repeats=repeats,
    )
    h_tflops = flops / (t_cute_h_us * 1e-6) / 1e12
    h_bw = hbm_lo / (t_cute_h_us * 1e-6) / 1e12
    print(f"  CuteDSL (heuristic):            {t_cute_h_us:>8.0f} us | "
          f"{h_tflops:>5.1f} TFLOPs ({h_tflops/WGMMA_PEAK_TFLOPS*100:>4.1f}% peak) | "
          f"{h_bw:>4.2f} TB/s ({h_bw/HBM_PEAK_TBPS*100:>4.1f}% HBM) | "
          f"{t_tri_us/t_cute_h_us:>4.2f}x vs Triton")

    # explicit tile sweep
    for bm, bn in cute_tiles:
        t_us, err = _try_cute_tile((bm, bn), x, cents, c_sq, out, repeats)
        if t_us is None:
            print(f"  CuteDSL (BM={bm},BN={bn:>3}):           SKIP   ({err.splitlines()[0]})")
            continue
        c_tflops = flops / (t_us * 1e-6) / 1e12
        c_bw = hbm_lo / (t_us * 1e-6) / 1e12
        print(f"  CuteDSL (BM={bm},BN={bn:>3}):           {t_us:>8.0f} us | "
              f"{c_tflops:>5.1f} TFLOPs ({c_tflops/WGMMA_PEAK_TFLOPS*100:>4.1f}% peak) | "
              f"{c_bw:>4.2f} TB/s ({c_bw/HBM_PEAK_TBPS*100:>4.1f}% HBM) | "
              f"{t_tri_us/t_us:>4.2f}x vs Triton")

    del x, cents, c_sq, out
    gc.collect()
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--shapes",
        type=str,
        default=None,
        help="Comma-separated triplets, e.g. '10000000,128,20000;5000000,256,8192'",
    )
    ap.add_argument("--dtype", choices=["fp16", "bf16"], default="bf16")
    ap.add_argument("--repeats", type=int, default=10)
    args = ap.parse_args()

    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]
    dtype_bytes = 2

    if args.shapes is not None:
        shapes = []
        for s in args.shapes.split(";"):
            n, d, k = (int(x) for x in s.split(","))
            shapes.append((n, d, k))
    else:
        shapes = [
            # The heavy headline workload requested in the chat.
            (10_000_000, 64,  20_000),
            (10_000_000, 128, 20_000),
            (10_000_000, 256, 20_000),
            # Larger-D heavy variants for completeness (D=384/512 less common
            # but force compute-boundedness so we can read FLOPs cleanly).
            (5_000_000,  256, 20_000),
            (2_500_000,  512, 20_000),
        ]

    # Tile candidates we want to A/B for very-large-K. The default
    # heuristic picks (128, 128) for D >= 96 K large; (128, 256) and
    # (128, 64) are the natural alternatives. We skip (64, 64) because
    # the heuristic never picks it for these shapes.
    cute_tiles = [(128, 128), (128, 256), (128, 64), (64, 128)]

    print(f"Hopper bf16 WGMMA peak ≈ {WGMMA_PEAK_TFLOPS} TFLOPs/s, "
          f"H200 HBM peak ≈ {HBM_PEAK_TBPS} TB/s")
    for N, D, K in shapes:
        run_one(N, D, K, dtype, args.repeats, dtype_bytes, cute_tiles)


if __name__ == "__main__":
    main()
