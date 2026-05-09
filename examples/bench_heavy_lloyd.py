"""Heavy-workload end-to-end Lloyd loop benchmark.

Measures CuteDSL Lloyd vs all-Triton Lloyd on N up to 10M, K up to 20K.
The Lloyd iter timing absorbs assign + sort + centroid update +
finalize. Reports per-iter wall time and assign-only TFLOPs to make it
easy to see how much of the assign speedup translates to end-to-end.
"""
from __future__ import annotations

import argparse
import gc
import time

import torch


WGMMA_PEAK_TFLOPS = 989.0


def _flops_per_iter(N: int, D: int, K: int) -> float:
    return 2.0 * N * K * D


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--dtype", choices=["fp16", "bf16"], default="bf16")
    args = ap.parse_args()

    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]

    shapes = [
        (10_000_000, 64, 20_000),
        (10_000_000, 128, 20_000),
        (10_000_000, 256, 20_000),
        (5_000_000, 256, 20_000),
        (2_500_000, 512, 20_000),
    ]

    print(f"\nLloyd loop bench: {args.iters} iters per shape, dtype={args.dtype}")
    print(f"{'Shape':<32} {'Triton (s)':>12} {'Cute (s)':>11} "
          f"{'Tri TFLOPs':>11} {'Cute TFLOPs':>12} {'Speedup':>10}")
    print("-" * 95)

    from flash_kmeans.cutedsl_impl import cutedsl_kmeans_Euclid
    from flash_kmeans import batch_kmeans_Euclid

    for N, D, K in shapes:
        torch.manual_seed(0)
        x = torch.randn(1, N, D, device="cuda", dtype=dtype)
        # Random init centroids (sampled from x rows).
        idx = torch.randint(0, N, (1, K), device="cuda")
        init = torch.gather(x, dim=1, index=idx[..., None].expand(-1, -1, D)).contiguous()

        # Warmup compile + L2 prime (small iter count).
        _ = cutedsl_kmeans_Euclid(x, K, max_iters=2, init_centroids=init.clone())
        _ = batch_kmeans_Euclid(x, n_clusters=K, max_iters=2, init_centroids=init.clone())
        torch.cuda.synchronize()

        # Bench
        t0 = time.time()
        ids_c, _, _ = cutedsl_kmeans_Euclid(
            x, K, max_iters=args.iters, init_centroids=init.clone()
        )
        torch.cuda.synchronize()
        t_cute = time.time() - t0

        t0 = time.time()
        ids_t, _, _ = batch_kmeans_Euclid(
            x, n_clusters=K, max_iters=args.iters, init_centroids=init.clone()
        )
        torch.cuda.synchronize()
        t_tri = time.time() - t0

        flops = _flops_per_iter(N, D, K) * args.iters
        tri_tflops = flops / t_tri / 1e12
        c_tflops = flops / t_cute / 1e12

        print(f"N={N:<8,} D={D:<3} K={K:<6,}      {t_tri:>12.2f} {t_cute:>11.2f} "
              f"{tri_tflops:>11.1f} {c_tflops:>12.1f} {t_tri/t_cute:>10.2f}x")

        del x, init, ids_c, ids_t
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
