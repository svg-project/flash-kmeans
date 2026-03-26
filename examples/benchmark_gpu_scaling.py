"""Benchmark multi-GPU scaling for kmeans_largeN.

Usage:
    # Auto-detect all visible GPUs and test 1, 2, ..., G:
    CUDA_VISIBLE_DEVICES=0,1,2,3 python examples/benchmark_gpu_scaling.py

    # Test specific GPU counts:
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python examples/benchmark_gpu_scaling.py \
        -n 100000000 -d 128 -k 8192 --max-iters 100 --gpus 1 2 4 8

    # fp32 mode:
    python examples/benchmark_gpu_scaling.py --dtype fp32
"""
from __future__ import annotations

import argparse
import time

import torch
import flash_kmeans.kmeans_large as _mod
from flash_kmeans.kmeans_large import kmeans_largeN


def _make_resolve_n(n):
    """Return a _resolve_devices replacement that uses the first n GPUs."""
    def _resolve(device):
        return [torch.device(f"cuda:{i}") for i in range(n)]
    return _resolve


def bench_one(x, k, max_iters, tol, num_gpus, warmup_iters=2, verbose=False):
    """Run kmeans_largeN on `num_gpus` GPUs and return wall-clock milliseconds."""
    init = x[torch.randperm(x.shape[0])[:k]]

    if num_gpus == 1:
        device = "cuda:0"
        ctx = None
    else:
        device = None
        # Temporarily patch _resolve_devices to use exactly num_gpus GPUs
        orig = _mod._resolve_devices
        _mod._resolve_devices = _make_resolve_n(num_gpus)
        ctx = orig  # save for restore

    try:
        # warmup
        kmeans_largeN(x, k, max_iters=warmup_iters, tol=-1,
                      init_centroids=init.clone(), device=device)
        torch.cuda.synchronize()

        # timed run
        start = time.time()
        kmeans_largeN(x, k, max_iters=max_iters, tol=tol,
                      init_centroids=init.clone(), device=device, verbose=verbose)
        torch.cuda.synchronize()
        elapsed_ms = (time.time() - start) * 1000
    finally:
        if ctx is not None:
            _mod._resolve_devices = ctx

    return elapsed_ms


def main():
    parser = argparse.ArgumentParser(description="Benchmark multi-GPU scaling for kmeans_largeN")
    parser.add_argument("-n", "--num-points", type=int, default=100_000_000)
    parser.add_argument("-d", "--dim", type=int, default=128)
    parser.add_argument("-k", "--num-clusters", type=int, default=8192)
    parser.add_argument("--max-iters", type=int, default=100)
    parser.add_argument("--tol", type=float, default=-1, help="negative = no early stop")
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "fp32"],
                        help="compute dtype (default: fp16)")
    parser.add_argument("--gpus", type=int, nargs="+", default=None,
                        help="GPU counts to test, e.g. --gpus 1 2 4 8. Default: 1..G")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    G_total = torch.cuda.device_count()
    assert G_total >= 1, "No CUDA devices found"

    gpu_counts = args.gpus if args.gpus else list(range(1, G_total + 1))
    for g in gpu_counts:
        assert 1 <= g <= G_total, f"--gpus {g} exceeds visible GPU count {G_total}"

    compute_dtype = torch.float16 if args.dtype == "fp16" else torch.float32

    print(f"Config: N={args.num_points:,}  D={args.dim}  K={args.num_clusters}  "
          f"iters={args.max_iters}  dtype={args.dtype}  GPUs visible={G_total}")
    print(f"Testing GPU counts: {gpu_counts}")
    print()

    # Generate data once (pinned, fp16 storage to save CPU memory at large N)
    torch.manual_seed(42)
    x = torch.randn(args.num_points, args.dim, device="cpu",
                     dtype=compute_dtype).pin_memory()
    print(f"Data generated: {x.shape}, dtype={x.dtype}, pinned={x.is_pinned()}")
    print()

    results = []  # (num_gpus, total_ms, ms_per_iter)

    for g in gpu_counts:
        elapsed = bench_one(x, args.num_clusters, args.max_iters, args.tol,
                            num_gpus=g, verbose=args.verbose)
        ms_per_iter = elapsed / args.max_iters
        results.append((g, elapsed, ms_per_iter))
        print(f"  GPUs={g}:  {elapsed:10.1f} ms total,  {ms_per_iter:8.2f} ms/iter")

    # Summary table
    print()
    print("=" * 60)
    print(f"{'GPUs':>6}  {'Total (ms)':>12}  {'ms/iter':>10}  {'Speedup':>8}")
    print("-" * 60)
    base_ms = results[0][1]
    for g, total_ms, ms_iter in results:
        speedup = base_ms / total_ms
        print(f"{g:>6}  {total_ms:>12.1f}  {ms_iter:>10.2f}  {speedup:>7.2f}x")
    print("=" * 60)


if __name__ == "__main__":
    main()
