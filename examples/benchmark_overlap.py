"""Benchmark: overlapped vs non-overlapped multi-GPU pipeline.

Demonstrates that the double-buffered stream overlap in kmeans_largeN
provides real speedup over a naive sequential implementation.

Usage:
    CUDA_VISIBLE_DEVICES=4 python examples/benchmark_overlap.py
    CUDA_VISIBLE_DEVICES=4,5 python examples/benchmark_overlap.py --gpus 2
    CUDA_VISIBLE_DEVICES=0,1,2,3 python examples/benchmark_overlap.py -n 100000000 --gpus 4
"""
from __future__ import annotations

import argparse
import time

import torch

from flash_kmeans.assign_euclid_triton import euclid_assign_triton
from flash_kmeans.centroid_update_triton import triton_centroid_update_sorted_euclid
from flash_kmeans.kmeans_large import kmeans_largeN


# ---------------------------------------------------------------------------
# Non-overlapped baseline: no double-buffering, no H2D/compute overlap
# ---------------------------------------------------------------------------
def kmeans_largeN_no_overlap(
    x, n_clusters, max_iters=100, tol=0.0, verbose=False,
    BLOCK_N=1048576, init_centroids=None, device=None, dtype=None,
):
    """Functionally identical to kmeans_largeN but deliberately non-overlapped.

    Differences from the overlapped version:
    1. Single stream per GPU — H2D transfer and compute run sequentially on the
       same stream, so the PCIe bus is idle during kernel execution and vice versa.
    2. GPUs are processed sequentially in the block loop (no parallel dispatch).
    3. Full torch.cuda.synchronize() barrier between iterations — no prefetching
       of next iteration's first block during the AllReduce.
    """
    if device is None:
        G = torch.cuda.device_count()
        devices = [torch.device(f"cuda:{i}") for i in range(G)]
    else:
        devices = [torch.device(device)]
    G = len(devices)
    dtype = dtype or x.dtype

    N, D = x.shape
    K = n_clusters
    num_blocks = (N + BLOCK_N - 1) // BLOCK_N

    # --- Partition blocks across GPUs ---
    blocks_per_gpu = [(num_blocks // G) + (1 if g < num_blocks % G else 0) for g in range(G)]
    block_start_g = [0] * G
    for g in range(1, G):
        block_start_g[g] = block_start_g[g - 1] + blocks_per_gpu[g - 1]
    point_start = [block_start_g[g] * BLOCK_N for g in range(G)]
    point_end = [min(point_start[g] + blocks_per_gpu[g] * BLOCK_N, N) for g in range(G)]

    active_gpus = [g for g in range(G) if blocks_per_gpu[g] > 0]
    primary_gpu = active_gpus[0]
    primary_dev = devices[primary_gpu]

    # --- Allocate per-GPU resources (single stream each) ---
    streams = {}
    centroids_g = {}
    cluster_ids_g = {}
    x_sq_cache_g = {}

    for g in active_gpus:
        dev = devices[g]
        with torch.cuda.device(dev):
            streams[g] = torch.cuda.Stream(device=dev)
            n_pts = point_end[g] - point_start[g]
            cluster_ids_g[g] = torch.empty((n_pts,), device=dev, dtype=torch.int32)
            x_sq_cache_g[g] = [None] * blocks_per_gpu[g]

    # --- Init centroids ---
    with torch.cuda.device(primary_dev):
        if init_centroids is not None:
            centroids = init_centroids.to(device=primary_dev, dtype=dtype)
        else:
            indices = torch.randint(0, N, (K,), device="cpu", dtype=torch.int32)
            centroids = x[indices].to(device=primary_dev, dtype=dtype)
    torch.cuda.synchronize()  # ensure centroids ready

    for g in active_gpus:
        dev = devices[g]
        with torch.cuda.device(dev):
            if g == primary_gpu:
                centroids_g[g] = centroids
            else:
                centroids_g[g] = centroids.to(dev)

    # Staging for multi-GPU reduce
    if G > 1:
        with torch.cuda.device(primary_dev):
            staging_sums = {g: torch.empty((K, D), device=primary_dev, dtype=torch.float32)
                           for g in active_gpus if g != primary_gpu}
            staging_cnts = {g: torch.empty((K,), device=primary_dev, dtype=torch.int32)
                           for g in active_gpus if g != primary_gpu}

    cluster_ids_cpu = torch.empty((N,), dtype=torch.int32, pin_memory=True) if G > 1 else None

    # --- Main loop ---
    for it in range(max_iters):
        start_time = time.time()

        # Phase 1: zero accumulators + c_sq  (synchronous, no overlap)
        centroid_sums_g = {}
        centroid_cnts_g = {}
        c_sq_g = {}
        for g in active_gpus:
            dev = devices[g]
            with torch.cuda.device(dev):
                centroid_sums_g[g] = torch.zeros((K, D), device=dev, dtype=torch.float32)
                centroid_cnts_g[g] = torch.zeros((K,), device=dev, dtype=torch.int32)
                c_sq_g[g] = (centroids_g[g] ** 2).sum(dim=-1)
        torch.cuda.synchronize()  # barrier: init done on all GPUs

        # Phase 2: process blocks — ONE stream per GPU, GPUs sequentially
        for g in active_gpus:
            dev = devices[g]
            ps = point_start[g]
            with torch.cuda.device(dev):
                with torch.cuda.stream(streams[g]):
                    for local_idx in range(blocks_per_gpu[g]):
                        n_start = ps + local_idx * BLOCK_N
                        n_end = min(n_start + BLOCK_N, N)

                        # --- H2D: synchronous copy (blocks stream until done) ---
                        x_block = x[n_start:n_end].to(dev, non_blocking=False, dtype=dtype)

                        if x_sq_cache_g[g][local_idx] is None:
                            x_sq_cache_g[g][local_idx] = (x_block ** 2).sum(dim=-1)

                        local_offset = local_idx * BLOCK_N
                        local_end = min(local_offset + BLOCK_N, point_end[g] - ps)

                        cluster_ids_block = euclid_assign_triton(
                            x_block.unsqueeze(0),
                            centroids_g[g].unsqueeze(0),
                            x_sq_cache_g[g][local_idx].unsqueeze(0),
                            out=cluster_ids_g[g][local_offset:local_end].unsqueeze(0),
                            c_sq=c_sq_g[g].unsqueeze(0),
                        )

                        triton_centroid_update_sorted_euclid(
                            x=x_block.unsqueeze(0),
                            cluster_ids=cluster_ids_block,
                            old_centroids=centroids_g[g].unsqueeze(0),
                            centroid_sums=centroid_sums_g[g].unsqueeze(0),
                            centroid_cnts=centroid_cnts_g[g].unsqueeze(0),
                            calculate_new=False,
                        )

        torch.cuda.synchronize()  # barrier: all blocks done

        # Phase 3: Reduce
        with torch.cuda.device(primary_dev):
            if G > 1:
                for g in active_gpus:
                    if g != primary_gpu:
                        staging_sums[g].copy_(centroid_sums_g[g])
                        staging_cnts[g].copy_(centroid_cnts_g[g])

                total_sums = centroid_sums_g[primary_gpu]
                total_cnts = centroid_cnts_g[primary_gpu]
                for g in active_gpus:
                    if g != primary_gpu:
                        total_sums = total_sums + staging_sums[g]
                        total_cnts = total_cnts + staging_cnts[g]
            else:
                total_sums = centroid_sums_g[primary_gpu]
                total_cnts = centroid_cnts_g[primary_gpu]

            mask = total_cnts > 0
            new_centroids = torch.where(
                mask.unsqueeze(-1),
                total_sums / total_cnts.clamp(min=1).unsqueeze(-1).float(),
                centroids_g[primary_gpu].float(),
            ).to(dtype)

            shift = (new_centroids - centroids_g[primary_gpu]).norm(dim=-1).max()

        torch.cuda.synchronize()  # barrier: reduce done

        if verbose:
            print(f"Iter {it}, center shift: {shift.item():.6f}, time: {time.time() - start_time:.2f}s")

        if shift < tol:
            break

        # Broadcast
        centroids_g[primary_gpu] = new_centroids.clone()
        for g in active_gpus:
            if g != primary_gpu:
                with torch.cuda.device(devices[g]):
                    centroids_g[g].copy_(new_centroids)

        torch.cuda.synchronize()  # barrier: broadcast done before next iteration

    if G == 1:
        return cluster_ids_g[primary_gpu], new_centroids
    else:
        for g in active_gpus:
            ps_g, pe_g = point_start[g], point_end[g]
            cluster_ids_cpu[ps_g:pe_g].copy_(cluster_ids_g[g][:pe_g - ps_g])
        return cluster_ids_cpu, new_centroids.cpu()


# ---------------------------------------------------------------------------
# Benchmark harness
# ---------------------------------------------------------------------------
def bench(fn, x, k, max_iters, init, device, warmup_iters=2, verbose=False):
    """Time a single run (after warmup) and return ms."""
    fn(x, k, max_iters=warmup_iters, tol=-1, init_centroids=init.clone(), device=device)
    torch.cuda.synchronize()

    start = time.time()
    fn(x, k, max_iters=max_iters, tol=-1, init_centroids=init.clone(),
       device=device, verbose=verbose)
    torch.cuda.synchronize()
    return (time.time() - start) * 1000


def main():
    parser = argparse.ArgumentParser(
        description="Compare overlapped vs non-overlapped multi-GPU kmeans_largeN")
    parser.add_argument("-n", "--num-points", type=int, default=100_000_000)
    parser.add_argument("-d", "--dim", type=int, default=128)
    parser.add_argument("-k", "--num-clusters", type=int, default=8192)
    parser.add_argument("--max-iters", type=int, default=10)
    parser.add_argument("--gpus", type=int, default=None,
                        help="number of GPUs to use (default: all visible)")
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "fp32"])
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    G_total = torch.cuda.device_count()
    num_gpus = args.gpus or G_total
    assert 1 <= num_gpus <= G_total, f"--gpus {num_gpus} > visible {G_total}"

    compute_dtype = torch.float16 if args.dtype == "fp16" else torch.float32

    if num_gpus == 1:
        device = "cuda:0"
    elif num_gpus == G_total:
        device = None
    else:
        # Monkey-patch to use first num_gpus GPUs
        import flash_kmeans.kmeans_large as _mod
        _orig = _mod._resolve_devices
        _mod._resolve_devices = lambda dev: [torch.device(f"cuda:{i}") for i in range(num_gpus)]
        device = None

    print(f"Config: N={args.num_points:,}  D={args.dim}  K={args.num_clusters}  "
          f"iters={args.max_iters}  dtype={args.dtype}  GPUs={num_gpus}")
    print()

    torch.manual_seed(42)
    x = torch.randn(args.num_points, args.dim, device="cpu",
                     dtype=compute_dtype).pin_memory()
    init = x[torch.randperm(args.num_points)[:args.num_clusters]]
    print(f"Data: {x.shape}, dtype={x.dtype}, pinned={x.is_pinned()}")
    print()

    # --- Non-overlapped ---
    ms_no = bench(kmeans_largeN_no_overlap, x, args.num_clusters, args.max_iters,
                  init, device, verbose=args.verbose)
    iter_no = ms_no / args.max_iters
    print(f"  Non-overlapped:  {ms_no:10.1f} ms total,  {iter_no:8.2f} ms/iter")

    # --- Overlapped ---
    ms_ov = bench(kmeans_largeN, x, args.num_clusters, args.max_iters,
                  init, device, verbose=args.verbose)
    iter_ov = ms_ov / args.max_iters
    print(f"  Overlapped:      {ms_ov:10.1f} ms total,  {iter_ov:8.2f} ms/iter")

    # --- Summary ---
    print()
    print("=" * 55)
    print(f"{'Method':<20} {'Total (ms)':>12} {'ms/iter':>10} {'Speedup':>10}")
    print("-" * 55)
    print(f"{'Non-overlapped':<20} {ms_no:>12.1f} {iter_no:>10.2f} {'1.00x':>10}")
    speedup = ms_no / ms_ov
    print(f"{'Overlapped':<20} {ms_ov:>12.1f} {iter_ov:>10.2f} {speedup:>9.2f}x")
    print("=" * 55)


if __name__ == "__main__":
    main()
