from __future__ import annotations

import time
from typing import Optional
import torch
import argparse
from flash_kmeans.kmeans_large import kmeans_largeN

try:
    from fastkmeans_patch import FastKMeans  # Apply monkey patch to fastkmeans

    def kmeans_Euclid_fastkmeans(
        x, n_clusters, max_iters=100, tol=0.0, init_centroids=None, verbose=False
    ):
        """
        Wrapper for fastkmeans to match the batch interface.
        Note: fastkmeans only supports Euclidean distance and processes each batch item separately.
        With monkey patch, we can pass torch.Tensor directly without numpy conversion.
        """
        # FastKMeans expects 2D input (N, D)
        # Use monkey patched version to accept tensor directly
        data = x
        kmeans = FastKMeans(
            d=data.shape[1],
            k=n_clusters,
            niter=max_iters,
            tol=tol,
            verbose=verbose,
            gpu=True,
            max_points_per_centroid=None,
            chunk_size_data=131072,
        )
        kmeans.train(data)
        labels = kmeans.predict(data)
        return labels, None, None
except ImportError:
    print("fastkmeans_patch not found, skipping fastkmeans wrapper.")
    kmeans_Euclid_fastkmeans = None


def kmeans_Euclid_flashkmeans(
    x, n_clusters, max_iters=100, tol=0.0, init_centroids=None, verbose=False
):
    return kmeans_largeN(
        x,
        n_clusters,
        max_iters=max_iters,
        tol=tol,
        init_centroids=init_centroids,
        verbose=verbose,
    )


def benchmark_kmeans(n, d, k, kmeans_func, max_iters=10, tol=-1.0, verbose=False):
    print(f"Benchmarking {kmeans_func.__name__} with N={n}, D={d}, K={k}, iter={max_iters}")
    
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    x = torch.randn(n, d, device='cpu', dtype=torch.float32, pin_memory=True)
    print("Data generated.")
    # warmup
    for _ in range(2):
        kmeans_func(x, k, max_iters=2, tol=tol, verbose=verbose)
    # benchmark
    torch.cuda.synchronize()
    start = time.time()
    test_runs = 2
    for _ in range(test_runs):
        kmeans_func(x, k, max_iters=max_iters, tol=tol, verbose=verbose)
    torch.cuda.synchronize()
    end = time.time()
    total_ms = (end - start) * 1000 / test_runs
    iter_ms = total_ms / max_iters
    print(f"Time taken: {total_ms:.2f} ms total, {iter_ms :.2f} ms/iter")
    return total_ms, iter_ms


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark KMeans implementations")
    parser.add_argument("--num-points", "-n", type=int, default=67108864, help="Number of points per batch")
    parser.add_argument("--dim", "-d", type=int, default=128, help="Dimensionality of points")
    parser.add_argument("--num-clusters", "-k", type=int, default=8192, help="Number of clusters")
    parser.add_argument("--max-iters", type=int, default=10, help="Maximum number of iterations")
    parser.add_argument("--tol", type=float, default=-1, help="Tolerance for center movement; negative disables early stopping")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    args = parser.parse_args()

    n = args.num_points
    d = args.dim
    k = args.num_clusters

    # settings = [ 
    #     # (N, K, D)
    #     (262144, 512, 128),
    #     (1048576, 1024, 128),
    #     (4194304, 2048, 128),
    #     (16777216, 4096, 128),
    #     (67108864, 8192, 128),
    #     (268435456, 16384, 128),
    #     (1073741824, 32768, 128),
    # ]

    benchmark_kmeans(n, d, k, kmeans_Euclid_flashkmeans, max_iters=args.max_iters, verbose=args.verbose)
    if kmeans_Euclid_fastkmeans is not None:
        benchmark_kmeans(n, d, k, kmeans_Euclid_fastkmeans, max_iters=args.max_iters, verbose=args.verbose)

    