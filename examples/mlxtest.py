# /// script
# dependencies = [
#   "mlx",
#   "torch",
#   "numpy",
# ]
# ///

import time
import mlx.core as mx
import torch
import numpy as np
import importlib.metadata

def mlx_get_indices(mask):
    """
    Hardware-efficient fallback for mx.nonzero:
    Maps False values to -1, sorts them to the front, and slices the valid indices.
    """
    N = mask.size
    indices = mx.where(mask, mx.arange(N), mx.array([-1]))
    sorted_indices = mx.sort(indices)
    num_true = mx.sum(mask.astype(mx.int32)).item()
    return sorted_indices[-num_true:] if num_true > 0 else mx.array([], dtype=mx.int32)

def flash_assign_mlx(X, centroids, chunk_size=32768):
    """
    FlashAssign: Avoids materializing the massive N x K distance matrix.
    Uses the Euclidean identity ||x-c||^2 = ||x||^2 + ||c||^2 - 2<x,c>.
    """
    N, D = X.shape
    x_sq = mx.sum(X**2, axis=1, keepdims=True)
    c_sq = mx.sum(centroids**2, axis=1)
    
    labels = []
    for i in range(0, N, chunk_size):
        x_chunk = X[i : i + chunk_size]
        # MLX JIT fuses these operations to minimize memory bandwidth usage
        dist = x_sq[i : i + chunk_size] + c_sq - 2 * mx.matmul(x_chunk, centroids.T)
        labels.append(mx.argmin(dist, axis=1))
    return mx.concatenate(labels)

def sort_inverse_update_mlx(X, labels, K):
    """
    Sort-Inverse Update: Replaces slow atomic 'scatter-adds' with 
    high-bandwidth contiguous memory prefix sums (cumsum).
    """
    N, D = X.shape
    
    # 1. Sort data by cluster assignment
    sort_idx = mx.argsort(labels)
    sorted_X = X[sort_idx]
    sorted_labels = labels[sort_idx]
    
    # 2. Identify contiguous segments (clusters)
    diff_mask = sorted_labels[:-1] != sorted_labels[1:]
    is_boundary = mx.concatenate([diff_mask, mx.array([True])])
    boundary_indices = mlx_get_indices(is_boundary)
    
    # 3. Perform contiguous reduction using cumulative sums
    cumsum_X = mx.cumsum(sorted_X, axis=0)
    sums_at_ends = cumsum_X[boundary_indices]
    sums_shifted = mx.concatenate([mx.zeros((1, D)), sums_at_ends[:-1]], axis=0)
    cluster_sums = sums_at_ends - sums_shifted
    
    # 4. Calculate cluster counts (sizes)
    counts_at_ends = boundary_indices + 1
    counts_shifted = mx.concatenate([mx.array([0]), counts_at_ends[:-1]])
    cluster_counts = (counts_at_ends - counts_shifted).reshape(-1, 1)
    
    # 5. Map results back to the K centroids
    new_centroids = mx.zeros((K, D))
    active_ids = sorted_labels[boundary_indices]
    new_centroids[active_ids] = cluster_sums / cluster_counts
    
    return new_centroids

def flash_kmeans_mlx(X, K, iters=5):
    # Initialization (Forgy method)
    centroids = X[:K]
    for _ in range(iters):
        labels = flash_assign_mlx(X, centroids)
        centroids = sort_inverse_update_mlx(X, labels, K)
        mx.eval(centroids)
    return centroids

def kmeans_pytorch(X_np, K, iters=5, device="mps"):
    X = torch.from_numpy(X_np).to(device)
    centroids = X[:K].clone()
    N, D = X.shape
    
    for _ in range(iters):
        # Bottleneck 1: Memory wall (O(N*K) allocation)
        dist = torch.cdist(X, centroids)
        labels = dist.argmin(dim=1)
        
        # Bottleneck 2: Atomic contention (scatter_add_ on GPU)
        new_centroids = torch.zeros((K, D), device=device)
        counts = torch.zeros((K, 1), device=device)
        new_centroids.scatter_add_(0, labels.unsqueeze(1).expand(-1, D), X)
        counts.scatter_add_(0, labels.unsqueeze(1), torch.ones((N, 1), device=device))
        centroids = new_centroids / counts.clamp(min=1)
        torch.mps.synchronize()
    return centroids

def run_benchmark():
    # Dataset configuration
    N, D, K = 1_000_000, 128, 1024
    ITERS = 5
    
    print(f"--- Flash-KMeans Benchmark (arXiv:2603.09229) ---")
    print(f"Config: {N:,} points, {D} dims, {K} clusters\n")
    
    data_np = np.random.randn(N, D).astype(np.float32)
    X_mlx = mx.array(data_np)
    
    # MLX Benchmark
    print("[MLX] Running Flash-KMeans...")
    _ = flash_kmeans_mlx(X_mlx[:10000], K, iters=1) # Warmup
    start = time.perf_counter()
    _ = flash_kmeans_mlx(X_mlx, K, iters=ITERS)
    mlx_time = (time.perf_counter() - start) / ITERS
    print(f"MLX Avg Iteration: {mlx_time:.4f}s")
    
    # PyTorch Benchmark
    if torch.backends.mps.is_available():
        print("\n[PyTorch MPS] Running Standard KMeans...")
        try:
            _ = kmeans_pytorch(data_np[:10000], K, iters=1) # Warmup
            start = time.perf_counter()
            _ = kmeans_pytorch(data_np, K, iters=ITERS)
            pt_time = (time.perf_counter() - start) / ITERS
            print(f"PyTorch MPS Avg Iteration: {pt_time:.4f}s")
            print(f"\nResult: MLX is {pt_time / mlx_time:.2f}x faster.")
        except Exception as e:
            print(f"PyTorch Failed: {e}")
            print("Note: PyTorch often OOMs here due to the 4GB distance matrix.")

if __name__ == "__main__":
    run_benchmark()