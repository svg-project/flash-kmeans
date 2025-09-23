import torch
import torch.nn.functional as F
from torch.cuda import nvtx
from flash_kmeans.assign_euclid_triton import euclid_assign_triton
from flash_kmeans.centroid_update_triton import triton_centroid_update_cosine, triton_centroid_update_euclid, triton_centroid_update_sorted_euclid
from tqdm import trange

# -------------------- Compiled single-iteration kernels --------------------

# 1. Euclidean
def _euclid_iter(x, x_sq, centroids):
    
    cluster_ids = euclid_assign_triton(x, centroids, x_sq)
    centroids_new = triton_centroid_update_sorted_euclid(x, cluster_ids, centroids)

    shift = (centroids_new - centroids).norm(dim=-1).max()
    return centroids_new, shift, cluster_ids

# 2. Cosine
def _cosine_iter(x_norm, centroids):
    cos_sim = torch.einsum('bnd,bkd->bnk', x_norm, centroids)
    cluster_ids = cos_sim.argmax(dim=-1)
    centroids_new = triton_centroid_update_cosine(x_norm, cluster_ids, centroids)
    # centroids_new = centroids_new.clone()
    shift = (centroids_new - centroids).norm(dim=-1).max()
    return centroids_new, shift, cluster_ids

# 3. Dot-product
def _dot_iter(x, centroids):
    sim = torch.einsum('bnd,bkd->bnk', x, centroids)
    cluster_ids = sim.argmax(dim=-1)
    centroids_new = triton_centroid_update_cosine(x, cluster_ids, centroids)
    # centroids_new = centroids_new.clone()
    shift = (centroids_new - centroids).norm(dim=-1).max()
    return centroids_new, shift, cluster_ids

COMPILE_FLAG = False

try:
    if COMPILE_FLAG:
        _euclid_iter_compiled = torch.compile(_euclid_iter, dynamic=True, mode="reduce-overhead")
        _cosine_iter_compiled = torch.compile(_cosine_iter, dynamic=True, mode="reduce-overhead")
        _dot_iter_compiled    = torch.compile(_dot_iter,    dynamic=True, mode="reduce-overhead")
    else:
        _euclid_iter_compiled = _euclid_iter
        _cosine_iter_compiled = _cosine_iter
        _dot_iter_compiled    = _dot_iter
except Exception:  # pragma: no cover
    _euclid_iter_compiled = _euclid_iter
    _cosine_iter_compiled = _cosine_iter
    _dot_iter_compiled    = _dot_iter

def batch_kmeans_Euclid(x, n_clusters, max_iters=100, tol=0.0, init_centroids=None, verbose=False):
    """
    Batched KMeans clustering in PyTorch using Euclidean distance.

    Args:
        x: Tensor of shape (B, N, D), batch_size B, N points per batch, D dims.
        n_clusters: Number of clusters.
        max_iters: Max number of iterations.
        tol: Relative tolerance for center movement.
        verbose: Print loss for each iter.
    Returns:
        cluster_ids: (B, N) LongTensor, cluster assignment for each point.
        centroids: (B, n_clusters, D) final cluster centers.
    """
    B, N, D = x.shape

    # Pre-compute squared L2 norm of all points (constant during iterations)
    x_sq = (x ** 2).sum(dim=-1)  # (B, N)

    if init_centroids is None:
        # Randomly select initial centers from x
        indices = torch.randint(0, N, (B, n_clusters), device=x.device)
        centroids = torch.gather(
            x,
            dim=1,
            index=indices[..., None].expand(-1, -1, D)
        )  # (B, n_clusters, D)
    else:
        centroids = init_centroids

    centroids = centroids.view(B, n_clusters, D)

    for it in range(max_iters):
        # ---- compiled single iteration ----
        centroids_new, center_shift, cluster_ids = _euclid_iter_compiled(x, x_sq, centroids)

        # 4. Check for convergence
        if verbose:
            print(f"Iter {it}, center shift: {center_shift.item():.6f}")
        if center_shift < tol:
            break
        centroids = centroids_new.clone()

    return cluster_ids, centroids, it + 1


def batch_kmeans_Cosine(x, n_clusters, max_iters=100, tol=0.0, init_centroids=None, verbose=False):
    """
    Batched KMeans clustering in PyTorch using Cosine similarity.

    Args:
        x: Tensor of shape (B, N, D), batch_size B, N points per batch, D dims.
        n_clusters: Number of clusters.
        max_iters: Max number of iterations.
        tol: Relative tolerance for center movement.
        verbose: Print loss for each iter.
    Returns:
        cluster_ids: (B, N) LongTensor, cluster assignment for each point.
        centroids: (B, n_clusters, D) final cluster centers.
    """
    B, N, D = x.shape

    # Normalize input vectors for cosine similarity
    x_norm = F.normalize(x, p=2, dim=-1)  # (B, N, D)

    if init_centroids is None:
        # Randomly select initial centers from x_norm
        indices = torch.randint(0, N, (B, n_clusters), device=x.device)
        centroids = torch.gather(
            x_norm,
            dim=1,
            index=indices[..., None].expand(-1, -1, D)
        ) # (B, n_clusters, D)
    else:
        centroids = init_centroids

    centroids = centroids.view(B, n_clusters, D)
    centroids = F.normalize(centroids, p=2, dim=-1)  # Ensure centroids are normalized

    for it in range(max_iters):
        # ---- compiled single iteration ----
        centroids_new, center_shift, cluster_ids = _cosine_iter_compiled(x_norm, centroids)

        # 4. Check for convergence
        if verbose:
            print(f"Iter {it}, center shift: {center_shift.item():.6f}")
        if center_shift < tol:
            break
        centroids = centroids_new.clone()

    return cluster_ids, centroids, it + 1


def batch_kmeans_Dot(x, n_clusters, max_iters=100, tol=0.0, init_centroids=None, verbose=False):
    """
    Batched KMeans clustering in PyTorch using raw dot-product as similarity.

    """
    B, N, D = x.shape

    if init_centroids is None:
        # 随机初始化中心
        indices = torch.randint(0, N, (B, n_clusters), device=x.device)
        centroids = torch.gather(
            x,
            dim=1,
            index=indices[..., None].expand(-1, -1, D)
        )
    else:
        centroids = init_centroids

    centroids = centroids.view(B, n_clusters, D)

    for it in range(max_iters):
        # ---- compiled single iteration ----
        centroids_new, center_shift, cluster_ids = _dot_iter_compiled(x, centroids)

        # 4. Check for convergence
        if verbose:
            print(f"Iter {it} (dot), center shift: {center_shift.item():.6f}")
        if center_shift < tol:
            break
        centroids = centroids_new.clone()

    return cluster_ids, centroids, it + 1


if __name__ == "__main__":
    # 用法示例
    B, N, D = 32, 74256, 128  # 32 个 batch，每个 batch 10 万点，128 维
    dtype = torch.float16
    x = torch.randn(B, N, D, device="cuda", dtype=dtype)  # 大 batch 用 GPU 跑
    n_clusters = 1000
    max_iters = 2

    print("=== Testing Euclidean Distance K-Means ===")
    cluster_ids_euclid, centroids_euclid, n_iters_euclid = batch_kmeans_Euclid(x, n_clusters, max_iters=max_iters, verbose=True)
    print(f"Euclidean - cluster_ids shape: {cluster_ids_euclid.shape}, centroids shape: {centroids_euclid.shape}")

    print("\n=== Testing Cosine Similarity K-Means ===")
    cluster_ids_cosine, centroids_cosine, n_iters_cosine = batch_kmeans_Cosine(x, n_clusters, max_iters=max_iters, verbose=True)
    print(f"Cosine - cluster_ids shape: {cluster_ids_cosine.shape}, centroids shape: {centroids_cosine.shape}")

    print("\n=== Testing Dot-Product K-Means ===")
    cluster_ids_dot, centroids_dot, n_iters_dot = batch_kmeans_Dot(x, n_clusters, max_iters=max_iters, verbose=True)
    print(f"Dot - cluster_ids shape: {cluster_ids_dot.shape}, centroids shape: {centroids_dot.shape}")

    # Profile the time cost with rounds=100
    rounds = 200
    import time

    print(f"\n=== Speed Comparison (averaged over {rounds} rounds) ===")

    # Test Euclidean Distance K-Means
    euclid_start = torch.cuda.Event(enable_timing=True)
    euclid_end = torch.cuda.Event(enable_timing=True)
    euclid_start.record()
    for i in range(rounds):
        cluster_ids_euclid, centroids_euclid, n_iters_euclid = batch_kmeans_Euclid(x, n_clusters, init_centroids=centroids_euclid, max_iters=max_iters, verbose=False)
    euclid_end.record(); torch.cuda.synchronize()
    euclid_time = euclid_start.elapsed_time(euclid_end) / rounds
    euclid_time_per_iter = euclid_time / n_iters_euclid
    print(f"Euclidean Distance K-Means: {euclid_time:.2f} ms per run, total {n_iters_euclid} iterations, {euclid_time_per_iter:.2f} ms per iter")
    print(f"Euclidean Distance TFLOPS: {2 * B * N * D * n_clusters * n_iters_euclid / euclid_time / 1e12:.2f}")
    
    # Test Cosine Similarity K-Means
    cosine_start = torch.cuda.Event(enable_timing=True)
    cosine_end = torch.cuda.Event(enable_timing=True)
    cosine_start.record()
    for i in range(rounds):
        cluster_ids_cosine, centroids_cosine, n_iters_cosine = batch_kmeans_Cosine(x, n_clusters, max_iters=max_iters, init_centroids=centroids_cosine, verbose=False)
    cosine_end.record(); torch.cuda.synchronize()
    cosine_time = cosine_start.elapsed_time(cosine_end) / rounds
    cosine_time_per_iter = cosine_time / n_iters_cosine
    print(f"Cosine Similarity K-Means: {cosine_time:.2f} ms per run, total {n_iters_cosine} iterations, {cosine_time_per_iter:.2f} ms per iter")
    print(f"Cosine Similarity TFLOPS: {2 * B * N * D * n_clusters * n_iters_cosine / cosine_time / 1e12:.2f}")

    # Test Dot-Product K-Means
    dot_start = torch.cuda.Event(enable_timing=True)
    dot_end = torch.cuda.Event(enable_timing=True)
    dot_start.record()
    for i in range(rounds):
        cluster_ids_dot, centroids_dot, n_iters_dot = batch_kmeans_Dot(x, n_clusters, max_iters=max_iters, init_centroids=centroids_dot, verbose=False)
    dot_end.record(); torch.cuda.synchronize()
    dot_time = dot_start.elapsed_time(dot_end) / rounds
    dot_time_per_iter = dot_time / n_iters_dot
    print(f"Dot-Product K-Means: {dot_time:.2f} ms per run, total {n_iters_dot} iterations, {dot_time_per_iter:.2f} ms per iter")
    print(f"Dot-Product TFLOPS: {2 * B * N * D * n_clusters * n_iters_dot / dot_time / 1e12:.2f}")
