import torch
from flash_kmeans import batch_kmeans_Euclid
from flash_kmeans.centroid_update_triton import triton_centroid_update_euclid
import time
import argparse

def _euclid_iter_torch(x, x_sq, centroids):
    cent_sq = (centroids ** 2).sum(dim=-1)
    cross = torch.einsum('bnd,bkd->bnk', x, centroids)
    dist_sq = (x_sq[:,:,None] + cent_sq[:,None,:] - 2.0 * cross).clamp_min_(0.0)
    cluster_ids = dist_sq.argmin(dim=-1)
    centroids_new = triton_centroid_update_euclid(x, cluster_ids, centroids)
    shift = (centroids_new - centroids).norm(dim=-1).max()
    return centroids_new, shift, cluster_ids

def batch_kmeans_Euclid_torch(x, n_clusters, max_iters=100, tol=0.0, init_centroids=None, verbose=False):
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
        centroids_new, center_shift, cluster_ids = _euclid_iter_torch(x, x_sq, centroids)

        # 4. Check for convergence
        if verbose:
            print(f"Iter {it}, center shift: {center_shift.item():.6f}")
        if center_shift < tol:
            break
        centroids = centroids_new.clone()

    return cluster_ids, centroids, it + 1


# https://github.com/DeMoriarty/fast_pytorch_kmeans 
try:
    from fast_pytorch_kmeans import KMeans
    def batch_kmeans_Euclid_fast_torch(x, n_clusters, max_iters=100, tol=0.0, init_centroids=None, verbose=False):

        kmeans = KMeans(n_clusters=n_clusters, mode='euclidean', verbose=0, tol=tol, max_iter=max_iters)
        all_labels = []
        for i in range(x.shape[0]):
            labels = kmeans.fit_predict(x[i])
            all_labels.append(labels)
        labels = torch.stack(all_labels)
        return labels, None, None
except ImportError:
    print("fast_pytorch_kmeans is not installed")
    batch_kmeans_Euclid_fast_torch = None


def benchmark_kmeans(b, n, d, k, kmeans_func, max_iters=100, tol=0.0):
    x = torch.randn(b, n, d, device='cuda', dtype=torch.float16)
    # warmup
    for _ in range(10):
        kmeans_func(x, k, max_iters=max_iters, tol=tol)
    # benchmark
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(10):
        kmeans_func(x, k, max_iters=max_iters, tol=tol)
    torch.cuda.synchronize()
    end = time.time()
    return (end - start) / 10 * 1000

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark KMeans implementations")
    parser.add_argument("--batch-size", "-b", type=int, default=32, help="Batch size")
    parser.add_argument("--num-points", "-n", type=int, default=74256, help="Number of points per batch")
    parser.add_argument("--dim", "-d", type=int, default=128, help="Dimensionality of points")
    parser.add_argument("--num-clusters", "-k", type=int, default=1000, help="Number of clusters")
    parser.add_argument("--max-iters", type=int, default=100, help="Maximum number of iterations")
    parser.add_argument("--tol", type=float, default=-1, help="Tolerance for center movement; negative disables early stopping")
    args = parser.parse_args()

    b = args.batch_size
    n = args.num_points
    d = args.dim
    k = args.num_clusters
    max_iters = args.max_iters
    tol = args.tol

    if batch_kmeans_Euclid_fast_torch is not None:
        print("fast_pytorch_kmeans")
        fast_time = benchmark_kmeans(b, n, d, k, batch_kmeans_Euclid_fast_torch, max_iters=max_iters, tol=tol)
        print(f"fast_pytorch_kmeans time for {b}x{n}x{d}x{k} with {max_iters} iterations: {fast_time} ms")
        print(f"fast_pytorch_kmeans for 1 iteration: {fast_time / max_iters} ms")
    else:
        print("fast_pytorch_kmeans is not installed")
        print("Skipping fast_pytorch_kmeans benchmark")
    print()
    print("batched torch kmeans")
    torch_time = benchmark_kmeans(b, n, d, k, batch_kmeans_Euclid_torch, max_iters=max_iters, tol=tol)
    print(f"batched torch kmeans time for {b}x{n}x{d}x{k} with {max_iters} iterations: {torch_time} ms")
    print(f"batched torch kmeans for 1 iteration: {torch_time / max_iters} ms")
    print()
    print("flash_kmeans")
    flash_time = benchmark_kmeans(b, n, d, k, batch_kmeans_Euclid, max_iters=max_iters, tol=tol)
    print(f"flash_kmeans time for {b}x{n}x{d}x{k} with {max_iters} iterations: {flash_time} ms")
    print(f"flash_kmeans for 1 iteration: {flash_time / max_iters} ms")
    print()