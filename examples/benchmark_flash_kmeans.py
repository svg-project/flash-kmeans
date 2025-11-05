import torch
from flash_kmeans import batch_kmeans_Euclid, batch_kmeans_Cosine
from flash_kmeans.centroid_update_triton import triton_centroid_update_euclid, triton_centroid_update_cosine
from flash_kmeans.torch_fallback import batch_kmeans_Euclid_torch_native
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

def _cosine_iter_torch(x, centroids):
    cos_sim = torch.einsum('bnd,bkd->bnk', x, centroids)
    cluster_ids = cos_sim.argmax(dim=-1)
    centroids_new = triton_centroid_update_cosine(x, cluster_ids, centroids)
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

def batch_kmeans_Cosine_torch(x, n_clusters, max_iters=100, tol=0.0, init_centroids=None, verbose=False):
    """
    Batched KMeans clustering in PyTorch using Cosine distance.

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
        centroids_new, center_shift, cluster_ids = _cosine_iter_torch(x, centroids)

        # Check for convergence
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

    def batch_kmeans_Cosine_fast_torch(x, n_clusters, max_iters=100, tol=0.0, init_centroids=None, verbose=False):
        kmeans = KMeans(n_clusters=n_clusters, mode='cosine', verbose=0, tol=tol, max_iter=max_iters)
        all_labels = []
        for i in range(x.shape[0]):
            labels = kmeans.fit_predict(x[i])
            all_labels.append(labels)
        labels = torch.stack(all_labels)
        return labels, None, None
except ImportError:
    print("fast_pytorch_kmeans is not installed")
    batch_kmeans_Euclid_fast_torch = None


# https://github.com/AnswerDotAI/fastkmeans
try:
    from fastkmeans_patch import FastKMeans  # Apply monkey patch to fastkmeans
    def batch_kmeans_Euclid_fastkmeans(x, n_clusters, max_iters=100, tol=0.0, init_centroids=None, verbose=False):
        """
        Wrapper for fastkmeans to match the batch interface.
        Note: fastkmeans only supports Euclidean distance and processes each batch item separately.
        With monkey patch, we can pass torch.Tensor directly without numpy conversion.
        """
        all_labels = []
        for i in range(x.shape[0]):
            # FastKMeans expects 2D input (N, D)
            # Use monkey patched version to accept tensor directly
            data = x[i]
            kmeans = FastKMeans(d=data.shape[1], k=n_clusters, niter=max_iters, tol=tol, verbose=verbose, gpu=True, max_points_per_centroid=None)
            kmeans.train(data)
            labels = kmeans.predict(data)
            all_labels.append(labels)
        labels = torch.stack(all_labels)
        return labels, None, None
    def batch_kmeans_Euclid_fastkmeans_torch(x, n_clusters, max_iters=100, tol=0.0, init_centroids=None, verbose=False):
        all_labels = []
        for i in range(x.shape[0]):
            # FastKMeans expects 2D input (N, D)
            # Use monkey patched version to accept tensor directly
            data = x[i]
            kmeans = FastKMeans(d=data.shape[1], k=n_clusters, niter=max_iters, tol=tol, verbose=verbose, gpu=True, max_points_per_centroid=None, use_triton=False)
            kmeans.train(data)
            labels = kmeans.predict(data)
            all_labels.append(labels)
        labels = torch.stack(all_labels)
        return labels, None, None
except ImportError:
    print("fastkmeans is not installed")
    batch_kmeans_Euclid_fastkmeans = None


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

def benchmark_kmeans_all(b, n, d, k, kmeans_func_list, max_iters=100, tol=0.0, output_file="results.jsonl"):
    with open(output_file, "a") as output:
        for kmeans_func in kmeans_func_list:
            print("Benchmarking:", kmeans_func.__name__)
            try:
                t = benchmark_kmeans(b, n, d, k, kmeans_func, max_iters=max_iters, tol=tol)
                print(f"Time for {b}x{n}x{d}x{k} with {max_iters} iterations: {t:.2f} ms")
                print(f"For 1 iteration: {t / max_iters:.2f} ms")
                # write to output json
            except Exception as e:
                t = -1
                print(f"Error during benchmarking: {e}")
            finally:
                output.write(f'{{"method": "{kmeans_func.__name__}", "batch_size": {b}, "num_points": {n}, "dim": {d}, "num_clusters": {k}, "max_iters": {max_iters}, "tol": {tol}, "time_ms": {t}, "time_per_iter_ms": {t / max_iters}}}\n')
                output.flush()
            print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark KMeans implementations")
    parser.add_argument("--batch-size", "-b", type=int, default=32, help="Batch size")
    parser.add_argument("--num-points", "-n", type=int, default=74256, help="Number of points per batch")
    parser.add_argument("--dim", "-d", type=int, default=128, help="Dimensionality of points")
    parser.add_argument("--num-clusters", "-k", type=int, default=1000, help="Number of clusters")
    parser.add_argument("--max-iters", type=int, default=100, help="Maximum number of iterations")
    parser.add_argument("--tol", type=float, default=-1, help="Tolerance for center movement; negative disables early stopping")
    parser.add_argument("--distance-mode", type=str, default="euclid", choices=["euclid", "cosine"], help="Distance metric to use")
    parser.add_argument("--output-file", type=str, default="results.jsonl", help="Output file for benchmark results")
    
    args = parser.parse_args()

    b = args.batch_size
    n = args.num_points
    d = args.dim
    k = args.num_clusters
    max_iters = args.max_iters
    tol = args.tol

    if args.distance_mode == "euclid":
        kmeans_func_list = [batch_kmeans_Euclid_torch, batch_kmeans_Euclid_torch_native, batch_kmeans_Euclid]
        if batch_kmeans_Euclid_fast_torch is not None:
            kmeans_func_list.insert(0, batch_kmeans_Euclid_fast_torch)
        if batch_kmeans_Euclid_fastkmeans is not None:
            kmeans_func_list.append(batch_kmeans_Euclid_fastkmeans_torch)
            kmeans_func_list.append(batch_kmeans_Euclid_fastkmeans)
    elif args.distance_mode == "cosine":
        kmeans_func_list = [batch_kmeans_Cosine_torch, batch_kmeans_Cosine]
        if batch_kmeans_Cosine_fast_torch is not None:
            kmeans_func_list.insert(0, batch_kmeans_Cosine_fast_torch)
    else:
        raise ValueError("Invalid distance mode")

    benchmark_kmeans_all(b, n, d, k, kmeans_func_list, max_iters=max_iters, tol=tol, output_file=args.output_file)

