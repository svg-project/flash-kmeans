import torch
import torch.nn.functional as F

def euclid_assign_torch_native_chunked(x, centroids, x_sq, chunk_size_N=32768, chunk_size_K=1024):
    """
    Torch naive implementation for assignment with chunking to avoid OOM.
    
    Args:
        x: (B, N, D) input points
        centroids: (B, K, D) cluster centers
        x_sq: (B, N) pre-computed ||x||^2
        chunk_size_N: chunk size along N dimension
        chunk_size_K: chunk size along K dimension
        
    Returns:
        cluster_ids: (B, N) int32 cluster assignment per point
    """
    B, N, D = x.shape
    K = centroids.shape[1]
    assert centroids.shape[2] == x.shape[2], "Dimension mismatch between x and centroids"
    assert x.shape[0] == centroids.shape[0], "Batch size mismatch between x and centroids"

    cent_sq = (centroids ** 2).sum(dim=-1)  # (B, K)

    cluster_ids = torch.empty((B, N), dtype=torch.int32, device=x.device)

    # Process in chunks to avoid OOM
    for n_start in range(0, N, chunk_size_N):
        n_end = min(n_start + chunk_size_N, N)
        x_chunk = x[:, n_start:n_end, :]  # (B, n_chunk, D)
        x_sq_chunk = x_sq[:, n_start:n_end]  # (B, n_chunk)

        dists_chunk = torch.empty((B, n_end - n_start, K), device=x.device)

        for k_start in range(0, K, chunk_size_K):
            k_end = min(k_start + chunk_size_K, K)
            cent_chunk = centroids[:, k_start:k_end, :]  # (B, k_chunk, D)
            cent_sq_chunk = cent_sq[:, k_start:k_end]  # (B, k_chunk)

            # Compute squared distances
            dists_partial = (
                x_sq_chunk.unsqueeze(-1)  # (B, n_chunk, 1)
                - 2 * torch.bmm(x_chunk, cent_chunk.transpose(1, 2))  # (B, n_chunk, k_chunk)
                + cent_sq_chunk.unsqueeze(-2)  # (B, 1, k_chunk)
            )  # (B, n_chunk, k_chunk)

            dists_chunk[:, :, k_start:k_end] = dists_partial

        # Assign cluster ids
        cluster_ids[:, n_start:n_end] = torch.argmin(dists_chunk, dim=-1)
    
    return cluster_ids

def torch_loop_centroid_update(x_norm: torch.Tensor, cluster_ids: torch.Tensor, old_centroids: torch.Tensor, mode = 'euclid'):
    """Reference Python implementation (double for-loop)"""
    assert mode in ['euclid', 'cosine'], "Mode must be 'euclid' or 'cosine'"
    B, N, D = x_norm.shape
    K = old_centroids.shape[1]
    new_centroids = torch.zeros_like(old_centroids)
    for b in range(B):
        for k in range(K):
            mask = cluster_ids[b] == k
            if mask.any():
                if mode == 'euclid':
                    new_centroids[b, k] = x_norm[b][mask].mean(dim=0, dtype=x_norm.dtype)
                else:  # cosine
                    new_centroids[b, k] = F.normalize(x_norm[b][mask].mean(dim=0, dtype=x_norm.dtype), p=2, dim=0)
            else:
                new_centroids[b, k] = old_centroids[b, k]
    return new_centroids

def _centroid_update_torch_native(x, cluster_ids, old_centroids, mode = 'euclid'):
    """
    Torch naive implementation for centroid update.
    
    Args:
        x: (B, N, D) input points
        cluster_ids: (B, N) cluster assignment per point
        old_centroids: (B, K, D) previous centroids
        mode: 'euclid' or 'cosine'
        
    Returns:
        centroids_new: (B, K, D) updated centroids
    """
    B, N, D = x.shape
    K = old_centroids.shape[1]
    assert mode in ['euclid', 'cosine'], "Mode must be 'euclid' or 'cosine'"
    

    cluster_sums = torch.zeros((B, K, D), device=x.device, dtype=torch.float32)
    cluster_counts = torch.zeros((B, K), device=x.device, dtype=torch.float32)
    
    for b in range(B):
        # Accumulate sums and counts by torch
        cluster_sums[b].index_add_(0, cluster_ids[b], x[b].to(torch.float32))
        ones = torch.ones((N,), device=x.device, dtype=torch.float32)
        cluster_counts[b].index_add_(0, cluster_ids[b], ones)



    cluster_counts.unsqueeze_(-1)  # Avoid division by zero
    empty_mask = (cluster_counts == 0)
    cluster_counts.clamp_min_(1.0)
    centroids_new = cluster_sums / cluster_counts
    centroids = torch.where(empty_mask, old_centroids, centroids_new)
    if mode == 'cosine':
        centroids = F.normalize(centroids, p=2, dim=-1)
    return centroids.to(x.dtype)


def _euclid_iter_torch_naive(x, x_sq, centroids, chunk_size_N=32768, chunk_size_K=1024):
    """
    One iteration of KMeans using pure PyTorch (fallback when Triton is not available).
    
    Args:
        x: (B, N, D) input points
        x_sq: (B, N) pre-computed ||x||^2
        centroids: (B, K, D) cluster centers
        chunk_size: chunk size for assignment to avoid OOM
        
    Returns:
        centroids_new: (B, K, D) updated centroids
        shift: scalar, max centroid movement
        cluster_ids: (B, N) cluster assignments
    """
    # Assignment step: find nearest centroid for each point
    cluster_ids = euclid_assign_torch_native_chunked(x, centroids, x_sq, chunk_size_N, chunk_size_K)

    # Update step: recompute centroids
    centroids_new = _centroid_update_torch_native(x, cluster_ids, centroids)
    
    # Compute shift
    shift = (centroids_new - centroids).norm(dim=-1).max()
    
    return centroids_new, shift, cluster_ids

def batch_kmeans_Euclid_torch_native(x, n_clusters, max_iters=100, tol=0.0, init_centroids=None, verbose=False, chunk_size_N=32768, chunk_size_K=1024):
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
        centroids_new, center_shift, cluster_ids = _euclid_iter_torch_naive(x, x_sq, centroids, chunk_size_N, chunk_size_K)

        if verbose:
            print(f"Iter {it}, center shift: {center_shift.item():.6f}")
        if center_shift < tol:
            break
        centroids = centroids_new.clone()

    return cluster_ids, centroids, it + 1

if __name__ == "__main__":
    torch.manual_seed(0)

    # Simple test accuracy
    B, N, D, K = 32, 74256, 128, 1000
    x = torch.randn(B, N, D, device="cuda")
    cent = torch.randn(B, K, D, device="cuda")
    x_sq = (x.to(torch.float32) ** 2).sum(-1)
    centroids = torch.randn(B, K, D, device='cuda')


    ## test _euclid_assign_torch_chunked

    # torch ref
    # dist = (
    #     x_sq.unsqueeze(-1) + (cent.to(torch.float32) ** 2).sum(-1).unsqueeze(1) - 2.0 * torch.einsum("bnd,bkd->bnk", x, cent).to(torch.float32)
    # ).clamp_min_(0.0)
    # ref_ids = dist.argmin(dim=-1)
    # _euclid_assign_torch_chunked
    impl_ids = euclid_assign_torch_native_chunked(x, cent, x_sq) 

    # torch.testing.assert_close(ref_ids.to(torch.float32), impl_ids.to(torch.float32))





