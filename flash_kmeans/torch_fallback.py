import math

import torch
import torch.nn.functional as F


def _kmeanspp_sequential(x, n_clusters, x_sq=None, weights=None):
    """
    Standard batched kmeans++ initialization (Arthur & Vassilvitskii, 2007).

    K sequential rounds, each sampling one centroid proportional to min distance².
    Used internally by scalable kmeans++ for the final candidate reduction step.

    Args:
        x: (B, N, D) input points.
        n_clusters: number of clusters K.
        x_sq: (B, N) precomputed ||x||^2, optional.
        weights: (B, N) per-sample weights, optional.

    Returns:
        centroids: (B, K, D) initial centroids.
    """
    B, N, D = x.shape
    device = x.device
    if x_sq is None:
        x_sq = (x ** 2).sum(dim=-1)

    centroids = torch.empty((B, n_clusters, D), device=device, dtype=x.dtype)
    batch_arange = torch.arange(B, device=device)
    w_f = weights.float() if weights is not None else None

    # First centroid
    if w_f is not None:
        w_probs = w_f / w_f.sum(dim=-1, keepdim=True).clamp_min(1e-30)
        first_idx = torch.multinomial(w_probs, 1).squeeze(-1)
    else:
        first_idx = torch.randint(0, N, (B,), device=device)
    centroids[:, 0] = x[batch_arange, first_idx]

    if n_clusters == 1:
        return centroids

    c = centroids[:, 0:1, :]
    c_sq = (c ** 2).sum(dim=-1)
    min_dists = (x_sq - 2 * torch.bmm(x, c.transpose(1, 2)).squeeze(-1) + c_sq).float()
    min_dists.clamp_min_(0)

    for k in range(1, n_clusters):
        probs = min_dists * w_f if w_f is not None else min_dists
        prob_sums = probs.sum(dim=-1, keepdim=True)
        # Avoid CPU-GPU sync: use torch.where instead of if zero_rows.any()
        probs = torch.where(prob_sums > 0, probs / prob_sums.clamp_min(1e-30), 1.0 / N)

        idx = torch.multinomial(probs, 1).squeeze(-1)
        centroids[:, k] = x[batch_arange, idx]

        new_c = centroids[:, k:k+1, :]
        new_c_sq = (new_c ** 2).sum(dim=-1)
        new_dists = (x_sq - 2 * torch.bmm(x, new_c.transpose(1, 2)).squeeze(-1) + new_c_sq).float()
        new_dists.clamp_min_(0)
        torch.minimum(min_dists, new_dists, out=min_dists)

    return centroids


def _update_min_dists_batched(x, x_sq, new_cands, min_dists, max_bytes=512 * 1024 * 1024):
    """Update min_dists in-place with distances to new_cands, chunking along
    the candidate dimension to keep peak memory under max_bytes."""
    B, N, _ = x.shape
    n_cands = new_cands.shape[1]
    # Chunk size so (B, N, chunk) float32 tensor fits in max_bytes
    chunk_l = max(1, max_bytes // (B * N * 4))

    for l_start in range(0, n_cands, chunk_l):
        l_end = min(l_start + chunk_l, n_cands)
        chunk = new_cands[:, l_start:l_end, :]
        chunk_sq = (chunk ** 2).sum(dim=-1)  # (B, chunk)
        # (B, N, chunk) distances in input dtype, cast to float32
        dists = (
            x_sq.unsqueeze(-1)
            - 2 * torch.bmm(x, chunk.transpose(1, 2))
            + chunk_sq.unsqueeze(-2)
        ).float()
        dists.clamp_min_(0)
        chunk_min = dists.min(dim=-1).values  # (B, N)
        torch.minimum(min_dists, chunk_min, out=min_dists)


def standard_kmeans_pp(x, n_clusters, x_sq=None, weights=None, n_local_trials=None):
    """
    Standard kmeans++ with greedy local trials (Arthur & Vassilvitskii, 2007).
    Matches scikit-learn's _kmeans_plusplus implementation.

    Each step samples n_local_trials candidates and picks the one that
    minimizes the total potential (sum of weighted min distances²).

    Args:
        x: (B, N, D) input points.
        n_clusters: K.
        x_sq: (B, N) precomputed ||x||^2, optional.
        weights: (B, N) per-sample weights, optional.
        n_local_trials: candidates per step. Default: 2 + int(log(K)).

    Returns:
        centroids: (B, K, D) initial centroids.
    """
    B, N, D = x.shape
    device = x.device
    if x_sq is None:
        x_sq = (x ** 2).sum(dim=-1)

    if n_local_trials is None:
        n_local_trials = 2 + int(math.log(n_clusters))

    centroids = torch.empty((B, n_clusters, D), device=device, dtype=x.dtype)
    batch_arange = torch.arange(B, device=device)
    w_f = weights.float() if weights is not None else None

    # First centroid
    if w_f is not None:
        w_probs = w_f / w_f.sum(dim=-1, keepdim=True).clamp_min(1e-30)
        first_idx = torch.multinomial(w_probs, 1).squeeze(-1)
    else:
        first_idx = torch.randint(0, N, (B,), device=device)
    centroids[:, 0] = x[batch_arange, first_idx]

    if n_clusters == 1:
        return centroids

    # Initial min distances: dist² from each point to first centroid
    c = centroids[:, 0:1, :]  # (B, 1, D)
    c_sq = (c ** 2).sum(dim=-1)  # (B, 1)
    closest_dist_sq = (
        x_sq - 2 * torch.bmm(x, c.transpose(1, 2)).squeeze(-1) + c_sq
    ).float()
    closest_dist_sq.clamp_min_(0)

    for k in range(1, n_clusters):
        # Sampling probabilities: dist² * weight
        # Avoids allocating ones tensor for unweighted case
        weighted_dists = closest_dist_sq * w_f if w_f is not None else closest_dist_sq
        prob_sums = weighted_dists.sum(dim=-1, keepdim=True)
        # Avoid CPU-GPU sync: use torch.where instead of if zero_rows.any()
        probs = torch.where(prob_sums > 0, weighted_dists / prob_sums.clamp_min(1e-30), 1.0 / N)

        # Sample n_local_trials candidates (with replacement, matching sklearn)
        n_trials = min(n_local_trials, N)
        candidate_ids = torch.multinomial(probs, n_trials, replacement=True)  # (B, n_trials)
        candidates = torch.gather(
            x, 1, candidate_ids.unsqueeze(-1).expand(-1, -1, D)
        )  # (B, n_trials, D)

        # Distances from each candidate to all points: (B, n_trials, N)
        cand_sq = (candidates ** 2).sum(dim=-1)  # (B, n_trials)
        dist_to_cands = (
            x_sq.unsqueeze(1)
            - 2 * torch.bmm(candidates, x.transpose(1, 2))
            + cand_sq.unsqueeze(-1)
        ).float()
        dist_to_cands.clamp_min_(0)

        # For each candidate, new min distances
        new_min_dists = torch.min(
            closest_dist_sq.unsqueeze(1), dist_to_cands
        )  # (B, n_trials, N)

        # Potential per candidate: sum of weighted new_min_dists
        # Weighted: bmm with weight vector. Unweighted: simple sum (avoids bmm with ones).
        if w_f is not None:
            candidates_pot = torch.bmm(
                new_min_dists, w_f.unsqueeze(-1)
            ).squeeze(-1)  # (B, n_trials)
        else:
            candidates_pot = new_min_dists.sum(dim=-1)  # (B, n_trials)

        # Pick best candidate per batch (lowest potential)
        best_trial = candidates_pot.argmin(dim=-1)  # (B,)
        best_id = candidate_ids[batch_arange, best_trial]
        centroids[:, k] = x[batch_arange, best_id]
        closest_dist_sq = new_min_dists[batch_arange, best_trial]  # (B, N)

    return centroids


def scalable_kmeans_pp(x, n_clusters, x_sq=None, weights=None,
                       oversampling_factor=2.0, n_rounds=8):
    """
    Scalable K-Means++ (K-Means||) initialization (Bahmani et al., 2012).
    Matches cuml/cuvs implementation.

    Performs n_rounds passes, each sampling l = oversampling_factor * K
    candidates in parallel, then reduces via weighted sequential kmeans++.

    Args:
        x: (B, N, D) input points.
        n_clusters: K.
        x_sq: (B, N) precomputed ||x||^2, optional.
        weights: (B, N) per-sample weights, optional.
        oversampling_factor: candidates per round = oversampling_factor * K (default 2.0).
        n_rounds: number of oversampling rounds (default 8, matching cuml).

    Returns:
        centroids: (B, K, D) initial centroids.
    """
    B, N, D = x.shape
    device = x.device
    l = max(1, int(oversampling_factor * n_clusters))

    if x_sq is None:
        x_sq = (x ** 2).sum(dim=-1)

    w_f = weights.float() if weights is not None else None
    batch_arange = torch.arange(B, device=device)

    # --- Step 1: first center ---
    if w_f is not None:
        w_probs = w_f / w_f.sum(dim=-1, keepdim=True).clamp_min(1e-30)
        first_idx = torch.multinomial(w_probs, 1).squeeze(-1)
    else:
        first_idx = torch.randint(0, N, (B,), device=device)

    # For very small K, sequential is cheaper than the oversampling overhead
    if n_clusters <= 3:
        centroids = torch.empty((B, 1, D), device=device, dtype=x.dtype)
        centroids[:, 0] = x[batch_arange, first_idx]
        if n_clusters == 1:
            return centroids
        return _kmeanspp_sequential(x, n_clusters, x_sq, weights)

    candidates_list = [x[batch_arange, first_idx].unsqueeze(1)]  # [(B, 1, D)]

    # Initial min distances
    c = candidates_list[0]
    c_sq = (c ** 2).sum(dim=-1)
    min_dists = (x_sq - 2 * torch.bmm(x, c.transpose(1, 2)).squeeze(-1) + c_sq).float()
    min_dists.clamp_min_(0)

    # --- Step 2: oversampling rounds ---
    for _ in range(n_rounds):
        probs = min_dists * w_f if w_f is not None else min_dists
        prob_sums = probs.sum(dim=-1, keepdim=True)
        # Avoid CPU-GPU sync: use torch.where instead of if zero_rows.any()
        probs = torch.where(prob_sums > 0, probs / prob_sums.clamp_min(1e-30), 1.0 / N)

        n_samples = min(l, N)
        new_idx = torch.multinomial(probs, n_samples, replacement=False)  # (B, l)
        new_cands = torch.gather(
            x, 1, new_idx.unsqueeze(-1).expand(-1, -1, D)
        )  # (B, l, D)
        candidates_list.append(new_cands)

        # Update min distances with all new candidates at once (chunked)
        _update_min_dists_batched(x, x_sq, new_cands, min_dists)

    # --- Step 3: collect candidates ---
    all_candidates = torch.cat(candidates_list, dim=1)  # (B, C, D)
    C = all_candidates.shape[1]

    if C <= n_clusters:
        # Degenerate: pad with random points
        extra_idx = torch.randint(0, N, (B, n_clusters - C), device=device)
        extra = torch.gather(x, 1, extra_idx.unsqueeze(-1).expand(-1, -1, D))
        return torch.cat([all_candidates, extra], dim=1)

    # --- Step 4: weight candidates by assigned-point count ---
    assignments = euclid_assign_torch_native_chunked(
        x, all_candidates, x_sq
    )  # (B, N) int32

    cand_weights = torch.zeros((B, C), device=device, dtype=torch.float32)
    if w_f is not None:
        cand_weights.scatter_add_(1, assignments.long(), w_f)
    else:
        cand_weights.scatter_add_(
            1, assignments.long(),
            torch.ones((B, N), device=device, dtype=torch.float32),
        )

    # --- Step 5: reduce C candidates to K centroids ---
    # Greedy weighted kmeans++ on candidates (matches cuml's reduction). The
    # greedy local-trials variant is used (not plain sequential) so the final
    # reduction is as robust as standard_kmeans_pp; a plain reduction
    # occasionally picks two candidates from one cluster and leaves another
    # uncovered, producing rare but severe quality blow-ups.
    cand_sq = (all_candidates.float() ** 2).sum(-1)
    return standard_kmeans_pp(all_candidates, n_clusters, cand_sq, weights=cand_weights)


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

def batch_kmeans_Euclid_torch_native(x, n_clusters, max_iters=100, tol=0.0, init_centroids=None, verbose=False, chunk_size_N=32768, chunk_size_K=1024, init="random"):
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
        if init == "scalable-kmeans++":
            centroids = scalable_kmeans_pp(x, n_clusters, x_sq)
        elif init == "standard-kmeans++":
            centroids = standard_kmeans_pp(x, n_clusters, x_sq)
        else:
            # Randomly select initial centers from x (without replacement, matching sklearn)
            uniform = torch.ones(B, N, device=x.device)
            indices = torch.multinomial(uniform, n_clusters, replacement=False)
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





