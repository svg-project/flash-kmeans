"""Pure-PyTorch k-means fallback (device agnostic).

Kept for API compatibility with the original release and as a reference
implementation for tests; the production path is the C++/CUDA extension.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def euclid_assign_torch_native_chunked(
    x: torch.Tensor,
    centroids: torch.Tensor,
    x_sq: torch.Tensor | None = None,
    chunk_size_N: int = 32768,
    chunk_size_K: int = 1024,
) -> torch.Tensor:
    """Nearest-centroid labels for (B, N, D) input, computed in chunks."""
    B, N, D = x.shape
    K = centroids.shape[1]
    cent_sq = (centroids.to(torch.float32) ** 2).sum(dim=-1)
    labels = torch.empty((B, N), dtype=torch.int32, device=x.device)
    for n_start in range(0, N, chunk_size_N):
        n_end = min(n_start + chunk_size_N, N)
        x_chunk = x[:, n_start:n_end].to(torch.float32)
        x_sq_chunk = (
            x_chunk.pow(2).sum(-1) if x_sq is None else x_sq[:, n_start:n_end].float()
        )
        dists = torch.empty((B, n_end - n_start, K), device=x.device,
                            dtype=torch.float32)
        for k_start in range(0, K, chunk_size_K):
            k_end = min(k_start + chunk_size_K, K)
            c = centroids[:, k_start:k_end].to(torch.float32)
            dists[:, :, k_start:k_end] = (
                x_sq_chunk.unsqueeze(-1)
                - 2.0 * torch.bmm(x_chunk, c.transpose(1, 2))
                + cent_sq[:, k_start:k_end].unsqueeze(1)
            )
        labels[:, n_start:n_end] = dists.argmin(dim=-1)
    return labels


def _torch_update(x, labels, old, mode):
    B, N, D = x.shape
    K = old.shape[1]
    sums = torch.zeros((B, K, D), device=x.device, dtype=torch.float32)
    counts = torch.zeros((B, K), device=x.device, dtype=torch.float32)
    ones = torch.ones((N,), device=x.device, dtype=torch.float32)
    for b in range(B):
        sums[b].index_add_(0, labels[b].long(), x[b].to(torch.float32))
        counts[b].index_add_(0, labels[b].long(), ones)
    empty = counts == 0
    new = sums / counts.clamp_min(1.0).unsqueeze(-1)
    new = torch.where(empty.unsqueeze(-1), old.to(torch.float32), new)
    if mode == "cosine":
        new = F.normalize(new, p=2, dim=-1)
    return new.to(x.dtype)


def batch_kmeans_Euclid_torch_native(
    x,
    n_clusters,
    max_iters=100,
    tol=0.0,
    init_centroids=None,
    verbose=False,
    chunk_size_N=32768,
    chunk_size_K=1024,
):
    """Batched Euclidean k-means in pure PyTorch (CPU or CUDA)."""
    B, N, D = x.shape
    K = int(n_clusters)
    if init_centroids is None:
        idx = torch.randint(0, N, (B, K), device=x.device)
        centroids = torch.gather(x, 1, idx[..., None].expand(-1, -1, D))
    else:
        centroids = init_centroids.view(B, K, D)

    labels = torch.zeros((B, N), dtype=torch.int32, device=x.device)
    it = 0
    for it in range(max_iters):
        labels = euclid_assign_torch_native_chunked(
            x, centroids, None, chunk_size_N, chunk_size_K
        )
        new = _torch_update(x, labels, centroids, "euclid")
        shift = (new.to(torch.float32) - centroids.to(torch.float32)).norm(
            dim=-1
        ).max().item()
        if verbose:
            print(f"Iter {it}, center shift: {shift:.6f}")
        if shift < tol:
            return labels, centroids, it + 1
        centroids = new
    return labels, centroids, it + 1
