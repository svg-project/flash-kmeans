"""Pure-PyTorch reference implementation used by the equivalence tests.

This mirrors the semantics of the original flash-kmeans Python fallback:
  * assignment   : ||x - c||^2 = ||x||^2 - 2 x.c + ||c||^2 (fp32 accumulation)
  * centroid     : fp32 mean of assigned points, empty clusters keep the old
                   centroid, optional L2 normalization (cosine / dot modes)
  * convergence  : max per-(batch, cluster) centroid movement < tol
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def ref_assign_euclid(x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    x_sq = (x.to(torch.float32) ** 2).sum(-1)
    c_sq = (c.to(torch.float32) ** 2).sum(-1)
    dist = (
        x_sq.unsqueeze(-1)
        + c_sq.unsqueeze(1)
        - 2.0 * torch.bmm(x.to(torch.float32), c.to(torch.float32).transpose(1, 2))
    )
    return dist.argmin(dim=-1).to(torch.int32)


def ref_assign_dot(x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    sim = torch.bmm(x.to(torch.float32), c.to(torch.float32).transpose(1, 2))
    return sim.argmax(dim=-1).to(torch.int32)


def ref_update(
    x: torch.Tensor,
    labels: torch.Tensor,
    old: torch.Tensor,
    mode: str = "euclid",
) -> torch.Tensor:
    B, N, D = x.shape
    K = old.shape[1]
    sums = torch.zeros((B, K, D), device=x.device, dtype=torch.float32)
    counts = torch.zeros((B, K), device=x.device, dtype=torch.float32)
    ones = torch.ones((N,), device=x.device, dtype=torch.float32)
    for b in range(B):
        sums[b].index_add_(0, labels[b], x[b].to(torch.float32))
        counts[b].index_add_(0, labels[b], ones)
    empty = counts == 0
    new = sums / counts.clamp_min(1.0).unsqueeze(-1)
    new = torch.where(empty.unsqueeze(-1), old.to(torch.float32), new)
    if mode in ("cosine", "dot"):
        new = new / new.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return new.to(x.dtype)


def ref_kmeans(
    x: torch.Tensor,
    n_clusters: int,
    max_iters: int = 100,
    tol: float = 0.0,
    init_centroids: torch.Tensor | None = None,
    mode: str = "euclid",
    seed: int = 0,
):
    """Returns (labels, centroids, n_iters) like ``batch_kmeans_Euclid``."""
    if x.dim() == 2:
        x = x.unsqueeze(0)
    B, N, D = x.shape
    K = int(n_clusters)

    if init_centroids is None:
        g = torch.Generator(device=x.device).manual_seed(int(seed))
        idx = torch.randint(0, N, (B, K), device=x.device, generator=g)
        c = torch.gather(x, 1, idx[..., None].expand(-1, -1, D)).clone()
    else:
        c = init_centroids.to(x.dtype).clone()

    if mode == "cosine":
        x_src = F.normalize(x.to(torch.float32), dim=-1).to(x.dtype)
        c = F.normalize(c.to(torch.float32), dim=-1).to(x.dtype)
    else:
        x_src = x

    labels = torch.zeros((B, N), device=x.device, dtype=torch.int32)
    it = 0
    for it in range(max_iters):
        if mode == "euclid":
            labels = ref_assign_euclid(x_src, c)
        else:
            labels = ref_assign_dot(x_src, c)
        new = ref_update(x_src, labels, c, mode)
        shift = (new.to(torch.float32) - c.to(torch.float32)).norm(dim=-1).max().item()
        if shift < tol:
            return labels, c, it + 1
        c = new
    return labels, c, it + 1
