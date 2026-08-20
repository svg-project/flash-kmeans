"""CuTe DSL implementation of batched Euclidean k-means.

This subpackage is imported only when ``backend="cute"`` is selected, so the
optional CuTe dependencies do not affect the default Triton installation.
"""
from __future__ import annotations

from typing import Optional

import torch

from .arch import get_lloyd_module

FEATURE_DIM = 128
MAX_CLUSTERS = 1024


def _validate_inputs(
    x: torch.Tensor,
    n_clusters: int,
    max_iters: int,
    init_centroids: Optional[torch.Tensor],
) -> None:
    if x.ndim != 3:
        raise ValueError(f"x must have shape (B, N, D), got {tuple(x.shape)}")
    if not x.is_cuda:
        raise ValueError("The CuTe backend requires x to be a CUDA tensor.")
    if x.dtype != torch.bfloat16:
        raise ValueError(
            f"The CuTe backend requires x.dtype=torch.bfloat16, got {x.dtype}."
        )
    if x.shape[-1] != FEATURE_DIM:
        raise ValueError(
            f"The CuTe backend requires D={FEATURE_DIM}, got D={x.shape[-1]}."
        )
    if not 1 <= int(n_clusters) <= MAX_CLUSTERS:
        raise ValueError(
            f"The CuTe backend supports at most {MAX_CLUSTERS} clusters "
            f"(and at least 1), got {n_clusters}."
        )
    if max_iters < 1:
        raise ValueError(f"max_iters must be at least 1, got {max_iters}.")
    if init_centroids is not None:
        expected = (x.shape[0], int(n_clusters), FEATURE_DIM)
        if tuple(init_centroids.shape) != expected:
            raise ValueError(
                f"init_centroids must have shape {expected}, got "
                f"{tuple(init_centroids.shape)}."
            )
        if init_centroids.device != x.device:
            raise ValueError("init_centroids must be on the same device as x.")


@torch.no_grad()
def batch_kmeans_Euclid(
    x: torch.Tensor,
    n_clusters: int,
    max_iters: int = 100,
    tol: float = 0.0,
    init_centroids: Optional[torch.Tensor] = None,
    verbose: bool = False,
):
    """Run batched Euclidean k-means with fused CuTe DSL kernels."""
    _validate_inputs(x, n_clusters, max_iters, init_centroids)
    x = x.contiguous()
    module = get_lloyd_module(x.device)

    # Keep imports below dependency and architecture validation so importing
    # flash_kmeans never requires the optional CuTe stack.
    import cuda.bindings.driver as cuda
    from cutlass import Int32

    batch, n_samples, dim = x.shape
    n_clusters = int(n_clusters)
    if init_centroids is None:
        indices = torch.randint(
            0, n_samples, (batch, n_clusters), device=x.device
        )
        centroids = torch.gather(
            x, 1, indices[..., None].expand(-1, -1, dim)
        ).contiguous()
    else:
        centroids = init_centroids.to(dtype=torch.bfloat16).contiguous().clone()

    centroid_sq = (centroids.float() ** 2).sum(-1).contiguous()
    ids = torch.empty(batch, n_samples, dtype=torch.int32, device=x.device)
    best = torch.empty(batch, n_samples, dtype=torch.float32, device=x.device)
    counts = torch.zeros(batch, n_clusters, dtype=torch.int32, device=x.device)
    sums = torch.zeros(
        batch, n_clusters, dim, dtype=torch.float32, device=x.device
    )
    x_sq = torch.empty(batch, n_samples, dtype=torch.float32, device=x.device)
    views = module._views(x, centroids, centroid_sq, ids, best, counts, sums, x_sq)
    assign, finalize = module._get_compiled(
        n_samples, n_clusters, batch, x.device, views
    )
    xv, cv, csqv, iv, bv, countsv, sumsv, xsqv = views
    stream = cuda.CUstream(torch.cuda.current_stream(x.device).cuda_stream)

    check_shift = bool(verbose) or tol > 0.0
    n_iters = 0
    for iteration in range(max_iters):
        previous = centroids.clone() if check_shift else None
        assign(xv, cv, csqv, iv, bv, countsv, sumsv, xsqv, Int32(0), stream)
        finalize(
            sumsv,
            countsv,
            cv,
            csqv,
            Int32(1 if iteration < max_iters - 1 else 0),
            stream,
        )
        n_iters = iteration + 1
        if check_shift:
            center_shift = (centroids - previous).norm(dim=-1).max()
            if verbose:
                print(f"Iter {iteration}, center shift: {center_shift.item():.6f}")
            if tol > 0.0 and center_shift < tol:
                # Match the existing Triton return contract: labels and
                # centroids both describe the assignment side of this step.
                centroids.copy_(previous)
                break

    return ids, centroids, n_iters


@torch.no_grad()
def euclid_assign_cute(x: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
    """Assign points to fixed centroids using the CuTe backend."""
    n_clusters = centroids.shape[1]
    _validate_inputs(x, n_clusters, 1, centroids)
    x = x.contiguous()
    centroids = centroids.to(dtype=torch.bfloat16).contiguous()
    module = get_lloyd_module(x.device)

    import cuda.bindings.driver as cuda
    from cutlass import Int32

    batch, n_samples, dim = x.shape
    centroid_sq = (centroids.float() ** 2).sum(-1).contiguous()
    ids = torch.empty(batch, n_samples, dtype=torch.int32, device=x.device)
    best = torch.empty(batch, n_samples, dtype=torch.float32, device=x.device)
    counts = torch.zeros(batch, n_clusters, dtype=torch.int32, device=x.device)
    # The assign-only specialization does not read these two buffers, but they
    # remain in the common kernel ABI.
    sums = torch.empty(
        batch, n_clusters, dim, dtype=torch.float32, device=x.device
    )
    x_sq = torch.empty(batch, n_samples, dtype=torch.float32, device=x.device)
    views = module._views(x, centroids, centroid_sq, ids, best, counts, sums, x_sq)
    assign, _ = module._get_compiled(
        n_samples,
        n_clusters,
        batch,
        x.device,
        views,
        fuse_sums=False,
        topj=1,
    )
    stream = cuda.CUstream(torch.cuda.current_stream(x.device).cuda_stream)
    assign(*views, Int32(0), stream)
    return ids


__all__ = ["batch_kmeans_Euclid", "euclid_assign_cute"]
