"""CuTe DSL implementation of batched Euclidean k-means.

This subpackage is imported only when ``backend="cute"`` is selected, so the
optional CuTe dependencies do not affect the default Triton installation.
"""
from __future__ import annotations

from typing import Optional

import torch

from .arch import get_lloyd_module

FEATURE_DIM = 128
MIN_BATCH_SIZE = 1
MIN_SAMPLES = 2
MIN_CLUSTERS = 2
MAX_CLUSTERS = 1024
ALIGNMENT_BYTES = 16  # every kernel operand is compiled with assumed_align=16


def _aligned(tensor: torch.Tensor) -> torch.Tensor:
    """A contiguous, 16-byte-aligned view of ``tensor`` (copying only if needed)."""
    tensor = tensor.contiguous()
    if tensor.data_ptr() % ALIGNMENT_BYTES:
        tensor = tensor.clone()
    return tensor


def _unused(device, *shape):
    """A stand-in for a kernel output this backend does not consume.

    ``mBest`` (the packed score of the assigned centroid) is write-only, and
    ``mSums`` is untouched on the assign-only path. Their stores are dropped
    at trace time (``write_best`` and ``fuse_sums`` are compile-time
    constants), so the kernel never indexes these tensors -- but they still
    have to be passed, because they are part of the kernel ABI.

    ``mXsq`` deliberately does NOT go through here: see the comment at its
    allocation in ``batch_kmeans_Euclid``.
    """
    return torch.empty(*shape, dtype=torch.float32, device=device)


def _validate_inputs(
    x: torch.Tensor,
    n_clusters: int,
    max_iters: int,
    init_centroids: Optional[torch.Tensor],
) -> None:
    if x.ndim != 3:
        raise ValueError(f"x must have shape (B, N, D), got {tuple(x.shape)}")
    if x.shape[0] < MIN_BATCH_SIZE:
        raise ValueError(
            f"The CuTe backend requires at least {MIN_BATCH_SIZE} batch, "
            f"got B={x.shape[0]}."
        )
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
    if x.shape[1] < MIN_SAMPLES:
        raise ValueError(
            f"The CuTe backend requires at least {MIN_SAMPLES} samples, "
            f"got N={x.shape[1]}."
        )
    if not MIN_CLUSTERS <= int(n_clusters) <= MAX_CLUSTERS:
        raise ValueError(
            f"The CuTe backend supports at most {MAX_CLUSTERS} clusters "
            f"(and at least {MIN_CLUSTERS}), got {n_clusters}."
        )
    if max_iters < 1:
        raise ValueError(f"max_iters must be at least 1, got {max_iters}.")
    # The kernels are compiled with assumed_align=16 on every operand, and
    # .contiguous() is a no-op on a tensor that is already contiguous, so a
    # legal but oddly-offset view has to be rejected here. Left unchecked it
    # surfaces either as an opaque tvm-ffi "Misaligned Tensor data" or, when
    # tvm-ffi is absent, as an asynchronous CUDA misaligned-address fault that
    # poisons the whole context.
    if x.data_ptr() % ALIGNMENT_BYTES:
        raise ValueError(
            f"The CuTe backend requires x to be {ALIGNMENT_BYTES}-byte "
            "aligned; this tensor is a view at an unaligned offset. Pass "
            "x.clone() (or an unsliced tensor) instead."
        )
    if init_centroids is not None:
        expected = (x.shape[0], int(n_clusters), FEATURE_DIM)
        # Match the Triton backend, which just does init_centroids.view(B,K,D):
        # any shape holding the right number of elements is accepted, so a
        # (K, D) array from a previous 2-D fit works.
        if init_centroids.numel() != expected[0] * expected[1] * expected[2]:
            raise ValueError(
                f"init_centroids must hold {expected[0] * expected[1] * expected[2]} "
                f"elements (reshapeable to {expected}), got "
                f"{tuple(init_centroids.shape)} with {init_centroids.numel()}."
            )
        if not init_centroids.dtype.is_floating_point:
            raise ValueError(
                "init_centroids must be a floating-point tensor, got "
                f"{init_centroids.dtype}."
            )
        if init_centroids.device != x.device:
            raise ValueError("init_centroids must be on the same device as x.")


def _checked_centroid_sq(centroids: torch.Tensor) -> torch.Tensor:
    """``||c||^2`` in fp32, refused if it is not finite.

    The kernels seed the packed argmin with 3.0e38 and pad unused centroid
    columns with the same value. A non-finite ``c_sq`` makes every real column
    lose to that sentinel, and the sentinel is not itself a packed value, so it
    decodes to a nonsense cluster id. The kernels now clamp that id -- it can
    no longer run off the end of the histogram and centroid-sum buffers -- but
    the labels would still be meaningless, so reject the input here. This is
    the one non-finite case worth paying a sync for: it is free to detect
    (``c_sq`` is computed anyway) and it is silent otherwise.
    """
    centroid_sq = (centroids.float() ** 2).sum(-1).contiguous()
    if not bool(torch.isfinite(centroid_sq).all()):
        raise ValueError(
            "Centroid norms are not finite (||c||^2 overflowed fp32). The "
            "CuTe backend scores points as ||c||^2 - 2*x.c, so it needs "
            "finite centroid norms; rescale the input."
        )
    return centroid_sq


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
        centroids = _aligned(
            torch.gather(x, 1, indices[..., None].expand(-1, -1, dim))
        )
    else:
        centroids = _aligned(
            init_centroids.reshape(batch, n_clusters, dim)
            .to(dtype=torch.bfloat16)
            .clone()
        )

    centroid_sq = _checked_centroid_sq(centroids)
    ids = torch.empty(batch, n_samples, dtype=torch.int32, device=x.device)
    best = _unused(x.device, 1, 4)
    counts = torch.zeros(batch, n_clusters, dtype=torch.int32, device=x.device)
    sums = torch.zeros(
        batch, n_clusters, dim, dtype=torch.float32, device=x.device
    )
    # Full-size even though write_xsq=0 below means nothing is ever written to
    # it: shrinking it requires compiling out the ||x||^2 block, which costs
    # 2.7x on SM100 (16.1 -> 43.9 ms at B=80 N=75600 K=591 on B200) because
    # rx[] loses its second consumer and the batched row prefetch collapses.
    # See the comment at that block in lloyd_sm100.py.
    x_sq = torch.empty(batch, n_samples, dtype=torch.float32, device=x.device)
    views = module._views(x, centroids, centroid_sq, ids, best, counts, sums, x_sq)
    assign, finalize = module._get_compiled(
        n_samples, n_clusters, batch, x.device, views, write_best=False
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
    centroids = _aligned(centroids.to(dtype=torch.bfloat16))
    module = get_lloyd_module(x.device)

    import cuda.bindings.driver as cuda
    from cutlass import Int32

    batch, n_samples, dim = x.shape
    centroid_sq = _checked_centroid_sq(centroids)
    ids = torch.empty(batch, n_samples, dtype=torch.int32, device=x.device)
    best = _unused(x.device, 1, 4)
    counts = torch.zeros(batch, n_clusters, dtype=torch.int32, device=x.device)
    # The assign-only specialization never touches the centroid sums, and with
    # fuse_sums=False the ||x||^2 block is gone too, so both are placeholders.
    sums = _unused(x.device, 1, 1, 4)
    x_sq = _unused(x.device, 1, 4)
    views = module._views(x, centroids, centroid_sq, ids, best, counts, sums, x_sq)
    assign, _ = module._get_compiled(
        n_samples,
        n_clusters,
        batch,
        x.device,
        views,
        fuse_sums=False,
        write_best=False,
    )
    stream = cuda.CUstream(torch.cuda.current_stream(x.device).cuda_stream)
    assign(*views, Int32(0), stream)
    return ids


__all__ = ["batch_kmeans_Euclid", "euclid_assign_cute"]
