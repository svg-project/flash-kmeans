"""Thin Python wrappers around the flash-kmeans C++/CUDA core.

All numerical work (assignment, centroid update, the k-means iteration loop and
the large-N streaming pipeline) happens in ``_flash_kmeans_cpp``.  This module
only marshals torch tensors -- pointers, shapes, dtype codes and the current
CUDA stream -- into the extension, keeping the original public API and
signatures.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch

from . import _flash_kmeans_cpp as _C

__all__ = [
    "batch_kmeans_Euclid",
    "batch_kmeans_Cosine",
    "batch_kmeans_Dot",
    "euclid_assign_triton",
    "cosine_assign_triton",
    "triton_centroid_update_euclid",
    "triton_centroid_update_sorted_euclid",
    "triton_centroid_update_cosine",
    "triton_centroid_update_sorted_cosine",
    "kmeans_largeN",
    "kmeans_largeN_assign",
    "resolve_devices",
]

_F32, _F16, _BF16 = 0, 1, 2
_EUCLID, _COSINE, _DOT = 0, 1, 2

_DTYPE_CODES = {
    torch.float32: _F32,
    torch.float16: _F16,
    torch.bfloat16: _BF16,
}


def _code(dtype: torch.dtype) -> int:
    try:
        return _DTYPE_CODES[dtype]
    except KeyError:
        raise TypeError(
            f"flash-kmeans supports float32 / float16 / bfloat16, got {dtype}"
        ) from None


def _device_index(device: torch.device) -> int:
    if device.type != "cuda":
        raise RuntimeError("flash-kmeans requires CUDA tensors")
    return device.index if device.index is not None else torch.cuda.current_device()


def _stream() -> int:
    return torch.cuda.current_stream().cuda_stream


def _require_cuda(t: torch.Tensor, name: str) -> torch.Tensor:
    if t.device.type != "cuda":
        raise RuntimeError(f"flash-kmeans: {name} must be a CUDA tensor")
    return t


def resolve_devices(device) -> list[int]:
    """Resolve a device argument to a list of CUDA ordinals.

    ``device=None`` -> all visible GPUs (multi-GPU for large-N workloads).
    """
    if device is None:
        count = torch.cuda.device_count()
        if count == 0:
            raise RuntimeError("No CUDA devices available")
        return list(range(count))
    dev = torch.device(device)
    if dev.type != "cuda":
        raise ValueError(f"flash-kmeans only supports CUDA devices, got {dev}")
    return [dev.index if dev.index is not None else torch.cuda.current_device()]


# ---------------------------------------------------------------------------
# batched k-means
# ---------------------------------------------------------------------------


def _batch_kmeans(
    x: torch.Tensor,
    n_clusters: int,
    max_iters: int,
    tol: float,
    init_centroids: Optional[torch.Tensor],
    verbose: bool,
    seed: int,
    mode: int,
):
    _require_cuda(x, "x")
    if x.dim() == 2:
        x = x.unsqueeze(0)
    if x.dim() != 3:
        raise ValueError("x must have shape (B, N, D) or (N, D)")
    x = x.contiguous()
    B, N, D = (int(v) for v in x.shape)
    K = int(n_clusters)
    if min(B, N, D, K) <= 0:
        raise ValueError("B, N, D and n_clusters must all be positive")

    init = None
    if init_centroids is not None:
        init = init_centroids.to(device=x.device, dtype=x.dtype).contiguous()
        if init.dim() == 2:
            init = init.unsqueeze(0)
        if tuple(init.shape) != (B, K, D):
            raise ValueError(
                f"init_centroids must have shape ({B}, {K}, {D}), got {tuple(init.shape)}"
            )
    labels = torch.empty((B, N), dtype=torch.int32, device=x.device)
    centroids = torch.empty((B, K, D), dtype=x.dtype, device=x.device)
    n_iters = _C.batch_kmeans(
        x.data_ptr(),
        0 if init is None else init.data_ptr(),
        labels.data_ptr(),
        centroids.data_ptr(),
        B,
        N,
        D,
        K,
        int(max_iters),
        float(tol),
        int(mode),
        _code(x.dtype),
        int(seed),
        bool(verbose),
        _device_index(x.device),
        _stream(),
    )
    return labels, centroids, int(n_iters)


def batch_kmeans_Euclid(
    x: torch.Tensor,
    n_clusters: int,
    max_iters: int = 100,
    tol: float = 0.0,
    init_centroids: Optional[torch.Tensor] = None,
    verbose: bool = False,
    *,
    seed: int = 0,
    use_heuristic: bool = True,
):
    """Batched Euclidean k-means.  Returns (labels, centroids, n_iters)."""
    return _batch_kmeans(x, n_clusters, max_iters, tol, init_centroids, verbose,
                         seed, _EUCLID)


def batch_kmeans_Cosine(
    x: torch.Tensor,
    n_clusters: int,
    max_iters: int = 100,
    tol: float = 0.0,
    init_centroids: Optional[torch.Tensor] = None,
    verbose: bool = False,
    *,
    seed: int = 0,
):
    return _batch_kmeans(x, n_clusters, max_iters, tol, init_centroids, verbose,
                         seed, _COSINE)


def batch_kmeans_Dot(
    x: torch.Tensor,
    n_clusters: int,
    max_iters: int = 100,
    tol: float = 0.0,
    init_centroids: Optional[torch.Tensor] = None,
    verbose: bool = False,
    *,
    seed: int = 0,
):
    return _batch_kmeans(x, n_clusters, max_iters, tol, init_centroids, verbose,
                         seed, _DOT)


# ---------------------------------------------------------------------------
# assignment
# ---------------------------------------------------------------------------


def _assign(x: torch.Tensor, centroids: torch.Tensor, out, mode: int):
    _require_cuda(x, "x")
    _require_cuda(centroids, "centroids")
    x = x.contiguous()
    centroids = centroids.contiguous()
    B, N, D = (int(v) for v in x.shape)
    K = int(centroids.shape[1])
    if out is None:
        out = torch.empty((B, N), dtype=torch.int32, device=x.device)
    _C.euclid_assign(
        x.data_ptr(),
        centroids.data_ptr(),
        out.data_ptr(),
        B,
        N,
        D,
        K,
        int(mode),
        _code(x.dtype),
        _device_index(x.device),
        _stream(),
    )
    return out


def euclid_assign_triton(
    x: torch.Tensor,
    centroids: torch.Tensor,
    x_sq: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    c_sq: Optional[torch.Tensor] = None,
    *,
    BLOCK_N: int = 128,
    BLOCK_K: int = 128,
    num_warps: Optional[int] = None,
    num_stages: Optional[int] = None,
    config: Optional[dict] = None,
    use_heuristic: bool = True,
) -> torch.Tensor:
    """Nearest-centroid labels (int32).  ``x_sq``/``c_sq`` are accepted for API
    compatibility; the C++ kernel derives them internally."""
    return _assign(x, centroids, out, _EUCLID)


def cosine_assign_triton(
    x: torch.Tensor,
    centroids: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    *,
    BLOCK_N: int = 128,
    BLOCK_K: int = 128,
) -> torch.Tensor:
    """Argmax cosine-similarity labels; inputs are expected to be normalized."""
    return _assign(x, centroids, out, _COSINE)


# ---------------------------------------------------------------------------
# centroid update
# ---------------------------------------------------------------------------


def _centroid_update(
    x: torch.Tensor,
    cluster_ids: torch.Tensor,
    old_centroids: torch.Tensor,
    centroid_sums: Optional[torch.Tensor],
    centroid_cnts: Optional[torch.Tensor],
    calculate_new: bool,
    mode: int,
):
    _require_cuda(x, "x")
    x = x.contiguous()
    cluster_ids = cluster_ids.contiguous()
    if cluster_ids.dtype != torch.int32:
        cluster_ids = cluster_ids.to(torch.int32)
    old = old_centroids.contiguous()
    B, N, D = (int(v) for v in x.shape)
    K = int(old.shape[1])
    out = torch.empty_like(old) if calculate_new else old
    sums_ptr = 0 if centroid_sums is None else centroid_sums.data_ptr()
    counts_ptr = 0 if centroid_cnts is None else centroid_cnts.data_ptr()
    _C.centroid_update(
        x.data_ptr(),
        cluster_ids.data_ptr(),
        old.data_ptr(),
        sums_ptr,
        counts_ptr,
        out.data_ptr(),
        B,
        N,
        D,
        K,
        int(mode),
        _code(x.dtype),
        bool(calculate_new),
        _device_index(x.device),
        _stream(),
    )
    return out if calculate_new else None


def triton_centroid_update_euclid(
    x: torch.Tensor, cluster_ids: torch.Tensor, old_centroids: torch.Tensor
) -> torch.Tensor:
    return _centroid_update(x, cluster_ids, old_centroids, None, None, True, _EUCLID)


def triton_centroid_update_sorted_euclid(
    x: torch.Tensor,
    cluster_ids: torch.Tensor,
    old_centroids: torch.Tensor,
    *,
    BLOCK_N: int = 256,
    centroid_sums: Optional[torch.Tensor] = None,
    centroid_cnts: Optional[torch.Tensor] = None,
    calculate_new: bool = True,
):
    return _centroid_update(x, cluster_ids, old_centroids, centroid_sums,
                            centroid_cnts, calculate_new, _EUCLID)


def triton_centroid_update_cosine(
    x_norm: torch.Tensor, cluster_ids: torch.Tensor, old_centroids: torch.Tensor
) -> torch.Tensor:
    return _centroid_update(x_norm, cluster_ids, old_centroids, None, None, True, _COSINE)


def triton_centroid_update_sorted_cosine(
    x_norm: torch.Tensor,
    cluster_ids: torch.Tensor,
    old_centroids: torch.Tensor,
    *,
    BLOCK_N: int = 256,
) -> torch.Tensor:
    return _centroid_update(x_norm, cluster_ids, old_centroids, None, None, True, _COSINE)


# ---------------------------------------------------------------------------
# large-N streaming (CPU data)
# ---------------------------------------------------------------------------


def kmeans_largeN(
    x: torch.Tensor,
    n_clusters: int,
    max_iters: int = 100,
    tol: float = 0.0,
    verbose: bool = False,
    BLOCK_N: int = 1048576,
    init_centroids: Optional[torch.Tensor] = None,
    device=None,
    dtype: Optional[torch.dtype] = None,
    seed: int = 0,
):
    """Streaming k-means for CPU-resident data, optionally across all GPUs.

    Returns ``(cluster_ids, centroids)``.  ``cluster_ids``/``centroids`` live on
    the single GPU when one device is used and in host memory (pinned) when the
    work is partitioned across several GPUs, matching the original behaviour.
    """
    if x.device.type != "cpu":
        raise ValueError("kmeans_largeN expects x to be on the CPU")
    if x.dim() != 2:
        raise ValueError("kmeans_largeN expects x with shape (N, D)")
    dtype = dtype or x.dtype
    _code(dtype)
    x = x.to(dtype)
    if not x.is_contiguous():
        x = x.contiguous()
    if torch.cuda.is_available() and not x.is_pinned():
        x = x.pin_memory()
    N, D = (int(v) for v in x.shape)
    K = int(n_clusters)

    devices = resolve_devices(device)
    if len(devices) == 1:
        dev = torch.device(f"cuda:{devices[0]}")
        labels = torch.empty((N,), dtype=torch.int32, device=dev)
        centroids = torch.empty((K, D), dtype=dtype, device=dev)
        labels_to_cpu = False
        centroids_to_cpu = False
    else:
        labels = torch.empty((N,), dtype=torch.int32, pin_memory=True)
        centroids = torch.empty((K, D), dtype=dtype, pin_memory=True)
        labels_to_cpu = True
        centroids_to_cpu = True

    init = None
    if init_centroids is not None:
        primary = torch.device(f"cuda:{devices[0]}")
        init = init_centroids.to(device=primary, dtype=dtype).contiguous()

    _C.kmeans_large_n(
        x.data_ptr(),
        0 if init is None else init.data_ptr(),
        labels.data_ptr(),
        labels_to_cpu,
        centroids.data_ptr(),
        centroids_to_cpu,
        N,
        D,
        K,
        int(max_iters),
        float(tol),
        _code(dtype),
        int(BLOCK_N),
        [int(d) for d in devices],
        int(seed),
        bool(verbose),
    )
    return labels, centroids


def kmeans_largeN_assign(
    x: torch.Tensor,
    centroids: torch.Tensor,
    BLOCK_N: int = 1048576,
    device=None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Assign labels for large CPU-resident ``x`` against ``centroids``."""
    if x.device.type != "cpu":
        raise ValueError("kmeans_largeN_assign expects x to be on the CPU")
    if x.dim() != 2:
        raise ValueError("kmeans_largeN_assign expects x with shape (N, D)")
    dtype = dtype or x.dtype
    _code(dtype)
    x = x.to(dtype)
    if not x.is_contiguous():
        x = x.contiguous()
    if torch.cuda.is_available() and not x.is_pinned():
        x = x.pin_memory()
    N, D = (int(v) for v in x.shape)
    K = int(centroids.shape[0])

    devices = resolve_devices(device)
    centroids_on_device = centroids.device.type == "cuda"
    if centroids_on_device:
        cent = centroids.to(device=f"cuda:{devices[0]}", dtype=dtype).contiguous()
    else:
        cent = centroids.to(dtype=dtype).contiguous()

    if len(devices) == 1:
        labels = torch.empty((N,), dtype=torch.int32,
                             device=torch.device(f"cuda:{devices[0]}"))
        labels_to_cpu = False
    else:
        labels = torch.empty((N,), dtype=torch.int32, pin_memory=True)
        labels_to_cpu = True

    _C.kmeans_large_n_assign(
        x.data_ptr(),
        cent.data_ptr(),
        bool(centroids_on_device),
        labels.data_ptr(),
        labels_to_cpu,
        N,
        D,
        K,
        _code(dtype),
        int(BLOCK_N),
        [int(d) for d in devices],
    )
    return labels
