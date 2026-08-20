"""Backend-dispatched public Euclidean k-means API."""
from __future__ import annotations

from typing import Optional
import warnings

import torch

from .torch_fallback import (
    batch_kmeans_Euclid_torch_native,
    euclid_assign_torch_native_chunked,
)

BACKENDS = ("triton", "cute", "torch")


class _TritonBackendUnavailable(RuntimeError):
    pass


def _validate_backend(backend: str) -> str:
    if backend not in BACKENDS:
        choices = ", ".join(repr(choice) for choice in BACKENDS)
        raise ValueError(f"backend must be one of {choices}, got {backend!r}.")
    return backend


def _require_triton_backend():
    try:
        from .kmeans_triton_impl import batch_kmeans_Euclid as implementation
    except Exception as exc:
        raise _TritonBackendUnavailable(
            "flash_kmeans Triton kernels are not available. Ensure Triton and "
            "the package runtime dependencies are installed."
        ) from exc
    return implementation


def _batch_kmeans_euclid_triton(**kwargs):
    return _require_triton_backend()(**kwargs)


def batch_kmeans_Euclid(
    x: torch.Tensor,
    n_clusters: int,
    max_iters: int = 100,
    tol: float = 0.0,
    init_centroids: Optional[torch.Tensor] = None,
    verbose: bool = False,
    *,
    use_heuristic: bool = True,
    backend: str = "triton",
):
    """Run batched Euclidean k-means with the selected backend.

    ``backend="triton"`` is the default and preserves the existing behavior.
    The optional CuTe backend currently supports CUDA bf16 inputs with B>=1,
    D=128, N>=2, and 2 to 1024 clusters on SM90, SM100, and SM120 GPUs.
    """
    backend = _validate_backend(backend)
    common = dict(
        x=x,
        n_clusters=n_clusters,
        max_iters=max_iters,
        tol=tol,
        init_centroids=init_centroids,
        verbose=verbose,
    )
    if backend == "triton":
        try:
            return _batch_kmeans_euclid_triton(
                **common, use_heuristic=use_heuristic
            )
        except _TritonBackendUnavailable as exc:
            warnings.warn(
                f"Falling back to PyTorch implementation: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            return batch_kmeans_Euclid_torch_native(**common)
    if backend == "cute":
        from .cute_backend import batch_kmeans_Euclid as cute_kmeans

        return cute_kmeans(**common)
    return batch_kmeans_Euclid_torch_native(**common)


def euclid_assign(
    x: torch.Tensor,
    centroids: torch.Tensor,
    x_sq: Optional[torch.Tensor],
    *,
    backend: str,
    chunk_size_data: int = 32768,
    chunk_size_centroids: int = 1024,
) -> torch.Tensor:
    """Assign points to centroids using the selected backend."""
    backend = _validate_backend(backend)
    if backend == "triton":
        from .assign_euclid_triton import euclid_assign_triton

        return euclid_assign_triton(x, centroids, x_sq)
    if backend == "cute":
        from .cute_backend import euclid_assign_cute

        return euclid_assign_cute(x, centroids)
    return euclid_assign_torch_native_chunked(
        x,
        centroids,
        x_sq,
        chunk_size_N=chunk_size_data,
        chunk_size_K=chunk_size_centroids,
    )


__all__ = ["BACKENDS", "batch_kmeans_Euclid"]
