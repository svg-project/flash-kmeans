
from __future__ import annotations

from typing import Optional

import torch

try:
    from flash_kmeans.kmeans_triton_impl import batch_kmeans_Euclid 
    from flash_kmeans.assign_euclid_triton import euclid_assign_triton
    _HAS_TRITON_IMPL = True
except Exception:
    _HAS_TRITON_IMPL = False


def _require_triton_cuda():
    if not _HAS_TRITON_IMPL:
        raise RuntimeError(
            "flash_kmeans Triton kernels are not available. "
            "Ensure the package modules are importable."
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run the Triton-backed k-means implementation.")


class FlashKMeans:
    """
    Fast batched K-Means clustering implemented with Triton GPU kernels.

    Parameters
    ----------
    d : int
        Feature dimensionality (n_features).
    k : int
        Number of clusters. (n_clusters)
    niter : int, default=25
        Maximum iterations.
    tol : float, default=1e-8
        Convergence tolerance on centroid shift.
    seed : int, default=0
        Random seed for centroid initialization.
    verbose : bool, default=False
        Whether to print per-iteration info.
    dtype : torch.dtype, optional
        Compute Data type for algorithm.
    device : torch.device | None
        Target device. Defaults to "cuda:0" when available.
        Currently, only CUDA devices are supported.
    """

    def __init__(
        self,
        d: int,
        k: int,
        niter: int = 25,
        tol: float = 1e-8,
        seed: int = 0,
        verbose: bool = False,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        self.d = int(d)
        self.k = int(k)
        self.niter = int(niter)
        self.tol = float(tol)
        self.seed = int(seed)
        self.verbose = bool(verbose)
        self.dtype = dtype

        # default device
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        # # Model state
        # # Centroids are stored on CPU in float32 to minimize GPU memory usage post-training.
        # # Shape is always (B, K, D), where B=1 for unbatched training.
        # self.centroids: Optional[torch.Tensor] = None

    def train(self, data: torch.Tensor):
        """
        Fit KMeans on data and store centroids.

        Parameters
        ----------
        data : torch.Tensor
            Accepts Shape:
            - (n_samples, n_features)
            - (batch_size, n_samples, n_features)

        """
        _require_triton_cuda()

        if data.ndim == 2:
            N, D = data.shape
            B = None
            x_b = data.unsqueeze(0)  # (1, N, D)
        elif data.ndim == 3:
            B, N, D = data.shape
            x_b = data
        else:
            raise ValueError("data must be of shape (n_samples, n_features) or (batch_size, n_samples, n_features)")

        # Ensure CUDA + dtype
        compute_dtype = self.dtype or x_b.dtype 
        x_b = x_b.to(device=self.device, dtype=compute_dtype, copy=False)

        # Set random seed
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

        # Run batched Triton KMeans (Euclidean)
        cluster_ids_b, centroids_b, _ = batch_kmeans_Euclid(
            x_b,
            self.k,
            max_iters=self.niter,
            tol=self.tol,
            init_centroids=None,
            verbose=self.verbose,
        )
 
        self.centroids_b = centroids_b
        self.cluster_ids_b = cluster_ids_b
        self._batch_size = B

    def fit(self, data: torch.Tensor):
        """Alias for train; returns self."""
        self.train(data)
        return self

    def predict(self, data: torch.Tensor) -> torch.LongTensor:
        """
        Assign each point to the nearest centroid using the Triton assign kernel.

        Parameters
        ----------
        data : torch.Tensor
            Accepts Shape:
            - (n_samples, n_features)
            - (batch_size, n_samples, n_features)

        If model was trained batched (batch_size>1), prediction must be provided with the same batch_size.
        """
        _require_triton_cuda()

        if self.centroids_b is None or self._batch_size is None:
            raise RuntimeError("Model not trained. Call train() or fit() first.")

        # Normalize input shape
        if data.ndim == 2:
            B = None
            N, D = data.shape
            x_b = data.unsqueeze(0)  # (1, N, D)
        elif data.ndim == 3:
            B, N, D = data.shape
            x_b = data
        else:
            raise ValueError("data must be of shape (n_samples, n_features) or (batch_size, n_samples, n_features)")

        if B != self._batch_size:
            raise ValueError(
                f"Model was trained with batch size B={self._batch_size}, "
                f"but predict received B={B}. Provide matching batch size."
            )

        # Prepare tensors for kernel call
        compute_dtype = self.dtype or x_b.dtype 
        x_b = x_b.to(device=self.device, dtype=compute_dtype, copy=False)
 
        x_sq = (x_b ** 2).sum(dim=-1)
        # Call Triton assignment kernel
        labels_b = euclid_assign_triton(x_b, self.centroids_b, x_sq)

        if B is None:
            return labels_b.squeeze(0)  # (N,)
        return labels_b  # (B, N)

    def fit_predict(self, data: torch.Tensor) -> torch.tensor:
        """
        Fit KMeans on data and store centroids.

        Parameters
        ----------
        data : torch.Tensor
            Input data for clustering.
            data shape accepts:
            - (n_samples, n_features)
            - (batch_size, n_samples, n_features)

        
        Returns
        -------
        labels : torch.LongTensor (int64)
            Shape depending on input:
            - (n_samples,) if input was (n_samples, n_features)
            - (batch_size, n_samples) if input was (batch_size, n_samples, n_features)

        """
        # cluster_ids: (B, N)
        self.train(data)
        return self.cluster_ids_b.squeeze(0) if self._batch_size is None else self.cluster_ids_b
