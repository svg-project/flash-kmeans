"""Drop-in ``FlashKMeans`` estimator backed by the C++/CUDA core.

The constructor, attributes and method semantics match the original
Triton-based implementation; the compute itself lives in
``flash_kmeans::batch_kmeans`` / ``flash_kmeans::kmeans_large_n`` (C++/CUDA).
"""

from __future__ import annotations

from typing import Optional

import torch

from . import ops


class FlashKMeans:
    """
    Fast batched K-Means clustering implemented with C++/CUDA kernels.

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
    use_triton : bool, default=True
        Kept for API compatibility.  The backend is always the C++/CUDA core.
    seed : int, default=0
        Random seed for centroid initialization.
    chunk_size_data : int, default=32768
        Kept for API compatibility (the C++ kernel manages its own tiling).
    chunk_size_centroids : int, default=1024
        Kept for API compatibility.
    chunk_size_data_cpu : int, default=1048576
        Chunk size along n_samples when streaming CPU data to the GPU.
    verbose : bool, default=False
        Whether to print per-iteration info.
    dtype : torch.dtype, optional
        Compute dtype for the algorithm.
    device : torch.device | None
        Target device. None means "cuda:0" for in-memory data and all visible
        GPUs for the large-N streaming path.
    """

    def __init__(
        self,
        d: int,
        k: int,
        niter: int = 25,
        tol: float = 1e-8,
        use_triton: bool = True,
        seed: int = 0,
        chunk_size_data: int = 32768,
        chunk_size_centroids: int = 1024,
        chunk_size_data_cpu: int = 1048576,
        verbose: bool = False,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        self.d = int(d)
        self.k = int(k)
        self.niter = int(niter)
        self.tol = float(tol)
        self.use_triton = bool(use_triton)
        self.seed = int(seed)
        self.chunk_size_data = int(chunk_size_data)
        self.chunk_size_centroids = int(chunk_size_centroids)
        self.chunk_size_data_cpu = int(chunk_size_data_cpu)
        self.verbose = bool(verbose)
        self.dtype = dtype

        self._raw_device = device
        if device is None:
            self.device = torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device(device)

        self.centroids_b: Optional[torch.Tensor] = None
        self.cluster_ids_b: Optional[torch.Tensor] = None
        self._batch_size: Optional[int] = None

    # ------------------------------------------------------------------
    # fitting
    # ------------------------------------------------------------------

    def train(self, data: torch.Tensor):
        """
        Fit KMeans on data and store centroids.

        Parameters
        ----------
        data : torch.Tensor
            Shape (n_samples, n_features) or (batch_size, n_samples, n_features).
            CPU tensors larger than ``chunk_size_data_cpu`` are streamed to the
            GPU in chunks (``kmeans_largeN`` path).
        """
        if data.ndim == 2:
            N, D = data.shape
            B = None
            x_b = data.unsqueeze(0)
        elif data.ndim == 3:
            B, N, D = data.shape
            x_b = data
        else:
            raise ValueError(
                "data must be of shape (n_samples, n_features) or "
                "(batch_size, n_samples, n_features)"
            )

        if data.device.type == "cpu" and N > self.chunk_size_data_cpu:
            assert B is None, "Batched data with large N on CPU is not supported yet."
            labels, centroids = ops.kmeans_largeN(
                x_b[0],
                self.k,
                max_iters=self.niter,
                tol=self.tol,
                verbose=self.verbose,
                BLOCK_N=self.chunk_size_data_cpu,
                device=self._raw_device,
                dtype=self.dtype,
                seed=self.seed,
            )
            self.cluster_ids_b = labels.unsqueeze_(0)
            self.centroids_b = centroids.unsqueeze_(0)
        else:
            compute_dtype = self.dtype or x_b.dtype
            x_b = x_b.to(device=self.device, dtype=compute_dtype, copy=False)
            labels_b, centroids_b, _ = ops.batch_kmeans_Euclid(
                x_b,
                self.k,
                max_iters=self.niter,
                tol=self.tol,
                init_centroids=None,
                verbose=self.verbose,
                seed=self.seed,
            )
            self.cluster_ids_b = labels_b
            self.centroids_b = centroids_b

        self._batch_size = B
        return self

    def fit(self, data: torch.Tensor):
        """Alias for train; returns self."""
        return self.train(data)

    # ------------------------------------------------------------------
    # inference
    # ------------------------------------------------------------------

    def predict(self, data: torch.Tensor) -> torch.Tensor:
        """
        Assign each point to the nearest centroid.

        Parameters
        ----------
        data : torch.Tensor
            Shape (n_samples, n_features) or (batch_size, n_samples, n_features).
            If the model was trained batched (batch_size > 1), the same batch
            size is required.
        """
        if self.centroids_b is None:
            raise RuntimeError("Model not trained. Call train() or fit() first.")

        if data.ndim == 2:
            B = None
            N, D = data.shape
            x_b = data.unsqueeze(0)
        elif data.ndim == 3:
            B, N, D = data.shape
            x_b = data
        else:
            raise ValueError(
                "data must be of shape (n_samples, n_features) or "
                "(batch_size, n_samples, n_features)"
            )

        if B != self._batch_size:
            raise ValueError(
                f"Model was trained with batch size B={self._batch_size}, "
                f"but predict received B={B}. Provide matching batch size."
            )

        if data.device.type == "cpu" and N > self.chunk_size_data_cpu:
            assert B is None, "Batched data with large N on CPU is not supported yet."
            return ops.kmeans_largeN_assign(
                x_b[0],
                self.centroids_b[0],
                dtype=self.dtype,
                BLOCK_N=self.chunk_size_data_cpu,
                device=self._raw_device,
            )

        compute_dtype = self.dtype or x_b.dtype
        x_b = x_b.to(device=self.device, dtype=compute_dtype, copy=False)
        labels_b = ops.euclid_assign_triton(x_b, self.centroids_b)
        if B is None:
            return labels_b.squeeze(0)
        return labels_b

    def fit_predict(self, data: torch.Tensor) -> torch.Tensor:
        """
        Fit KMeans on data and return the cluster labels.

        Returns (n_samples,) for 2D input and (batch_size, n_samples) for 3D.
        """
        self.train(data)
        if self._batch_size is None:
            return self.cluster_ids_b.squeeze(0)
        return self.cluster_ids_b
