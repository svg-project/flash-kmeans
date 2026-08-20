
from __future__ import annotations

import warnings
from typing import Optional

import torch

from flash_kmeans.kmeans_api import (
    BACKENDS,
    _require_triton_backend,
    batch_kmeans_Euclid,
    euclid_assign,
)
from flash_kmeans.kmeans_large import kmeans_largeN, kmeans_largeN_assign
from flash_kmeans.torch_fallback import batch_kmeans_Euclid_torch_native


def _require_triton_cuda():
    _require_triton_backend()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run the Triton-backed k-means implementation.")


class FlashKMeans:
    """
    Fast batched K-Means clustering with selectable GPU backends.

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
        Backward-compatible backend switch. Ignored when ``backend`` is set.
    seed : int, default=0
        Random seed for centroid initialization.
    chunk_size_data : int, default=32768
        Only used when fallback to PyTorch implementation.
        Chunk size along the data dimension for assignment/update steps.
    chunk_size_centroids : int, default=1024
        Only used when fallback to PyTorch implementation.
        Chunk size along the centroid dimension for assignment/update steps.
    chunk_size_data_cpu : int, default=1048576
        Only when n_samples is too large to fit into GPU memory, this parameter controls
        the chunk size of n_samples when copying data from CPU to GPU in chunks.
    verbose : bool, default=False
        Whether to print per-iteration info.
    dtype : torch.dtype, optional
        Compute Data type for algorithm.
    device : torch.device | None
        Target device. Defaults to "cuda:0" when available.
        Currently, only CUDA devices are supported.
    backend : {"triton", "cute", "torch"} | None
        Implementation to use. ``None`` preserves the existing ``use_triton``
        behavior. The CuTe backend requires CUDA bf16 data with D=128.
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
        backend: Optional[str] = None,
    ):
        self.d = int(d)
        self.k = int(k)
        self.niter = int(niter)
        self.tol = float(tol)
        backend_was_explicit = backend is not None
        if backend is None:
            backend = "triton" if use_triton else "torch"
        elif backend not in BACKENDS:
            choices = ", ".join(repr(choice) for choice in BACKENDS)
            raise ValueError(f"backend must be one of {choices}, got {backend!r}.")
        self.backend = backend
        self.use_triton = backend == "triton"
        self.seed = int(seed)
        self.chunk_size_data = int(chunk_size_data)
        self.chunk_size_centroids = int(chunk_size_centroids)
        self.chunk_size_data_cpu = int(chunk_size_data_cpu)
        self.verbose = bool(verbose)
        self.dtype = dtype

        if self.backend == "triton":
            try:
                _require_triton_cuda()
            except RuntimeError as e:
                if backend_was_explicit:
                    raise
                warnings.warn(
                    f"Falling back to PyTorch implementation: {e}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self.backend = "torch"
                self.use_triton = False
        elif self.backend == "cute" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is required to run the CuTe-backed k-means implementation.")

        # Store raw device for largeN multi-GPU path (None = auto-detect all GPUs)
        self._raw_device = device
        # default device for single-GPU / in-memory paths
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        self.centroids_b = None
        self.cluster_ids_b = None


    def train(self, data: torch.Tensor):
        """
        Fit KMeans on data and store centroids.

        Parameters
        ----------
        data : torch.Tensor
            Accepts Shape:
            - (n_samples, n_features)
            - (batch_size, n_samples, n_features)

            if data is from GPU, it will process directly on GPU.
            if data is from CPU, it will copy & process data on GPU by chunk_size_data_cpu.

        """

        if data.ndim == 2:
            N, D = data.shape
            B = None
            x_b = data.unsqueeze(0)  # (1, N, D)
        elif data.ndim == 3:
            B, N, D = data.shape
            x_b = data
        else:
            raise ValueError("data must be of shape (n_samples, n_features) or (batch_size, n_samples, n_features)")

        # Set random seed
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

        if data.device.type == "cpu" and N > self.chunk_size_data_cpu:
            # handle for large N on CPU
            assert B is None, "Batched data with large N on CPU is not supported yet."
            if self.backend != "triton":
                raise NotImplementedError(
                    "Large CPU-resident datasets currently require backend='triton'."
                )
            cluster_ids_b, centroids_b  = kmeans_largeN(
                x_b[0],
                self.k,
                max_iters=self.niter,
                tol=self.tol,
                verbose=self.verbose,
                dtype=self.dtype,
                BLOCK_N=self.chunk_size_data_cpu,
                device=self._raw_device,
            )
            centroids_b.unsqueeze_(0)
            cluster_ids_b.unsqueeze_(0)
        else:
            # Ensure CUDA + dtype
            compute_dtype = self.dtype or x_b.dtype
            x_b = x_b.to(device=self.device, dtype=compute_dtype, copy=False)

            if self.backend != "torch":
                cluster_ids_b, centroids_b, _ = batch_kmeans_Euclid(
                    x_b,
                    self.k,
                    max_iters=self.niter,
                    tol=self.tol,
                    init_centroids=None,
                    verbose=self.verbose,
                    backend=self.backend,
                )
            else:
                # Run batched PyTorch KMeans (Euclidean)
                cluster_ids_b, centroids_b, _ = batch_kmeans_Euclid_torch_native(
                    x_b,
                    self.k,
                    max_iters=self.niter,
                    tol=self.tol,
                    init_centroids=None,
                    verbose=self.verbose,
                    chunk_size_N=self.chunk_size_data,
                    chunk_size_K=self.chunk_size_centroids,
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
        Assign each point to the nearest centroid using the selected backend.

        Parameters
        ----------
        data : torch.Tensor
            Accepts Shape:
            - (n_samples, n_features)
            - (batch_size, n_samples, n_features)

        If model was trained batched (batch_size>1), prediction must be provided with the same batch_size.
        """

        if self.centroids_b is None:
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
        
        if data.device.type == "cpu" and N > self.chunk_size_data_cpu:
            # handle for large N on CPU
            assert B is None, "Batched data with large N on CPU is not supported yet."
            if self.backend != "triton":
                raise NotImplementedError(
                    "Large CPU-resident datasets currently require backend='triton'."
                )
            labels = kmeans_largeN_assign(
                x_b[0],
                self.centroids_b[0],
                dtype=self.dtype,
                BLOCK_N=self.chunk_size_data_cpu,
                device=self._raw_device,
            )
            return labels  # (N,)
    
        # Prepare tensors for kernel call
        compute_dtype = self.dtype or x_b.dtype 
        x_b = x_b.to(device=self.device, dtype=compute_dtype, copy=False)
 
        x_sq = None
        if self.backend != "cute":
            # Chunked to avoid materializing a full (B, N, D) temp.
            N_ = x_b.shape[1]
            x_sq = torch.empty(x_b.shape[:-1], device=x_b.device, dtype=x_b.dtype)
            _CHUNK = 1 << 20
            for i in range(0, N_, _CHUNK):
                x_sq[:, i:i + _CHUNK] = (x_b[:, i:i + _CHUNK] ** 2).sum(dim=-1)

        labels_b = euclid_assign(
            x_b,
            self.centroids_b,
            x_sq,
            backend=self.backend,
            chunk_size_data=self.chunk_size_data,
            chunk_size_centroids=self.chunk_size_centroids,
        )

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
