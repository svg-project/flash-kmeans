
from __future__ import annotations

from typing import Optional
from flash_kmeans.torch_fallback import euclid_assign_torch_native_chunked, batch_kmeans_Euclid_torch_native

try:
    from flash_kmeans.kmeans_triton_impl import batch_kmeans_Euclid_weighted as _batch_kmeans_Euclid_weighted
    _HAS_WEIGHTED = True
except Exception:
    _HAS_WEIGHTED = False
import torch

try:
    from flash_kmeans.kmeans_triton_impl import batch_kmeans_Euclid 
    from flash_kmeans.assign_euclid_triton import euclid_assign_triton
    from flash_kmeans.kmeans_large import kmeans_largeN, kmeans_largeN_assign
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
    use_triton : bool, default=True
        Whether to use triton implementation. If False, falls back to PyTorch implementation.
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
    init : str, default="random"
        Centroid initialization method.
        - "random": uniform random selection from data points.
        - "scalable-kmeans++": scalable kmeans++ / K-Means|| (Bahmani et al., 2012), matches cuml.
        - "standard-kmeans++": standard greedy kmeans++ (Arthur & Vassilvitskii, 2007), matches scikit-learn.
    n_init : int | str, default="auto"
        Number of times k-means is run with different initializations. The
        result with the lowest inertia is kept. "auto" = 10 for random init,
        1 for kmeans++ (matching sklearn).
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
        use_triton: bool = True,
        seed: int = 0,
        chunk_size_data: int = 32768,
        chunk_size_centroids: int = 1024,
        chunk_size_data_cpu: int = 1048576,
        verbose: bool = False,
        init: str = "random",
        n_init: int | str = "auto",
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
        self.init = init
        self.dtype = dtype

        if n_init == "auto":
            self.n_init = 10 if init == "random" else 1
        else:
            self.n_init = int(n_init)

        if self.use_triton:
            try:
                _require_triton_cuda()
            except RuntimeError as e:
                Warning(f"Falling back to PyTorch implementation: {e}")
                self.use_triton = False

        # Store raw device for largeN multi-GPU path (None = auto-detect all GPUs)
        self._raw_device = device
        # default device for single-GPU / in-memory paths
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device


    def train(self, data: torch.Tensor, weights: torch.Tensor = None):
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

        weights : torch.Tensor, optional
            Per-sample weights for weighted k-means.
            Shape: (n_samples,) or (batch_size, n_samples)

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

        # Normalize weights shape
        if weights is not None:
            if weights.ndim == 1:
                weights_b = weights.unsqueeze(0)
            else:
                weights_b = weights
            weights_b = weights_b.to(device=self.device, dtype=torch.float32, copy=False)
        else:
            weights_b = None

        best_inertia = None  # (B_int,) per-batch inertia
        best_centroids_b = None
        best_cluster_ids_b = None
        B_int = x_b.shape[0]

        for run_i in range(self.n_init):
            torch.manual_seed(self.seed + run_i)
            torch.cuda.manual_seed_all(self.seed + run_i)

            cluster_ids_b, centroids_b = self._run_kmeans_once(
                x_b, N, B, data, weights_b,
            )

            if self.n_init > 1:
                # Per-batch inertia so we can pick the best run independently
                # for each batch element, not just the best run overall.
                D = x_b.shape[-1]
                x_eval = x_b.to(device=self.device, dtype=centroids_b.dtype, copy=False)
                assigned = centroids_b.gather(
                    1, cluster_ids_b.unsqueeze(-1).expand(-1, -1, D)
                )
                inertia = ((x_eval - assigned) ** 2).sum(dim=(-1, -2))  # (B_int,)

                if best_inertia is None:
                    best_inertia = inertia
                    best_centroids_b = centroids_b
                    best_cluster_ids_b = cluster_ids_b
                else:
                    improved = inertia < best_inertia  # (B_int,)
                    best_inertia = torch.where(improved, inertia, best_inertia)
                    # Update centroids and assignments only for improved batches
                    mask_c = improved[:, None, None].expand_as(centroids_b)
                    mask_id = improved[:, None].expand_as(cluster_ids_b)
                    best_centroids_b = torch.where(mask_c, centroids_b, best_centroids_b)
                    best_cluster_ids_b = torch.where(mask_id, cluster_ids_b, best_cluster_ids_b)
            else:
                best_centroids_b = centroids_b
                best_cluster_ids_b = cluster_ids_b

        self.centroids_b = best_centroids_b
        self.cluster_ids_b = best_cluster_ids_b
        self._batch_size = B

    def _run_kmeans_once(self, x_b, N, B, data, weights_b):
        """Run a single k-means pass and return (cluster_ids_b, centroids_b)."""
        if weights_b is not None:
            assert _HAS_WEIGHTED, "Weighted k-means requires Triton implementation"
            compute_dtype = self.dtype or x_b.dtype
            x_b = x_b.to(device=self.device, dtype=compute_dtype, copy=False)
            cluster_ids_b, centroids_b, _ = _batch_kmeans_Euclid_weighted(
                x_b,
                self.k,
                weights_b,
                max_iters=self.niter,
                tol=self.tol,
                init_centroids=None,
                verbose=self.verbose,
                init=self.init,
            )
        elif data.device.type == "cpu" and N > self.chunk_size_data_cpu:
            assert B is None, "Batched data with large N on CPU is not supported yet."
            assert self.use_triton, "process large N data requires triton implementation."
            cluster_ids_b, centroids_b = kmeans_largeN(
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
            compute_dtype = self.dtype or x_b.dtype
            x_b = x_b.to(device=self.device, dtype=compute_dtype, copy=False)

            if self.use_triton:
                cluster_ids_b, centroids_b, _ = batch_kmeans_Euclid(
                    x_b,
                    self.k,
                    max_iters=self.niter,
                    tol=self.tol,
                    init_centroids=None,
                    verbose=self.verbose,
                    init=self.init,
                )
            else:
                cluster_ids_b, centroids_b, _ = batch_kmeans_Euclid_torch_native(
                    x_b,
                    self.k,
                    max_iters=self.niter,
                    tol=self.tol,
                    init_centroids=None,
                    verbose=self.verbose,
                    chunk_size_N=self.chunk_size_data,
                    chunk_size_K=self.chunk_size_centroids,
                    init=self.init,
                )
        return cluster_ids_b, centroids_b

    def fit(self, data: torch.Tensor, weights: torch.Tensor = None):
        """Alias for train; returns self."""
        self.train(data, weights=weights)
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
            assert self.use_triton, "process large N data requires triton implementation." 
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
 
        # Chunked to avoid materializing a full (B, N, D) temp.
        N_ = x_b.shape[1]
        x_sq = torch.empty(x_b.shape[:-1], device=x_b.device, dtype=x_b.dtype)
        _CHUNK = 1 << 20
        for i in range(0, N_, _CHUNK):
            x_sq[:, i:i + _CHUNK] = (x_b[:, i:i + _CHUNK] ** 2).sum(dim=-1)

        if self.use_triton:
            # Call Triton assignment kernel
            labels_b = euclid_assign_triton(x_b, self.centroids_b, x_sq)
        else:
            # Call PyTorch assignment fallback
            labels_b = euclid_assign_torch_native_chunked(
                x_b,
                self.centroids_b,
                x_sq,
                chunk_size_N=self.chunk_size_data,
                chunk_size_K=self.chunk_size_centroids,
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
