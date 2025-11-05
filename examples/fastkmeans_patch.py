"""
Monkey patch for FastKMeans to support torch.Tensor input without numpy conversion.
"""

import torch
import numpy as np

try:
    from fastkmeans import FastKMeans
    from fastkmeans.kmeans import _kmeans_torch_double_chunked, _get_device
    from fastkmeans.triton_kernels import triton_kmeans
    
    def _patched_train(self, data):
        """Patched train to support torch.Tensor input directly without numpy conversion"""
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        np.random.seed(self.seed)

        # Move data to PyTorch CPU Tensor
        if isinstance(data, np.ndarray):
            data_torch = torch.from_numpy(data)
        elif isinstance(data, torch.Tensor):
            data_torch = data
        data_norms_torch = (data_torch**2).sum(dim=1)

        device = _get_device(self.device)
        if device == "cuda" and self.pin_gpu_memory:
            data_torch = data_torch.pin_memory()
            data_norms_torch = data_norms_torch.pin_memory()

        centroids, _ = _kmeans_torch_double_chunked(
            data_torch,
            data_norms_torch,
            k=self.k,
            max_iters=self.niter,
            tol=self.tol,
            device=device,
            dtype=self.dtype,
            chunk_size_data=self.chunk_size_data,
            chunk_size_centroids=self.chunk_size_centroids,
            max_points_per_centroid=self.max_points_per_centroid,
            verbose=self.verbose,
            use_triton=self.use_triton,
        )
        self.centroids = centroids.numpy()
    
    def _patched_predict(self, data):
        """Patched predict to support torch.Tensor input and return torch.Tensor"""
        if self.centroids is None:
            raise RuntimeError("Must call train() or fit() before predict().")

        if isinstance(data, np.ndarray):
            data_torch = torch.from_numpy(data)
        elif isinstance(data, torch.Tensor):
            data_torch = data
        data_norms_torch = (data_torch**2).sum(dim=1)

        # We'll do a chunked assignment pass, similar to the main loop, but no centroid updates
        centroids_torch = torch.from_numpy(self.centroids)
        centroids_torch = centroids_torch.to(device=self.device, dtype=torch.float32)
        centroid_norms = (centroids_torch**2).sum(dim=1)

        n_samples = data_torch.shape[0]
        labels = torch.empty(n_samples, dtype=torch.long, device=self.device)

        start_idx = 0
        while start_idx < n_samples:
            end_idx = min(start_idx + self.chunk_size_data, n_samples)

            data_chunk = data_torch[start_idx:end_idx].to(device=self.device, dtype=torch.float32, non_blocking=True)
            data_chunk_norms = data_norms_torch[start_idx:end_idx].to(
                device=self.device, dtype=torch.float32, non_blocking=True
            )
            batch_size = data_chunk.size(0)
            best_ids = torch.zeros((batch_size,), device=self.device, dtype=torch.long)

            if self.use_triton:
                triton_kmeans(
                    data_chunk,
                    data_chunk_norms,
                    centroids_torch,
                    centroid_norms,
                    best_ids,
                )
            else:
                best_dist = torch.full((batch_size,), float("inf"), device=self.device, dtype=torch.float32)
                c_start = 0
                k = centroids_torch.shape[0]
                while c_start < k:
                    c_end = min(c_start + self.chunk_size_centroids, k)
                    centroid_chunk = centroids_torch[c_start:c_end]
                    centroid_chunk_norms = centroid_norms[c_start:c_end]

                    dist_chunk = data_chunk_norms.unsqueeze(1) + centroid_chunk_norms.unsqueeze(0)
                    dist_chunk = dist_chunk.addmm_(data_chunk, centroid_chunk.t(), alpha=-2.0, beta=1.0)

                    local_min_vals, local_min_ids = torch.min(dist_chunk, dim=1)
                    improved_mask = local_min_vals < best_dist
                    best_dist[improved_mask] = local_min_vals[improved_mask]
                    best_ids[improved_mask] = c_start + local_min_ids[improved_mask]
                    c_start = c_end

            labels[start_idx:end_idx] = best_ids
            start_idx = end_idx

        return labels
    
    # Apply monkey patches
    FastKMeans.train = _patched_train
    FastKMeans.predict = _patched_predict
    
    print("✓ FastKMeans monkey patched: torch.Tensor input/output without numpy conversion")
    
except ImportError as e:
    print(f"✗ fastkmeans not available, skipping monkey patch: {e}")
except Exception as e:
    print(f"✗ Failed to monkey patch FastKMeans: {e}")
