"""Compatibility shim for the original batched k-means module path."""

from .ops import batch_kmeans_Cosine, batch_kmeans_Dot, batch_kmeans_Euclid

__all__ = ["batch_kmeans_Euclid", "batch_kmeans_Cosine", "batch_kmeans_Dot"]
