"""Compatibility shim for the original large-N module path."""

from .ops import kmeans_largeN, kmeans_largeN_assign

__all__ = ["kmeans_largeN", "kmeans_largeN_assign"]
