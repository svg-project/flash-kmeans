"""flash-kmeans: fast batched k-means clustering with C++/CUDA kernels.

The public API is unchanged from the Triton-based release; the compute kernels
and the k-means iteration loops are implemented in C++/CUDA and exposed through
nanobind (``flash_kmeans._flash_kmeans_cpp``).
"""

from .interface import FlashKMeans
from .ops import (
    batch_kmeans_Cosine,
    batch_kmeans_Dot,
    batch_kmeans_Euclid,
    cosine_assign_triton,
    euclid_assign_triton,
    kmeans_largeN,
    kmeans_largeN_assign,
    triton_centroid_update_cosine,
    triton_centroid_update_euclid,
    triton_centroid_update_sorted_cosine,
    triton_centroid_update_sorted_euclid,
)

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
    "FlashKMeans",
    "kmeans_largeN",
    "kmeans_largeN_assign",
]

__version__ = "0.4.0"
