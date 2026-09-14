"""Compatibility shim for the original Triton module path.

The centroid updates are implemented in the C++/CUDA extension.
"""

from .ops import (
    triton_centroid_update_cosine,
    triton_centroid_update_euclid,
    triton_centroid_update_sorted_cosine,
    triton_centroid_update_sorted_euclid,
)

__all__ = [
    "triton_centroid_update_euclid",
    "triton_centroid_update_sorted_euclid",
    "triton_centroid_update_cosine",
    "triton_centroid_update_sorted_cosine",
]
