from .interface import FlashKMeans
from .kmeans_api import batch_kmeans_Euclid

try:
    from .kmeans_triton_impl import (
        batch_kmeans_Cosine,
        batch_kmeans_Dot,
    )
    from .centroid_update_triton import (
        triton_centroid_update_euclid,
        triton_centroid_update_sorted_euclid,
    )
    from .kmeans_large import kmeans_largeN, kmeans_largeN_assign
except Exception:
    import warnings

    warnings.warn(
        "Triton kmeans implementation not found, falling back to torch native implementation.",
        RuntimeWarning,
        stacklevel=2,
    )

    def no_torch_fallback():
        raise ImportError(
            "Triton kmeans implementation not found, and this function has no torch fallback."
        )

    triton_centroid_update_euclid = no_torch_fallback
    triton_centroid_update_sorted_euclid = no_torch_fallback
    batch_kmeans_Cosine = no_torch_fallback
    batch_kmeans_Dot = no_torch_fallback
    kmeans_largeN_assign = no_torch_fallback
    kmeans_largeN = no_torch_fallback

__all__ = [
    "batch_kmeans_Euclid",
    "batch_kmeans_Cosine",
    "batch_kmeans_Dot",
    "triton_centroid_update_euclid",
    "triton_centroid_update_sorted_euclid",
    "FlashKMeans",
    "kmeans_largeN",
    "kmeans_largeN_assign",
]

__version__ = "0.3.1"
