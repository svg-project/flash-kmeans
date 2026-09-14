"""Compatibility shim for the original Triton module path.

The kernels now live in the C++/CUDA extension; this module keeps the old
import path working for downstream code.
"""

from .ops import cosine_assign_triton, euclid_assign_triton

__all__ = ["euclid_assign_triton", "cosine_assign_triton"]
