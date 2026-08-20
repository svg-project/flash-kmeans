"""Runtime dependency and architecture dispatch for the CuTe backend."""
from __future__ import annotations

import functools
import importlib.util
from typing import Optional, Union

import torch

MIN_CUTLASS_DSL_VERSION = (4, 6)
SUPPORTED_ARCHS = ("sm90", "sm100", "sm120")


@functools.lru_cache(maxsize=1)
def _check_dependencies() -> None:
    try:
        import cutlass
    except ImportError as exc:
        raise RuntimeError(
            "The CuTe backend requires the optional CuTe dependencies. "
            "Install them with `pip install 'flash-kmeans[cute]'`."
        ) from exc

    parts = cutlass.__version__.split(".")
    try:
        major, minor = int(parts[0]), int(parts[1])
    except (IndexError, ValueError):
        return
    if (major, minor) < MIN_CUTLASS_DSL_VERSION:
        required = ".".join(str(x) for x in MIN_CUTLASS_DSL_VERSION)
        raise RuntimeError(
            "The CuTe backend requires nvidia-cutlass-dsl "
            f">= {required}.0 (found {cutlass.__version__}). Install a "
            "compatible version with `pip install 'flash-kmeans[cute]'`."
        )
    if importlib.util.find_spec("quack") is None:
        raise RuntimeError(
            "The CuTe backend requires quack-kernels >= 0.6.1. Install the "
            "compatible optional dependencies with "
            "`pip install 'flash-kmeans[cute]'`."
        )


def get_arch(device: Optional[Union[torch.device, int, str]] = None) -> str:
    if not torch.cuda.is_available():
        raise RuntimeError("The CuTe backend requires CUDA.")
    device = torch.device(device) if device is not None else torch.device("cuda")
    major, minor = torch.cuda.get_device_capability(device)
    arch = f"sm{major}{minor}"
    if arch not in SUPPORTED_ARCHS:
        supported = ", ".join(SUPPORTED_ARCHS)
        raise RuntimeError(
            f"The CuTe backend does not support {arch}; supported architectures: "
            f"{supported}."
        )
    return arch


@functools.lru_cache(maxsize=len(SUPPORTED_ARCHS))
def _module_for_arch(arch: str):
    if arch == "sm90":
        from . import lloyd_sm90 as module
    elif arch == "sm100":
        from . import lloyd_sm100 as module
    else:
        from . import lloyd_sm120 as module
    return module


def get_lloyd_module(device: Optional[Union[torch.device, int, str]] = None):
    _check_dependencies()
    return _module_for_arch(get_arch(device))
