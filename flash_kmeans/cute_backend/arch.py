"""Runtime dependency and architecture dispatch for the CuTe backend."""
from __future__ import annotations

import functools
import importlib.util
import sys
from typing import Optional, Union

import torch

MIN_CUTLASS_DSL_VERSION = (4, 6)

# Compute-capability major version -> kernel module. Only the Blackwell
# datacenter and RTX PRO families are supported:
#
#   major 10 -> lloyd_sm100: B200 (sm_100), B300 / GB300 (sm_103)
#   major 12 -> lloyd_sm120: RTX PRO 6000 series (sm_120)
#
# The minor version selects a variant inside a family rather than a different
# kernel: every member of a family exposes the instructions these kernels are
# built on (tcgen05 + TMA on 10.x, mma.sync + TMA on 12.x), and the DSL
# compiles for the launching device's exact target.
_ARCH_BY_CAPABILITY_MAJOR = {10: "sm100", 12: "sm120"}
SUPPORTED_ARCHS = tuple(dict.fromkeys(_ARCH_BY_CAPABILITY_MAJOR.values()))
SUPPORTED_DEVICES = "sm_10x (B200, B300, GB300) and sm_12x (RTX PRO 6000 series)"


@functools.lru_cache(maxsize=1)
def _check_dependencies() -> None:
    if sys.version_info < (3, 10):
        raise RuntimeError("The CuTe backend requires Python 3.10 or newer.")
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
    """Return the CuTe kernel family for ``device``.

    Raises ``RuntimeError`` when ``device`` is not a CUDA device of a
    supported architecture, so an unsupported GPU fails with a clear message
    instead of compiling kernels it cannot run.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("The CuTe backend requires CUDA.")
    device = torch.device(device) if device is not None else torch.device("cuda")
    if device.type != "cuda":
        raise RuntimeError(
            f"The CuTe backend requires a CUDA device, got a {device.type!r} "
            "device."
        )
    major, minor = torch.cuda.get_device_capability(device)
    arch = _ARCH_BY_CAPABILITY_MAJOR.get(major)
    if arch is None:
        raise RuntimeError(
            f"The CuTe backend does not support "
            f"{torch.cuda.get_device_name(device)} (sm_{major}{minor}); "
            f"supported architectures: {SUPPORTED_DEVICES}."
        )
    return arch


@functools.lru_cache(maxsize=len(SUPPORTED_ARCHS))
def _module_for_arch(arch: str):
    if arch == "sm100":
        from . import lloyd_sm100 as module
    elif arch == "sm120":
        from . import lloyd_sm120 as module
    else:
        # Unreachable via get_arch; guards against adding a capability to
        # _ARCH_BY_CAPABILITY_MAJOR without wiring up its kernel module.
        raise RuntimeError(f"Unknown CuTe kernel family {arch!r}.")
    return module


def get_lloyd_module(device: Optional[Union[torch.device, int, str]] = None):
    _check_dependencies()
    return _module_for_arch(get_arch(device))
