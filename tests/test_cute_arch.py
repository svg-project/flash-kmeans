"""Device/architecture gating for the CuTe backend.

These tests run without CUDA and without the optional CuTe dependencies: they
only exercise ``flash_kmeans.cute_backend.arch``, which imports nothing beyond
torch. The supported set is Blackwell only -- sm_10x (B200, B300, GB300) and
sm_12x (RTX PRO 6000 series).
"""
from unittest import mock

import pytest
import torch

from flash_kmeans.cute_backend import arch

SUPPORTED = [
    ((10, 0), "sm100", "NVIDIA B200"),
    ((10, 3), "sm100", "NVIDIA B300"),
    ((10, 3), "sm100", "NVIDIA GB300"),
    ((12, 0), "sm120", "NVIDIA RTX PRO 6000 Blackwell Server Edition"),
]

UNSUPPORTED = [
    ((7, 0), "Tesla V100-SXM2-16GB"),
    ((7, 5), "NVIDIA GeForce RTX 2080 Ti"),
    ((8, 0), "NVIDIA A100-SXM4-80GB"),
    ((8, 6), "NVIDIA GeForce RTX 3090"),
    ((8, 9), "NVIDIA L40S"),
    ((9, 0), "NVIDIA H100 80GB HBM3"),  # SM90 is deliberately not supported
    ((11, 0), "NVIDIA Future 11.x"),
    ((13, 0), "NVIDIA Future 13.x"),
]


@pytest.mark.parametrize(("capability", "expected", "name"), SUPPORTED)
def test_supported_capabilities_select_the_right_kernel_family(capability, expected, name):
    with mock.patch.object(torch.cuda, "is_available", return_value=True), \
         mock.patch.object(torch.cuda, "get_device_capability", return_value=capability), \
         mock.patch.object(torch.cuda, "get_device_name", return_value=name):
        assert arch.get_arch("cuda:0") == expected


@pytest.mark.parametrize(("capability", "name"), UNSUPPORTED)
def test_unsupported_capabilities_are_rejected(capability, name):
    with mock.patch.object(torch.cuda, "is_available", return_value=True), \
         mock.patch.object(torch.cuda, "get_device_capability", return_value=capability), \
         mock.patch.object(torch.cuda, "get_device_name", return_value=name):
        with pytest.raises(RuntimeError) as excinfo:
            arch.get_arch("cuda:0")

    message = str(excinfo.value)
    # The error must name the device, its capability, and the supported set,
    # so a user on the wrong GPU knows immediately why it refused.
    assert name in message
    assert f"sm_{capability[0]}{capability[1]}" in message
    assert arch.SUPPORTED_DEVICES in message


def test_sm90_is_not_in_the_supported_set():
    assert arch.SUPPORTED_ARCHS == ("sm100", "sm120")
    assert "sm90" not in arch.SUPPORTED_ARCHS


def test_get_arch_requires_cuda():
    with mock.patch.object(torch.cuda, "is_available", return_value=False):
        with pytest.raises(RuntimeError, match="requires CUDA"):
            arch.get_arch()


def test_get_arch_rejects_a_non_cuda_device():
    with mock.patch.object(torch.cuda, "is_available", return_value=True):
        with pytest.raises(RuntimeError, match="requires a CUDA device"):
            arch.get_arch("cpu")


def test_module_for_arch_rejects_an_unknown_family():
    with pytest.raises(RuntimeError, match="Unknown CuTe kernel family"):
        arch._module_for_arch("sm90")


def test_flash_kmeans_rejects_an_unsupported_device_at_construction():
    from flash_kmeans import FlashKMeans

    with mock.patch.object(torch.cuda, "is_available", return_value=True), \
         mock.patch.object(torch.cuda, "get_device_capability", return_value=(9, 0)), \
         mock.patch.object(torch.cuda, "get_device_name", return_value="NVIDIA H100 80GB HBM3"):
        with pytest.raises(RuntimeError, match="does not support"):
            FlashKMeans(d=128, k=8, backend="cute", device=torch.device("cuda:0"))


def test_flash_kmeans_accepts_a_supported_device_at_construction():
    from flash_kmeans import FlashKMeans

    with mock.patch.object(torch.cuda, "is_available", return_value=True), \
         mock.patch.object(torch.cuda, "get_device_capability", return_value=(10, 3)), \
         mock.patch.object(torch.cuda, "get_device_name", return_value="NVIDIA B300"):
        model = FlashKMeans(d=128, k=8, backend="cute", device=torch.device("cuda:0"))

    assert model.backend == "cute"
