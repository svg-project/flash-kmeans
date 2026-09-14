"""Large-N streaming pipeline tests (CPU data -> GPU chunks)."""

from __future__ import annotations

import pytest
import torch

from flash_kmeans import (
    batch_kmeans_Euclid,
    euclid_assign_triton,
    kmeans_largeN,
    kmeans_largeN_assign,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required"
)


def make_cpu(N, D, dtype, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(N, D, generator=g).to(dtype).contiguous()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_large_n_matches_batch_pipeline(dtype):
    N, D, K = 100_000, 64, 64
    x_cpu = make_cpu(N, D, dtype).pin_memory()
    g = torch.Generator(device="cuda").manual_seed(0)
    init = torch.randn(K, D, device="cuda", dtype=dtype, generator=g)

    labels, centroids = kmeans_largeN(
        x_cpu, K, max_iters=3, tol=0.0, BLOCK_N=32768, init_centroids=init
    )
    x_gpu = x_cpu.to("cuda")
    b_labels, b_cent, _ = batch_kmeans_Euclid(
        x_gpu.unsqueeze(0), K, max_iters=3, tol=0.0, init_centroids=init.unsqueeze(0)
    )
    agree = (labels == b_labels.squeeze(0)).float().mean().item()
    assert agree >= 0.995, f"largeN vs batch label agreement {agree}"
    diff = (centroids.float() - b_cent.squeeze(0).float()).abs().max().item()
    assert diff <= 1e-2, f"largeN vs batch centroid diff {diff}"


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_large_n_assign_is_exact(dtype):
    """Assignment uses no cross-chunk accumulation, so it must be exact."""
    N, D, K = 200_000, 128, 96
    x_cpu = make_cpu(N, D, dtype, seed=3).pin_memory()
    g = torch.Generator(device="cuda").manual_seed(1)
    cent = torch.randn(K, D, device="cuda", dtype=dtype, generator=g)

    labels = kmeans_largeN_assign(x_cpu, cent, BLOCK_N=65536, device="cuda:0")
    b_labels = euclid_assign_triton(x_cpu.to("cuda").unsqueeze(0), cent.unsqueeze(0))
    assert torch.equal(labels, b_labels.squeeze(0))


def test_large_n_single_device_placements():
    N, D, K = 50_000, 32, 16
    x_cpu = make_cpu(N, D, torch.float32)
    labels, centroids = kmeans_largeN(x_cpu, K, max_iters=2, BLOCK_N=16384,
                                      device="cuda:0")
    assert labels.device.type == "cuda" and labels.dtype == torch.int32
    assert labels.shape == (N,)
    assert centroids.device.type == "cuda" and centroids.shape == (K, D)


def test_large_n_is_deterministic_with_seed():
    N, D, K = 50_000, 32, 16
    x_cpu = make_cpu(N, D, torch.float32, seed=5)
    a = kmeans_largeN(x_cpu, K, max_iters=2, BLOCK_N=16384, seed=42, device="cuda:0")
    b = kmeans_largeN(x_cpu, K, max_iters=2, BLOCK_N=16384, seed=42, device="cuda:0")
    assert torch.equal(a[0], b[0])
    assert torch.allclose(a[1], b[1], atol=1e-4, rtol=1e-4)


def test_large_n_non_pinned_input_works():
    N, D, K = 20_000, 16, 8
    x_cpu = make_cpu(N, D, torch.float32, seed=7)  # not pinned
    labels, centroids = kmeans_largeN(x_cpu, K, max_iters=1, BLOCK_N=8192,
                                      device="cuda:0")
    assert labels.shape == (N,)
    assert centroids.shape == (K, D)
