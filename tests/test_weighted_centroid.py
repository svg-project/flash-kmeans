"""Correctness tests for the weighted centroid-update kernel.

Covers the cases the kernel must not regress on:
  - power-of-two and NON-power-of-two D (the kernel streams D in BLOCK_D tiles
    with masking, so D=96/192/80 must work, not just D=64/128/256),
  - large D + fp32,
  - all-ones weights reduce to the unweighted kernel,
  - empty clusters fall back to old_centroids.
"""

import pytest
import torch

from flash_kmeans.centroid_update_triton import (
    triton_centroid_update_sorted_euclid,
    triton_centroid_update_sorted_euclid_weighted,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA"
)


def _torch_weighted_reference(x, cluster_ids, old_centroids, weights):
    """Per-cluster weighted mean in fp32; empty clusters keep old_centroids."""
    B, N, D = x.shape
    K = old_centroids.shape[1]
    xf, wf = x.float(), weights.float()
    out = torch.empty((B, K, D), device=x.device, dtype=torch.float32)
    for b in range(B):
        for k in range(K):
            mask = cluster_ids[b] == k
            wsum = wf[b][mask].sum()
            if wsum <= 0:
                out[b, k] = old_centroids[b, k].float()
            else:
                out[b, k] = (xf[b][mask] * wf[b][mask, None]).sum(0) / wsum
    return out.to(x.dtype)


# (B, N, K, D, dtype) — includes non-power-of-two D and large-D/fp32.
CASES = [
    (4, 8000, 32, 64, torch.float16),
    (4, 8000, 32, 128, torch.float16),
    (4, 8000, 32, 96, torch.float16),    # non-pow2
    (4, 8000, 32, 192, torch.float16),   # non-pow2
    (2, 8000, 32, 80, torch.float16),    # non-pow2 head dim
    (2, 8000, 32, 256, torch.float16),
    (2, 8000, 16, 1024, torch.float32),  # large D + fp32
]


@pytest.mark.parametrize("B,N,K,D,dtype", CASES)
def test_weighted_matches_torch_reference(B, N, K, D, dtype):
    torch.manual_seed(0)
    dev = "cuda"
    x = torch.randn(B, N, D, device=dev, dtype=dtype)
    cluster_ids = torch.randint(0, K, (B, N), device=dev)
    old_c = torch.randn(B, K, D, device=dev, dtype=dtype)
    weights = torch.rand(B, N, device=dev) + 0.1

    got = triton_centroid_update_sorted_euclid_weighted(x, cluster_ids, old_c, weights)
    ref = _torch_weighted_reference(x, cluster_ids, old_c, weights)

    tol = 5e-2 if dtype == torch.float16 else 1e-4
    assert (got.float() - ref.float()).abs().max().item() < tol


@pytest.mark.parametrize("B,N,K,D,dtype", CASES)
def test_ones_weights_match_unweighted(B, N, K, D, dtype):
    torch.manual_seed(0)
    dev = "cuda"
    x = torch.randn(B, N, D, device=dev, dtype=dtype)
    cluster_ids = torch.randint(0, K, (B, N), device=dev)
    old_c = torch.randn(B, K, D, device=dev, dtype=dtype)
    ones = torch.ones(B, N, device=dev)

    got_w = triton_centroid_update_sorted_euclid_weighted(x, cluster_ids, old_c, ones)
    got_u = triton_centroid_update_sorted_euclid(x, cluster_ids, old_c)

    tol = 5e-2 if dtype == torch.float16 else 1e-4
    assert (got_w.float() - got_u.float()).abs().max().item() < tol


def test_empty_cluster_falls_back_to_old_centroids():
    torch.manual_seed(0)
    dev = "cuda"
    B, N, K, D = 2, 8000, 32, 96  # non-pow2 D on purpose
    x = torch.randn(B, N, D, device=dev, dtype=torch.float16)
    cluster_ids = torch.randint(0, K, (B, N), device=dev)
    cluster_ids[cluster_ids == 0] = 1  # cluster 0 empty in every batch
    old_c = torch.randn(B, K, D, device=dev, dtype=torch.float16)
    weights = torch.rand(B, N, device=dev) + 0.1

    got = triton_centroid_update_sorted_euclid_weighted(x, cluster_ids, old_c, weights)
    torch.testing.assert_close(got[:, 0].float(), old_c[:, 0].float(), atol=1e-3, rtol=0)
