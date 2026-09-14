"""Functional-equivalence tests against the pure-PyTorch reference.

Every test compares the C++/CUDA implementation with ``tests/reference.py``
(same inputs, same explicit initialization), checking both the assignment
(labels) and the resulting centroids.
"""

from __future__ import annotations

import pytest
import torch

from flash_kmeans import (
    batch_kmeans_Cosine,
    batch_kmeans_Dot,
    batch_kmeans_Euclid,
    euclid_assign_triton,
    triton_centroid_update_euclid,
    triton_centroid_update_sorted_euclid,
)

from reference import ref_assign_euclid, ref_kmeans, ref_update

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required"
)

DTYPES = [torch.float32, torch.float16, torch.bfloat16]
# fp32 kernels accumulate in fp32; 2-byte dtypes lose precision in the
# comparison reference as well, so allow a slightly larger tolerance.
ATOL = {torch.float32: 1e-4, torch.float16: 5e-2, torch.bfloat16: 5e-2}
AGREEMENT = {torch.float32: 1.0, torch.float16: 0.995, torch.bfloat16: 0.99}


def make_inputs(B, N, D, K, dtype, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(B, N, D, device="cuda", dtype=dtype, generator=g)
    init = torch.randn(B, K, D, device="cuda", dtype=dtype, generator=g)
    return x, init


def check_equal(new_labels, new_cent, ref_labels, ref_cent, dtype):
    agree = (new_labels == ref_labels).float().mean().item()
    assert agree >= AGREEMENT[dtype], f"label agreement {agree}"
    diff = (new_cent.float() - ref_cent.float()).abs().max().item()
    assert diff <= ATOL[dtype], f"centroid max diff {diff}"


def inertia(x, labels, cent):
    assigned = cent.float().gather(
        1, labels.long().unsqueeze(-1).expand(-1, -1, x.shape[-1])
    )
    return ((x.float() - assigned) ** 2).sum().item()


def check_objective(x, labels, cent, ref_labels, ref_cent):
    """Multi-iteration runs are chaotic: compare the reached optimum instead."""
    ref_i = inertia(x, ref_labels, ref_cent)
    new_i = inertia(x, labels, cent)
    rel = abs(new_i - ref_i) / ref_i
    assert rel <= 1e-3, f"inertia relative difference {rel}"
    agree = (labels == ref_labels).float().mean().item()
    assert agree >= 0.9, f"label agreement {agree}"


@pytest.mark.parametrize("mode", ["euclid", "cosine", "dot"])
@pytest.mark.parametrize("dtype", DTYPES)
def test_single_iteration_matches_reference(mode, dtype):
    B, N, D, K = 2, 4096, 64, 32
    x, init = make_inputs(B, N, D, K, dtype)
    fn = {
        "euclid": batch_kmeans_Euclid,
        "cosine": batch_kmeans_Cosine,
        "dot": batch_kmeans_Dot,
    }[mode]

    labels, cent, iters = fn(x, K, max_iters=1, tol=0.0, init_centroids=init)
    ref_labels, ref_cent, riters = ref_kmeans(
        x, K, max_iters=1, tol=0.0, init_centroids=init, mode=mode
    )
    assert iters == riters
    check_equal(labels, cent, ref_labels, ref_cent, dtype)


@pytest.mark.parametrize("mode", ["euclid", "cosine", "dot"])
@pytest.mark.parametrize("dtype", DTYPES)
def test_converged_objective_matches_reference(mode, dtype):
    B, N, D, K = 2, 4096, 64, 32
    x, init = make_inputs(B, N, D, K, dtype)
    fn = {
        "euclid": batch_kmeans_Euclid,
        "cosine": batch_kmeans_Cosine,
        "dot": batch_kmeans_Dot,
    }[mode]

    labels, cent, iters = fn(x, K, max_iters=8, tol=0.0, init_centroids=init)
    ref_labels, ref_cent, riters = ref_kmeans(
        x, K, max_iters=8, tol=0.0, init_centroids=init, mode=mode
    )
    assert iters == riters
    check_objective(x, labels, cent, ref_labels, ref_cent)


@pytest.mark.parametrize("dtype", DTYPES)
def test_assignment_matches_reference(dtype):
    B, N, D, K = 3, 2048, 32, 16
    x, cent = make_inputs(B, N, D, K, dtype)
    labels = euclid_assign_triton(x, cent)
    ref = ref_assign_euclid(x, cent)
    agree = (labels == ref).float().mean().item()
    assert agree >= AGREEMENT[dtype], f"label agreement {agree}"
    assert labels.dtype == torch.int32


@pytest.mark.parametrize("dtype", DTYPES)
def test_centroid_update_matches_reference(dtype):
    B, N, D, K = 2, 2048, 32, 16
    x, old = make_inputs(B, N, D, K, dtype)
    labels = ref_assign_euclid(x, old)
    new = triton_centroid_update_sorted_euclid(x, labels, old)
    ref = ref_update(x, labels, old, mode="euclid")
    diff = (new.float() - ref.float()).abs().max().item()
    assert diff <= ATOL[dtype], f"centroid max diff {diff}"

    new2 = triton_centroid_update_euclid(x, labels, old)
    assert torch.allclose(new.float(), new2.float(), atol=ATOL[dtype], rtol=1e-4)


@pytest.mark.parametrize("D", [1, 3, 7, 16, 31, 64, 127, 128, 129, 256, 512, 513, 1024])
def test_assignment_dimension_coverage(D):
    """Exercise odd / non-power-of-two / large feature dimensions."""
    B, N, K = 2, 1024, 24
    x, cent = make_inputs(B, N, D, K, torch.float32)
    labels = euclid_assign_triton(x, cent)
    ref = ref_assign_euclid(x, cent)
    agree = (labels == ref).float().mean().item()
    assert agree >= 0.995, f"D={D}: label agreement {agree}"


@pytest.mark.parametrize("D", [8, 64, 513, 2048])
def test_batch_kmeans_dimension_coverage(D):
    B, N, K = 2, 1500, 20
    x, init = make_inputs(B, N, D, K, torch.float32, seed=D)
    labels, cent, _ = batch_kmeans_Euclid(x, K, max_iters=6, init_centroids=init)
    rlabels, rcent, _ = ref_kmeans(x, K, max_iters=6, init_centroids=init)
    if D <= 128:
        check_equal(labels, cent, rlabels, rcent, torch.float32)
    else:
        # long fp32 dot products accumulate in different orders, which makes
        # multi-iteration trajectories chaotic for large D
        check_objective(x, labels, cent, rlabels, rcent)


def test_convergence_stops_early():
    B, N, D, K = 2, 1024, 16, 8
    x, init = make_inputs(B, N, D, K, torch.float32)
    labels, cent, iters = batch_kmeans_Euclid(
        x, K, max_iters=100, tol=1e9, init_centroids=init
    )
    assert iters == 1
    rlabels, rcent, riters = ref_kmeans(
        x, K, max_iters=100, tol=1e9, init_centroids=init
    )
    assert iters == riters
    check_equal(labels, cent, rlabels, rcent, torch.float32)


def test_empty_clusters_keep_old_centroid():
    # K > N guarantees empty clusters (random init also duplicates rows).
    B, N, D, K = 1, 5, 8, 16
    x, init = make_inputs(B, N, D, K, torch.float32)
    labels, cent, _ = batch_kmeans_Euclid(x, K, max_iters=3, init_centroids=init)
    rlabels, rcent, _ = ref_kmeans(x, K, max_iters=3, init_centroids=init)
    check_equal(labels, cent, rlabels, rcent, torch.float32)


def test_seeded_random_init_is_deterministic():
    B, N, D, K = 2, 1024, 32, 16
    x, _ = make_inputs(B, N, D, K, torch.float32)
    a = batch_kmeans_Euclid(x, K, max_iters=5, seed=123)
    b = batch_kmeans_Euclid(x, K, max_iters=5, seed=123)
    assert torch.equal(a[0], b[0])
    # centroid accumulation uses atomicAdd, so allow fp32 round-off noise
    assert torch.allclose(a[1], b[1], atol=1e-4, rtol=1e-4)
    c = batch_kmeans_Euclid(x, K, max_iters=5, seed=124)
    assert not torch.equal(a[1], c[1])
