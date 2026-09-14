"""FlashKMeans API tests (drop-in compatibility of the Python surface)."""

from __future__ import annotations

import pytest
import torch

from flash_kmeans import FlashKMeans, batch_kmeans_Euclid, euclid_assign_triton

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required"
)


def test_fit_predict_2d():
    N, D, K = 8192, 32, 16
    x = torch.randn(N, D, device="cuda")
    km = FlashKMeans(d=D, k=K, niter=10, seed=0, device=torch.device("cuda:0"))
    labels = km.fit_predict(x)
    assert labels.shape == (N,)
    assert labels.dtype == torch.int32
    assert km.centroids_b.shape == (1, K, D)
    assert km.cluster_ids_b.shape == (1, N)
    assert km.predict(x).shape == (N,)


def test_fit_predict_3d_batched():
    B, N, D, K = 3, 2048, 24, 12
    x = torch.randn(B, N, D, device="cuda")
    km = FlashKMeans(d=D, k=K, niter=8, device="cuda:0")
    labels = km.fit_predict(x)
    assert labels.shape == (B, N)
    assert km.predict(x).shape == (B, N)
    with pytest.raises(ValueError):
        km.predict(torch.randn(B + 1, N, D, device="cuda"))


def test_train_predict_matches_raw_api():
    B, N, D, K = 2, 4096, 32, 16
    x = torch.randn(B, N, D, device="cuda")
    km = FlashKMeans(d=D, k=K, niter=10, seed=7, device="cuda:0")
    km.train(x)
    labels = km.predict(x)
    raw_labels, raw_cent, _ = batch_kmeans_Euclid(x, K, max_iters=10, seed=7)
    # same deterministic C++ RNG; fp32 atomic accumulation may flip points on
    # exact distance ties, so allow a tiny margin
    agree = (km.cluster_ids_b == raw_labels).float().mean().item()
    assert agree >= 0.999, f"agreement {agree}"
    assert torch.allclose(km.centroids_b, raw_cent, atol=1e-4, rtol=1e-4)
    ref = euclid_assign_triton(x, raw_cent)
    assert torch.equal(labels, ref)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_dtype_support(dtype):
    N, D, K = 4096, 32, 8
    x = torch.randn(N, D, device="cuda", dtype=dtype)
    km = FlashKMeans(d=D, k=K, niter=5, dtype=dtype, device="cuda:0")
    labels = km.fit_predict(x)
    assert labels.dtype == torch.int32
    assert km.centroids_b.dtype == dtype


def test_cpu_large_n_path():
    N, D, K = 20_000, 32, 16
    x = torch.randn(N, D)  # CPU
    km = FlashKMeans(
        d=D, k=K, niter=4, chunk_size_data_cpu=4096, device="cuda:0", seed=0
    )
    labels = km.fit_predict(x)
    assert labels.shape == (N,)
    assert km.centroids_b.shape == (1, K, D)
    # predict must equal an independent assignment against the stored centroids
    labels2 = km.predict(x)
    ref = euclid_assign_triton(x.to("cuda").unsqueeze(0), km.centroids_b).squeeze(0)
    assert torch.equal(labels2, ref)


def test_use_triton_flag_is_accepted():
    N, D, K = 2048, 16, 8
    x = torch.randn(N, D, device="cuda")
    km = FlashKMeans(d=D, k=K, niter=3, use_triton=False, device="cuda:0")
    assert km.fit_predict(x).shape == (N,)


def test_predict_before_fit_raises():
    km = FlashKMeans(d=8, k=4)
    with pytest.raises(RuntimeError):
        km.predict(torch.randn(16, 8, device="cuda"))
