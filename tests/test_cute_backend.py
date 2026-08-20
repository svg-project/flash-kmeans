import importlib.util

import pytest
import torch

from flash_kmeans import FlashKMeans, batch_kmeans_Euclid


def _cute_is_available():
    if (
        not torch.cuda.is_available()
        or importlib.util.find_spec("cutlass") is None
        or importlib.util.find_spec("quack") is None
    ):
        return False
    return torch.cuda.get_device_capability()[0] in (9, 10, 12)


pytestmark = pytest.mark.skipif(
    not _cute_is_available(), reason="CuTe dependencies and a supported GPU are required"
)


def _centroid_oracle(x, ids, old_centroids):
    batch, _, dim = x.shape
    n_clusters = old_centroids.shape[1]
    counts = torch.zeros(batch, n_clusters, dtype=torch.int32, device=x.device)
    counts.scatter_add_(1, ids.long(), torch.ones_like(ids, dtype=torch.int32))
    sums = torch.zeros(
        batch, n_clusters, dim, dtype=torch.float32, device=x.device
    )
    sums.scatter_add_(
        1, ids.long()[..., None].expand(-1, -1, dim), x.float()
    )
    means = sums / counts.clamp_min(1)[..., None]
    return torch.where((counts == 0)[..., None], old_centroids.float(), means)


def _inertia(x, ids, centroids):
    own = torch.gather(
        centroids.float(),
        1,
        ids.long()[..., None].expand(-1, -1, x.shape[-1]),
    )
    return ((x.float() - own) ** 2).sum(-1).mean()


def test_cute_one_iteration_matches_oracle_and_triton():
    torch.manual_seed(0)
    x = torch.randn(2, 384, 128, device="cuda", dtype=torch.bfloat16)
    indices = torch.tensor(
        [[0, 64, 128, 192, 256, 320], [1, 65, 129, 193, 257, 321]],
        device=x.device,
    )
    initial = torch.gather(
        x, 1, indices[..., None].expand(-1, -1, x.shape[-1])
    ).contiguous()

    cute_ids, cute_centroids, n_iters = batch_kmeans_Euclid(
        x, 6, max_iters=1, init_centroids=initial, backend="cute"
    )
    triton_ids, _, _ = batch_kmeans_Euclid(
        x, 6, max_iters=1, init_centroids=initial, backend="triton"
    )
    torch.cuda.synchronize()

    expected = _centroid_oracle(x, cute_ids, initial).to(torch.bfloat16)
    torch.testing.assert_close(cute_centroids, expected, atol=0.05, rtol=0.0)
    assert (cute_ids == triton_ids).float().mean().item() > 0.995
    assert n_iters == 1


def test_cute_flash_kmeans_fit_and_predict():
    torch.manual_seed(1)
    x = torch.randn(384, 128, device="cuda", dtype=torch.bfloat16)
    model = FlashKMeans(
        d=128, k=6, niter=1, tol=0.0, dtype=torch.bfloat16, backend="cute"
    ).fit(x)

    predicted = model.predict(x)
    assert model.cluster_ids_b.shape == (1, 384)
    assert model.centroids_b.shape == (1, 6, 128)
    assert predicted.shape == (384,)
    assert predicted.dtype == torch.int32
    distances = (
        (x.float()[:, None, :] - model.centroids_b[0].float()[None, :, :]) ** 2
    ).sum(-1)
    assert (predicted == distances.argmin(-1)).float().mean().item() > 0.995


def test_cute_multi_iteration_inertia_matches_triton():
    torch.manual_seed(3)
    x = torch.randn(2, 384, 128, device="cuda", dtype=torch.bfloat16)
    indices = torch.randint(0, x.shape[1], (2, 6), device=x.device)
    initial = torch.gather(
        x, 1, indices[..., None].expand(-1, -1, x.shape[-1])
    ).contiguous()

    triton_ids, triton_centroids, _ = batch_kmeans_Euclid(
        x, 6, max_iters=5, tol=0.0, init_centroids=initial, backend="triton"
    )
    cute_ids, cute_centroids, _ = batch_kmeans_Euclid(
        x, 6, max_iters=5, tol=0.0, init_centroids=initial, backend="cute"
    )
    triton_inertia = _inertia(x, triton_ids, triton_centroids)
    cute_inertia = _inertia(x, cute_ids, cute_centroids)

    relative_difference = (cute_inertia - triton_inertia).abs() / triton_inertia
    assert relative_difference.item() < 5e-3


def test_cute_honors_tolerance():
    torch.manual_seed(2)
    centers = torch.randn(1, 6, 128, device="cuda", dtype=torch.bfloat16)
    x = centers[:, :, None, :].expand(-1, -1, 64, -1).reshape(1, 384, 128)

    _, converged, n_iters = batch_kmeans_Euclid(
        x,
        6,
        max_iters=5,
        tol=1e-8,
        init_centroids=centers,
        backend="cute",
    )

    assert n_iters == 1
    torch.testing.assert_close(converged, centers)


def test_cute_tolerance_return_matches_triton_contract():
    torch.manual_seed(4)
    x = torch.randn(2, 384, 128, device="cuda", dtype=torch.bfloat16)
    indices = torch.randint(0, x.shape[1], (2, 6), device=x.device)
    initial = torch.gather(
        x, 1, indices[..., None].expand(-1, -1, x.shape[-1])
    ).contiguous()

    _, triton_centroids, triton_iters = batch_kmeans_Euclid(
        x, 6, max_iters=5, tol=1e9, init_centroids=initial, backend="triton"
    )
    _, cute_centroids, cute_iters = batch_kmeans_Euclid(
        x, 6, max_iters=5, tol=1e9, init_centroids=initial, backend="cute"
    )

    assert cute_iters == triton_iters == 1
    torch.testing.assert_close(cute_centroids, triton_centroids)


def test_cute_supports_batch_larger_than_cuda_grid_y_limit():
    batch = 65536
    x = torch.randn(batch, 2, 128, device="cuda", dtype=torch.bfloat16)
    initial = x.clone()

    ids, centroids, n_iters = batch_kmeans_Euclid(
        x, 2, max_iters=1, tol=0.0, init_centroids=initial, backend="cute"
    )

    assert ids.shape == (batch, 2)
    assert centroids.shape == (batch, 2, 128)
    assert n_iters == 1


def test_cute_rejects_single_sample_train_and_predict():
    x = torch.randn(1, 1, 128, device="cuda", dtype=torch.bfloat16)

    with pytest.raises(ValueError, match="at least 2 samples"):
        batch_kmeans_Euclid(x, 2, max_iters=1, backend="cute")

    model = FlashKMeans(
        d=128, k=2, niter=1, dtype=torch.bfloat16, backend="cute"
    )
    model.fit(torch.randn(2, 128, device="cuda", dtype=torch.bfloat16))
    with pytest.raises(ValueError, match="at least 2 samples"):
        model.predict(x.squeeze(0))


@pytest.mark.parametrize(
    ("shape", "dtype", "clusters", "message"),
    [
        ((0, 2, 128), torch.bfloat16, 2, "at least 1 batch"),
        ((1, 32, 64), torch.bfloat16, 4, "requires D=128"),
        ((1, 32, 128), torch.float16, 4, "requires x.dtype=torch.bfloat16"),
        ((1, 32, 128), torch.bfloat16, 1, "at least 2"),
        ((1, 32, 128), torch.bfloat16, 1025, "at most 1024 clusters"),
    ],
)
def test_cute_rejects_unsupported_inputs(shape, dtype, clusters, message):
    x = torch.randn(*shape, device="cuda", dtype=dtype)
    with pytest.raises(ValueError, match=message):
        batch_kmeans_Euclid(x, clusters, max_iters=1, backend="cute")
