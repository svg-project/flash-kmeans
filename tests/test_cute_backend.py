import importlib.util

import pytest
import torch

from flash_kmeans import FlashKMeans, batch_kmeans_Euclid
from flash_kmeans.cute_backend import euclid_assign_cute
from flash_kmeans.cute_backend.arch import get_arch, get_lloyd_module


def _cute_is_available():
    if (
        importlib.util.find_spec("cutlass") is None
        or importlib.util.find_spec("quack") is None
    ):
        return False
    # Derive the gate from the production dispatch so these tests can never
    # claim to cover a device the backend refuses, or skip one it supports.
    try:
        get_arch()
    except RuntimeError:
        return False
    return True


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


def _assign_oracle(x, centroids, chunk=128):
    """Exact fp32 nearest-centroid assignment, chunked to bound the temp.

    Uses the squared-difference form rather than the ``|x|^2 - 2x.c + |c|^2``
    expansion the kernel evaluates, so this is an independent reference.
    """
    centroids_f = centroids.float()[:, None, :, :]
    out = []
    for i in range(0, x.shape[1], chunk):
        block = x[:, i:i + chunk].float()[:, :, None, :]
        out.append(((block - centroids_f) ** 2).sum(-1).argmin(-1))
    return torch.cat(out, dim=1).int()


def _random_init(x, n_clusters, seed):
    """Distinct initial centroids, sampled without replacement.

    Duplicate centroids would make every point a tie between them, and the
    kernel's packed argmin does not break ties the way ``argmin`` does (the
    cluster id lives in the low mantissa bits, so the winner depends on the
    sign of the score). That disagreement says nothing about correctness, so
    keep it out of the oracle comparisons.
    """
    batch, n_samples, dim = x.shape
    generator = torch.Generator(device=x.device).manual_seed(seed)
    indices = torch.stack(
        [
            torch.randperm(n_samples, device=x.device, generator=generator)[:n_clusters]
            for _ in range(batch)
        ]
    )
    return torch.gather(x, 1, indices[..., None].expand(-1, -1, dim)).contiguous()


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

    labels, converged, n_iters = batch_kmeans_Euclid(
        x,
        6,
        max_iters=5,
        tol=1e-8,
        init_centroids=centers,
        backend="cute",
    )

    assert n_iters == 1
    torch.testing.assert_close(converged, centers)
    # `converged == centers` alone is guaranteed by the tol-break restoring the
    # previous centroids, so it cannot fail. Assert the part that can: the data
    # is 64 exact copies of each centre, so the partition must be exact.
    expected = torch.arange(6, device=x.device).repeat_interleave(64)
    assert torch.equal(labels[0].long(), expected)


def test_cute_tolerance_break_returns_the_centroids_the_labels_describe():
    # The contract is that a tol break returns the centroids the labels were
    # assigned against, not the post-update ones. Comparing cute's centroids to
    # triton's cannot detect a violation here -- with a break at iteration 1
    # both are still bit-identical to the caller's init_centroids. Assert the
    # relationship between the two returned values instead.
    torch.manual_seed(4)
    x = torch.randn(2, 2048, 128, device="cuda", dtype=torch.bfloat16)
    initial = _random_init(x, 64, seed=4)

    _, triton_centroids, triton_iters = batch_kmeans_Euclid(
        x, 64, max_iters=5, tol=1e9, init_centroids=initial, backend="triton"
    )
    cute_ids, cute_centroids, cute_iters = batch_kmeans_Euclid(
        x, 64, max_iters=5, tol=1e9, init_centroids=initial, backend="cute"
    )
    torch.cuda.synchronize()

    assert cute_iters == triton_iters == 1
    torch.testing.assert_close(cute_centroids, triton_centroids)
    agreement = (cute_ids == _assign_oracle(x, cute_centroids)).float().mean().item()
    assert agreement > 0.99, (
        f"labels disagree with the returned centroids ({agreement:.4f}); the "
        "tol break returned the post-update centroids"
    )


def test_cute_supports_batch_larger_than_cuda_grid_y_limit():
    batch = 65536
    x = torch.randn(batch, 2, 128, device="cuda", dtype=torch.bfloat16)
    initial = x.clone()

    ids, centroids, n_iters = batch_kmeans_Euclid(
        x, 2, max_iters=1, tol=0.0, init_centroids=initial, backend="cute"
    )
    torch.cuda.synchronize()

    assert ids.shape == (batch, 2)
    assert centroids.shape == (batch, 2, 128)
    assert n_iters == 1
    # This is the only grid-saturating case in the suite, so check values and
    # not just shapes. Every point IS a centroid (K == N == 2), so the labels
    # must be exactly [0, 1] and the update must reproduce x bit for bit.
    expected_ids = torch.arange(2, device=x.device, dtype=torch.int32).expand(batch, 2)
    assert torch.equal(ids, expected_ids)
    assert torch.equal(centroids, x)


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
    ("n_clusters", "n_samples"),
    [
        (2, 2),        # the smallest launch the backend accepts
        (3, 257),      # N straddles the 128-row tile
        (128, 130),    # K exactly one centroid block
        (591, 1024),   # the production geometry's K (not a multiple of 128)
        (1024, 1024),  # the largest K the packed argmin can encode
    ],
)
def test_cute_one_iteration_matches_fp32_oracle_across_shapes(n_clusters, n_samples):
    torch.manual_seed(11)
    x = torch.randn(2, n_samples, 128, device="cuda", dtype=torch.bfloat16)
    initial = _random_init(x, n_clusters, seed=n_clusters * 1000 + n_samples)

    ids, centroids, n_iters = batch_kmeans_Euclid(
        x, n_clusters, max_iters=1, tol=0.0, init_centroids=initial, backend="cute"
    )
    torch.cuda.synchronize()

    assert n_iters == 1
    assert ids.shape == (2, n_samples)
    # A packed-argmin decode that escaped [0, K) would mean the histogram and
    # centroid-sum atomics wrote out of bounds.
    assert int(ids.min()) >= 0
    assert int(ids.max()) < n_clusters

    agreement = (ids == _assign_oracle(x, initial)).float().mean().item()
    assert agreement > 0.99, f"assignment agreement {agreement:.4f} vs fp32 oracle"

    expected = _centroid_oracle(x, ids, initial).to(torch.bfloat16)
    torch.testing.assert_close(centroids, expected, atol=0.05, rtol=0.0)


@pytest.mark.parametrize("n_clusters", [2, 32, 64, 65, 96, 128, 129, 256, 591, 1024])
def test_cute_predict_matches_fp32_oracle_across_clusters(n_clusters):
    # The assign-only path splits the 128 accumulator columns across FOUR
    # epilogue groups on SM100 (vs two for the fused Lloyd path), so its
    # cross-group argmin merge has to reduce over all of them. Merging only
    # one peer silently drops columns 64..127, which stays invisible for
    # K <= 64 because those columns are padding -- hence the sweep past 64.
    torch.manual_seed(17)
    x = torch.randn(2, 2048, 128, device="cuda", dtype=torch.bfloat16)
    centroids = _random_init(x, n_clusters, seed=n_clusters)

    labels = euclid_assign_cute(x, centroids)
    torch.cuda.synchronize()

    assert labels.shape == (2, 2048)
    assert int(labels.min()) >= 0
    assert int(labels.max()) < n_clusters
    agreement = (labels == _assign_oracle(x, centroids)).float().mean().item()
    assert agreement > 0.99, (
        f"K={n_clusters}: predict agreement {agreement:.4f} vs fp32 oracle"
    )


def test_cute_keeps_the_previous_centroid_for_empty_clusters():
    torch.manual_seed(12)
    x = torch.randn(1, 2048, 128, device="cuda", dtype=torch.bfloat16)
    # 16 identical centroids: at most one can win any point, so the rest must
    # survive the update unchanged instead of collapsing to zero.
    initial = x[:, :1].expand(-1, 16, -1).contiguous()

    ids, centroids, _ = batch_kmeans_Euclid(
        x, 16, max_iters=1, tol=0.0, init_centroids=initial, backend="cute"
    )
    torch.cuda.synchronize()

    counts = torch.zeros(1, 16, dtype=torch.int64, device=x.device)
    counts.scatter_add_(1, ids.long(), torch.ones_like(ids, dtype=torch.int64))
    assert int((counts == 0).sum()) > 0, "expected this setup to leave empty clusters"

    expected = _centroid_oracle(x, ids, initial).to(torch.bfloat16)
    torch.testing.assert_close(centroids, expected, atol=0.05, rtol=0.0)


def test_cute_is_deterministic_across_repeated_calls():
    torch.manual_seed(13)
    x = torch.randn(4, 8192, 128, device="cuda", dtype=torch.bfloat16)
    initial = _random_init(x, 591, seed=13)

    runs = [
        batch_kmeans_Euclid(
            x, 591, max_iters=5, tol=0.0, init_centroids=initial, backend="cute"
        )
        for _ in range(3)
    ]
    torch.cuda.synchronize()

    for ids, centroids, n_iters in runs[1:]:
        assert torch.equal(ids, runs[0][0])
        assert torch.equal(centroids, runs[0][1])
        assert n_iters == runs[0][2]


def test_cute_accepts_non_contiguous_input():
    torch.manual_seed(14)
    # (N, B, D) storage viewed as (B, N, D): D stays innermost-contiguous.
    source = torch.randn(4096, 4, 128, device="cuda", dtype=torch.bfloat16)
    x = source.permute(1, 0, 2)
    assert not x.is_contiguous()
    initial = _random_init(x.contiguous(), 64, seed=14)

    ids, centroids, _ = batch_kmeans_Euclid(
        x, 64, max_iters=2, tol=0.0, init_centroids=initial, backend="cute"
    )
    ref_ids, ref_centroids, _ = batch_kmeans_Euclid(
        x.contiguous(), 64, max_iters=2, tol=0.0, init_centroids=initial, backend="cute"
    )
    torch.cuda.synchronize()

    assert torch.equal(ids, ref_ids)
    assert torch.equal(centroids, ref_centroids)


def test_cute_module_matches_the_device_architecture():
    device = torch.device("cuda", torch.cuda.current_device())
    assert get_lloyd_module(device).__name__.endswith(get_arch(device))


@pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason="requires at least two CUDA devices"
)
def test_cute_runs_on_a_non_default_device():
    # The compile cache is keyed on (arch, geometry, K, num_sms), not on the
    # device index, so a kernel first compiled for cuda:0 gets reused for
    # cuda:1. Check that reuse actually produces the right answer there.
    if torch.cuda.get_device_capability(0) != torch.cuda.get_device_capability(1):
        pytest.skip("the two devices are different architectures")

    results = []
    for index in (0, 1, 0):
        device = torch.device("cuda", index)
        torch.manual_seed(15)
        x = torch.randn(2, 4096, 128, device=device, dtype=torch.bfloat16)
        initial = _random_init(x, 64, seed=15)
        ids, centroids, _ = batch_kmeans_Euclid(
            x, 64, max_iters=3, tol=0.0, init_centroids=initial, backend="cute"
        )
        torch.cuda.synchronize(device)

        assert ids.device == device
        assert centroids.device == device
        assert int(ids.max()) < 64
        results.append((ids.cpu(), centroids.cpu(), _inertia(x, ids, centroids).item()))

    assert torch.equal(results[0][0], results[1][0])
    assert torch.equal(results[0][1], results[1][1])
    assert results[0][2] == pytest.approx(results[1][2], rel=1e-6)


@pytest.mark.parametrize("bad", ["nan", "inf"])
@pytest.mark.parametrize("n_clusters", [8, 59, 129])
def test_cute_non_finite_input_stays_in_bounds(bad, n_clusters):
    # The argmin seed and the padded-column sentinel are both a plain 3.0e38,
    # which is NOT a packed value: it decodes to bits(3.0e38) & 1023 == 486.
    # A single non-finite element poisons a whole row (the MMA spreads it over
    # all 128 columns, and fmin drops NaNs), so before the decode was clamped
    # that row emitted id 486 and drove the mHist / mSums atomics past the end
    # of both buffers -- inside PyTorch's allocator pool, so silently.
    torch.manual_seed(19)
    x = torch.randn(1, 256, 128, device="cuda", dtype=torch.bfloat16)
    initial = x[:, :n_clusters].contiguous().clone()
    x[0, 5, 3] = float("nan") if bad == "nan" else float("inf")

    ids, _, _ = batch_kmeans_Euclid(
        x, n_clusters, max_iters=1, tol=0.0, init_centroids=initial, backend="cute"
    )
    counts = torch.bincount(ids.flatten().long(), minlength=n_clusters)
    torch.cuda.synchronize()

    assert int(ids.min()) >= 0
    assert int(ids.max()) < n_clusters
    # every point must still be accounted for in exactly one cluster
    assert int(counts.sum()) == x.shape[1]
    assert counts.numel() == n_clusters


def test_cute_rejects_centroids_whose_norm_overflows_fp32():
    torch.manual_seed(20)
    x = torch.randn(1, 256, 128, device="cuda", dtype=torch.bfloat16)
    huge = (x[:, :8].float() * 3e19).to(torch.bfloat16).contiguous()

    with pytest.raises(ValueError, match="not finite"):
        batch_kmeans_Euclid(
            x, 8, max_iters=1, tol=0.0, init_centroids=huge, backend="cute"
        )


def test_cute_rejects_unaligned_x_with_an_actionable_message():
    # The kernels bake assumed_align=16 on every operand and .contiguous() is a
    # no-op on an already-contiguous view, so an odd base offset has to be
    # caught here. Unchecked it is an opaque tvm-ffi error, or -- without
    # tvm-ffi -- an async CUDA misaligned-address fault that kills the context.
    flat = torch.randn(256 * 128 + 8, device="cuda", dtype=torch.bfloat16)
    unaligned = flat[1 : 1 + 256 * 128].view(1, 256, 128)
    assert unaligned.is_contiguous() and unaligned.data_ptr() % 16

    with pytest.raises(ValueError, match="16-byte aligned"):
        batch_kmeans_Euclid(unaligned, 6, max_iters=1, backend="cute")

    # and the remedy the message suggests has to actually work
    ids, _, _ = batch_kmeans_Euclid(
        unaligned.clone(), 6, max_iters=1, tol=0.0, backend="cute"
    )
    torch.cuda.synchronize()
    assert int(ids.max()) < 6


def test_cute_predict_accepts_unaligned_centroids():
    torch.manual_seed(21)
    x = torch.randn(1, 256, 128, device="cuda", dtype=torch.bfloat16)
    flat = torch.randn(6 * 128 + 8, device="cuda", dtype=torch.bfloat16)
    centroids = flat[1 : 1 + 6 * 128].view(1, 6, 128)
    assert centroids.data_ptr() % 16

    labels = euclid_assign_cute(x, centroids)
    torch.cuda.synchronize()
    agreement = (labels == _assign_oracle(x, centroids)).float().mean().item()
    assert agreement > 0.99


@pytest.mark.parametrize("shape", [(1, 4, 128), (4, 128), (4 * 128,)])
def test_cute_accepts_any_reshapeable_init_centroids(shape):
    # Triton just does init_centroids.view(B, K, D), so (K, D) from a previous
    # 2-D fit is a natural thing to pass; the CuTe backend used to demand the
    # exact (B, K, D) shape while the README claimed interface parity.
    torch.manual_seed(22)
    x = torch.randn(1, 256, 128, device="cuda", dtype=torch.bfloat16)
    initial = x[0, :4].reshape(shape).contiguous()

    ids, centroids, _ = batch_kmeans_Euclid(
        x, 4, max_iters=1, tol=0.0, init_centroids=initial, backend="cute"
    )
    torch.cuda.synchronize()
    assert ids.shape == (1, 256)
    assert centroids.shape == (1, 4, 128)
    assert int(ids.max()) < 4


def test_cute_rejects_non_floating_init_centroids():
    torch.manual_seed(23)
    x = torch.randn(1, 256, 128, device="cuda", dtype=torch.bfloat16)
    initial = x[0, :4].unsqueeze(0).to(torch.int32)

    with pytest.raises(ValueError, match="floating-point"):
        batch_kmeans_Euclid(
            x, 4, max_iters=1, init_centroids=initial, backend="cute"
        )


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
