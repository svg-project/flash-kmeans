from unittest import mock
import subprocess
import sys

import pytest
import torch

from flash_kmeans import FlashKMeans, batch_kmeans_Euclid
from flash_kmeans import kmeans_api


def test_default_backend_forwards_to_triton_unchanged():
    sentinel = object()
    fake = mock.Mock(return_value=sentinel)
    x = torch.empty(1, 2, 3)

    with mock.patch.object(kmeans_api, "_batch_kmeans_euclid_triton", fake):
        result = batch_kmeans_Euclid(
            x,
            7,
            max_iters=4,
            tol=0.25,
            init_centroids=None,
            verbose=True,
            use_heuristic=False,
        )

    assert result is sentinel
    fake.assert_called_once_with(
        x=x,
        n_clusters=7,
        max_iters=4,
        tol=0.25,
        init_centroids=None,
        verbose=True,
        use_heuristic=False,
    )


def test_torch_backend_runs_without_cuda():
    torch.manual_seed(0)
    x = torch.randn(2, 32, 4)
    ids, centroids, n_iters = batch_kmeans_Euclid(
        x, 3, max_iters=2, backend="torch"
    )

    assert ids.shape == (2, 32)
    assert centroids.shape == (2, 3, 4)
    assert n_iters == 2


def test_invalid_backend_is_rejected():
    with pytest.raises(ValueError, match="backend must be one of"):
        batch_kmeans_Euclid(torch.empty(1, 2, 3), 2, backend="invalid")


def test_flash_kmeans_legacy_backend_mapping():
    assert FlashKMeans(4, 2, use_triton=False).backend == "torch"
    assert FlashKMeans(4, 2, use_triton=False, backend="torch").backend == "torch"


def test_flash_kmeans_explicit_backend_wins_over_legacy_flag():
    if not torch.cuda.is_available():
        pytest.skip("explicit CuTe selection requires CUDA")
    model = FlashKMeans(128, 2, use_triton=False, backend="cute")
    assert model.backend == "cute"
    assert not model.use_triton


def test_import_does_not_load_cute_dependencies():
    code = "import sys, flash_kmeans; assert 'cutlass' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)


def test_torch_backend_survives_a_missing_triton_runtime_dependency():
    code = """
import sys
sys.modules['tqdm'] = None
import torch
from flash_kmeans import FlashKMeans, batch_kmeans_Euclid
assert FlashKMeans(4, 2, backend='torch').backend == 'torch'
x = torch.randn(1, 16, 4)
ids, centers, n_iters = batch_kmeans_Euclid(
    x, 2, max_iters=1, backend='torch'
)
assert ids.shape == (1, 16)
assert centers.shape == (1, 2, 4)
assert n_iters == 1
"""
    subprocess.run([sys.executable, "-c", code], check=True)
