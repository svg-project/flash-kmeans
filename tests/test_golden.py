"""Golden-output regression tests.

The fixtures in ``tests/data`` were produced by the upstream Triton
implementation (see ``_generate_golden.py``); inputs are regenerated from the
stored CPU seed.  This keeps functional equivalence covered even when the
upstream checkout is not available.
"""

from __future__ import annotations

import pathlib

import pytest
import torch

from flash_kmeans import batch_kmeans_Cosine, batch_kmeans_Dot, batch_kmeans_Euclid

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required"
)

DATA = pathlib.Path(__file__).resolve().parent / "data"

CASES = {
    "euclid_fp32": (batch_kmeans_Euclid, "golden_euclid_fp32.pt"),
    "euclid_fp16": (batch_kmeans_Euclid, "golden_euclid_fp16.pt"),
    "cosine_fp32": (batch_kmeans_Cosine, "golden_cosine_fp32.pt"),
    "dot_fp32": (batch_kmeans_Dot, "golden_dot_fp32.pt"),
}

# Upstream fp32 kernels use tf32 for the dot products in some paths, so after a
# few iterations the partitions drift slightly; the objective (inertia) is the
# primary equivalence criterion, agreement is a sanity bound.
AGREEMENT = {"euclid_fp32": 0.9, "euclid_fp16": 0.9, "cosine_fp32": 0.9, "dot_fp32": 0.9}


def rebuild_inputs(golden):
    dtype = getattr(torch, golden["dtype"])
    g = torch.Generator().manual_seed(golden["seed"])
    x = torch.randn(golden["B"], golden["N"], golden["D"], generator=g).to(dtype).cuda()
    init = torch.randn(golden["B"], golden["K"], golden["D"], generator=g).to(dtype).cuda()
    return x, init


def inertia(x, labels, centroids):
    assigned = centroids.float().gather(
        1, labels.long().unsqueeze(-1).expand(-1, -1, x.shape[-1])
    )
    return ((x.float() - assigned) ** 2).sum().item()


@pytest.mark.parametrize("name", sorted(CASES))
def test_matches_upstream_golden(name):
    fn, filename = CASES[name]
    path = DATA / filename
    if not path.exists():
        pytest.skip(f"missing golden fixture {path}")
    golden = torch.load(path, map_location="cuda", weights_only=False)
    x, init = rebuild_inputs(golden)
    ref_labels = golden["labels"].cuda()
    ref_cent = golden["centroids"].cuda()

    labels, centroids, n_iters = fn(
        x, golden["K"], max_iters=golden["iters"], tol=0.0, init_centroids=init
    )
    assert n_iters == golden["iters"]

    ref_inertia = inertia(x, ref_labels, ref_cent)
    rel = abs(inertia(x, labels, centroids) - ref_inertia) / ref_inertia
    assert rel <= 1e-3, f"{name}: inertia relative difference {rel}"
    agree = (labels == ref_labels).float().mean().item()
    assert agree >= AGREEMENT[name], f"{name}: label agreement {agree}"
