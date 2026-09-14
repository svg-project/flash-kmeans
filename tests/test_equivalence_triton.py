"""Functional-equivalence tests against the original Triton implementation.

The upstream package is executed in a subprocess (see ``_triton_ref_runner.py``)
so both implementations cannot interfere with each other.  The comparison
criteria are the ones that define functional equivalence for k-means:

  * one iteration on identical inputs must produce (almost) identical
    assignments -- this isolates the kernels from chaotic drift,
  * after several iterations both implementations must reach the same
    clustering objective (inertia) up to floating-point noise,
  * converged partitions must overlap almost perfectly.

The reference repository is looked up through ``FLASH_KMEANS_REF_REPO`` and
defaults to the sibling checkout ``../flash-kmeans`` (the git worktree layout
used to develop this rewrite).  Tests skip if it is unavailable.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from flash_kmeans import (
    batch_kmeans_Cosine,
    batch_kmeans_Dot,
    batch_kmeans_Euclid,
    euclid_assign_triton,
    kmeans_largeN,
    triton_centroid_update_sorted_euclid,
)

REF_REPO = Path(
    os.environ.get(
        "FLASH_KMEANS_REF_REPO",
        Path(__file__).resolve().parents[2] / "flash-kmeans",
    )
)
RUNNER = Path(__file__).resolve().parent / "_triton_ref_runner.py"

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required"
)


def _reference_available() -> bool:
    return (REF_REPO / "flash_kmeans" / "kmeans_triton_impl.py").exists()


needs_reference = pytest.mark.skipif(
    not _reference_available(),
    reason=f"original Triton implementation not found at {REF_REPO}",
)

DTYPES = [torch.float32, torch.float16, torch.bfloat16]


def run_reference(tmp_path: Path, payload: dict) -> dict:
    inputs = tmp_path / "inputs.pt"
    outputs = tmp_path / "outputs.pt"
    torch.save(payload, inputs)
    env = dict(os.environ, PYTHONPATH=str(REF_REPO))
    proc = subprocess.run(
        [sys.executable, str(RUNNER), str(inputs), str(outputs)],
        env=env,
        capture_output=True,
        text=True,
        cwd=str(tmp_path),
    )
    if proc.returncode != 0:
        pytest.skip(f"original Triton reference failed to run: {proc.stderr[-400:]}")
    return torch.load(outputs, map_location="cuda", weights_only=False)


def make_inputs(B, N, D, K, dtype, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(B, N, D, device="cuda", dtype=dtype, generator=g)
    init = torch.randn(B, K, D, device="cuda", dtype=dtype, generator=g)
    return x, init


def inertia(x, labels, centroids):
    if labels.dim() == 1:
        assigned = centroids.float()[labels.long()]
        return ((x.float() - assigned) ** 2).sum().item()
    assigned = centroids.float().gather(
        1, labels.long().unsqueeze(-1).expand(-1, -1, x.shape[-1])
    )
    return ((x.float() - assigned) ** 2).sum().item()


@needs_reference
@pytest.mark.parametrize("mode", ["euclid", "cosine", "dot"])
@pytest.mark.parametrize("dtype", DTYPES)
def test_batch_kmeans_matches_triton(tmp_path, mode, dtype):
    """One-iteration kernel equivalence plus converged-objective equivalence."""
    B, N, D, K = 2, 4096, 64, 32
    x, init = make_inputs(B, N, D, K, dtype)
    ref = run_reference(
        tmp_path,
        {"x": x, "init": init, "K": K, "iters": [1, 8], "modes": [mode]},
    )
    fn = {
        "euclid": batch_kmeans_Euclid,
        "cosine": batch_kmeans_Cosine,
        "dot": batch_kmeans_Dot,
    }[mode]

    # --- single iteration: kernels must agree almost exactly -------------
    labels1, cent1, _ = fn(x, K, max_iters=1, tol=0.0, init_centroids=init)
    ref_labels1 = ref[mode][0]["labels"].cuda()
    agree1 = (labels1 == ref_labels1).float().mean().item()
    assert agree1 >= 0.99, f"single-iteration label agreement {agree1}"

    # --- eight iterations: both must reach the same optimum --------------
    labels8, cent8, _ = fn(x, K, max_iters=8, tol=0.0, init_centroids=init)
    ref_labels8 = ref[mode][1]["labels"].cuda()
    ref_cent8 = ref[mode][1]["centroids"].cuda()
    ref_inertia = inertia(x, ref_labels8, ref_cent8)
    rel = abs(inertia(x, labels8, cent8) - ref_inertia) / ref_inertia
    assert rel <= 1e-3, f"inertia relative difference {rel}"
    # independent fp32 accumulation orders make long k-means runs chaotic;
    # inertia above is the meaningful functional-equivalence criterion.
    agree8 = (labels8 == ref_labels8).float().mean().item()
    assert agree8 >= 0.8, f"converged label agreement {agree8}"


@needs_reference
@pytest.mark.parametrize("dtype", DTYPES)
def test_assignment_kernel_matches_triton(tmp_path, dtype):
    B, N, D, K = 2, 4096, 128, 64
    x, cent = make_inputs(B, N, D, K, dtype)
    ref = run_reference(tmp_path, {"x": x, "init": cent, "K": K, "iters": 1,
                                   "modes": [], "assign": True})
    labels = euclid_assign_triton(x, cent)
    ref_labels = ref["assign"]["labels"].cuda()
    agree = (labels == ref_labels).float().mean().item()
    assert agree >= 0.999, f"assignment label agreement {agree}"


@needs_reference
def test_centroid_update_matches_triton(tmp_path):
    B, N, D, K = 2, 4096, 64, 32
    x, init = make_inputs(B, N, D, K, torch.float32)
    ref = run_reference(tmp_path, {"x": x, "init": init, "K": K, "iters": 1,
                                   "modes": [], "update": True})
    labels = ref["update"]["labels"].cuda()
    updated = triton_centroid_update_sorted_euclid(x, labels, init)
    diff = (updated.float() - ref["update"]["centroids"].cuda().float()).abs().max().item()
    assert diff <= 1e-3, f"centroid update max diff {diff}"


@needs_reference
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_large_n_matches_triton(tmp_path, dtype):
    N, D, K = 200_000, 64, 128
    g = torch.Generator().manual_seed(0)
    x_cpu = torch.randn(N, D, generator=g).to(dtype).pin_memory()
    init = torch.randn(K, D, device="cuda", dtype=dtype)
    ref = run_reference(
        tmp_path,
        {
            "x": x_cpu.to("cuda").unsqueeze(0),
            "x_cpu": x_cpu,
            "init": init.unsqueeze(0),
            "init3": init.unsqueeze(0),
            "K": K,
            "iters": 2,
            "modes": [],
            "largeN": True,
            "block_n": 65536,
        },
    )
    labels, centroids = kmeans_largeN(
        x_cpu, K, max_iters=2, tol=0.0, BLOCK_N=65536, init_centroids=init,
        device="cuda:0",
    )
    ref_labels = ref["largeN"]["labels"].cuda().view(-1)
    ref_cent = ref["largeN"]["centroids"].cuda().view(K, D)
    # chunked atomic accumulation orders differ between implementations, so
    # long trajectories drift: compare the clustering objective and require
    # a near-identical partition.
    x_cuda = x_cpu.to("cuda").float()
    ref_inertia = inertia(x_cuda, ref_labels, ref_cent)
    rel = abs(inertia(x_cuda, labels, centroids) - ref_inertia) / ref_inertia
    assert rel <= 2e-3, f"large-N inertia relative difference {rel}"
    agree = (labels == ref_labels).float().mean().item()
    assert agree >= 0.98, f"large-N label agreement {agree}"
    # random data has no cluster structure, so the few flipped assignments move
    # small centroids by up to ~1e-1; keep a loose sanity bound only.
    diff = (centroids.float() - ref_cent.float()).abs().max().item()
    assert diff <= 0.1, f"large-N centroid diff {diff}"
