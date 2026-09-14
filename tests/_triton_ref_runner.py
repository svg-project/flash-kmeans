"""Run the original Triton implementation on serialized inputs.

Executed as a subprocess with ``PYTHONPATH`` pointing at the upstream
``flash-kmeans`` checkout so that ``import flash_kmeans`` resolves to the
Triton-based reference, not to the C++ rewrite under test.
"""

from __future__ import annotations

import sys

import torch

assert len(sys.argv) == 3, "usage: _triton_ref_runner.py <inputs.pt> <outputs.pt>"
inputs_path, outputs_path = sys.argv[1], sys.argv[2]

# An editable install of the rewrite registers an import hook that takes
# precedence over PYTHONPATH; drop it so the upstream Triton package is used.
sys.meta_path = [
    finder
    for finder in sys.meta_path
    if "editable" not in type(finder).__module__
    and "scikit_build" not in type(finder).__module__
]

import flash_kmeans as fk  # noqa: E402  (must come after subprocess path setup)

if "flash-kmeans-cpp" in fk.__file__:
    raise RuntimeError(f"resolved the rewrite instead of the reference: {fk.__file__}")

inputs = torch.load(inputs_path, map_location="cuda")
x = inputs["x"]
init = inputs.get("init")
K = inputs["K"]
iters = inputs.get("iters", 1)
modes = inputs.get("modes", ["euclid", "cosine", "dot"])
iters_list = list(iters) if isinstance(iters, (list, tuple)) else [iters]

fns = {
    "euclid": fk.batch_kmeans_Euclid,
    "cosine": fk.batch_kmeans_Cosine,
    "dot": fk.batch_kmeans_Dot,
}

out = {}
for mode in modes:
    runs = []
    for n_iter in iters_list:
        labels, centroids, n_iters = fns[mode](
            x, K, max_iters=n_iter, tol=0.0, init_centroids=init
        )
        runs.append(
            {
                "labels": labels.cpu(),
                "centroids": centroids.cpu(),
                "iters": n_iters,
            }
        )
    out[mode] = runs

if inputs.get("assign", False):
    from flash_kmeans.assign_euclid_triton import euclid_assign_triton

    x_sq = (x.to(torch.float32) ** 2).sum(-1).to(x.dtype)
    out["assign"] = {"labels": euclid_assign_triton(x, init, x_sq).cpu()}

if inputs.get("update", False):
    labels, _, _ = fk.batch_kmeans_Euclid(x, K, max_iters=1, tol=0.0, init_centroids=init)
    updated = fk.triton_centroid_update_sorted_euclid(x, labels, init)
    out["update"] = {"labels": labels.cpu(), "centroids": updated.cpu()}

if inputs.get("largeN", False):
    x_cpu = inputs["x_cpu"].cpu()
    init3 = inputs.get("init3")
    init2 = init3[0] if (init3 is not None and init3.dim() == 3) else init3
    labels, centroids = fk.kmeans_largeN(
        x_cpu,
        K,
        max_iters=iters_list[0],
        tol=0.0,
        BLOCK_N=inputs.get("block_n", 65536),
        init_centroids=init2,
        device="cuda:0",
    )
    out["largeN"] = {"labels": labels.cpu(), "centroids": centroids.cpu()}

torch.save(out, outputs_path)
print("reference ok")
