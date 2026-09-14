"""Generate frozen golden outputs with the upstream Triton implementation.

Run this only when refreshing the regression fixtures::

    FLASH_KMEANS_REF_REPO=/path/to/flash-kmeans \
        pixi run python tests/_generate_golden.py

Inputs are regenerated from a CPU seed inside the test, so only the labels and
centroids are stored; the resulting ``tests/data/golden_*.pt`` fixtures are a
few KiB each and let the suite verify functional equivalence even on machines
without the upstream checkout.
"""

from __future__ import annotations

import os
import pathlib
import sys

import torch

REF_REPO = os.environ.get(
    "FLASH_KMEANS_REF_REPO",
    str(pathlib.Path(__file__).resolve().parents[2] / "flash-kmeans"),
)
sys.path.insert(0, REF_REPO)
# drop editable-install hooks so the upstream package is really imported
sys.meta_path = [
    f
    for f in sys.meta_path
    if "editable" not in type(f).__module__ and "scikit_build" not in type(f).__module__
]

import flash_kmeans as fk  # noqa: E402

assert "flash-kmeans-cpp" not in fk.__file__, f"resolved the rewrite: {fk.__file__}"

DATA = pathlib.Path(__file__).resolve().parent / "data"
DATA.mkdir(exist_ok=True)

CASES = [
    ("euclid_fp32", "batch_kmeans_Euclid", torch.float32),
    ("euclid_fp16", "batch_kmeans_Euclid", torch.float16),
    ("cosine_fp32", "batch_kmeans_Cosine", torch.float32),
    ("dot_fp32", "batch_kmeans_Dot", torch.float32),
]

B, N, D, K, ITERS = 2, 1024, 32, 16, 4


def main() -> None:
    for name, fn_name, dtype in CASES:
        g = torch.Generator().manual_seed(0)  # CPU generator: machine independent
        x = torch.randn(B, N, D, generator=g).to(dtype).cuda()
        init = torch.randn(B, K, D, generator=g).to(dtype).cuda()
        fn = getattr(fk, fn_name)
        labels, centroids, iters = fn(
            x, K, max_iters=ITERS, tol=0.0, init_centroids=init
        )
        path = DATA / f"golden_{name}.pt"
        torch.save(
            {
                "seed": 0,
                "B": B,
                "N": N,
                "D": D,
                "K": K,
                "iters": ITERS,
                "dtype": str(dtype).replace("torch.", ""),
                "labels": labels.cpu(),
                "centroids": centroids.cpu(),
                "source": f"upstream Triton ({fn_name}, {dtype})",
            },
            path,
        )
        size_kb = path.stat().st_size / 1024
        print(f"wrote {path.name} ({size_kb:.1f} KiB, {fn_name}, {dtype})")


if __name__ == "__main__":
    main()
