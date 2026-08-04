#!/usr/bin/env python
"""Capability check for the flattened launch grid.

The old 2D grid put the batch dimension on `grid.y`, which CUDA caps at 65535.
Any problem with B > 65535 therefore fails to launch on `main`. The flattened
1D grid moves everything onto `grid.x` (cap 2^31-1), so it launches.

Run on both branches:
    main            -> expect a launch failure (the bug this PR fixes)
    grid-fix branch -> expect PASS

    python benchmarks/grid_fix/bench_large_b.py
"""
import torch

from flash_kmeans.assign_euclid_triton import (
    euclid_assign_triton,
    cosine_assign_triton,
    _need_split_d,
)
from flash_kmeans.centroid_update_triton import triton_centroid_update_sorted_euclid

DTYPE = torch.float16

# B > 65535 so the old grid.y cap is exceeded. Two D regimes so both the
# small-D (single feature tile) and split-D (D tiled) assign paths are
# exercised -- the split-D kernels used to keep the batch dim on grid.y too.
# N/K are kept tiny so B (not the per-batch work) dominates the footprint.
#   D=32  -> small-D path
#   D=768 -> split-D path (D > _SMALL_D_MAX=512)
CASES = [
    ("small-D", dict(B=70_000, N=64, K=8, D=32)),
    ("split-D", dict(B=70_000, N=32, K=8, D=768)),
]


def run_case(tag, B, N, K, D):
    dev = "cuda"
    split = _need_split_d(D, DTYPE, torch.device(dev))
    print(f"[{tag}] B={B} (> 65535), N={N}, K={K}, D={D} "
          f"(dispatch: {'split-D' if split else 'small-D'})")
    x = torch.randn(B, N, D, device=dev, dtype=DTYPE)
    centroids = torch.randn(B, K, D, device=dev, dtype=DTYPE)
    # Accumulate in fp32 without materialising a full fp32 copy of x (which
    # would be 2x the already-large x for big B*D).
    x_sq = (x * x).sum(-1, dtype=torch.float32)
    cluster_ids = torch.randint(0, K, (B, N), device=dev, dtype=torch.int64)

    ok = True
    for name, fn in [
        ("euclid_assign", lambda: euclid_assign_triton(x, centroids, x_sq)),
        ("cosine_assign", lambda: cosine_assign_triton(x, centroids)),
        ("euclid_update", lambda: triton_centroid_update_sorted_euclid(x, cluster_ids, centroids)),
    ]:
        try:
            out = fn()
            torch.cuda.synchronize()
            print(f"  [PASS] {name} launched, out shape {tuple(out.shape)}")
        except Exception as e:
            ok = False
            msg = str(e).splitlines()[0] if str(e) else type(e).__name__
            print(f"  [FAIL] {name}: {msg}")
    print()
    # Free before the next (larger) case so we don't OOM stacking allocations.
    del x, centroids, x_sq, cluster_ids
    torch.cuda.empty_cache()
    return ok


def main():
    print(f"GPU: {torch.cuda.get_device_name(0)}\n")
    # List (not generator) so all() can't short-circuit: every case must run
    # even after a failure, so `main` demonstrates both D regimes failing.
    results = [run_case(tag, **shape) for tag, shape in CASES]
    ok = all(results)
    print(f"=> {'ALL LAUNCHED (grid fix works)' if ok else 'LAUNCH FAILED (expected on main)'}")


if __name__ == "__main__":
    main()
