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

from flash_kmeans.assign_euclid_triton import euclid_assign_triton, cosine_assign_triton
from flash_kmeans.centroid_update_triton import triton_centroid_update_sorted_euclid

# B > 65535 so the old grid.y cap is exceeded; keep N/D/K tiny to stay small.
B, N, K, D = 70_000, 64, 8, 32
DTYPE = torch.float16


def main():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Case: B={B} (> 65535), N={N}, K={K}, D={D}\n")
    dev = "cuda"
    x = torch.randn(B, N, D, device=dev, dtype=DTYPE)
    centroids = torch.randn(B, K, D, device=dev, dtype=DTYPE)
    x_sq = (x.float() ** 2).sum(-1)
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

    print(f"\n=> {'ALL LAUNCHED (grid fix works)' if ok else 'LAUNCH FAILED (expected on main)'}")


if __name__ == "__main__":
    main()
