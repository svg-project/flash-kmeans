#!/usr/bin/env python
"""Regression benchmark for the flattened 1D launch-grid change.

The grid change only rewrites how `program_id` is decoded and how the grid is
launched for the *non-split* assign kernels (`euclid`/`cosine`) and the shared
`_centroid_update_chunk_kernel`. The math each program does is unchanged, so
this script exists to prove throughput is not regressed on the paths every
current user hits.

Run it on `main` and on the grid-fix branch, then compare the two JSON files:

    git checkout main
    python benchmarks/grid_fix/bench_grid.py --out /tmp/grid_main.json
    git checkout <grid-fix-branch>
    python benchmarks/grid_fix/bench_grid.py --out /tmp/grid_fix.json
    python benchmarks/grid_fix/bench_grid.py --compare /tmp/grid_main.json /tmp/grid_fix.json

All shapes use D <= 512 so they stay on the non-split kernels (the ones the fix
touches). The split-D kernels keep their 2D launch and are out of scope here.
"""
import argparse
import json
import subprocess

import torch

from flash_kmeans.assign_euclid_triton import euclid_assign_triton, cosine_assign_triton
from flash_kmeans.centroid_update_triton import (
    triton_centroid_update_sorted_euclid,
    triton_centroid_update_sorted_cosine,
)

# (B, N, K, D) — a spread of batch/point/cluster/dim sizes on the non-split path.
SHAPES = [
    (1, 1_000_000, 256, 128),   # single big batch, large N
    (8, 131_072, 256, 128),     # medium batch
    (64, 16_384, 256, 128),     # many batches
    (1, 4_000_000, 1024, 64),   # very large N, K=1024
    (256, 8_192, 128, 128),     # large B (2D grid.y would be 256 here — still legal)
    (4, 262_144, 512, 256),     # larger D
]

DTYPE = torch.float16
WARMUP = 5
ITERS = 20


def _time_ms(fn, warmup=WARMUP, iters=ITERS):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def _mpts_per_s(B, N, ms):
    """Millions of points processed per second (B*N points per call)."""
    return (B * N) / (ms * 1e-3) / 1e6


def bench_case(B, N, K, D):
    dev = "cuda"
    x = torch.randn(B, N, D, device=dev, dtype=DTYPE)
    centroids = torch.randn(B, K, D, device=dev, dtype=DTYPE)
    x_sq = (x.float() ** 2).sum(-1)
    cluster_ids = torch.randint(0, K, (B, N), device=dev, dtype=torch.int64)

    # normalized copy for cosine centroid update
    x_norm = torch.nn.functional.normalize(x.float(), dim=-1).to(DTYPE)

    results = {}

    results["euclid_assign"] = _time_ms(
        lambda: euclid_assign_triton(x, centroids, x_sq)
    )
    results["cosine_assign"] = _time_ms(
        lambda: cosine_assign_triton(x, centroids)
    )
    results["euclid_update"] = _time_ms(
        lambda: triton_centroid_update_sorted_euclid(x, cluster_ids, centroids)
    )
    results["cosine_update"] = _time_ms(
        lambda: triton_centroid_update_sorted_cosine(x_norm, cluster_ids, centroids)
    )

    del x, centroids, x_sq, cluster_ids, x_norm
    torch.cuda.empty_cache()
    return results


def run(out_path):
    commit = subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"]
    ).decode().strip()
    branch = subprocess.check_output(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"]
    ).decode().strip()

    gpu = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu}  |  branch: {branch}  |  commit: {commit}\n")

    all_results = {"gpu": gpu, "branch": branch, "commit": commit, "cases": {}}

    header = f"{'shape (B,N,K,D)':>26} | {'euclid_assign':>14} | {'cosine_assign':>14} | {'euclid_update':>14} | {'cosine_update':>14}"
    print(header)
    print("-" * len(header))
    for shape in SHAPES:
        r = bench_case(*shape)
        all_results["cases"][str(shape)] = r
        B, N = shape[0], shape[1]
        print(
            f"{str(shape):>26} | "
            f"{r['euclid_assign']:>7.3f} ms {_mpts_per_s(B,N,r['euclid_assign']):>4.0f}M/s | "
            f"{r['cosine_assign']:>7.3f} ms {_mpts_per_s(B,N,r['cosine_assign']):>4.0f}M/s | "
            f"{r['euclid_update']:>7.3f} ms {_mpts_per_s(B,N,r['euclid_update']):>4.0f}M/s | "
            f"{r['cosine_update']:>7.3f} ms {_mpts_per_s(B,N,r['cosine_update']):>4.0f}M/s"
        )

    if out_path:
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nSaved -> {out_path}")


def compare(before_path, after_path):
    with open(before_path) as f:
        before = json.load(f)
    with open(after_path) as f:
        after = json.load(f)

    print(f"BEFORE: {before['branch']} @ {before['commit']}")
    print(f"AFTER : {after['branch']} @ {after['commit']}")
    print(f"GPU   : {after['gpu']}\n")
    print("Numbers are AFTER/BEFORE latency ratio (<1.0 = faster, >1.0 = slower).\n")

    kernels = ["euclid_assign", "cosine_assign", "euclid_update", "cosine_update"]
    header = f"{'shape (B,N,K,D)':>26} | " + " | ".join(f"{k:>14}" for k in kernels)
    print(header)
    print("-" * len(header))

    worst = 0.0
    for shape, b in before["cases"].items():
        a = after["cases"].get(shape)
        if a is None:
            continue
        ratios = []
        cells = []
        for k in kernels:
            ratio = a[k] / b[k]
            ratios.append(ratio)
            worst = max(worst, ratio)
            cells.append(f"{ratio:>13.3f}x")
        print(f"{shape:>26} | " + " | ".join(cells))

    print(f"\nWorst-case slowdown ratio across all cases/kernels: {worst:.3f}x")
    if worst <= 1.05:
        print("=> No meaningful regression (within 5% noise band).")
    else:
        print("=> Potential regression >5%; inspect the case above.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=None, help="write results JSON to this path")
    p.add_argument(
        "--compare", nargs=2, metavar=("BEFORE", "AFTER"),
        help="compare two result JSON files instead of benchmarking",
    )
    args = p.parse_args()

    if args.compare:
        compare(*args.compare)
    else:
        run(args.out)


if __name__ == "__main__":
    main()
