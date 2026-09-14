"""Kernel benchmark and performance-baseline generator.

Usage
-----
    pixi run python benchmarks/bench_kmeans.py                # print timings
    pixi run python benchmarks/bench_kmeans.py --update       # refresh
        tests/perf_baselines.json for the current GPU
    FLASH_KMEANS_PROFILE=1 pixi run python benchmarks/bench_kmeans.py
        # additionally print the per-phase breakdown from the C++
        # instrumentation (assignment, accumulate, finalize, ...)
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib

import torch

from flash_kmeans import batch_kmeans_Euclid, euclid_assign_triton

BASELINES = pathlib.Path(__file__).resolve().parents[1] / "tests" / "perf_baselines.json"

BATCH = dict(B=32, N=74256, K=1000, iters=5)

CASES = {
    "batch_fp16_D128": "end-to-end batch k-means, fp16, D=128, K=1000 per iteration",
    "assign_fp16_D128": "assignment only, fp16, D=128, K=1000",
    "assign_fp32_D64": "assignment only, fp32, D=64, K=4096",
}


def time_callable(fn, warmup: int = 2, reps: int = 5) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(reps):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / reps


def make_inputs(dtype, D, B, N, K):
    g = torch.Generator(device="cuda").manual_seed(0)
    x = torch.randn(B, N, D, device="cuda", dtype=dtype, generator=g)
    init = torch.randn(B, K, D, device="cuda", dtype=dtype, generator=g)
    return x, init


def measure_all() -> dict[str, float]:
    B, N, K, iters = BATCH["B"], BATCH["N"], BATCH["K"], BATCH["iters"]
    g = torch.Generator(device="cuda").manual_seed(0)

    x16 = torch.randn(B, N, 128, device="cuda", dtype=torch.float16, generator=g)
    c16 = torch.randn(B, K, 128, device="cuda", dtype=torch.float16, generator=g)
    x32 = torch.randn(B, N, 64, device="cuda", dtype=torch.float32, generator=g)
    c32 = torch.randn(B, 4096, 64, device="cuda", dtype=torch.float32, generator=g)

    results = {}
    total = time_callable(
        lambda: batch_kmeans_Euclid(x16, K, max_iters=iters, init_centroids=c16)
    )
    results["batch_fp16_D128"] = total / iters
    results["assign_fp16_D128"] = time_callable(lambda: euclid_assign_triton(x16, c16))
    results["assign_fp32_D64"] = time_callable(lambda: euclid_assign_triton(x32, c32))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--update",
        action="store_true",
        help="write the measurements into tests/perf_baselines.json",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    torch.cuda.init()
    device_name = torch.cuda.get_device_name(0)
    if os.environ.get("FLASH_KMEANS_PROFILE"):
        print("(per-phase profile enabled via FLASH_KMEANS_PROFILE)")

    results = measure_all()
    flops = 2 * BATCH["B"] * BATCH["N"] * 128 * BATCH["K"] * BATCH["iters"]
    print(f"\ndevice: {device_name}")
    for key, ms in results.items():
        print(f"  {key:20s} {ms:9.3f} ms   ({CASES[key]})")
    print(
        f"  assign fp16 D=128 throughput: "
        f"{2 * BATCH['B'] * BATCH['N'] * 128 * BATCH['K'] / (results['assign_fp16_D128'] / 1e3) / 1e12:.1f} TFLOP/s"
    )
    print(
        f"  batch throughput: "
        f"{flops / (results['batch_fp16_D128'] * BATCH['iters'] / 1e3) / 1e12:.1f} TFLOP/s"
    )

    if args.update:
        baselines = {}
        if BASELINES.exists():
            baselines = json.loads(BASELINES.read_text())
        baselines[device_name] = {
            "tolerance": 1.3,
            "cases": {k: round(v, 4) for k, v in results.items()},
        }
        BASELINES.write_text(json.dumps(baselines, indent=2, sort_keys=True) + "\n")
        print(f"\nupdated {BASELINES}")


if __name__ == "__main__":
    main()
