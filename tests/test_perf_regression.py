"""Performance regression tests.

These compare the current build against recorded baselines for the same GPU
(``tests/perf_baselines.json``).  They only run when ``FLASH_KMEANS_PERF=1``
is set (see ``conftest.py``), because timings depend on the machine, clocks
and other load.

Refresh the baselines with::

    pixi run python benchmarks/bench_kmeans.py --update
"""

from __future__ import annotations

import json
import pathlib

import pytest
import torch

from flash_kmeans import batch_kmeans_Euclid, euclid_assign_triton

pytestmark = [
    pytest.mark.perf,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required"),
]

BASELINES = pathlib.Path(__file__).resolve().parent / "perf_baselines.json"


def load_baseline():
    if not BASELINES.exists():
        pytest.skip(f"no baselines at {BASELINES}")
    table = json.loads(BASELINES.read_text())
    name = torch.cuda.get_device_name(0)
    if name not in table:
        pytest.skip(f"no baseline recorded for {name}")
    return table[name]


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


@pytest.fixture(scope="module")
def measured():
    baseline = load_baseline()
    g = torch.Generator(device="cuda").manual_seed(0)
    B, N, K, iters = 32, 74256, 1000, 5
    x16 = torch.randn(B, N, 128, device="cuda", dtype=torch.float16, generator=g)
    c16 = torch.randn(B, K, 128, device="cuda", dtype=torch.float16, generator=g)
    x32 = torch.randn(B, N, 64, device="cuda", dtype=torch.float32, generator=g)
    c32 = torch.randn(B, 4096, 64, device="cuda", dtype=torch.float32, generator=g)

    results = {}
    results["batch_fp16_D128"] = (
        time_callable(
            lambda: batch_kmeans_Euclid(x16, K, max_iters=iters, init_centroids=c16)
        )
        / iters
    )
    results["assign_fp16_D128"] = time_callable(lambda: euclid_assign_triton(x16, c16))
    results["assign_fp32_D64"] = time_callable(lambda: euclid_assign_triton(x32, c32))
    return baseline, results


@pytest.mark.parametrize(
    "case", ["batch_fp16_D128", "assign_fp16_D128", "assign_fp32_D64"]
)
def test_within_baseline(measured, case):
    baseline, results = measured
    expected = baseline["cases"][case]
    tolerance = baseline.get("tolerance", 1.3)
    actual = results[case]
    limit = expected * tolerance
    assert actual <= limit, (
        f"{case}: {actual:.3f} ms is more than {tolerance}x the recorded "
        f"baseline ({expected:.3f} ms on {torch.cuda.get_device_name(0)})"
    )
    print(f"{case}: {actual:.3f} ms (baseline {expected:.3f} ms)")
