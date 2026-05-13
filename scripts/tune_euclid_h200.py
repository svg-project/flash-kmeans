"""Autotune harness for the no-xsq Triton Euclid assign kernels (H200).

Sweeps every config in ``_TUNE_CONFIGS`` (small-D kernel, key=N,K) and
``_TUNE_CONFIGS_SPLIT_D`` (split-D kernel, key=N,K,D) over a focused
shape grid that mirrors the coverage of the existing per-arch heuristic
tables in ``flash_kmeans/assign_euclid_triton.py``. The winning configs
(by mean elapsed time over ``repeats`` timed iters after ``warmup``
warmups) are emitted to a JSON file.

Run as::

    python scripts/tune_euclid_h200.py --kind smallD --gpu 4 --out small_g4.json
    # … or shard across multiple GPUs with --shard / --num-shards

The main launcher ``--all`` shards cells across GPUs 4-7 via subprocesses
and merges the per-GPU JSONs at the end.
"""
from __future__ import annotations

import argparse
import gc
import itertools
import json
import os
import subprocess
import sys
import time
from typing import List, Tuple

# Defer torch / triton import to subprocess workers so the main launcher
# stays fast.


# ---------------------------------------------------------------------------
# Shape grids — trimmed to keep wall-clock under ~1 hour per kind across 4
# GPUs while still covering every (D-bucket × K-bucket) cell the current
# H200 heuristic tables branch on.
# ---------------------------------------------------------------------------

SMALLD_DS = [64, 128, 256, 512]
SMALLD_KS = [256, 4096, 65536, 200000]
SMALLD_NS = [65536, 1048576]
SMALLD_DTYPES = ["fp16", "fp32"]

LARGED_DS = [1024, 2048, 4096]
LARGED_KS = [256, 4096, 65536]
LARGED_NS = [65536, 1048576]
LARGED_DTYPES = ["fp16", "fp32"]


def smallD_cells():
    for D, K, N, dt in itertools.product(SMALLD_DS, SMALLD_KS, SMALLD_NS, SMALLD_DTYPES):
        yield (N, K, D, dt)


def largeD_cells():
    for D, K, N, dt in itertools.product(LARGED_DS, LARGED_KS, LARGED_NS, LARGED_DTYPES):
        yield (N, K, D, dt)


_DTYPE_MAP = {"fp16": "torch.float16", "bf16": "torch.bfloat16", "fp32": "torch.float32"}


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------


def _bench_us(fn, repeats: int, warmup: int):
    import torch
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(repeats):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / repeats * 1000  # us


def _bench_one_smallD(N, K, D, dtype_s, repeats, warmup, configs, verbose=False):
    import torch
    import triton
    from flash_kmeans.assign_euclid_triton import (
        _euclid_assign_kernel,
        _dtype_bytes,
        _smem_limit,
        _smem_bytes,
    )

    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[dtype_s]
    device = torch.device("cuda")

    g = torch.Generator(device="cuda")
    g.manual_seed(0)
    x = torch.randn(1, N, D, device=device, dtype=dtype, generator=g)
    cents = torch.randn(1, K, D, device=device, dtype=dtype, generator=g)
    c_sq = (cents.float() ** 2).sum(-1)
    out = torch.empty((1, N), device=device, dtype=torch.int32)

    stride_x_b, stride_x_n, stride_x_d = x.stride()
    stride_c_b, stride_c_k, stride_c_d = cents.stride()
    stride_csq_b, stride_csq_k = c_sq.stride()
    stride_out_b, stride_out_n = out.stride()

    dtype_bytes = _dtype_bytes(dtype)
    smem = _smem_limit(device)

    best = None
    best_t = float("inf")
    for cfg in configs:
        BN = int(cfg["BLOCK_N"])
        BK = int(cfg["BLOCK_K"])
        W = int(cfg["num_warps"])
        S = int(cfg["num_stages"])
        if _smem_bytes(D, BN, BK, S, dtype_bytes) > smem:
            continue

        grid = (triton.cdiv(N, BN), 1)

        def fn():
            _euclid_assign_kernel[grid](
                x, cents, c_sq, out,
                1, N, K, D,
                stride_x_b, stride_x_n, stride_x_d,
                stride_c_b, stride_c_k, stride_c_d,
                stride_csq_b, stride_csq_k,
                stride_out_b, stride_out_n,
                BLOCK_N=BN, BLOCK_K=BK,
                num_warps=W, num_stages=S,
            )

        try:
            t = _bench_us(fn, repeats=repeats, warmup=warmup)
        except Exception as exc:
            if verbose:
                print(f"  skip BN={BN} BK={BK} W={W} S={S}: {str(exc).splitlines()[0][:80]}")
            continue

        if verbose:
            print(f"  BN={BN:>3} BK={BK:>3} W={W} S={S}: {t:>7.1f} us")
        if t < best_t:
            best_t = t
            best = {"BLOCK_N": BN, "BLOCK_K": BK, "num_warps": W, "num_stages": S}

    del x, cents, c_sq, out
    gc.collect()
    torch.cuda.empty_cache()
    return best, best_t


def _bench_one_splitD(N, K, D, dtype_s, repeats, warmup, configs, verbose=False):
    import torch
    import triton
    from flash_kmeans.assign_euclid_triton import (
        _euclid_assign_kernel_split_d,
        _dtype_bytes,
        _smem_limit,
        _smem_bytes_split_d,
    )

    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[dtype_s]
    device = torch.device("cuda")

    g = torch.Generator(device="cuda")
    g.manual_seed(0)
    x = torch.randn(1, N, D, device=device, dtype=dtype, generator=g)
    cents = torch.randn(1, K, D, device=device, dtype=dtype, generator=g)
    c_sq = (cents.float() ** 2).sum(-1)
    out = torch.empty((1, N), device=device, dtype=torch.int32)

    stride_x_b, stride_x_n, stride_x_d = x.stride()
    stride_c_b, stride_c_k, stride_c_d = cents.stride()
    stride_csq_b, stride_csq_k = c_sq.stride()
    stride_out_b, stride_out_n = out.stride()

    dtype_bytes = _dtype_bytes(dtype)
    smem = _smem_limit(device)

    best = None
    best_t = float("inf")
    for cfg in configs:
        BN = int(cfg["BLOCK_N"])
        BK = int(cfg["BLOCK_K"])
        BD = int(cfg["BLOCK_D"])
        W = int(cfg["num_warps"])
        S = int(cfg["num_stages"])
        if BD > D:
            continue
        if _smem_bytes_split_d(BD, BN, BK, S, dtype_bytes) > smem:
            continue

        grid = (triton.cdiv(N, BN), 1)

        def fn():
            _euclid_assign_kernel_split_d[grid](
                x, cents, c_sq, out,
                1, N, K, D,
                stride_x_b, stride_x_n, stride_x_d,
                stride_c_b, stride_c_k, stride_c_d,
                stride_csq_b, stride_csq_k,
                stride_out_b, stride_out_n,
                BLOCK_N=BN, BLOCK_K=BK, BLOCK_D=BD,
                num_warps=W, num_stages=S,
            )

        try:
            t = _bench_us(fn, repeats=repeats, warmup=warmup)
        except Exception as exc:
            if verbose:
                print(f"  skip BN={BN} BK={BK} BD={BD} W={W} S={S}: {str(exc).splitlines()[0][:80]}")
            continue

        if verbose:
            print(f"  BN={BN:>3} BK={BK:>3} BD={BD:>3} W={W} S={S}: {t:>7.1f} us")
        if t < best_t:
            best_t = t
            best = {"BLOCK_N": BN, "BLOCK_K": BK, "BLOCK_D": BD,
                    "num_warps": W, "num_stages": S}

    del x, cents, c_sq, out
    gc.collect()
    torch.cuda.empty_cache()
    return best, best_t


def run_worker(kind: str, cells: List[Tuple[int, int, int, str]],
               out_path: str, repeats: int, warmup: int, verbose: bool):
    """Worker entry. Iterates over assigned cells and writes results."""
    from flash_kmeans.assign_euclid_triton import (
        _TUNE_CONFIGS, _TUNE_CONFIGS_SPLIT_D,
    )

    smallD_cfgs = [c.kwargs | {"num_warps": c.num_warps, "num_stages": c.num_stages}
                   for c in _TUNE_CONFIGS]
    largeD_cfgs = [c.kwargs | {"num_warps": c.num_warps, "num_stages": c.num_stages}
                   for c in _TUNE_CONFIGS_SPLIT_D]

    results = []
    t0 = time.time()
    for i, (N, K, D, dt) in enumerate(cells):
        cell_start = time.time()
        if kind == "smallD":
            best, t_us = _bench_one_smallD(N, K, D, dt, repeats, warmup, smallD_cfgs,
                                            verbose=verbose)
        else:
            best, t_us = _bench_one_splitD(N, K, D, dt, repeats, warmup, largeD_cfgs,
                                            verbose=verbose)
        dt_cell = time.time() - cell_start
        elapsed = time.time() - t0
        print(f"[{i+1}/{len(cells)}] N={N} K={K} D={D} dt={dt}: "
              f"best={best} ({t_us:.1f} us)  cell_time={dt_cell:.1f}s  elapsed={elapsed:.1f}s",
              flush=True)
        results.append({
            "N": N, "K": K, "D": D, "dtype": dt,
            "best": best, "time_us": t_us,
        })
        # Incremental save (resilient to crashes).
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)


# ---------------------------------------------------------------------------
# Multi-GPU launcher
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["smallD", "largeD"], required=False)
    ap.add_argument("--gpu", type=int, default=None,
                    help="(Worker mode) GPU index inside CUDA_VISIBLE_DEVICES.")
    ap.add_argument("--shard", type=int, default=0,
                    help="(Worker mode) 0-based shard id of cells.")
    ap.add_argument("--num-shards", type=int, default=1,
                    help="(Worker mode) total number of shards.")
    ap.add_argument("--out", type=str, default=None,
                    help="(Worker mode) output JSON path.")
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--all", action="store_true",
                    help="Launcher: shard across GPUs 4..7 for both kinds, write merged JSONs.")
    ap.add_argument("--gpus", type=str, default="4,5,6,7",
                    help="(Launcher) comma-separated GPU IDs to use.")
    ap.add_argument("--outdir", type=str, default="scripts/tune_results",
                    help="(Launcher) directory for per-shard + merged JSONs.")
    args = ap.parse_args()

    if args.all:
        return launcher(args)

    # --- Worker mode ---
    if args.kind is None:
        ap.error("--kind required in worker mode")
    cells = list(smallD_cells() if args.kind == "smallD" else largeD_cells())
    # Shard by stride so each shard gets a mix of small/large cells.
    my = [c for i, c in enumerate(cells) if i % args.num_shards == args.shard]
    out_path = args.out or f"tune_{args.kind}_shard{args.shard}.json"
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    print(f"Worker shard={args.shard}/{args.num_shards} kind={args.kind} "
          f"cells={len(my)} (of {len(cells)})  out={out_path}")
    run_worker(args.kind, my, out_path, args.repeats, args.warmup, args.verbose)


def launcher(args):
    gpus = [int(g) for g in args.gpus.split(",")]
    os.makedirs(args.outdir, exist_ok=True)

    failed = []

    # Run kinds sequentially so two workers never share a GPU.
    for kind in ("smallD", "largeD"):
        procs = []
        for shard_idx, gpu in enumerate(gpus):
            out = os.path.join(args.outdir, f"tune_{kind}_g{gpu}.json")
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            cmd = [
                sys.executable, __file__,
                "--kind", kind,
                "--shard", str(shard_idx),
                "--num-shards", str(len(gpus)),
                "--out", out,
                "--repeats", str(args.repeats),
                "--warmup", str(args.warmup),
            ]
            print(f"Launching GPU {gpu} kind={kind} -> {out}", flush=True)
            p = subprocess.Popen(cmd, env=env)
            procs.append((kind, gpu, out, p))

        for kind_p, gpu, out, p in procs:
            rc = p.wait()
            if rc != 0:
                failed.append((kind_p, gpu, out, rc))
                print(f"  GPU {gpu} kind={kind_p} FAILED rc={rc}")

    if failed:
        print(f"WARN: {len(failed)} workers failed; merged JSON may be partial.")

    # Merge per-kind.
    for kind in ("smallD", "largeD"):
        merged = []
        for gpu in gpus:
            p = os.path.join(args.outdir, f"tune_{kind}_g{gpu}.json")
            if not os.path.exists(p):
                continue
            with open(p) as f:
                merged.extend(json.load(f))
        merged_path = os.path.join(args.outdir, f"tune_{kind}_h200.json")
        with open(merged_path, "w") as f:
            json.dump(merged, f, indent=2)
        print(f"Merged {len(merged)} cells -> {merged_path}")


if __name__ == "__main__":
    main()
