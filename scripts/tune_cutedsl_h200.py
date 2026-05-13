"""Autotune harness for the no-xsq CuteDSL FA3 Euclid assign kernel (H200).

For each (N, K, D, dtype) cell we let ``cutedsl_impl._autotune_pick``
sweep every fitting ``(BM, BN, use_ws)`` combination (12 candidates max,
filtered to those that fit SMEM), pick the winner, and we emit it to
a JSON file. Cells are sharded across GPUs 4..7 via subprocesses.

Run as::

    python scripts/tune_cutedsl_h200.py --all --gpus 4,5,6,7
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

# Defer torch / cutlass import to subprocess workers.

# ---------------------------------------------------------------------------
# Shape grid — same as Triton small-D (the CuteDSL kernel is small-D only).
# ---------------------------------------------------------------------------

DS = [64, 128, 256, 512]
KS = [256, 4096, 65536]
NS = [65536, 1048576]
DTYPES = ["bf16", "fp16"]


def cells():
    for D, K, N, dt in itertools.product(DS, KS, NS, DTYPES):
        yield (N, K, D, dt)


def _bench_one(N, K, D, dtype_s, repeats, warmup):
    import torch
    from flash_kmeans.cutedsl_impl import (
        _autotune_pick, _try_init_cutedsl, _kernel_cache, _dlpack_cache,
    )

    if not _try_init_cutedsl():
        raise RuntimeError("cutedsl init failed")

    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[dtype_s]
    device = torch.device("cuda")

    g = torch.Generator(device="cuda")
    g.manual_seed(0)
    x = torch.randn(N, D, device=device, dtype=dtype, generator=g).contiguous()
    cents = torch.randn(K, D, device=device, dtype=dtype, generator=g).contiguous()
    c_sq = (cents.float() ** 2).sum(-1).contiguous()
    out = torch.empty(N, device=device, dtype=torch.int32)

    # Run the autotune sweep. _autotune_pick handles per-config compile +
    # bench and returns the winner.
    BM, BN, use_ws, t_us = _autotune_pick(
        x, cents, c_sq, out,
        warmup=warmup, repeats=repeats, verbose=False,
    )

    # Clear caches so they don't accumulate across cells (different shape =
    # different compiled handle, but the dlpack cache keys on data_ptr too).
    _kernel_cache.clear()
    _dlpack_cache.clear()
    del x, cents, c_sq, out
    gc.collect()
    torch.cuda.empty_cache()

    return {"BM": int(BM), "BN": int(BN), "use_ws": bool(use_ws)}, float(t_us)


def run_worker(cells_list, out_path, repeats, warmup):
    results = []
    t0 = time.time()
    for i, (N, K, D, dt) in enumerate(cells_list):
        cell_start = time.time()
        try:
            best, t_us = _bench_one(N, K, D, dt, repeats, warmup)
        except Exception as exc:
            print(f"[{i+1}/{len(cells_list)}] N={N} K={K} D={D} dt={dt}: FAILED ({exc!r})",
                  flush=True)
            results.append({"N": N, "K": K, "D": D, "dtype": dt,
                            "best": None, "time_us": None, "error": str(exc)})
            with open(out_path, "w") as f:
                json.dump(results, f, indent=2)
            continue
        dt_cell = time.time() - cell_start
        elapsed = time.time() - t0
        print(f"[{i+1}/{len(cells_list)}] N={N} K={K} D={D} dt={dt}: "
              f"best={best} ({t_us:.1f} us)  cell_time={dt_cell:.1f}s  elapsed={elapsed:.1f}s",
              flush=True)
        results.append({
            "N": N, "K": K, "D": D, "dtype": dt,
            "best": best, "time_us": t_us,
        })
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--gpus", type=str, default="4,5,6,7")
    ap.add_argument("--outdir", type=str, default="scripts/tune_results")
    args = ap.parse_args()

    if args.all:
        gpus = [int(g) for g in args.gpus.split(",")]
        os.makedirs(args.outdir, exist_ok=True)

        procs = []
        for shard_idx, gpu in enumerate(gpus):
            out = os.path.join(args.outdir, f"tune_cutedsl_g{gpu}.json")
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            cmd = [
                sys.executable, __file__,
                "--shard", str(shard_idx),
                "--num-shards", str(len(gpus)),
                "--out", out,
                "--repeats", str(args.repeats),
                "--warmup", str(args.warmup),
            ]
            print(f"Launching GPU {gpu} -> {out}", flush=True)
            procs.append((gpu, out, subprocess.Popen(cmd, env=env)))

        failed = []
        for gpu, out, p in procs:
            rc = p.wait()
            if rc != 0:
                failed.append((gpu, out, rc))
                print(f"  GPU {gpu} FAILED rc={rc}")
        if failed:
            print(f"WARN: {len(failed)} workers failed.")

        merged = []
        for gpu in gpus:
            p = os.path.join(args.outdir, f"tune_cutedsl_g{gpu}.json")
            if not os.path.exists(p):
                continue
            with open(p) as f:
                merged.extend(json.load(f))
        merged_path = os.path.join(args.outdir, "tune_cutedsl_h200.json")
        with open(merged_path, "w") as f:
            json.dump(merged, f, indent=2)
        print(f"Merged {len(merged)} cells -> {merged_path}")
        return

    all_cells = list(cells())
    my = [c for i, c in enumerate(all_cells) if i % args.num_shards == args.shard]
    out_path = args.out or f"tune_cutedsl_shard{args.shard}.json"
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    print(f"Worker shard={args.shard}/{args.num_shards} cells={len(my)} (of {len(all_cells)})  out={out_path}",
          flush=True)
    run_worker(my, out_path, args.repeats, args.warmup)


if __name__ == "__main__":
    main()
