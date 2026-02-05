from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time


def _measure_worker(n: int, d: int, k: int, rounds: int, dtype: str, use_heuristic: bool):
    import torch
    from flash_kmeans import batch_kmeans_Euclid

    dtype_map = {"fp16": torch.float16, "fp32": torch.float32}
    torch_dtype = dtype_map[dtype]

    torch.manual_seed(0)
    B = 1
    x = torch.randn(B, n, d, device="cuda", dtype=torch_dtype)

    torch.cuda.synchronize()
    start_time = time.time()
    cluster_ids, centers, _ = batch_kmeans_Euclid(
        x,
        n_clusters=k,
        tol=0.0,
        init_centroids=None,
        verbose=False,
        max_iters=1,
        use_heuristic=use_heuristic,
    )
    torch.cuda.synchronize()
    compile_s = time.time() - start_time

    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)
    start_evt.record()
    for _ in range(rounds):
        cluster_ids, centers, _ = batch_kmeans_Euclid(
            x,
            n_clusters=k,
            tol=0.0,
            init_centroids=centers,
            verbose=False,
            max_iters=1,
            use_heuristic=use_heuristic,
        )
    end_evt.record()
    torch.cuda.synchronize()
    iter_ms = start_evt.elapsed_time(end_evt) / rounds

    mode = "heuristic" if use_heuristic else "autotune"
    result = {
        "mode": mode,
        "compile_s": compile_s,
        "iter_ms": iter_ms,
    }
    print(json.dumps(result))


def _run_subprocess(mode: str, args) -> dict:
    env = os.environ.copy()
    cache_root = os.path.join(tempfile.gettempdir(), "flash_kmeans_triton_cache")
    cache_dir = os.path.join(cache_root, mode)
    os.makedirs(cache_dir, exist_ok=True)
    env["TRITON_CACHE_DIR"] = cache_dir

    cmd = [
        sys.executable,
        os.path.abspath(__file__),
        "--worker",
        "--mode",
        mode,
        "--n",
        str(args.n),
        "--d",
        str(args.d),
        "--k",
        str(args.k),
        "--rounds",
        str(args.rounds),
        "--dtype",
        args.dtype,
    ]
    output = subprocess.check_output(cmd, env=env, text=True)
    for line in output.strip().splitlines()[::-1]:
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    raise RuntimeError(f"No JSON result in subprocess output:\n{output}")


def main():
    parser = argparse.ArgumentParser(description="Compare heuristic vs autotune compile/latency")
    parser.add_argument("--n", type=int, default=1000000)
    parser.add_argument("--d", type=int, default=512)
    parser.add_argument("--k", type=int, default=200000)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--dtype", choices=["fp16", "fp32"], default="fp16")
    parser.add_argument("--mode", choices=["heuristic", "autotune", "both"], default="both")
    parser.add_argument("--worker", action="store_true", help="Internal flag for subprocess runs")
    args = parser.parse_args()

    if args.worker:
        use_heuristic = args.mode == "heuristic"
        _measure_worker(args.n, args.d, args.k, args.rounds, args.dtype, use_heuristic)
        return

    modes = [args.mode] if args.mode != "both" else ["heuristic", "autotune"]
    results = []
    for mode in modes:
        results.append(_run_subprocess(mode, args))

    for r in results:
        print(
            f"[{r['mode']}] compile {r['compile_s']:.2f}s, "
            f"iter {r['iter_ms']:.3f} ms"
        )


if __name__ == "__main__":
    main()