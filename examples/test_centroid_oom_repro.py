"""Reproduce issue #19: fp64 large-D centroid update OOM at kernel launch.

The chunk kernel reserves per-launch local-memory scratch proportional to its
per-program feature tile. The old kernel loaded a full [BLOCK_N, next_pow2(D)]
tile, which for D=1536 fp64 spills heavily; when little free GPU memory remains
(as on a node already holding a large dataset), cuLaunchKernel fails with
CUDA_ERROR_OUT_OF_MEMORY. The split-D kernel bounds the tile to
[BLOCK_N, BLOCK_D], shrinking the scratch by ~16x so the launch succeeds.

Run under memory pressure (leave only `--free-gb` free):
    python examples/test_centroid_oom_repro.py --free-gb 6
"""

from __future__ import annotations

import argparse
import torch

from flash_kmeans.centroid_update_triton import triton_centroid_update_sorted_euclid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--free-gb", type=float, default=6.0,
                    help="GiB of free GPU memory to leave before the launch")
    ap.add_argument("--D", type=int, default=1536)
    ap.add_argument("--N", type=int, default=262144)
    ap.add_argument("--K", type=int, default=4096)
    args = ap.parse_args()

    device = "cuda"
    dtype = torch.float64

    x = torch.randn(1, args.N, args.D, device=device, dtype=dtype)
    ids = torch.randint(0, args.K, (1, args.N), device=device, dtype=torch.int32)
    old = torch.randn(1, args.K, args.D, device=device, dtype=dtype)

    # Squeeze free memory down to `--free-gb` with a filler allocation to
    # emulate a GPU already holding a large working set.
    free, total = torch.cuda.mem_get_info()
    target_free = int(args.free_gb * (1024 ** 3))
    filler = None
    if free > target_free:
        nbytes = free - target_free
        filler = torch.empty(nbytes // 4, device=device, dtype=torch.float32)
    free_after, _ = torch.cuda.mem_get_info()
    print(f"total={total/1e9:.1f}GB  free_before={free/1e9:.1f}GB  "
          f"free_after_filler={free_after/1e9:.1f}GB")

    try:
        out = triton_centroid_update_sorted_euclid(x, ids, old)
        torch.cuda.synchronize()
        print(f"LAUNCH OK  -> centroids {tuple(out.shape)} {out.dtype}")
    except Exception as e:
        print(f"LAUNCH FAILED -> {type(e).__name__}: {str(e)[:120]}")
    finally:
        del filler


if __name__ == "__main__":
    main()
