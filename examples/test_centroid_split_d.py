"""Correctness + performance harness for the sorted centroid-update kernels.

Runs a dtype x D sweep against an fp32-accumulated torch reference (which
matches the Triton kernel's fp32 accumulation semantics), plus a small perf
suite on the shapes that matter for the documented benchmarks.

Usage:
    python examples/test_centroid_split_d.py            # correctness + perf
    python examples/test_centroid_split_d.py --tag old  # label the run
"""

from __future__ import annotations

import argparse
import torch
import torch.nn.functional as F

from flash_kmeans.centroid_update_triton import (
    triton_centroid_update_sorted_euclid,
    triton_centroid_update_sorted_cosine,
)

try:
    from flash_kmeans.centroid_update_triton import _choose_block_d, _dtype_bytes, _pad_d
    _HAS_HELPERS = True
except Exception:
    _HAS_HELPERS = False


DTYPES = [torch.float16, torch.bfloat16, torch.float32, torch.float64]
D_LIST = [64, 128, 192, 256, 768, 1024, 1536, 2048]


def _ref_euclid(x, cluster_ids, old_centroids, K):
    """fp32-accumulated reference matching kernel semantics."""
    B, N, D = x.shape
    xf = x.float()
    sums = torch.zeros((B, K, D), device=x.device, dtype=torch.float32)
    cnts = torch.zeros((B, K), device=x.device, dtype=torch.float32)
    idx = cluster_ids.long()
    for b in range(B):
        sums[b].index_add_(0, idx[b], xf[b])
        cnts[b].index_add_(0, idx[b], torch.ones(N, device=x.device))
    means = sums / cnts.clamp(min=1.0).unsqueeze(-1)
    empty = (cnts == 0).unsqueeze(-1)
    means = torch.where(empty, old_centroids.float(), means)
    return means


def _ref_cosine(x, cluster_ids, old_centroids, K):
    means = _ref_euclid(x, cluster_ids, old_centroids, K)
    return F.normalize(means, p=2, dim=-1)


def _rand_ids(B, N, K, device):
    # Mix of balanced + skewed assignments to exercise variable run lengths.
    ids = torch.randint(0, K, (B, N), device=device, dtype=torch.int32)
    return ids


def correctness():
    torch.manual_seed(0)
    device = "cuda"
    B, N, K = 2, 9001, 257
    print(f"\n=== Correctness (B={B}, N={N}, K={K}) ===")
    worst = 0.0
    n_fail = 0
    for dtype in DTYPES:
        for D in D_LIST:
            x = torch.randn(B, N, D, device=device, dtype=dtype)
            ids = _rand_ids(B, N, K, device)
            old = torch.randn(B, K, D, device=device, dtype=dtype)

            bd = _choose_block_d(D, 256, _dtype_bytes(dtype)) if _HAS_HELPERS else _pad_d(D) if _HAS_HELPERS else -1

            # Euclid
            try:
                out = triton_centroid_update_sorted_euclid(x, ids, old)
                ref = _ref_euclid(x, ids, old, K)
                # Compare where reference is finite / non-empty-handled
                diff = (out.float() - ref).abs()
                denom = ref.abs().clamp(min=1e-2)
                rel = (diff / denom).max().item()
                tol = 3e-2 if dtype in (torch.float16, torch.bfloat16) else 2e-3
                ok = rel <= tol
                worst = max(worst, rel if rel == rel else 0.0)
                status = "OK " if ok else "FAIL"
                if not ok:
                    n_fail += 1
                print(f"  euclid {str(dtype).replace('torch.',''):>9} D={D:>4} BLOCK_D={bd:>4} rel={rel:.2e} tol={tol:.0e} {status}")
            except Exception as e:
                n_fail += 1
                print(f"  euclid {str(dtype).replace('torch.',''):>9} D={D:>4} BLOCK_D={bd:>4} EXCEPTION: {type(e).__name__}: {str(e)[:80]}")

            # Cosine
            try:
                outc = triton_centroid_update_sorted_cosine(x, ids, old)
                refc = _ref_cosine(x, ids, old, K)
                diffc = (outc.float() - refc).abs().max().item()
                tolc = 3e-2 if dtype in (torch.float16, torch.bfloat16) else 3e-3
                okc = diffc <= tolc
                if not okc:
                    n_fail += 1
                status = "OK " if okc else "FAIL"
                print(f"  cosine {str(dtype).replace('torch.',''):>9} D={D:>4} BLOCK_D={bd:>4} absmax={diffc:.2e} tol={tolc:.0e} {status}")
            except Exception as e:
                n_fail += 1
                print(f"  cosine {str(dtype).replace('torch.',''):>9} D={D:>4} BLOCK_D={bd:>4} EXCEPTION: {type(e).__name__}: {str(e)[:80]}")

    print(f"\n  worst rel(euclid)={worst:.2e}  failures={n_fail}")
    return n_fail


def _time(fn, warmup=5, iters=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def perf(tag):
    torch.manual_seed(0)
    device = "cuda"
    print(f"\n=== Perf [{tag}] (ms/call, lower is better) ===")
    configs = [
        # (B, N, D, K, dtype)  -- main documented benchmark shape
        (32, 74256, 128, 1000, torch.float16),
        (32, 74256, 128, 1000, torch.float32),
        # large-N single batch (kmeans_largeN style chunk)
        (1, 1_048_576, 128, 8192, torch.float32),
        # large-D cases (the issue #19 regime)
        (1, 262144, 1536, 4096, torch.float32),
        (1, 262144, 1536, 4096, torch.float64),
    ]
    for (B, N, D, K, dtype) in configs:
        try:
            x = torch.randn(B, N, D, device=device, dtype=dtype)
            ids = _rand_ids(B, N, K, device)
            old = torch.randn(B, K, D, device=device, dtype=dtype)
            bd = _choose_block_d(D, 256, _dtype_bytes(dtype)) if _HAS_HELPERS else -1
            t = _time(lambda: triton_centroid_update_sorted_euclid(x, ids, old))
            print(f"  euclid B={B} N={N} D={D:>4} K={K} {str(dtype).replace('torch.',''):>9} BLOCK_D={bd:>4}: {t:.3f} ms")
        except Exception as ex:
            print(f"  euclid B={B} N={N} D={D:>4} K={K} {str(dtype).replace('torch.',''):>9}: EXCEPTION {type(ex).__name__}: {str(ex)[:80]}")
        finally:
            del x, ids, old
            torch.cuda.empty_cache()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="run")
    ap.add_argument("--skip-perf", action="store_true")
    ap.add_argument("--skip-correctness", action="store_true")
    args = ap.parse_args()

    nf = 0
    if not args.skip_correctness:
        nf = correctness()
    if not args.skip_perf:
        perf(args.tag)
    print(f"\nDONE tag={args.tag} failures={nf}")
