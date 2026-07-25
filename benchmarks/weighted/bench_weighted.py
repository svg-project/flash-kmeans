#!/usr/bin/env python
"""Correctness + throughput for the new weighted centroid-update kernel.

Correctness is checked three ways against a pure-torch reference:
  1. random positive weights           -> matches torch weighted mean per cluster
  2. all-ones weights                   -> matches the *unweighted* triton kernel
  3. an empty cluster                   -> falls back to old_centroids for that k

Throughput reports the weighted kernel latency next to the unweighted kernel so
the extra cost of carrying per-point weights is visible.

    python benchmarks/weighted/bench_weighted.py
"""
import torch

from flash_kmeans.centroid_update_triton import (
    triton_centroid_update_sorted_euclid,
    triton_centroid_update_sorted_euclid_weighted,
)

DTYPE = torch.float16
WARMUP = 5
ITERS = 20


def torch_weighted_reference(x, cluster_ids, old_centroids, weights):
    """Per-cluster weighted mean in fp32; empty clusters keep old_centroids."""
    B, N, D = x.shape
    K = old_centroids.shape[1]
    xf = x.float()
    wf = weights.float()
    out = torch.empty((B, K, D), device=x.device, dtype=torch.float32)
    for b in range(B):
        for k in range(K):
            mask = cluster_ids[b] == k
            wsum = wf[b][mask].sum()
            if wsum <= 0:
                out[b, k] = old_centroids[b, k].float()
            else:
                out[b, k] = (xf[b][mask] * wf[b][mask, None]).sum(0) / wsum
    return out.to(x.dtype)


def _time_ms(fn):
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(ITERS):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / ITERS


def check(name, ok, extra=""):
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}] {name} {extra}")
    return ok


def _correctness_for_shape(B, N, K, D, dtype, dev):
    """Run the three correctness checks for one (shape, dtype) and return ok."""
    torch.manual_seed(0)
    all_ok = True
    tag = f"D={D:<4} {str(dtype).split('.')[-1]:>7}"

    # 1) random positive weights vs torch reference
    x = torch.randn(B, N, D, device=dev, dtype=dtype)
    cluster_ids = torch.randint(0, K, (B, N), device=dev, dtype=torch.int64)
    old_c = torch.randn(B, K, D, device=dev, dtype=dtype)
    weights = torch.rand(B, N, device=dev) + 0.1  # positive

    got = triton_centroid_update_sorted_euclid_weighted(x, cluster_ids, old_c, weights)
    ref = torch_weighted_reference(x, cluster_ids, old_c, weights)
    # tolerance scaled by dtype (fp16 centroids quantize coarsely)
    tol = 5e-2 if dtype == torch.float16 else 1e-4
    max_abs = (got.float() - ref.float()).abs().max().item()
    all_ok &= check(f"{tag} | random weights vs torch reference",
                    max_abs < tol, f"(max abs diff = {max_abs:.2e})")

    # 2) all-ones weights == unweighted kernel
    ones = torch.ones(B, N, device=dev)
    got_w = triton_centroid_update_sorted_euclid_weighted(x, cluster_ids, old_c, ones)
    got_u = triton_centroid_update_sorted_euclid(x, cluster_ids, old_c)
    max_abs2 = (got_w.float() - got_u.float()).abs().max().item()
    all_ok &= check(f"{tag} | all-ones weights == unweighted kernel",
                    max_abs2 < tol, f"(max abs diff = {max_abs2:.2e})")

    # 3) empty cluster falls back to old_centroids
    cluster_ids2 = cluster_ids.clone()
    cluster_ids2[cluster_ids2 == 0] = 1  # cluster 0 now empty in every batch
    got_e = triton_centroid_update_sorted_euclid_weighted(x, cluster_ids2, old_c, weights)
    fell_back = torch.allclose(got_e[:, 0].float(), old_c[:, 0].float(), atol=1e-3)
    all_ok &= check(f"{tag} | empty cluster -> old_centroids fallback", fell_back)
    return all_ok


def correctness():
    print("Correctness (D includes non-power-of-two + large-D/fp32):")
    dev = "cuda"
    # (B, N, K, D, dtype) — cover pow2, non-pow2, and large-D/fp32 (issue #19).
    cases = [
        (4, 20_000, 64, 128, torch.float16),   # pow2 baseline
        (4, 20_000, 64, 64,  torch.float16),   # pow2 small
        (4, 20_000, 64, 96,  torch.float16),   # non-pow2 (would crash pre-fix)
        (4, 20_000, 64, 192, torch.float16),   # non-pow2
        (2, 20_000, 64, 80,  torch.float16),   # non-pow2 head dim
        (2, 20_000, 64, 1024, torch.float32),  # large D + fp32 (untiled would spill)
    ]
    all_ok = True
    for B, N, K, D, dtype in cases:
        all_ok &= _correctness_for_shape(B, N, K, D, dtype, dev)
    print(f"\n  => {'ALL PASSED' if all_ok else 'FAILURES PRESENT'}\n")
    return all_ok


def throughput():
    print("Throughput (weighted vs unweighted centroid update):")
    dev = "cuda"
    shapes = [
        (1, 1_000_000, 256, 128),
        (8, 131_072, 256, 128),
        (4, 262_144, 512, 256),
        (4, 262_144, 512, 96),   # non-pow2
        (2, 262_144, 512, 192),  # non-pow2
    ]
    header = f"{'shape (B,N,K,D)':>26} | {'unweighted':>18} | {'weighted':>18} | {'overhead':>9}"
    print(header)
    print("-" * len(header))
    for B, N, K, D in shapes:
        x = torch.randn(B, N, D, device=dev, dtype=DTYPE)
        cluster_ids = torch.randint(0, K, (B, N), device=dev, dtype=torch.int64)
        old_c = torch.randn(B, K, D, device=dev, dtype=DTYPE)
        weights = torch.rand(B, N, device=dev) + 0.1

        t_u = _time_ms(lambda: triton_centroid_update_sorted_euclid(x, cluster_ids, old_c))
        t_w = _time_ms(lambda: triton_centroid_update_sorted_euclid_weighted(x, cluster_ids, old_c, weights))
        mpts_u = (B * N) / (t_u * 1e-3) / 1e6
        mpts_w = (B * N) / (t_w * 1e-3) / 1e6
        print(
            f"{str((B,N,K,D)):>26} | "
            f"{t_u:>7.3f} ms {mpts_u:>5.0f}M/s | "
            f"{t_w:>7.3f} ms {mpts_w:>5.0f}M/s | "
            f"{(t_w/t_u - 1)*100:>7.1f}%"
        )
        del x, cluster_ids, old_c, weights
        torch.cuda.empty_cache()


def main():
    print(f"GPU: {torch.cuda.get_device_name(0)}\n")
    ok = correctness()
    print()
    throughput()
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
