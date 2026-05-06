"""Correctness sweep for non-power-of-2 D across the assign and centroid-update
kernels, both euclid and cosine paths.

The kernels in this repo materialise the feature dimension as a single tile in
several places, which made non-power-of-2 D crash with `tl.arange's range must
be a power of 2`. This file pins down the fixed behaviour by comparing the
Triton path to a torch fp32 reference on awkward D values.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from flash_kmeans import batch_kmeans_Euclid, batch_kmeans_Cosine
from flash_kmeans.assign_euclid_triton import (
    cosine_assign_triton,
    euclid_assign_triton,
    _need_split_d,
)
from flash_kmeans.centroid_update_triton import (
    triton_centroid_update_sorted_euclid,
    triton_centroid_update_sorted_cosine,
)


# Mix awkward (non-pow2) and pow2 controls.
D_LIST = [16, 17, 33, 96, 100, 192, 200, 256, 320, 384, 511, 513, 768, 1000, 1024, 1500, 2049]
DTYPES = [torch.float16, torch.bfloat16, torch.float32]


def _euclid_ref(x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    xf, cf = x.float(), c.float()
    return (
        (xf * xf).sum(-1, keepdim=True)
        + (cf * cf).sum(-1).unsqueeze(1)
        - 2.0 * torch.einsum("bnd,bkd->bnk", xf, cf)
    ).clamp_min_(0.0)


def _check_argmin(ref_dist, tri_ids, dtype, label):
    ref_ids = ref_dist.argmin(dim=-1).int()
    same = (tri_ids == ref_ids)
    if (~same).any():
        d_ref = ref_dist.gather(-1, ref_ids.long().unsqueeze(-1)).squeeze(-1)
        d_tri = ref_dist.gather(-1, tri_ids.long().unsqueeze(-1)).squeeze(-1)
        rel = (d_tri - d_ref).abs() / (d_ref.abs() + 1e-3)
        tol = 2e-2 if dtype in (torch.float16, torch.bfloat16) else 5e-3
        bad = (~same) & (rel > tol)
        nbad = int(bad.sum())
        if nbad > 0:
            raise AssertionError(
                f"{label}: {nbad} bad picks (worst rel diff {rel[bad].max().item():.4g}, tol {tol})"
            )
    pct = same.float().mean().item() * 100
    print(f"  {label}: argmin match={pct:.2f}%")


def _check_argmax(ref_score, tri_ids, dtype, label):
    ref_ids = ref_score.argmax(dim=-1).int()
    same = (tri_ids == ref_ids)
    if (~same).any():
        s_ref = ref_score.gather(-1, ref_ids.long().unsqueeze(-1)).squeeze(-1)
        s_tri = ref_score.gather(-1, tri_ids.long().unsqueeze(-1)).squeeze(-1)
        gap = (s_ref - s_tri).abs()
        tol = 2e-2 if dtype in (torch.float16, torch.bfloat16) else 5e-3
        bad = (~same) & (gap > tol)
        nbad = int(bad.sum())
        if nbad > 0:
            raise AssertionError(
                f"{label}: {nbad} bad picks (worst score gap {gap[bad].max().item():.4g}, tol {tol})"
            )
    pct = same.float().mean().item() * 100
    print(f"  {label}: argmax match={pct:.2f}%")


def _check_centroid_update_euclid(x, cluster_ids, old_cents, label):
    """Compare sorted-Triton centroid update to a torch reference (fp32 sums)."""
    B, N, D = x.shape
    K = old_cents.shape[1]
    tri = triton_centroid_update_sorted_euclid(x, cluster_ids.to(torch.int64), old_cents)

    xf = x.float()
    sums = torch.zeros((B, K, D), device=x.device, dtype=torch.float32)
    cnts = torch.zeros((B, K), device=x.device, dtype=torch.float32)
    sums.scatter_add_(1, cluster_ids.long().unsqueeze(-1).expand(-1, -1, D), xf)
    cnts.scatter_add_(1, cluster_ids.long(), torch.ones_like(cluster_ids, dtype=torch.float32))
    # Capture empty mask BEFORE the clamp (clamp_min_ mutates cnts in place).
    empty = (cnts == 0).unsqueeze(-1)
    means = sums / cnts.clamp(min=1.0).unsqueeze(-1)
    ref = torch.where(empty, old_cents.float(), means).to(x.dtype)

    diff = (tri.float() - ref.float()).abs()
    tol = 5e-3 if x.dtype in (torch.float16, torch.bfloat16) else 1e-4
    max_diff = diff.max().item()
    if max_diff > tol:
        # Save the failing inputs for offline debug.
        torch.save(
            {
                "x": x.cpu(),
                "cluster_ids": cluster_ids.cpu(),
                "old_cents": old_cents.cpu(),
                "tri": tri.cpu(),
                "ref": ref.cpu(),
            },
            f"/tmp/cupd_fail_D{D}_{x.dtype}.pt",
        )
        raise AssertionError(f"{label}: centroid max diff {max_diff:.4g} > tol {tol}")
    print(f"  {label}: centroid max diff={max_diff:.4g}")


def main():
    torch.manual_seed(0)
    device = torch.device("cuda")

    for dtype in DTYPES:
        for D in D_LIST:
            B, N, K = 1, 8193, 1031
            x = torch.randn(B, N, D, device=device, dtype=dtype)
            c = torch.randn(B, K, D, device=device, dtype=dtype)
            x_sq = (x.float() ** 2).sum(-1)

            # ---- Euclid assign ----
            ids = euclid_assign_triton(x, c, x_sq)
            split = _need_split_d(D, dtype, x.device)
            label = f"euclid {str(dtype).replace('torch.', ''):>9} D={D:>4} split_d={split}"
            ref_dist = _euclid_ref(x, c)
            _check_argmin(ref_dist, ids, dtype, label)

            # ---- Cosine assign ----
            xn = F.normalize(x, dim=-1)
            cn = F.normalize(c, dim=-1)
            ref_score = torch.einsum("bnd,bkd->bnk", xn.float(), cn.float())
            cids = cosine_assign_triton(xn, cn)
            label = f"cosine {str(dtype).replace('torch.', ''):>9} D={D:>4} split_d={split}"
            _check_argmax(ref_score, cids, dtype, label)

            # ---- Centroid update (sorted, euclid) ----
            cluster_ids = torch.randint(0, K, (B, N), device=device, dtype=torch.int32)
            old_cents = torch.randn(B, K, D, device=device, dtype=dtype)
            label = f"cupd-eu {str(dtype).replace('torch.', ''):>9} D={D:>4}"
            _check_centroid_update_euclid(x, cluster_ids, old_cents, label)

    # ---- End-to-end: batch_kmeans_Euclid for a few non-pow2 D values ----
    print("\nEnd-to-end batch_kmeans_Euclid sanity:")
    for D in [100, 192, 384, 768]:
        x = torch.randn(1, 4096, D, device=device, dtype=torch.float16)
        cluster_ids, centroids, n_iters = batch_kmeans_Euclid(
            x, n_clusters=64, max_iters=5, tol=0.0
        )
        assert cluster_ids.shape == (1, 4096)
        assert centroids.shape == (1, 64, D)
        print(f"  D={D:4d}: cluster_ids ok, centroids ok, n_iters={n_iters}")

    print("\nALL OK")


if __name__ == "__main__":
    main()
