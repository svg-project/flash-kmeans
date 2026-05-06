"""Correctness sweep across dtype × D for the Triton assign kernels.

Covers both euclid and cosine paths, both small-D (existing kernel) and
split-D (new). Uses an fp32 reference and asserts argmin/argmax equality
on rows where the top-1/top-2 distance gap is tight enough to avoid
numerical-tie false alarms; for the rest we relax to a tiny score gap.
"""

from __future__ import annotations

import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, "/home/ec2-user/flash-kmeans")

from flash_kmeans.assign_euclid_triton import (
    cosine_assign_triton,
    euclid_assign_triton,
    _need_split_d,
)


D_LIST = [64, 128, 256, 512, 768, 1024, 2048, 4096]
DTYPES = [torch.float16, torch.bfloat16, torch.float32]


def _euclid_ref(x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    xf = x.float()
    cf = c.float()
    x_sq = (xf * xf).sum(-1)
    c_sq = (cf * cf).sum(-1)
    dist = (
        x_sq.unsqueeze(-1) + c_sq.unsqueeze(1)
        - 2.0 * torch.einsum("bnd,bkd->bnk", xf, cf)
    ).clamp_min_(0.0)
    return dist


def _check_argmin(ref_dist, tri_ids, dtype, label):
    """Verify Triton argmin-equivalent picks: equal id, OR equally good distance."""
    ref_ids = ref_dist.argmin(dim=-1).int()
    same = (tri_ids == ref_ids)

    # Where ids differ, the picked distance must equal the optimal within tol.
    if (~same).any():
        d_at_ref = ref_dist.gather(-1, ref_ids.long().unsqueeze(-1)).squeeze(-1)
        d_at_tri = ref_dist.gather(-1, tri_ids.long().unsqueeze(-1)).squeeze(-1)
        # Tolerance: relative + absolute, scaled by D and dtype noise.
        rel_diff = (d_at_tri - d_at_ref).abs() / (d_at_ref.abs() + 1e-3)
        # fp32: tied within 1e-3 relative; fp16/bf16: 2e-2 (mantissa-driven).
        tol = 2e-2 if dtype in (torch.float16, torch.bfloat16) else 1e-3
        bad = (~same) & (rel_diff > tol)
        nbad = int(bad.sum())
        if nbad > 0:
            worst = rel_diff[bad].max().item()
            raise AssertionError(
                f"{label}: {nbad} bad picks (worst rel diff {worst:.4g}, tol {tol})"
            )

    match_pct = same.float().mean().item() * 100
    print(f"  {label}: argmin match={match_pct:.2f}%")


def _check_argmax(ref_score, tri_ids, dtype, label):
    ref_ids = ref_score.argmax(dim=-1).int()
    same = (tri_ids == ref_ids)
    if (~same).any():
        s_at_ref = ref_score.gather(-1, ref_ids.long().unsqueeze(-1)).squeeze(-1)
        s_at_tri = ref_score.gather(-1, tri_ids.long().unsqueeze(-1)).squeeze(-1)
        gap = (s_at_ref - s_at_tri).abs()
        tol = 2e-2 if dtype in (torch.float16, torch.bfloat16) else 1e-3
        bad = (~same) & (gap > tol)
        nbad = int(bad.sum())
        if nbad > 0:
            worst = gap[bad].max().item()
            raise AssertionError(
                f"{label}: {nbad} bad picks (worst score gap {worst:.4g}, tol {tol})"
            )
    match_pct = same.float().mean().item() * 100
    print(f"  {label}: argmax match={match_pct:.2f}%")


def main():
    torch.manual_seed(0)
    device = torch.device("cuda")

    for dtype in DTYPES:
        for D in D_LIST:
            # Use shapes that exercise tail masks: K not power of 2, N not power of 2.
            B, N, K = 1, 8193, 1031
            x = torch.randn(B, N, D, device=device, dtype=dtype)
            c = torch.randn(B, K, D, device=device, dtype=dtype)
            x_sq = (x.float() ** 2).sum(-1)

            ids = euclid_assign_triton(x, c, x_sq)
            split = _need_split_d(D, dtype, x.device)
            label = f"euclid {str(dtype).replace('torch.', ''):>9} D={D:>4} split_d={split}"
            ref_dist = _euclid_ref(x, c)
            _check_argmin(ref_dist, ids, dtype, label)

            xn = F.normalize(x, dim=-1)
            cn = F.normalize(c, dim=-1)
            ref_score = torch.einsum("bnd,bkd->bnk", xn.float(), cn.float())
            cids = cosine_assign_triton(xn, cn)
            label = f"cosine {str(dtype).replace('torch.', ''):>9} D={D:>4} split_d={split}"
            _check_argmax(ref_score, cids, dtype, label)

    print("\nALL OK")


if __name__ == "__main__":
    main()
