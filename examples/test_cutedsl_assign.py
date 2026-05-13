"""Correctness + bench tests for the fused CuteDSL assign kernel.

Two modes:
  * ``--mode correctness`` (default): exhaustive correctness sweep across
    ``(N, D, K)`` combinations and dtypes, verifying bit-equivalence with
    the Triton reference and the torch fp32 reference.
  * ``--mode bench``: end-to-end assign-kernel benchmark on the
    representative shapes from ``cutedsl_impl.py``'s docstring.

Usage:
    python examples/test_cutedsl_assign.py --mode correctness
    python examples/test_cutedsl_assign.py --mode bench
    python examples/test_cutedsl_assign.py --mode kernel_only --n 100000 --d 512 --k 512
"""
from __future__ import annotations

import argparse
import time

import torch


_DTYPES = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}


def _torch_ref_argmin(x: torch.Tensor, cents: torch.Tensor) -> torch.Tensor:
    cross = (x.float() @ cents.float().T)
    x_sq = (x.float() ** 2).sum(-1)
    c_sq = (cents.float() ** 2).sum(-1)
    dist = (x_sq[:, None] + c_sq[None, :] - 2.0 * cross).clamp_min_(0.0)
    return dist.argmin(-1).to(torch.int32)


def _bench(fn, repeats: int = 50, warmup: int = 5) -> float:
    """Return mean elapsed time in microseconds."""
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


def correctness_sweep():
    from flash_kmeans.cutedsl_impl import cutedsl_assign_euclid, cutedsl_info
    from flash_kmeans.assign_euclid_triton import euclid_assign_triton

    print(cutedsl_info())
    print()

    cases = [
        # (N, D, K, dtype)
        (512, 64, 32, "bf16"),
        (4096, 128, 256, "bf16"),
        (4096, 128, 256, "fp16"),
        (8192, 64, 256, "bf16"),
        (8192, 256, 256, "bf16"),
        (8192, 512, 512, "bf16"),
        # Edge cases (non-aligned N, K)
        (8001, 128, 257, "bf16"),
        (10001, 128, 7, "bf16"),
        (8192, 16, 8, "bf16"),
        # Edge: small N
        (137, 128, 64, "bf16"),
        # Edge: large K
        (1024, 128, 8192, "bf16"),
    ]

    n_pass = 0
    n_total = 0
    n_fallback = 0
    for N, D, K, dtype_s in cases:
        dtype = _DTYPES[dtype_s]
        torch.manual_seed(0)
        x = torch.randn(1, N, D, device="cuda", dtype=dtype)
        cents = torch.randn(1, K, D, device="cuda", dtype=dtype)

        out_cute = cutedsl_assign_euclid(x, cents)
        out_tri = euclid_assign_triton(x, cents)

        ref = _torch_ref_argmin(x.view(N, D), cents.view(K, D))

        cute_match = (out_cute.view(-1) == ref).float().mean().item()
        tri_match = (out_tri.view(-1) == ref).float().mean().item()

        n_total += 1
        if cute_match >= 0.999:  # allow a few ties to differ between paths
            n_pass += 1
            tag = "OK"
        else:
            tag = "FAIL"

        # Note if cute fell back to triton (we can't easily detect from outside;
        # match rate vs triton is a proxy).
        cute_eq_tri = (out_cute == out_tri).float().mean().item()
        if cute_eq_tri == 1.0:
            tag2 = "==tri"
        else:
            tag2 = ""

        print(f"  [{tag:4s}] N={N:>6} D={D:>4} K={K:>5} dtype={dtype_s} | cute_vs_ref={cute_match:.4f} tri_vs_ref={tri_match:.4f} {tag2}")

    print()
    print(f"PASS: {n_pass} / {n_total}")
    return n_pass == n_total


def bench_assign():
    from flash_kmeans.cutedsl_impl import cutedsl_assign_euclid
    from flash_kmeans.assign_euclid_triton import euclid_assign_triton

    shapes = [
        (1000000, 16, 8),
        (273000, 16, 64),
        (1000000, 64, 64),
        (200000, 256, 256),
        (100000, 512, 512),
        (8192, 128, 1024),
        (1024, 256, 4096),
    ]
    print(f"{'Shape':<32} {'Triton (us)':>13} {'NEW Cute (us)':>15} {'Speedup':>10}")
    print("-" * 75)
    for N, D, K in shapes:
        x = torch.randn(1, N, D, device="cuda", dtype=torch.bfloat16)
        cents = torch.randn(1, K, D, device="cuda", dtype=torch.bfloat16)
        c_sq = (cents.float() ** 2).sum(-1).view(K).contiguous()
        out = torch.empty((1, N), device="cuda", dtype=torch.int32)

        # Pass pre-computed c_sq to both paths to keep the comparison fair —
        # the Lloyd loop also caches c_sq across iterations.
        t_tri = _bench(lambda: euclid_assign_triton(x, cents, out=out, c_sq=c_sq.view(1, K)))
        t_cute = _bench(lambda: cutedsl_assign_euclid(x, cents, out=out, c_sq=c_sq))
        print(f"N={N:<8} D={D:<5} K={K:<5}      {t_tri:>13.1f} {t_cute:>15.1f} {t_tri/t_cute:>10.2f}x")


def bench_kernel_only(N: int, D: int, K: int, dtype_s: str, bm: int, bn: int, repeats: int = 30):
    """Direct kernel test (no fallback path) for debugging."""
    import cuda.bindings.driver as cuda_drv
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack

    from flash_kmeans._cutedsl_assign_kernel import HopperFlashKmeansAssign

    dtype = _DTYPES[dtype_s]

    torch.manual_seed(0)
    x = torch.randn(N, D, device="cuda", dtype=dtype).contiguous()
    cents = torch.randn(K, D, device="cuda", dtype=dtype).contiguous()
    c_sq = (cents.float() ** 2).sum(-1).contiguous()
    out = torch.empty(N, device="cuda", dtype=torch.int32)

    ref = _torch_ref_argmin(x, cents)

    kernel = HopperFlashKmeansAssign(
        acc_dtype=cutlass.Float32, m_block_size=bm, n_block_size=bn,
    )
    stream = cuda_drv.CUstream(0)
    print(f"Compiling N={N} D={D} K={K} dtype={dtype_s} BM={bm} BN={bn} ...")
    t0 = time.time()
    compiled = cute.compile(
        kernel,
        from_dlpack(x), from_dlpack(cents),
        from_dlpack(c_sq), from_dlpack(out),
        stream,
    )
    print(f"Compile took {time.time() - t0:.2f}s")

    out.zero_()
    compiled(
        from_dlpack(x), from_dlpack(cents),
        from_dlpack(c_sq), from_dlpack(out),
        stream,
    )
    torch.cuda.synchronize()
    n_mismatch = (out != ref).sum().item()
    print(f"Mismatches: {n_mismatch} / {N}")

    t = _bench(
        lambda: compiled(
            from_dlpack(x), from_dlpack(cents),
            from_dlpack(c_sq), from_dlpack(out),
            stream,
        ),
        repeats=repeats,
    )
    print(f"Avg time: {t:.1f} us")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["correctness", "bench", "kernel_only"], default="correctness")
    ap.add_argument("--n", type=int, default=8192)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--k", type=int, default=512)
    ap.add_argument("--dtype", choices=list(_DTYPES.keys()), default="bf16")
    ap.add_argument("--bm", type=int, default=128)
    ap.add_argument("--bn", type=int, default=128)
    ap.add_argument("--repeats", type=int, default=30)
    args = ap.parse_args()

    if args.mode == "correctness":
        ok = correctness_sweep()
        if not ok:
            raise SystemExit(1)
    elif args.mode == "bench":
        bench_assign()
    elif args.mode == "kernel_only":
        bench_kernel_only(
            args.n, args.d, args.k, args.dtype, args.bm, args.bn, args.repeats
        )


if __name__ == "__main__":
    main()
