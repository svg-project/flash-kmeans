"""CuteDSL implementation of flash-kmeans (Hopper SM90).

Public API:
    cutedsl_assign_euclid(x, centroids, x_sq, out=None, c_sq=None, ...,
                          autotune=True, cute_tile=None)
    cutedsl_kmeans_Euclid(x, n_clusters, ...)
    cutedsl_finalize(...)
    cutedsl_info()

Autotune
========

By default ``autotune=True`` and the first call for a new ``(N, D, K,
dtype)`` shape sweeps every fitting ``(BM, BN, use_ws)`` config (6 tile
shapes × 2 WS modes = up to 12 candidates), compiles each, benches it,
and caches the winner. Subsequent calls for the same shape dispatch
the cached compiled handle in sub-millisecond. A typical autotune for
one shape costs ~10–20 s of wall clock (compile-dominated); a Lloyd
loop only pays it once. Pass ``autotune_verbose=True`` to see per-config
timings.

Pass ``cute_tile=(BM, BN)`` or ``cute_tile=(BM, BN, use_ws)`` to skip
autotune and force a config (useful for deterministic/no-warmup runs).
Pass ``autotune=False`` to fall back to the static ``_pick_tile``
heuristic.

WHAT'S NEW vs the previous "cuBLAS bf16 GEMM + CuteDSL argmin" pipeline
----------------------------------------------------------------------

The old design materialised the full ``(N, K)`` cross matrix in HBM via
a cuBLAS-Lt GEMM and then ran a CuteDSL argmin epilogue. That works for
small N·K but is asymptotically wasteful: the cross tensor is the same
size as the (B,N,K) distance matrix the rest of flash-kmeans is
specifically designed to avoid. For the operator's heaviest example
(N=100K, D=512, K=512, fp16) the cross tensor alone is 50MB —
significantly above L2 — so two HBM round-trips dominate, and on the
mid-D shapes (N=1M, D=64, K=64) the materialisation actually *lost* to
the Triton K-streaming kernel (220 us vs 98 us per the previous header
notes).

The new design is a **fused TMA + WGMMA kernel** modelled directly on
the CUTLASS Hopper FMHA reference (FA-style mainloop) and the Hopper
dense GEMM example. It replicates the K-streaming property of the
Triton kernel, but adds:

* TMA bulk-tensor loads for both X and centroid tiles (vs Triton's
  per-element ``tl.load`` masks).
* Hopper SM90 WGMMA tensor cores via the official
  ``cute.nvgpu.warpgroup.MmaF16BF16Op`` atom, with multistage SMEM
  pipeline through ``cutlass.pipeline.PipelineTmaAsync``.
* In-register fused epilogue per centroid tile: the WGMMA accumulator
  ``(BM, BN_centroid)`` is converted to fp32 in registers, the
  Euclidean distance ``x_sq + c_sq − 2·cross`` is computed in registers,
  and a per-row running argmin is updated in registers. Across all
  centroid tiles, only ``(BM,)`` int32 best-indices are written to GMEM
  per CTA.
* Optional **producer/consumer warp specialization** (FMHA pattern):
  one load WG (24 regs/thread) issues all TMAs while the consumer WGs
  (240 regs/thread, set via ``setmaxregister_decrease/increase``) run
  WGMMA + the argmin epilogue. Selected per-shape by autotune; on
  H200 the heavy K=20K compute-bound shapes consistently prefer the
  non-WS variant (BM=128, BN=256), while smaller-tile configs benefit
  from WS.

See ``flash_kmeans/_cutedsl_assign_kernel.py`` for both kernels — the
WS variant is selected via the ``use_ws=True`` constructor flag of
``HopperFlashKmeansAssign``.

Bench (H200, GPU 0, single-shot assign, bf16 input, fp32 acc, c_sq pre-computed):

    Shape (N, D, K)      | Triton  | NEW CuteDSL  | Speedup
    ---------------------+---------+--------------+----------
    1M    × 16   × 8     |  35 us  |  39 us (Tri) |  -       *
    273K  × 16   × 64    |  34 us  |  28 us       |  1.25x
    1M    × 64   × 64    |  66 us  |  47 us       |  1.39x
    8K    × 128  × 1024  |  36 us  |  27 us       |  1.33x
    200K  × 256  × 256   | 116 us  |  84 us       |  1.39x
    1K    × 256  × 4K    |  83 us  |  64 us       |  1.29x
    100K  × 512  × 512   | 160 us  | 135 us       |  1.19x

*  D=16 K=16 and below falls back to Triton because the launch overhead
   of the fused CuteDSL kernel is comparable to the actual work.

End-to-end Lloyd loop (30 iters, ms total):

    Shape (N, D, K)      | Triton  | NEW CuteDSL  | Speedup
    ---------------------+---------+--------------+----------
    1M    × 64   × 64    | 10.3 ms |   8.4 ms     |  1.23x
    8K    × 128  × 1024  | 10.9 ms |   8.1 ms     |  1.35x
    200K  × 256  × 256   | 10.0 ms |   9.0 ms     |  1.10x
    100K  × 512  × 512   | 14.7 ms |  11.4 ms     |  1.28x

Heavy assign-bound workloads (Lloyd 10 iters, full Lloyd not just assign):

    Shape (N, D, K)         | Triton  | CuteDSL  |  Tri TFLOPs | Cute TFLOPs | Speedup
    ------------------------+---------+----------+-------------+-------------+--------
    10M  × 64   × 20K       | 1.23 s  |  1.03 s  |    208      |    249      | 1.19x
    10M  × 128  × 20K       | 1.93 s  |  1.30 s  |    266      |    395      | 1.49x
    10M  × 256  × 20K       | 2.90 s  |  1.82 s  |    353      |    563      | 1.60x
    5M   × 256  × 20K       | 1.46 s  |  0.91 s  |    351      |    563      | 1.60x
    2.5M × 512  × 20K       | 1.46 s  |  1.10 s  |    350      |    467      | 1.33x

The TFLOPs reported above are computed against the cross-term GEMM
(``2·N·K·D`` FMAs/iter) and divided by total Lloyd iter time (so they
include sort + centroid update overhead). At ``D=256`` the kernel
reaches 57 % of the Hopper bf16 WGMMA peak (~989 TFLOPs/s), rivaling
hand-tuned cuBLAS dense GEMMs at the same shape. The gap to peak comes
from the per-tile in-register argmin epilogue (small but not free) and
from the absence of producer/consumer warp specialization (~10 % more
overlap available on FA3 path). The Triton K-streaming kernel — the
prior best — sits at 21–36 % of peak on the same shapes.

Constraints
-----------

The new fused kernel handles:
* B = 1 (B > 1 falls back to ``euclid_assign_triton``).
* fp16 / bf16 input and centroid dtype (matched), fp32 ``x_sq``/``c_sq``.
* D a multiple of 16 (WGMMA K-tile is 16 for fp16/bf16).
* SMEM-fits configs: roughly D × (BM + 2·BN) ≤ 224 KiB on H200.
  D > ~512 with the largest (BM, BN) tile may not fit; we shrink the
  tile and ultimately fall back to Triton if even (BM=64, BN=64) is
  too large.

Anything outside these constraints falls back to the corresponding
Triton kernel so callers can keep using a single code path.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch

# ---------------------------------------------------------------------------
# Lazy CuteDSL import / kernel loader
# ---------------------------------------------------------------------------

_CUTEDSL_AVAILABLE = False
_CUTE_IMPORT_ERROR: Optional[Exception] = None
_AssignKernel = None  # type: ignore


def _try_init_cutedsl() -> bool:
    global _CUTEDSL_AVAILABLE, _CUTE_IMPORT_ERROR, _AssignKernel
    if _CUTEDSL_AVAILABLE:
        return True
    if _CUTE_IMPORT_ERROR is not None:
        return False
    try:
        import cutlass  # noqa: F401
        import cutlass.cute as cute  # noqa: F401
        from cutlass.cute.runtime import from_dlpack  # noqa: F401

        from ._cutedsl_assign_kernel import HopperFlashKmeansAssign

        globals()["cutlass"] = cutlass
        globals()["cute"] = cute
        globals()["from_dlpack"] = from_dlpack
        _AssignKernel = HopperFlashKmeansAssign
        _CUTEDSL_AVAILABLE = True
        return True
    except Exception as e:  # pragma: no cover - depends on local install
        _CUTE_IMPORT_ERROR = e
        return False


# ---------------------------------------------------------------------------
# Triton-side imports (fallback).
# ---------------------------------------------------------------------------
from .assign_euclid_triton import euclid_assign_triton  # noqa: E402
from .centroid_update_triton import (  # noqa: E402
    triton_centroid_finalize,
    triton_lloyd_centroid_step_euclid,
)


# ---------------------------------------------------------------------------
# Tile-shape heuristic
#
# Picks (BM, BN) for the fused kernel from (N, K, D, dtype). The table
# was tuned on H200 against the docstring's representative shapes. We
# fall back to (64, 64) if a larger tile would not fit SMEM.
# ---------------------------------------------------------------------------

def _smem_fits(BM: int, BN: int, D: int, x_dtype_bytes: int, smem_capacity: int) -> bool:
    """Return True if the (BM, BN, D) tile fits in per-CTA SMEM."""
    # X = (BM, D), 1 stage. C = (BN, D), at least 1 stage. Plus mbar
    # bookkeeping (~1 KiB). Mirror the kernel's own SMEM accounting.
    x_bytes = BM * D * x_dtype_bytes
    c_bytes = BN * D * x_dtype_bytes  # 1 stage
    mbar = 1024
    return (x_bytes + c_bytes + mbar) <= smem_capacity


# ---------------------------------------------------------------------------
# Autotune candidate enumeration
#
# We sweep (BM, BN, use_ws). All shapes that fit SMEM and pass JIT compile
# are benched; the winner is cached per (N, K, D, dtype) shape. The non-WS
# kernel is the historic dense_gemm-style path; WS wraps it with an extra
# producer warp group and decoupled register budgets (FMHA pattern). They
# are complementary — neither dominates uniformly across (D, K, BN).
# ---------------------------------------------------------------------------

# All (BM, BN) options the kernel class itself supports.
_ALL_TILES: Tuple[Tuple[int, int], ...] = (
    (64, 64), (64, 128), (64, 256),
    (128, 64), (128, 128), (128, 256),
)


def _autotune_candidates(D: int, x_dtype_bytes: int, smem_capacity: int):
    """Return the list of (BM, BN, use_ws) configs worth compiling.

    Filters by SMEM fit (one centroid stage minimum). Both ``use_ws``
    values are tried for every fitting tile, so the search is up to
    ``2 × |fitting tiles|`` configs.
    """
    out = []
    for bm, bn in _ALL_TILES:
        if not _smem_fits(bm, bn, D, x_dtype_bytes, smem_capacity):
            continue
        for use_ws in (False, True):
            out.append((bm, bn, use_ws))
    return out


def _pick_tile(
    N: int, K: int, D: int, x_dtype: torch.dtype, device: torch.device
) -> Tuple[int, int]:
    """Pick (BM, BN) tile shapes for the fused assign kernel.

    Validated on H200 across small (K=8…1024) and heavy (K≈20K) shapes.
    The table is intentionally small — auto-tuning across many configs
    is expensive in CuteDSL JIT compile time (~1s/config), so a static
    heuristic is a better trade-off than per-call autotuning. To force
    a config, use the ``cute_tile`` arg of ``cutedsl_assign_euclid``.

    Empirical pattern (bf16, H200):

    * Heavy K (K ≥ 4096) is *compute-bound*: WGMMA is the bottleneck,
      bigger BN amortises mainloop bookkeeping over more centroids per
      tile. ``BN=256`` wins by 13–18 % over ``BN=128`` at D=128/256.
      ``BN=64`` is competitive at D=64 (the WGMMA is short) and forced
      at D=512 (SMEM budget).
    * Mid K (128 < K < 4096): heuristic table tuned in the original
      bench sweep. ``BN=128`` is the sweet spot.
    * Small K (K ≤ 128): only a couple of tiles per CTA — narrower BN
      avoids wasted compute on masked OOB columns.
    """
    smem_capacity = torch.cuda.get_device_properties(device).shared_memory_per_block_optin
    bytes_per = 2  # fp16/bf16

    is_heavy_k = K >= 4096

    if D >= 384:
        # Very wide D: BN=256 won't fit. (128, 64) is the largest
        # universally-fitting config; (64, ...) is the next fallback.
        candidates = [(128, 64), (64, 128), (64, 64)]
    elif D >= 192:
        # D ≈ 256.
        if is_heavy_k:
            candidates = [(128, 256), (128, 128), (128, 64), (64, 128), (64, 64)]
        else:
            candidates = [(128, 128), (128, 64), (64, 128), (64, 64)]
    elif D >= 96:
        # D = 128.
        if is_heavy_k:
            candidates = [(128, 256), (128, 128), (128, 64), (64, 128), (64, 64)]
        elif K <= 128:
            candidates = [(128, 64), (128, 128), (64, 64)]
        else:
            candidates = [(128, 128), (128, 64), (64, 128), (64, 64)]
    elif D >= 33:
        # D = 64. Heavy-K (K=20K bench): BN=64 = 252 TFLOPs >
        # BN=256 = 248 > BN=128 = 231 TFLOPs. The WGMMA at D=64 is
        # short (4 inner k-blocks) so BN=64 keeps mainloop tight and
        # avoids the per-tile epilogue scaling with BN.
        if is_heavy_k:
            candidates = [(128, 64), (128, 256), (128, 128), (64, 64)]
        elif K <= 64:
            candidates = [(128, 64), (128, 128), (64, 64)]
        else:
            candidates = [(128, 128), (128, 64)]
    else:
        # D ≤ 32. Tiny problem; tile choice barely matters.
        if K <= 8:
            candidates = [(128, 64), (128, 128), (64, 64)]
        else:
            candidates = [(128, 128), (128, 64), (64, 64)]

    for BM, BN in candidates:
        if _smem_fits(BM, BN, D, bytes_per, smem_capacity):
            return BM, BN
    # Should be unreachable at D <= 512 but leave a defensive fallback.
    return 64, 64


# ---------------------------------------------------------------------------
# Compiled-kernel cache. cute.compile takes ~0.8-1s per (shape, dtype,
# tile) combo; without caching, an inner Lloyd loop would re-JIT every
# iter. Key on the inputs that affect compilation: N, D, K, x_dtype,
# centroids stride/dtype, BM, BN. Buffer addresses (data_ptr) are NOT
# part of the key — the compiled handle dispatches against any tensor
# of the same shape/stride/dtype.
# ---------------------------------------------------------------------------

_kernel_cache: dict = {}
_dlpack_cache: dict = {}
# Autotune cache: maps (N, D, K, x_dtype, c_dtype) -> (BM, BN, use_ws)
_autotune_cache: dict = {}


def _cached_from_dlpack(t: torch.Tensor):
    from cutlass.cute.runtime import from_dlpack  # local import — see _try_init_cutedsl
    key = (t.data_ptr(), tuple(t.shape), tuple(t.stride()), t.dtype)
    val = _dlpack_cache.get(key)
    if val is None:
        val = from_dlpack(t)
        _dlpack_cache[key] = val
    return val


def _get_compiled_assign(
    x: torch.Tensor,
    centroids: torch.Tensor,
    x_sq: torch.Tensor,
    c_sq: torch.Tensor,
    out: torch.Tensor,
    BM: int,
    BN: int,
    use_ws: bool = False,
):
    """Compile (or fetch cached) the fused assign kernel for these tensors.

    Returns ``(compiled_callable, stream)``.
    """
    import cuda.bindings.driver as cuda_drv
    import cutlass
    import cutlass.cute as cute_mod
    from cutlass.cute.runtime import from_dlpack

    key = (
        tuple(x.shape), tuple(x.stride()), x.dtype,
        tuple(centroids.shape), tuple(centroids.stride()), centroids.dtype,
        BM, BN, bool(use_ws),
    )
    if key in _kernel_cache:
        return _kernel_cache[key]

    kernel = _AssignKernel(
        acc_dtype=cutlass.Float32,
        m_block_size=BM,
        n_block_size=BN,
        use_ws=bool(use_ws),
    )
    stream = cuda_drv.CUstream(0)
    compiled = cute_mod.compile(
        kernel,
        from_dlpack(x), from_dlpack(centroids),
        from_dlpack(x_sq), from_dlpack(c_sq), from_dlpack(out),
        stream,
    )
    _kernel_cache[key] = (compiled, stream)
    return compiled, stream


def _autotune_pick(
    x2d: torch.Tensor,
    c2d: torch.Tensor,
    x_sq_1d: torch.Tensor,
    c_sq: torch.Tensor,
    out_1d: torch.Tensor,
    *,
    warmup: int = 5,
    repeats: int = 10,
    verbose: bool = False,
) -> Tuple[int, int, bool, float]:
    """Compile + bench every fitting (BM, BN, use_ws) config, return the best.

    Returns ``(BM, BN, use_ws, time_us)``. The compiled kernel for the
    winner stays in ``_kernel_cache`` and is reused on subsequent calls.

    The autotune cost is O(num_configs × compile_time) ≈ 12 × 1.5 s = 18 s
    for the first call on a new shape, then sub-millisecond dispatch.
    Cache by (N, D, K, dtype) means a Lloyd loop pays it once.
    """
    N, D = x2d.shape
    K = c2d.shape[0]
    dtype_bytes = 2  # fp16/bf16
    smem_capacity = torch.cuda.get_device_properties(
        x2d.device
    ).shared_memory_per_block_optin

    candidates = _autotune_candidates(D, dtype_bytes, smem_capacity)

    best = None
    best_t = float("inf")
    for BM, BN, use_ws in candidates:
        try:
            compiled, stream = _get_compiled_assign(
                x2d, c2d, x_sq_1d, c_sq, out_1d, BM, BN, use_ws=use_ws
            )
        except Exception as exc:  # JIT compile failure (rare)
            if verbose:
                print(f"  autotune skip BM={BM} BN={BN} ws={use_ws}: {str(exc).splitlines()[0][:80]}")
            continue

        x_dl = _cached_from_dlpack(x2d)
        c_dl = _cached_from_dlpack(c2d)
        xs_dl = _cached_from_dlpack(x_sq_1d)
        cs_dl = _cached_from_dlpack(c_sq)
        o_dl = _cached_from_dlpack(out_1d)

        try:
            for _ in range(warmup):
                compiled(x_dl, c_dl, xs_dl, cs_dl, o_dl, stream)
            torch.cuda.synchronize()
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            for _ in range(repeats):
                compiled(x_dl, c_dl, xs_dl, cs_dl, o_dl, stream)
            e.record()
            torch.cuda.synchronize()
            t_us = s.elapsed_time(e) / repeats * 1000
        except Exception as exc:  # runtime kernel failure
            if verbose:
                print(f"  autotune skip BM={BM} BN={BN} ws={use_ws}: runtime error")
            continue

        if verbose:
            ws_tag = "WS " if use_ws else "non"
            print(f"  BM={BM:>3} BN={BN:>3} {ws_tag}: {t_us:>7.0f} us")

        if t_us < best_t:
            best_t = t_us
            best = (BM, BN, use_ws)

    if best is None:
        raise RuntimeError(
            f"autotune found no working config for N={N} D={D} K={K} "
            f"(this typically means even (BM=64, BN=64) doesn't fit SMEM)"
        )

    return best[0], best[1], best[2], best_t


# ---------------------------------------------------------------------------
# Public API: assign via fused TMA+WGMMA kernel.
# ---------------------------------------------------------------------------

def cutedsl_assign_euclid(
    x: torch.Tensor,            # (B, N, D)
    centroids: torch.Tensor,    # (B, K, D)
    x_sq: torch.Tensor,         # (B, N) fp32
    out: Optional[torch.Tensor] = None,
    c_sq: Optional[torch.Tensor] = None,
    *,
    cute_tile: Optional[Tuple] = None,
    autotune: bool = True,
    autotune_verbose: bool = False,
    # Legacy kwargs for compatibility with the old materialise-then-argmin
    # implementation. Ignored by the new fused kernel.
    x_bf16: Optional[torch.Tensor] = None,
    cross_buf: Optional[torch.Tensor] = None,
    c_sq_buf: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute nearest-centroid IDs, fused (no materialised cross matrix).

    Falls back to ``euclid_assign_triton`` when the input shape/dtype is
    outside the kernel's supported regime (B>1, non-fp16/bf16, D not
    a multiple of 16, SMEM too small, etc.). Callers do not need to
    branch.

    Parameters
    ----------
    cute_tile:
        Force a specific (BM, BN) or (BM, BN, use_ws) config and skip
        autotune. ``cute_tile=(128, 256)`` and ``cute_tile=(128, 256, True)``
        are both accepted.
    autotune:
        Default ``True``. On the first call for a new ``(N, D, K, dtype)``
        shape, compiles every fitting ``(BM, BN, use_ws)`` candidate
        (~12 configs × ~1.5 s each = ~18 s one-time cost) and benches
        each. The winner is cached and reused on subsequent calls — a
        Lloyd loop only pays the autotune cost once. Set to ``False`` to
        fall back to the static heuristic (``_pick_tile``) for
        deterministic / no-warmup behaviour.
    autotune_verbose:
        If True, prints per-config timings during the autotune sweep.
    """
    assert x.is_cuda and centroids.is_cuda and x_sq.is_cuda
    B, N, D = x.shape
    K = centroids.shape[1]
    assert centroids.shape == (B, K, D)
    assert x_sq.shape == (B, N)
    if out is None:
        out = torch.empty((B, N), device=x.device, dtype=torch.int32)

    # Triton expects c_sq with shape (B, K). Our CuteDSL kernel uses
    # the 1-D flattened (K,) shape. Normalise to (B, K) so all fallback
    # paths can forward c_sq verbatim.
    c_sq_for_tri = c_sq
    if c_sq is not None and c_sq.dim() == 1:
        c_sq_for_tri = c_sq.view(1, -1)

    # ---- fallback gates ---------------------------------------------------
    if not _try_init_cutedsl() or B != 1:
        return euclid_assign_triton(x, centroids, x_sq, out=out, c_sq=c_sq_for_tri)
    if x.dtype not in (torch.float16, torch.bfloat16):
        return euclid_assign_triton(x, centroids, x_sq, out=out, c_sq=c_sq_for_tri)
    if centroids.dtype != x.dtype:
        return euclid_assign_triton(x, centroids, x_sq, out=out, c_sq=c_sq_for_tri)
    if D % 16 != 0:
        return euclid_assign_triton(x, centroids, x_sq, out=out, c_sq=c_sq_for_tri)
    if D > 512:
        # The fused kernel is small-D only (the X tile fits in SMEM).
        # split-D (X-streaming) extension is tracked separately; for now
        # fall back to Triton for very wide features.
        return euclid_assign_triton(x, centroids, x_sq, out=out, c_sq=c_sq_for_tri)
    # Launch-bound tiny problems: D ≤ 16 and very small K. The CuteDSL
    # kernel has fixed mbarrier+TMA setup overhead (~5-10 us) that
    # Triton's vectorised launch dodges, so for shapes where the inner
    # work is in that ballpark we prefer Triton. Empirically the
    # crossover sits around (D ≤ 16, K ≤ 16) — every other shape we
    # tested (D=16 K=64 and up) wins on CuteDSL.
    if D <= 16 and K <= 16:
        return euclid_assign_triton(x, centroids, x_sq, out=out, c_sq=c_sq_for_tri)

    # ---- contiguity normalisation -----------------------------------------
    # The fused kernel needs k-major (D contiguous) X and centroids and
    # 1-D contiguous x_sq, c_sq, out.
    x2d = x.view(N, D)
    c2d = centroids.view(K, D)
    if not x2d.is_contiguous():
        x2d = x2d.contiguous()
    if not c2d.is_contiguous():
        c2d = c2d.contiguous()
    if c_sq is None:
        c_sq = (centroids.float() ** 2).sum(-1).view(K).contiguous()
    else:
        c_sq = c_sq.view(K)
        if not c_sq.is_contiguous():
            c_sq = c_sq.contiguous()
    x_sq_1d = x_sq.view(N)
    if not x_sq_1d.is_contiguous():
        x_sq_1d = x_sq_1d.contiguous()
    out_1d = out.view(N)
    if not out_1d.is_contiguous():
        out_1d = out_1d.contiguous()

    # ---- tile selection ---------------------------------------------------
    use_ws = False
    if cute_tile is not None:
        if len(cute_tile) == 2:
            BM, BN = cute_tile
        elif len(cute_tile) == 3:
            BM, BN, use_ws = cute_tile
        else:
            raise ValueError("cute_tile must be (BM, BN) or (BM, BN, use_ws)")
    elif autotune:
        # Cache key spans everything that affects kernel selection.
        ac_key = (N, D, K, x.dtype, centroids.dtype)
        cached = _autotune_cache.get(ac_key)
        if cached is None:
            try:
                BM, BN, use_ws, t_us = _autotune_pick(
                    x2d, c2d, x_sq_1d, c_sq, out_1d,
                    verbose=autotune_verbose,
                )
                _autotune_cache[ac_key] = (BM, BN, use_ws)
                if autotune_verbose:
                    ws_tag = "WS" if use_ws else "non-WS"
                    print(f"  autotune winner: BM={BM} BN={BN} {ws_tag} ({t_us:.0f} us)")
            except Exception:
                # Autotune crashed entirely (shouldn't happen at supported
                # shapes); fall back to heuristic + non-WS.
                BM, BN = _pick_tile(N, K, D, x.dtype, x.device)
                use_ws = False
        else:
            BM, BN, use_ws = cached
    else:
        BM, BN = _pick_tile(N, K, D, x.dtype, x.device)

    # ---- compile or fetch cached, then dispatch ---------------------------
    try:
        compiled, stream = _get_compiled_assign(
            x2d, c2d, x_sq_1d, c_sq, out_1d, BM, BN, use_ws=use_ws,
        )
    except Exception:
        return euclid_assign_triton(x, centroids, x_sq, out=out, c_sq=c_sq_for_tri)

    try:
        compiled(
            _cached_from_dlpack(x2d), _cached_from_dlpack(c2d),
            _cached_from_dlpack(x_sq_1d), _cached_from_dlpack(c_sq),
            _cached_from_dlpack(out_1d),
            stream,
        )
    except Exception:
        return euclid_assign_triton(x, centroids, x_sq, out=out, c_sq=c_sq_for_tri)

    return out


# ---------------------------------------------------------------------------
# CuteDSL Lloyd loop (bench harness).
# ---------------------------------------------------------------------------

def cutedsl_kmeans_Euclid(
    x: torch.Tensor,
    n_clusters: int,
    max_iters: int = 100,
    tol: float = 0.0,
    init_centroids: Optional[torch.Tensor] = None,
    verbose: bool = False,
):
    """Lloyd loop using fused CuteDSL assign + Triton centroid update.

    Identical interface to ``flash_kmeans.batch_kmeans_Euclid``. The
    centroid finalize step still uses the Triton kernel — see
    ``cutedsl_finalize`` for the rationale (CuteDSL atomic_add not yet
    stably exposed).
    """
    B, N, D = x.shape
    assert B == 1, "cutedsl path currently supports B=1 only"
    K = n_clusters

    x_sq = (x ** 2).sum(dim=-1)

    if init_centroids is None:
        indices = torch.randint(0, N, (B, K), device=x.device)
        centroids = torch.gather(
            x, dim=1,
            index=indices[..., None].expand(-1, -1, D),
        )
    else:
        centroids = init_centroids
    centroids = centroids.view(B, K, D).contiguous()

    centroids_a = centroids
    centroids_b = torch.empty_like(centroids_a)
    sums_buf = torch.zeros((B, K, D), device=x.device, dtype=torch.float32)
    cnts_buf = torch.zeros((B, K), device=x.device, dtype=torch.int32)
    shift_buf = torch.empty((B, K), device=x.device, dtype=torch.float32)
    cluster_ids_buf = torch.empty((B, N), device=x.device, dtype=torch.int32)

    cur, nxt = centroids_a, centroids_b
    for it in range(max_iters):
        cluster_ids = cutedsl_assign_euclid(
            x, cur, x_sq, out=cluster_ids_buf,
        )
        new_cents, _, max_shift = triton_lloyd_centroid_step_euclid(
            x, cluster_ids, cur,
            sums_buf=sums_buf,
            cnts_buf=cnts_buf,
            new_buf=nxt,
            shift_buf=shift_buf,
        )
        if verbose:
            print(f"Iter {it} (cutedsl), shift={max_shift.item():.6f}")
        if tol > 0.0 and max_shift.item() < tol:
            cur = new_cents
            break
        cur, nxt = nxt, cur

    return cluster_ids, cur, it + 1


# ---------------------------------------------------------------------------
# CuteDSL finalize — thin shim over Triton finalize.
# ---------------------------------------------------------------------------

def cutedsl_finalize(
    sums: torch.Tensor,
    cnts: torch.Tensor,
    old_centroids: torch.Tensor,
    *,
    out: Optional[torch.Tensor] = None,
    shift: Optional[torch.Tensor] = None,
):
    """CuteDSL drop-in for ``triton_centroid_finalize``.

    A hand-written CuteDSL finalize kernel would need
    ``cute.arch.atomic_add_f32`` to fuse the per-cluster shift reduction
    in a single launch. That API is not stably exposed in cutlass-dsl
    4.4.x / 4.5.x, so we delegate to ``triton_centroid_finalize``. Once
    that API surfaces (either through cute.arch or via a CCCL wrapper),
    re-implementing the finalize as a pure CuteDSL kernel is a single-
    day change.
    """
    return triton_centroid_finalize(sums, cnts, old_centroids, out=out, shift=shift)


# ---------------------------------------------------------------------------
# Convenience info function for bench scripts.
# ---------------------------------------------------------------------------

def cutedsl_info() -> str:
    if _try_init_cutedsl():
        import cutlass
        return f"CuteDSL fused-assign available (cutlass-dsl=={cutlass.__version__})"
    return f"CuteDSL UNAVAILABLE: {_CUTE_IMPORT_ERROR!r}"
