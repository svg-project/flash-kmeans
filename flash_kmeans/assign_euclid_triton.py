from typing import Optional
import torch
import triton
import triton.language as tl

# ===============================================================
# Triton kernel: compute nearest-centroid IDs (Euclidean distance)
# Inputs:
#   x           : (B, N, D)  float16 / float32
#   centroids   : (B, K, D)  same dtype as x
#   x_sq        : (B, N)     float32 – pre-computed ||x||^2 per point
# Output:
#   cluster_ids : (B, N)     int32   – nearest centroid index per point
# ===============================================================


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


# -----------------------------------------------------------------------------
# Auto-tuning setup – explore various tile sizes / warp counts
# -----------------------------------------------------------------------------

_TUNE_CONFIGS = [
    triton.Config({"BLOCK_N": BN, "BLOCK_K": BK}, num_stages=num_stages, num_warps=wp)
    for BN in [32, 64, 128]
    for BK in [32, 64, 128]
    for wp in [4, 8]
    for num_stages in [1, 2, 4]
]


def _cfg_keep(conf):
    """Basic heuristic to prune unbalanced configs."""
    BN = conf.kwargs["BLOCK_N"]
    BK = conf.kwargs["BLOCK_K"]
    # Avoid tiny tiles on many warps
    if BN * BK < 32 * 32 and conf.num_warps > 4:
        return False
    return True

_TUNE_CONFIGS = list(filter(_cfg_keep, _TUNE_CONFIGS))

_HALF_DTYPES = (torch.float16, torch.bfloat16)


def _dtype_bytes(dtype) -> int:
    """Element size in bytes for a torch / numpy-ish dtype.

    Falls back to 2 (fp16) when the dtype is unknown to keep prior behaviour
    (the heuristic was originally tuned with fp16 in mind).
    """
    if dtype is None:
        return 2
    if isinstance(dtype, torch.dtype):
        return torch.tensor([], dtype=dtype).element_size()
    # Allow callers to pass a raw byte size.
    if isinstance(dtype, int):
        return dtype
    return 2


def _is_half_dtype(dtype) -> bool:
    """True for fp16/bf16 (the original tuning regime).

    For these dtypes we skip the SMEM-fitting fallback entirely so heuristic
    selection on already-validated GPUs (H100/H200/A100) is byte-for-byte
    identical to the previous behaviour.
    """
    if dtype is None:
        return True
    if isinstance(dtype, torch.dtype):
        return dtype in _HALF_DTYPES
    return False


def _smem_bytes(D: int, BN: int, BK: int, num_stages: int, dtype_bytes: int) -> int:
    """Approximate dynamic shared-memory usage of `_euclid_assign_kernel`.

    The kernel materialises:
    - one ``x_tile`` of shape (BN, D) outside the K loop, and
    - ``num_stages`` copies of ``c_tile`` of shape (D, BK) for the software
      pipelined K loop.

    Other buffers (x_sq, c_sq, masks, accumulators) are negligible compared
    to these and are ignored.
    """
    return D * dtype_bytes * (BN + num_stages * BK)


def _smem_limit(device) -> int:
    """Per-block dynamic shared-memory budget for ``device``.

    Triton uses opt-in dynamic shared memory; prefer that attribute when
    available, fall back to the static limit, and finally to a conservative
    48 KiB for very old PyTorch builds.
    """
    props = torch.cuda.get_device_properties(device)
    for attr in (
        "shared_memory_per_block_optin",
        "max_shared_memory_per_block_optin",
        "shared_memory_per_block",
        "max_shared_memory_per_block",
    ):
        v = getattr(props, attr, None)
        if v:
            return int(v)
    return 48 * 1024


def _fit_config_to_smem(
    cfg: dict,
    D: int,
    dtype_bytes: int,
    smem_limit: int,
) -> dict:
    """Return a config that fits ``smem_limit`` and is closest to ``cfg``.

    The original config is returned unchanged whenever it already fits. If
    not, we enumerate all power-of-two ``(BLOCK_N, BLOCK_K, num_stages)``
    that are no larger than the original and pick the one that maximises
    work-per-program tile (``BLOCK_N * BLOCK_K * num_stages``), breaking
    ties towards the original aspect ratio. This avoids the pitfall of a
    pure greedy halving (e.g. shrinking BLOCK_K all the way to 16 when only
    a single halving was needed).

    Raises ``RuntimeError`` if even ``(BN=16, BK=16, S=1)`` does not fit –
    this only happens for absurdly large D combined with fp32 on tiny-SMEM
    GPUs.
    """
    BN0 = int(cfg["BLOCK_N"])
    BK0 = int(cfg["BLOCK_K"])
    W0 = int(cfg["num_warps"])
    S0 = int(cfg["num_stages"])

    if _smem_bytes(D, BN0, BK0, S0, dtype_bytes) <= smem_limit:
        return {"BLOCK_N": BN0, "BLOCK_K": BK0, "num_warps": W0, "num_stages": S0}

    def _pow2_down_to_16(v):
        out = []
        x = v
        while x >= 16:
            out.append(x)
            x //= 2
        return out

    best = None
    best_key = None
    for BN in _pow2_down_to_16(BN0):
        for BK in _pow2_down_to_16(BK0):
            for S in range(S0, 0, -1):
                if _smem_bytes(D, BN, BK, S, dtype_bytes) > smem_limit:
                    continue
                # Prefer larger total tile work, then closer aspect ratio
                # to the original, then larger BLOCK_N (more parallelism
                # along N), then larger num_stages (better pipelining).
                aspect_penalty = abs(
                    (BN / max(BK, 1)) - (BN0 / max(BK0, 1))
                )
                key = (BN * BK * S, -aspect_penalty, BN, S)
                if best_key is None or key > best_key:
                    best_key = key
                    best = (BN, BK, S)

    if best is None:
        raise RuntimeError(
            f"euclid_assign_triton: cannot fit kernel into shared memory "
            f"(D={D}, dtype_bytes={dtype_bytes}, smem_limit={smem_limit}). "
            f"Even BLOCK_N=16, BLOCK_K=16, num_stages=1 needs "
            f"{_smem_bytes(D, 16, 16, 1, dtype_bytes)} bytes."
        )

    BN, BK, S = best
    W = W0
    # Tiny tiles do not benefit from many warps and may even fail to compile
    # for some Triton versions; cap to 4.
    if BN * BK <= 32 * 32 and W > 4:
        W = 4

    return {"BLOCK_N": BN, "BLOCK_K": BK, "num_warps": W, "num_stages": S}


def _heuristic_euclid_config(
    N: int,
    K: int,
    D: int,
    *,
    device: Optional[torch.device] = None,
    dtype=None,
):
    """Architecture-aware heuristic config selection without autotune.

    Keep one unified heuristic entry and diverge inside by GPU family:
    - H200: existing hand-tuned heuristic
    - H100: heuristic derived from H100 grid tuning results
    - A100: heuristic derived from A100 grid tuning results
    - GB10: heuristic derived from GB10 grid tuning results
    - others: conservative fallback to reduce OOR risk

    For half-precision dtypes (fp16/bf16, the regime the per-arch tables
    were tuned in) the picked config is returned as-is so behaviour on
    H100/H200/A100 stays byte-for-byte identical. For wider dtypes (fp32
    and friends) we additionally pass the config through
    ``_fit_config_to_smem`` so the kernel does not OOR on small-SMEM GPUs.
    """
    if device is None:
        device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_properties(device).name.upper()

    if _is_half_dtype(dtype):
        # Original code path: trust the per-arch table without SMEM checks.
        def _finalize(cfg):
            return cfg
    else:
        dtype_bytes = _dtype_bytes(dtype)
        smem_limit = _smem_limit(device)

        def _finalize(cfg):
            return _fit_config_to_smem(cfg, D, dtype_bytes, smem_limit)

    if "H200" in gpu_name:
        # Keep the original H200 heuristic as-is.
        block_n = 128
        block_k = 64
        num_warps = 4
        num_stages = 1

        if D >= 512:
            block_n = 128
            block_k = 64
            num_warps = 8
            num_stages = 1
        elif D >= 256:
            block_n = 128
            block_k = 64
            num_warps = 4
            num_stages = 2
        else:
            # D <= 128
            if K >= 4096:
                block_k = 128
                if D >= 128:
                    num_warps = 8
                    num_stages = 2
                else:
                    num_warps = 4
                    num_stages = 4
            else:
                block_k = 64
                num_warps = 4
                num_stages = 1

        # D=64 with large K tends to prefer smaller BLOCK_N and deeper pipeline.
        if D <= 64 and K >= 4096:
            block_n = 64
            block_k = 128
            num_warps = 4
            num_stages = 4

        # Smaller N favors smaller BLOCK_N to reduce wasted work.
        if N < 65536:
            block_n = 64

        return _finalize({
            "BLOCK_N": block_n,
            "BLOCK_K": block_k,
            "num_warps": num_warps,
            "num_stages": num_stages,
        })

    if "H100" in gpu_name:
        # H100 tuned heuristic (more conservative on D=64 mid-K vs H200).
        block_n = 128
        block_k = 64
        num_warps = 4
        num_stages = 1

        if D >= 512:
            block_n = 128
            block_k = 64
            num_warps = 8
            num_stages = 1
        elif D >= 256:
            block_n = 128
            block_k = 64
            if K <= 1024:
                num_warps = 8
                num_stages = 1
            elif K <= 16384:
                num_warps = 4
                num_stages = 1
            else:
                num_warps = 8
                num_stages = 1
        else:
            # D <= 128
            if D <= 64:
                if K <= 1024:
                    block_k = 64
                    num_warps = 4
                    num_stages = 2
                elif K <= 16384:
                    block_k = 64
                    num_warps = 4
                    num_stages = 2
                elif K <= 65536:
                    block_k = 128
                    num_warps = 4
                    num_stages = 4
                else:
                    block_k = 64
                    num_warps = 4
                    num_stages = 4
            else:
                # D == 128
                if K <= 1024:
                    block_k = 64
                    num_warps = 4
                    num_stages = 1
                elif K <= 65536:
                    block_k = 128
                    num_warps = 8
                    num_stages = 2
                else:
                    block_k = 64
                    num_warps = 4
                    num_stages = 4

        if N < 65536:
            block_n = 64

        return _finalize({
            "BLOCK_N": block_n,
            "BLOCK_K": block_k,
            "num_warps": num_warps,
            "num_stages": num_stages,
        })

    if "A100" in gpu_name:
        # Robust default on A100 across tuned grid.
        block_n = 128
        block_k = 32
        num_warps = 4
        num_stages = 2

        if D == 128:
            # Small-N cases tend to prefer a larger K tile.
            if N <= 65536:
                block_k = 64
        elif D == 256:
            # D=256 benefits from deeper pipeline at larger K.
            if K >= 65536:
                block_k = 32
                num_stages = 4
            elif K >= 1024 and N <= 262144:
                block_k = 64
                num_stages = 4

        return _finalize({
            "BLOCK_N": block_n,
            "BLOCK_K": block_k,
            "num_warps": num_warps,
            "num_stages": num_stages,
        })

    if "GB10" in gpu_name:
        # GB10 (Grace Blackwell, ~80 SMs, ~99 KiB SMEM/SM) tuned heuristic.
        # Derived from a grid sweep over N in {65536, 262144, 1048576},
        # K in {256..200000}, D in {64,128,256,512}, B in {1, 32}, fp16.
        # Geomean slowdown vs. per-shape optimum is ~1% across the sweep
        # (worst case ~9% on sub-millisecond ops where timing noise
        # dominates). The selected config is post-processed by
        # _fit_config_to_smem so fp32 / large D inputs are shrunk to fit
        # GB10's modest shared-memory budget.
        if D >= 512:
            # D=512 strongly prefers a small BLOCK_N with 8 warps, except for
            # very small K where 4 warps is enough to saturate.
            if K <= 256:
                return _finalize({"BLOCK_N": 64, "BLOCK_K": 32, "num_warps": 4, "num_stages": 1})
            return _finalize({"BLOCK_N": 64, "BLOCK_K": 32, "num_warps": 8, "num_stages": 1})

        if D >= 256:
            if K <= 256:
                return _finalize({"BLOCK_N": 64, "BLOCK_K": 32, "num_warps": 4, "num_stages": 1})
            # Deeper pipeline + wider K tile pays off for K>=1024 at D=256.
            return _finalize({"BLOCK_N": 128, "BLOCK_K": 64, "num_warps": 4, "num_stages": 2})

        if D >= 128:
            if K <= 256:
                # Small K: a more square tile wins (BN=64, BK=64).
                return _finalize({"BLOCK_N": 64, "BLOCK_K": 64, "num_warps": 4, "num_stages": 1})
            if K <= 1024:
                # Transition region: small N likes a square tile, large N
                # benefits from BN=128 with deeper pipeline.
                if N <= 65536:
                    return _finalize({"BLOCK_N": 64, "BLOCK_K": 64, "num_warps": 4, "num_stages": 1})
                return _finalize({"BLOCK_N": 128, "BLOCK_K": 32, "num_warps": 4, "num_stages": 2})
            if K <= 65536:
                return _finalize({"BLOCK_N": 128, "BLOCK_K": 32, "num_warps": 4, "num_stages": 1})
            # K > 65536 (e.g. 200k) prefers the wider K tile to amortize loads.
            return _finalize({"BLOCK_N": 128, "BLOCK_K": 64, "num_warps": 4, "num_stages": 1})

        # D <= 64: BN=128, BK=32, 4 warps is robust across the full grid.
        # Only the tiniest shapes (small N + small K) shift toward a square
        # BN=64/BK=64 tile; larger N keeps the wider BN=128 default.
        if K <= 256 and N <= 65536:
            return _finalize({"BLOCK_N": 64, "BLOCK_K": 64, "num_warps": 4, "num_stages": 1})
        return _finalize({"BLOCK_N": 128, "BLOCK_K": 32, "num_warps": 4, "num_stages": 1})

    # Conservative fallback for unknown architectures (prioritize avoiding OOR).
    return _finalize({
        "BLOCK_N": 64,
        "BLOCK_K": 32,
        "num_warps": 4,
        "num_stages": 1,
    })


@triton.jit
def _euclid_assign_kernel(
    x_ptr,                 # *f16 / *f32 [B, N, D]
    c_ptr,                 # *f16 / *f32 [B, K, D]
    x_sq_ptr,              # *f32         [B, N]
    c_sq_ptr,              # *f32         [B, K]
    out_ptr,               # *i32         [B, N]
    B: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    D: tl.constexpr,
    stride_x_b: tl.constexpr,
    stride_x_n: tl.constexpr,
    stride_x_d: tl.constexpr,
    stride_c_b: tl.constexpr,
    stride_c_k: tl.constexpr,
    stride_c_d: tl.constexpr,
    stride_xsq_b: tl.constexpr,
    stride_xsq_n: tl.constexpr,
    stride_csq_b: tl.constexpr,
    stride_csq_k: tl.constexpr,
    stride_out_b: tl.constexpr,
    stride_out_n: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Each program handles a tile of BLOCK_N points for a given batch element.

    The kernel iterates over the centroid dimension K in chunks of BLOCK_K and
    maintains the running minimum distance as well as the corresponding index
    for every point in the tile.
    """
    pid_n = tl.program_id(0)          # tile index along N dimension
    pid_b = tl.program_id(1)          # batch index
    pid_b = pid_b.to(tl.int64)

    n_start = pid_n * BLOCK_N
    n_offsets = n_start + tl.arange(0, BLOCK_N)
    n_offsets = n_offsets.to(tl.int64)
    n_mask = n_offsets < N

    # ------------------------------------------------------------------
    # Load x tile  (BLOCK_N, D)
    # ------------------------------------------------------------------
    offs_d = tl.arange(0, D).to(tl.int64)
    # Compute pointer for x block: base + b*stride_x_b + n*stride_x_n + d*stride_x_d
    x_ptrs = (
        x_ptr
        + pid_b * stride_x_b
        + n_offsets[:, None] * stride_x_n
        + offs_d[None, :] * stride_x_d
    )
    x_tile = tl.load(x_ptrs, mask=n_mask[:, None], other=0.0)
    x_tile = x_tile  # compute in f32

    # Pre-load x_sq for the tile  (BLOCK_N,)
    xsq_ptrs = x_sq_ptr + pid_b * stride_xsq_b + n_offsets * stride_xsq_n
    x_sq_tile = tl.load(xsq_ptrs, mask=n_mask, other=0.0).to(tl.float32)

    # Init best distance / index
    best_dist = tl.full((BLOCK_N,), 3.4e38, tl.float32)  # large number
    best_idx = tl.zeros((BLOCK_N,), tl.int32)

    # ------------------------------------------------------------------
    # Iterate over centroids in chunks of BLOCK_K
    # ------------------------------------------------------------------
    for k_start in range(0, K, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        k_offsets = k_offsets.to(tl.int64)
        k_mask = k_offsets < K

        # Load centroid tile  (D, BLOCK_K)
        c_ptrs = (
            c_ptr
            + pid_b * stride_c_b
            + k_offsets[None, :] * stride_c_k
            + offs_d[:, None] * stride_c_d
        )
        c_tile = tl.load(c_ptrs, mask=k_mask[None, :], other=0.0)
        c_tile = c_tile

        # load c_sq for the tile  (BLOCK_K,)
        csq_ptrs = c_sq_ptr + pid_b * stride_csq_b + k_offsets * stride_csq_k
        cent_sq = tl.load(csq_ptrs, mask=k_mask, other=0.0).to(tl.float32)

        # # Compute centroid squared norms (BLOCK_K,)
        # cent_sq = tl.sum(c_tile * c_tile, axis=0).to(tl.float32)

        # Compute cross term (BLOCK_N, BLOCK_K) = x_tile @ c_tile
        cross = tl.dot(x_tile, c_tile).to(tl.float32)  # float32

        # Squared Euclidean distance
        dist = x_sq_tile[:, None] + cent_sq[None, :] - 2.0 * cross
        dist = tl.maximum(dist, 0.0)

        # Mask out invalid centroid columns before reduction
        dist = tl.where(k_mask[None, :], dist, 3.4e38)

        curr_min = tl.min(dist, axis=1)
        curr_idx = tl.argmin(dist, axis=1)

        update = curr_min < best_dist
        best_dist = tl.where(update, curr_min, best_dist)
        best_idx = tl.where(update, k_start + curr_idx, best_idx)

    # ------------------------------------------------------------------
    # Write results
    # ------------------------------------------------------------------
    out_ptrs = out_ptr + pid_b * stride_out_b + n_offsets * stride_out_n
    tl.store(out_ptrs, best_idx, mask=n_mask)

_euclid_assign_kernel_autotuned = triton.autotune(_TUNE_CONFIGS, key=["N", "K"])(_euclid_assign_kernel)

@triton.jit
def _cosine_assign_kernel(
    x_ptr,                 # *f16 / *f32 [B, N, D]
    c_ptr,                 # *f16 / *f32 [B, K, D]
    out_ptr,               # *i32         [B, N]
    B: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    D: tl.constexpr,
    stride_x_b: tl.constexpr,
    stride_x_n: tl.constexpr,
    stride_x_d: tl.constexpr,
    stride_c_b: tl.constexpr,
    stride_c_k: tl.constexpr,
    stride_c_d: tl.constexpr,
    stride_out_b: tl.constexpr,
    stride_out_n: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Each program handles a tile of BLOCK_N points for a given batch element.

    The kernel iterates over the centroid dimension K in chunks of BLOCK_K and
    maintains the running minimum distance as well as the corresponding index
    for every point in the tile.
    """
    pid_n = tl.program_id(0)          # tile index along N dimension
    pid_b = tl.program_id(1)          # batch index
    pid_b = pid_b.to(tl.int64)

    n_start = pid_n * BLOCK_N
    n_offsets = n_start + tl.arange(0, BLOCK_N)
    n_offsets = n_offsets.to(tl.int64)
    n_mask = n_offsets < N

    # ------------------------------------------------------------------
    # Load x tile  (BLOCK_N, D)
    # ------------------------------------------------------------------
    offs_d = tl.arange(0, D).to(tl.int64)
    # Compute pointer for x block: base + b*stride_x_b + n*stride_x_n + d*stride_x_d
    x_ptrs = (
        x_ptr
        + pid_b * stride_x_b
        + n_offsets[:, None] * stride_x_n
        + offs_d[None, :] * stride_x_d
    )
    x_tile = tl.load(x_ptrs, mask=n_mask[:, None], other=0.0)
    x_tile = x_tile  # compute in f32

    # Init best distance / index
    best_dist = tl.full((BLOCK_N,), -3.4e38, tl.float32)  # less is worse 
    best_idx = tl.zeros((BLOCK_N,), tl.int32)

    # ------------------------------------------------------------------
    # Iterate over centroids in chunks of BLOCK_K
    # ------------------------------------------------------------------
    for k_start in range(0, K, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        k_offsets = k_offsets.to(tl.int64)
        k_mask = k_offsets < K

        # Load centroid tile  (D, BLOCK_K)
        c_ptrs = (
            c_ptr
            + pid_b * stride_c_b
            + k_offsets[None, :] * stride_c_k
            + offs_d[:, None] * stride_c_d
        )
        c_tile = tl.load(c_ptrs, mask=k_mask[None, :], other=0.0)
        c_tile = c_tile

        # Compute cosine distance (BLOCK_N, BLOCK_K) = x_tile @ c_tile
        cross = tl.dot(x_tile, c_tile).to(tl.float32)  # float32

        # Mask out invalid centroid columns before reduction.
        # Use a sentinel below any real cosine/dot score so masked tail lanes
        # never win the argmax. (0.0 is a real cosine value: any input row whose
        # cosine to all real centroids is < 0 would otherwise route to a masked
        # lane, returning an out-of-range cluster id.)
        dist = tl.where(k_mask[None, :], cross, -torch.finfo(torch.float32).max)

        curr_max = tl.max(dist, axis=1)
        curr_idx = tl.argmax(dist, axis=1)

        update = curr_max > best_dist
        best_dist = tl.where(update, curr_max, best_dist)
        best_idx = tl.where(update, k_start + curr_idx, best_idx)

    # ------------------------------------------------------------------
    # Write results
    # ------------------------------------------------------------------
    out_ptrs = out_ptr + pid_b * stride_out_b + n_offsets * stride_out_n
    tl.store(out_ptrs, best_idx, mask=n_mask)

_cosine_assign_kernel_autotuned = triton.autotune(_TUNE_CONFIGS, key=["N", "K"])(_cosine_assign_kernel)


# ---------------------------------------------------------------
# Python wrapper
# ---------------------------------------------------------------

def euclid_assign_triton(
    x: torch.Tensor,
    centroids: torch.Tensor,
    x_sq: torch.Tensor,
    out: torch.Tensor = None,
    c_sq: torch.Tensor = None,
    *,
    BLOCK_N: int = 128,
    BLOCK_K: int = 128,
    num_warps: Optional[int] = None,
    num_stages: Optional[int] = None,
    config: Optional[dict] = None,
    use_heuristic: bool = True,
) -> torch.Tensor:
    """Return nearest-centroid indices using Triton kernel.

    Args:
        x         : (B, N, D) float16 / float32 (on CUDA)
        centroids : (B, K, D) same dtype/device as x
        x_sq      : (B, N)    float32 – ||x||^2 per point (on CUDA)
        out       : (B, N)    int32   – (option) pre-allocated output tensor (on CUDA)
        c_sq      : (B, K)    float32 – (option) ||centroids||^2 per centroid (on CUDA)

    Returns:
        cluster_ids (B, N) int32 (callers can cast to int64 if desired)
    Extra:
        config        : {"BLOCK_N","BLOCK_K","num_warps","num_stages"} to force a config
        use_heuristic : use a fixed heuristic config instead of autotune
    """
    assert x.is_cuda and centroids.is_cuda and x_sq.is_cuda, "All tensors must be on CUDA"
    # assert x.dtype in (torch.float16, torch.float32), "x must be fp16/fp32"
    assert centroids.dtype == x.dtype, "centroids dtype mismatch"

    B, N, D = x.shape
    K = centroids.shape[1]
    assert centroids.shape == (B, K, D), "centroids shape mismatch"
    assert x_sq.shape == (B, N), "x_sq shape mismatch"

    # x = x.contiguous()
    # centroids = centroids.contiguous()
    # x_sq = x_sq.contiguous()

    if out is None:
        out = torch.empty((B, N), device=x.device, dtype=torch.int32)
    if c_sq is None:
        c_sq = (centroids.to(torch.float32) ** 2).sum(-1)

    # Strides (in elements)
    stride_x_b, stride_x_n, stride_x_d = x.stride()
    stride_c_b, stride_c_k, stride_c_d = centroids.stride()
    stride_xsq_b, stride_xsq_n = x_sq.stride()
    stride_csq_b, stride_csq_k = c_sq.stride()
    stride_out_b, stride_out_n = out.stride()

    grid = lambda META: (triton.cdiv(N, META["BLOCK_N"]), B)

    selected_config = None
    if config is not None:
        selected_config = config
    elif num_warps is not None or num_stages is not None:
        if num_warps is None or num_stages is None:
            raise ValueError("num_warps and num_stages must be set together")
        selected_config = {
            "BLOCK_N": BLOCK_N,
            "BLOCK_K": BLOCK_K,
            "num_warps": num_warps,
            "num_stages": num_stages,
        }
    elif use_heuristic:
        selected_config = _heuristic_euclid_config(
            N, K, D, device=x.device, dtype=x.dtype
        )

    if selected_config is not None:
        _euclid_assign_kernel[grid](
            x,
            centroids,
            x_sq,
            c_sq,
            out,
            B,
            N,
            K,
            D,
            stride_x_b,
            stride_x_n,
            stride_x_d,
            stride_c_b,
            stride_c_k,
            stride_c_d,
            stride_xsq_b,
            stride_xsq_n,
            stride_csq_b,
            stride_csq_k,
            stride_out_b,
            stride_out_n,
            BLOCK_N=selected_config["BLOCK_N"],
            BLOCK_K=selected_config["BLOCK_K"],
            num_warps=selected_config["num_warps"],
            num_stages=selected_config["num_stages"],
        )
    else:
        _euclid_assign_kernel_autotuned[grid](
            x,
            centroids,
            x_sq,
            c_sq,
            out,
            B,
            N,
            K,
            D,
            stride_x_b,
            stride_x_n,
            stride_x_d,
            stride_c_b,
            stride_c_k,
            stride_c_d,
            stride_xsq_b,
            stride_xsq_n,
            stride_csq_b,
            stride_csq_k,
            stride_out_b,
            stride_out_n,
        )
    return out


def cosine_assign_triton(x: torch.Tensor, centroids: torch.Tensor, out: torch.Tensor = None,
                         *, BLOCK_N: int = 128, BLOCK_K: int = 128) -> torch.Tensor:
    """Return nearest(cosine similarity)-centroid indices using Triton kernel.

    Args:
        x         : (B, N, D) float16 / float32 (on CUDA)
        centroids : (B, K, D) same dtype/device as x

    Returns:
        cluster_ids (B, N) int32 (callers can cast to int64 if desired)
    """
    assert x.is_cuda and centroids.is_cuda, "All tensors must be on CUDA"
    # assert x.dtype in (torch.float16, torch.float32), "x must be fp16/fp32"
    assert centroids.dtype == x.dtype, "centroids dtype mismatch"

    B, N, D = x.shape
    K = centroids.shape[1]
    assert centroids.shape == (B, K, D), "centroids shape mismatch"

    # x = x.contiguous()
    # centroids = centroids.contiguous()
    # x_sq = x_sq.contiguous()

    if out is None:
        out = torch.empty((B, N), device=x.device, dtype=torch.int32)

    # Strides (in elements)
    stride_x_b, stride_x_n, stride_x_d = x.stride()
    stride_c_b, stride_c_k, stride_c_d = centroids.stride()
    stride_out_b, stride_out_n = out.stride()

    grid = lambda META: (triton.cdiv(N, META["BLOCK_N"]), B)

    _cosine_assign_kernel_autotuned[grid](
        x,
        centroids,
        out,
        B,
        N,
        K,
        D,
        stride_x_b,
        stride_x_n,
        stride_x_d,
        stride_c_b,
        stride_c_k,
        stride_c_d,
        stride_out_b,
        stride_out_n,
    )
    return out

# ---------------------------------------------------------------
# Quick correctness & performance check
# ---------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)

    B, N, D = 32, 74256, 128
    K = 1000
    out = torch.empty((B, N), device="cuda", dtype=torch.int32)
    dtype = torch.float16

    x = torch.randn(B, N, D, device="cuda", dtype=dtype)
    cent = torch.randn(B, K, D, device="cuda", dtype=dtype)
    x_sq = (x.to(torch.float32) ** 2).sum(-1)

    # Reference
    dist = (
        x_sq.unsqueeze(-1) + (cent.to(torch.float32) ** 2).sum(-1).unsqueeze(1) - 2.0 * torch.einsum("bnd,bkd->bnk", x, cent).to(torch.float32)
    ).clamp_min_(0.0)
    ref_ids = dist.argmin(dim=-1)

    tri_ids = euclid_assign_triton(x, cent, x_sq, out)

    print("Correct:", torch.equal(ref_ids.cpu(), tri_ids.cpu()))


    dist_cos = torch.einsum("bnd,bkd->bnk", x.to(torch.float32), cent.to(torch.float32))
    ref_ids_cos = dist_cos.argmax(dim=-1)
    tri_ids_cos = cosine_assign_triton(x, cent, out)

    print("Cosine Correct:", torch.equal(ref_ids_cos.cpu(), tri_ids_cos.cpu()))

    # Simple timing
    repeats = 20
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        euclid_assign_triton(x, cent, x_sq, out)
    end.record(); torch.cuda.synchronize()
    print(f"Avg time Triton: {start.elapsed_time(end)/repeats:.3f} ms for {B}x{N} points vs {K} centroids") 
    print(f"{ref_ids[10, 69344]=}, {tri_ids[10, 69344]=}, {dist[10, 69344, ref_ids[10, 69344]]=}, {dist[10, 69344, tri_ids[10, 69344]]=}")
    try:
        torch.testing.assert_close(ref_ids, tri_ids.to(ref_ids.dtype))
    except Exception as e:
        print("Assertion failed:", e)

    start.record()
    for _ in range(repeats):
        cosine_assign_triton(x, cent, out)
    end.record(); torch.cuda.synchronize()
    print(f"Avg time Triton Cosine: {start.elapsed_time(end)/repeats:.3f} ms for {B}x{N} points vs {K} centroids") 
    print(f"{ref_ids_cos[10, 69344]=}, {tri_ids_cos[10, 69344]=}, {dist_cos[10, 69344, ref_ids_cos[10, 69344]]=}, {dist_cos[10, 69344, tri_ids_cos[10, 69344]]=}")
    try:
        torch.testing.assert_close(ref_ids_cos, tri_ids_cos.to(ref_ids_cos.dtype))
    except Exception as e:
        print("Assertion failed:", e)
