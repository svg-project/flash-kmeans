import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from tqdm import trange


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _next_pow2(v: int) -> int:
    if v <= 1:
        return 1
    return 1 << (v - 1).bit_length()


def _pad_d(D: int) -> int:
    """Pad D up to the next power of two (min 16) for kernels that load the
    whole feature dimension as a single tile. ``tl.arange`` requires a
    power-of-two range, so non-power-of-two D (e.g. 192, 320, 768) needs to
    be expanded and the tail masked out at load/store time.
    """
    return max(16, _next_pow2(D))


def _dtype_bytes(dtype) -> int:
    """Element size in bytes for a torch dtype (fallback 4 for the unknown)."""
    if isinstance(dtype, torch.dtype):
        return torch.tensor([], dtype=dtype).element_size()
    if isinstance(dtype, int):
        return dtype
    return 4


# Per-program feature-tile budget (bytes) for the chunk kernel. The kernel
# materialises a ``[BLOCK_N, BLOCK_D]`` feature tile in registers/local memory
# for each cluster run; capping its byte size keeps the driver's per-launch
# local-memory scratch bounded. Without this cap a large D combined with a
# wide dtype (fp32/fp64) forces a huge ``[BLOCK_N, next_pow2(D)]`` tile that
# spills heavily and can fail ``cuLaunchKernel`` with CUDA OOM (issue #19).
# 256 KiB per program keeps D<=512 (all dtypes) and D<=1024 (fp16) in a single
# D pass -- byte-for-byte identical to the pre-split-D kernel -- while tiling
# only the genuinely large tiles that used to spill.
_CHUNK_TILE_BUDGET_BYTES = 256 * 1024


def _choose_block_d(D: int, BLOCK_N: int, dtype_bytes: int,
                    budget_bytes: int = _CHUNK_TILE_BUDGET_BYTES) -> int:
    """Pick the feature-dim tile ``BLOCK_D`` for ``_centroid_update_chunk_kernel``.

    Returns ``_pad_d(D)`` (a single D pass, identical to the original kernel)
    whenever the full padded feature vector fits the per-program tile budget.
    Otherwise returns the largest power-of-two ``BLOCK_D`` (>= 16) whose
    ``[BLOCK_N, BLOCK_D]`` tile fits the budget, so the inner D loop keeps the
    register/local-memory footprint bounded regardless of D or dtype.
    """
    D_pad = _pad_d(D)
    if BLOCK_N * D_pad * dtype_bytes <= budget_bytes:
        return D_pad
    bd = D_pad
    while bd > 16 and BLOCK_N * bd * dtype_bytes > budget_bytes:
        bd //= 2
    return max(bd, 16)


@triton.jit
def _centroid_update_kernel(
    x_ptr,                # *f16 / *f32 [B, N, D]
    cluster_ptr,          # *i32        [B, N]
    sum_ptr,              # *f32        [B, K, D]
    count_ptr,            # *i32        [B, K]
    # --- strides (elements) ---
    stride_x_b, stride_x_n, stride_x_d,
    stride_sum_b, stride_sum_k, stride_sum_d,
    stride_count_b, stride_count_k,
    B: tl.constexpr,
    N: tl.constexpr,
    D: tl.constexpr,
    K: tl.constexpr,
    BLOCK_D: tl.constexpr,   # number of dims processed per program
):
    """Each program processes 1 token across BLOCK_D dims using atomics with general strides."""
    pid = tl.program_id(axis=0)
    token_idx = pid  # range: [0, B*N)

    # Derive (b, n)
    b = (token_idx // N).to(tl.int64)
    n = (token_idx % N).to(tl.int64)

    # pointer to this token's feature vector
    x_offset = b * stride_x_b + n * stride_x_n
    x_tok_ptr = x_ptr + x_offset

    cluster_idx = tl.load(cluster_ptr + b * N + n)
    cluster_idx = tl.where(cluster_idx < K, cluster_idx, 0)
    cluster_idx = cluster_idx.to(tl.int64)

    # base ptr for centroid accum array
    centroid_base = b * stride_sum_b + cluster_idx * stride_sum_k

    offs = tl.arange(0, BLOCK_D).to(tl.int64)
    for d_start in range(0, D, BLOCK_D):
        mask = offs + d_start < D
        feats = tl.load(x_tok_ptr + (d_start + offs) * stride_x_d, mask=mask, other=0.0)
        feats = feats.to(tl.float32)
        dest_ptr = sum_ptr + centroid_base + (d_start + offs) * stride_sum_d
        tl.atomic_add(dest_ptr, feats, mask=mask)

    tl.atomic_add(count_ptr + b * stride_count_b + cluster_idx * stride_count_k, 1)


def triton_centroid_update_cosine(x_norm: torch.Tensor, cluster_ids: torch.Tensor, old_centroids: torch.Tensor):
    """Compute centroids using custom Triton kernel.

    Args:
        x_norm (Tensor): (B, N, D) normalized input vectors (float16/float32)
        cluster_ids (LongTensor): (B, N) cluster assignment per point
        old_centroids (Tensor): (B, K, D) previous centroids (same dtype as x_norm)

    Returns:
        Tensor: (B, K, D) updated and L2-normalized centroids (dtype == x_norm.dtype)
    """
    assert x_norm.is_cuda and cluster_ids.is_cuda, "Input tensors must be on CUDA device"
    B, N, D = x_norm.shape
    K = old_centroids.shape[1]
    assert cluster_ids.shape == (B, N)

    # Allocate accumulation buffers
    centroid_sums = torch.zeros((B, K, D), device=x_norm.device, dtype=torch.float32)
    centroid_counts = torch.zeros((B, K), device=x_norm.device, dtype=torch.int32)

    # Launch Triton kernel – one program per token
    total_tokens = B * N
    BLOCK_D = 128  # tuneable

    grid = (total_tokens,)
    _centroid_update_kernel[grid](
        x_norm,
        cluster_ids.to(torch.int32),
        centroid_sums,
        centroid_counts,
        x_norm.stride(0), x_norm.stride(1), x_norm.stride(2),
        centroid_sums.stride(0), centroid_sums.stride(1), centroid_sums.stride(2),
        centroid_counts.stride(0), centroid_counts.stride(1),
        B, N, D, K,
        BLOCK_D=BLOCK_D,
    )

    # Compute means; keep old centroid if empty cluster
    counts_f = centroid_counts.to(torch.float32).unsqueeze(-1).clamp(min=1.0)
    centroids = centroid_sums / counts_f

    # For clusters with zero count, revert to old centroids
    zero_mask = (centroid_counts == 0).unsqueeze(-1)
    centroids = torch.where(zero_mask, old_centroids.to(torch.float32), centroids)

    centroids = centroids.to(x_norm.dtype)
    centroids = F.normalize(centroids, p=2, dim=-1)
    return centroids


def torch_loop_centroid_update_cosine(x_norm: torch.Tensor, cluster_ids: torch.Tensor, old_centroids: torch.Tensor):
    """Reference Python implementation (double for-loop)"""
    B, N, D = x_norm.shape
    K = old_centroids.shape[1]
    new_centroids = torch.zeros_like(old_centroids)
    for b in range(B):
        for k in range(K):
            mask = cluster_ids[b] == k
            if mask.any():
                new_centroids[b, k] = F.normalize(x_norm[b][mask].mean(dim=0, dtype=x_norm.dtype), p=2, dim=0)
            else:
                new_centroids[b, k] = old_centroids[b, k]
    return new_centroids


def triton_centroid_update_euclid(x: torch.Tensor, cluster_ids: torch.Tensor, old_centroids: torch.Tensor):
    """Compute centroids for Euclidean KMeans using Triton.

    Args:
        x (Tensor): (B, N, D) input vectors (float16/float32)
        cluster_ids (LongTensor): (B, N) cluster assignment per point
        old_centroids (Tensor): (B, K, D) previous centroids (same dtype as x)

    Returns:
        Tensor: (B, K, D) updated centroids (dtype == x.dtype)
    """
    assert x.is_cuda and cluster_ids.is_cuda, "Input tensors must be on CUDA device"
    B, N, D = x.shape
    K = old_centroids.shape[1]
    assert cluster_ids.shape == (B, N)

    # Allocate accumulation buffers
    centroid_sums = torch.zeros((B, K, D), device=x.device, dtype=torch.float32)
    centroid_counts = torch.zeros((B, K), device=x.device, dtype=torch.int32)

    total_tokens = B * N
    BLOCK_D = 128  # tuneable
    grid = (total_tokens,)

    _centroid_update_kernel[grid](
        x,
        cluster_ids.to(torch.int32),
        centroid_sums,
        centroid_counts,
        x.stride(0), x.stride(1), x.stride(2),
        centroid_sums.stride(0), centroid_sums.stride(1), centroid_sums.stride(2),
        centroid_counts.stride(0), centroid_counts.stride(1),
        B, N, D, K,
        BLOCK_D=BLOCK_D,
    )

    # Compute means; keep old centroid if empty cluster
    counts_f = centroid_counts.to(torch.float32).unsqueeze(-1).clamp(min=1.0)
    centroids = centroid_sums / counts_f

    # For clusters with zero count, revert to old centroids
    zero_mask = (centroid_counts == 0).unsqueeze(-1)
    centroids = torch.where(zero_mask, old_centroids.to(torch.float32), centroids)

    return centroids.to(x.dtype)


# ------------------------------ NEW: chunk-wise centroid update (sorted ids) ------------------------------

@triton.jit
def _centroid_update_chunk_kernel(
    x_ptr,                # *f16 / *f32 [B, N, D] – ORIGINAL ORDER
    sorted_idx_ptr,       # *i32        [B, N]    – indices after sort
    sorted_cluster_ptr,   # *i32        [B, N]    – cluster ids in sorted order
    sum_ptr,              # *f32        [B, K, D]
    count_ptr,            # *i32        [B, K]
    # strides
    stride_x_b, stride_x_n, stride_x_d,
    stride_idx_b, stride_idx_n, stride_cluster_b, stride_cluster_n,
    stride_sum_b, stride_sum_k, stride_sum_d,
    stride_count_b, stride_count_k,
    B: tl.constexpr,
    N: tl.constexpr,
    D: tl.constexpr,
    K: tl.constexpr,
    BLOCK_N: tl.constexpr,   # how many tokens (points) each program processes
    BLOCK_D: tl.constexpr,   # feature-dim tile; inner loop streams D in BLOCK_D chunks
):
    """Each program processes **BLOCK_N consecutive, already-sorted tokens**.

    Because the tokens are sorted by cluster id, identical ids appear in
    contiguous runs.  We therefore accumulate a local sum/count for the
    current run and perform **a single atomic update per run**, instead of
    per-token.

    The feature dimension is streamed in ``BLOCK_D`` chunks (split-D), so the
    per-program ``[BLOCK_N, BLOCK_D]`` tile — and therefore the register /
    local-memory footprint — is bounded by ``BLOCK_D`` rather than D. This
    avoids the launch-time OOM that the old single-tile ``[BLOCK_N, next_pow2(D)]``
    load hit for large D + wide dtypes (issue #19), and drops the
    next-power-of-two padding waste (the inner loop iterates over the real D and
    masks only the final partial tile).
    """
    # FIX: 1D flattened grid — no 65535 limit on either B or N.
    flat_id = tl.program_id(axis=0)
    n_tiles = tl.cdiv(N, BLOCK_N)
    pid_b     = flat_id // n_tiles
    pid_chunk = flat_id % n_tiles

    b = pid_b.to(tl.int64)
    chunk_start = (pid_chunk * BLOCK_N).to(tl.int64)

    # Nothing to do – out of range
    if chunk_start >= N:
        return

    # base pointers for this batch
    idx_batch_base     = sorted_idx_ptr + b * stride_idx_b
    cid_batch_base     = sorted_cluster_ptr + b * stride_cluster_b
    x_batch_base       = x_ptr + b * stride_x_b  # for pointer arithmetic

    offs_token = tl.arange(0, BLOCK_N).to(tl.int64)
    # Hoisted feature-lane base; the (constexpr) d_start offset is added per
    # D-tile below. Keeping the arange out of the runtime cluster loop avoids
    # rebuilding it every iteration (matters for the single-pass small-D case).
    base_dim   = tl.arange(0, BLOCK_D).to(tl.int64)

    # first token index & validity mask
    token_idx  = chunk_start + offs_token
    valid_tok  = token_idx < N
    first_token_idx = chunk_start
    last_token_idx = tl.minimum(chunk_start + BLOCK_N, N) - 1

    # Load first cluster id to initialise the running accumulator
    first_id = tl.load(cid_batch_base + first_token_idx)
    last_id = tl.load(cid_batch_base + last_token_idx)
    all_ids = tl.load(cid_batch_base + token_idx * stride_cluster_n, mask=valid_tok, other=-1)

    all_tokens_idxs = tl.load(idx_batch_base + token_idx * stride_idx_n, mask=valid_tok, other=-1) # [BLOCK_N]
    all_tokens_idxs = all_tokens_idxs.to(tl.int64)

    for cid in range(first_id, last_id + 1):
        cluster_mask = all_ids == cid
        cluster_size = tl.sum(cluster_mask.to(tl.int32))
        if cluster_size != 0:
            for d_start in range(0, D, BLOCK_D):
                offs_dim = d_start + base_dim
                d_mask = offs_dim < D
                row_ptrs = x_batch_base + all_tokens_idxs[:, None] * stride_x_n + offs_dim[None, :] * stride_x_d
                cluster_feats = tl.load(
                    row_ptrs,
                    mask=cluster_mask[:, None] & d_mask[None, :],
                    other=0.0,
                )  # [BLOCK_N, BLOCK_D]
                cluster_feats = cluster_feats.to(tl.float32)
                sum_feats = tl.sum(cluster_feats, axis=0)  # [BLOCK_D]
                dest_ptr = sum_ptr + b * stride_sum_b + cid * stride_sum_k + offs_dim * stride_sum_d
                tl.atomic_add(dest_ptr, sum_feats, mask=d_mask)
            tl.atomic_add(count_ptr + b * stride_count_b + cid * stride_count_k, cluster_size)


# ---------------------------------------------------------------------------------------------

def triton_centroid_update_sorted_cosine(x_norm: torch.Tensor, cluster_ids: torch.Tensor, old_centroids: torch.Tensor,
                                         *, BLOCK_N: int = 256):
    """Fast centroid update assuming **cluster_ids are sorted along N**.

    This helper will sort the assignments (together with `x_norm`) and launch the
    chunk kernel above.  Compared to the naive per-token kernel it performs *one
    atomic add per run of identical ids* instead of per token, providing large
    speed-ups when clusters are reasonably sized.
    """
    assert x_norm.is_cuda and cluster_ids.is_cuda, "Inputs must be on CUDA"
    B, N, D = x_norm.shape
    K = old_centroids.shape[1]
    assert cluster_ids.shape == (B, N)

    # -------- sort per-batch --------
    sorted_cluster_ids, sorted_idx = torch.sort(cluster_ids, dim=-1)
    sorted_idx_int = sorted_idx.to(torch.int32)

    # accumulation buffers
    centroid_sums = torch.zeros((B, K, D), device=x_norm.device, dtype=torch.float32)
    centroid_cnts = torch.zeros((B, K),    device=x_norm.device, dtype=torch.int32)

    grid = (B * triton.cdiv(N, BLOCK_N),)
    _centroid_update_chunk_kernel[grid](
        x_norm,
        sorted_idx_int,
        sorted_cluster_ids.to(torch.int32),
        centroid_sums,
        centroid_cnts,
        x_norm.stride(0), x_norm.stride(1), x_norm.stride(2),
        sorted_idx_int.stride(0), sorted_idx_int.stride(1),
        sorted_cluster_ids.stride(0), sorted_cluster_ids.stride(1),
        centroid_sums.stride(0), centroid_sums.stride(1), centroid_sums.stride(2),
        centroid_cnts.stride(0), centroid_cnts.stride(1),
        B, N, D, K,
        BLOCK_N=BLOCK_N,
        BLOCK_D=_choose_block_d(D, BLOCK_N, _dtype_bytes(x_norm.dtype)),
    )

    # finalise – convert to means, handle empty clusters, renormalise
    counts_f = centroid_cnts.to(torch.float32).unsqueeze(-1).clamp(min=1.0)
    centroids = centroid_sums / counts_f
    empty_mask = (centroid_cnts == 0).unsqueeze(-1)
    centroids = torch.where(empty_mask, old_centroids.to(torch.float32), centroids)
    centroids = centroids.to(x_norm.dtype)
    centroids = F.normalize(centroids, p=2, dim=-1)
    return centroids

def triton_centroid_update_sorted_euclid(x: torch.Tensor, cluster_ids: torch.Tensor, old_centroids: torch.Tensor,
                                         *, BLOCK_N: int = 256, centroid_sums: torch.Tensor = None, centroid_cnts: torch.Tensor = None, calculate_new: bool = True):
    """Fast centroid update for *Euclidean* KMeans assuming cluster IDs are pre-sorted.

    Parameters
    ----------
    x : Tensor [B, N, D]
        Input feature vectors (no normalization assumed).
    cluster_ids : LongTensor [B, N]
        Cluster assignment for each point.
    old_centroids : Tensor [B, K, D]
        Previous centroids (used to fill empty clusters).
    BLOCK_N : int, optional
        Tokens per Triton program (affects occupancy/perf).
    centroid_sums : Tensor [B, K, D], optional
        Pre-allocated accumulation buffer for sums.  If None, a new buffer is created.
    centroid_cnts : Tensor [B, K], optional
        Pre-allocated accumulation buffer for counts.  If None, a new buffer is created.
    calculate_new : bool, default=True
        If True, compute and return the new centroids.  If False, only update the
        accumulation buffers.

    Returns
    _________
        centroids_new : Tensor [B, K, D] or None
            Updated centroids if `calculate_new` is True; otherwise None.
    """
    assert x.is_cuda and cluster_ids.is_cuda, "Inputs must be on CUDA device"
    B, N, D = x.shape
    K = old_centroids.shape[1]

    # Batch-wise sort of cluster assignments
    sorted_cluster_ids, sorted_idx = torch.sort(cluster_ids, dim=-1)
    sorted_idx_int = sorted_idx.to(torch.int32)

    if centroid_sums is None:
        centroid_sums = torch.zeros((B, K, D), device=x.device, dtype=torch.float32)
    else:
        assert centroid_sums.shape == (B, K, D)
    
    if centroid_cnts is None:
        centroid_cnts = torch.zeros((B, K),    device=x.device, dtype=torch.int32)
    else:
        assert centroid_cnts.shape == (B, K)

    grid = (B * triton.cdiv(N, BLOCK_N),)
    _centroid_update_chunk_kernel[grid](
        x,                       # original features
        sorted_idx_int,          # gather indices
        sorted_cluster_ids.to(torch.int32),
        centroid_sums,
        centroid_cnts,
        x.stride(0), x.stride(1), x.stride(2),
        sorted_idx_int.stride(0), sorted_idx_int.stride(1),
        sorted_cluster_ids.stride(0), sorted_cluster_ids.stride(1),
        centroid_sums.stride(0), centroid_sums.stride(1), centroid_sums.stride(2),
        centroid_cnts.stride(0), centroid_cnts.stride(1),
        B, N, D, K,
        BLOCK_N=BLOCK_N,
        BLOCK_D=_choose_block_d(D, BLOCK_N, _dtype_bytes(x.dtype)),
    )

    if calculate_new:
        # Convert sums to means; replace empty clusters with old centroids
        counts_f = centroid_cnts.to(torch.float32).unsqueeze(-1).clamp(min=1.0)
        centroids = centroid_sums / counts_f
        empty_mask = (centroid_cnts == 0).unsqueeze(-1)
        centroids = torch.where(empty_mask, old_centroids.to(torch.float32), centroids)
        return centroids.to(x.dtype)
    else:
        return None
# ------------------------------ END new implementation ------------------------------


@triton.jit
def _centroid_update_chunk_weighted_kernel(
    x_ptr,                # *f16 / *f32 [B, N, D] – ORIGINAL ORDER
    sorted_idx_ptr,       # *i32        [B, N]
    sorted_cluster_ptr,   # *i32        [B, N]
    weight_ptr,           # *f32        [B, N] – per-point weights in ORIGINAL order
    sum_ptr,              # *f32        [B, K, D]
    weight_sum_ptr,       # *f32        [B, K]
    # strides
    stride_x_b, stride_x_n, stride_x_d,
    stride_idx_b, stride_idx_n, stride_cluster_b, stride_cluster_n,
    stride_w_b, stride_w_n,
    stride_sum_b, stride_sum_k, stride_sum_d,
    stride_ws_b, stride_ws_k,
    B: tl.constexpr,
    N: tl.constexpr,
    D: tl.constexpr,
    K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Weighted variant of _centroid_update_chunk_kernel.

    Multiplies features by per-point weight inside the kernel to avoid
    materialising a full weighted-data copy on the host side.
    Accumulates float32 weight sums instead of int counts.

    The feature dimension is streamed in ``BLOCK_D`` chunks (split-D, matching
    ``_centroid_update_chunk_kernel``), so the per-program ``[BLOCK_N, BLOCK_D]``
    tile is bounded regardless of D or dtype, and non-power-of-two D is handled
    by masking the final partial tile rather than requiring a power-of-two
    ``tl.arange(0, D)``.
    """
    # FIX: 1D flattened grid — no 65535 limit on either B or N.
    flat_id = tl.program_id(axis=0)
    n_tiles = tl.cdiv(N, BLOCK_N)
    pid_b     = flat_id // n_tiles
    pid_chunk = flat_id % n_tiles

    b = pid_b.to(tl.int64)
    chunk_start = (pid_chunk * BLOCK_N).to(tl.int64)

    if chunk_start >= N:
        return

    idx_batch_base = sorted_idx_ptr + b * stride_idx_b
    cid_batch_base = sorted_cluster_ptr + b * stride_cluster_b
    x_batch_base   = x_ptr + b * stride_x_b
    w_batch_base   = weight_ptr + b * stride_w_b

    offs_token = tl.arange(0, BLOCK_N).to(tl.int64)
    # Hoisted feature-lane base; the (constexpr) d_start offset is added per
    # D-tile below (see _centroid_update_chunk_kernel).
    base_dim   = tl.arange(0, BLOCK_D).to(tl.int64)

    token_idx  = chunk_start + offs_token
    valid_tok  = token_idx < N
    first_token_idx = chunk_start
    last_token_idx  = tl.minimum(chunk_start + BLOCK_N, N) - 1

    first_id = tl.load(cid_batch_base + first_token_idx)
    last_id  = tl.load(cid_batch_base + last_token_idx)
    all_ids  = tl.load(cid_batch_base + token_idx * stride_cluster_n,
                       mask=valid_tok, other=-1)

    all_tokens_idxs = tl.load(idx_batch_base + token_idx * stride_idx_n,
                              mask=valid_tok, other=-1)
    all_tokens_idxs = all_tokens_idxs.to(tl.int64)

    # Load per-point weights (original order) once for the whole chunk
    all_weights = tl.load(w_batch_base + all_tokens_idxs * stride_w_n,
                          mask=valid_tok, other=0.0)
    all_weights = all_weights.to(tl.float32)

    for cid in range(first_id, last_id + 1):
        cluster_mask = all_ids == cid
        cluster_size = tl.sum(cluster_mask.to(tl.int32))
        if cluster_size != 0:
            # Per-token weights for this cluster (D-independent; computed once).
            cluster_weights = tl.where(cluster_mask, all_weights, 0.0)
            for d_start in range(0, D, BLOCK_D):
                offs_dim = d_start + base_dim
                d_mask = offs_dim < D
                row_ptrs = (x_batch_base
                            + all_tokens_idxs[:, None] * stride_x_n
                            + offs_dim[None, :] * stride_x_d)
                cluster_feats = tl.load(
                    row_ptrs,
                    mask=cluster_mask[:, None] & d_mask[None, :],
                    other=0.0,
                )  # [BLOCK_N, BLOCK_D]
                cluster_feats = cluster_feats.to(tl.float32)

                weighted_feats = cluster_feats * cluster_weights[:, None]
                sum_feats = tl.sum(weighted_feats, axis=0)  # [BLOCK_D]

                dest_ptr = (sum_ptr + b * stride_sum_b
                            + cid * stride_sum_k + offs_dim * stride_sum_d)
                tl.atomic_add(dest_ptr, sum_feats, mask=d_mask)

            w_sum = tl.sum(cluster_weights)
            tl.atomic_add(weight_sum_ptr + b * stride_ws_b
                          + cid * stride_ws_k, w_sum)


def triton_centroid_update_sorted_euclid_weighted(
    x: torch.Tensor,
    cluster_ids: torch.Tensor,
    old_centroids: torch.Tensor,
    weights: torch.Tensor,
    *,
    BLOCK_N: int = 256,
):
    """Weighted centroid update using a dedicated Triton kernel.

    Avoids materialising a full weighted-data copy by multiplying features
    by per-point weights inside the kernel.  Also eliminates the
    fp16->fp32->fp16->fp32 precision round-trip of the old approach.

    Parameters
    ----------
    x : Tensor [B, N, D]
        Input feature vectors.
    cluster_ids : LongTensor [B, N]
        Cluster assignment for each point.
    old_centroids : Tensor [B, K, D]
        Previous centroids (used to fill empty clusters).
    weights : Tensor [B, N]
        Per-sample weights (positive).
    """
    assert x.is_cuda and cluster_ids.is_cuda and weights.is_cuda
    B, N, D = x.shape
    K = old_centroids.shape[1]

    sorted_cluster_ids, sorted_idx = torch.sort(cluster_ids, dim=-1)
    sorted_idx_int = sorted_idx.to(torch.int32)

    centroid_sums = torch.zeros((B, K, D), device=x.device, dtype=torch.float32)
    weight_sums   = torch.zeros((B, K),    device=x.device, dtype=torch.float32)
    weights_f32   = weights.float()

    grid = (B * triton.cdiv(N, BLOCK_N),)
    _centroid_update_chunk_weighted_kernel[grid](
        x,
        sorted_idx_int,
        sorted_cluster_ids.to(torch.int32),
        weights_f32,
        centroid_sums,
        weight_sums,
        x.stride(0), x.stride(1), x.stride(2),
        sorted_idx_int.stride(0), sorted_idx_int.stride(1),
        sorted_cluster_ids.stride(0), sorted_cluster_ids.stride(1),
        weights_f32.stride(0), weights_f32.stride(1),
        centroid_sums.stride(0), centroid_sums.stride(1), centroid_sums.stride(2),
        weight_sums.stride(0), weight_sums.stride(1),
        B, N, D, K,
        BLOCK_N=BLOCK_N,
        # The weighted kernel keeps two fp32 [BLOCK_N, BLOCK_D] tiles live at once
        # (cluster_feats and cluster_feats * weights), vs one in the unweighted
        # kernel, so it needs a tighter per-tile budget to avoid spilling. A
        # quarter of the shared budget caps BLOCK_D at 128 for D=256 fp16, which
        # matches the unweighted kernel's throughput (a full-D tile is ~2x slower).
        BLOCK_D=_choose_block_d(
            D, BLOCK_N, _dtype_bytes(x.dtype),
            budget_bytes=_CHUNK_TILE_BUDGET_BYTES // 4,
        ),
    )

    centroids = centroid_sums / weight_sums.unsqueeze(-1).clamp(min=1e-8)
    empty_mask = (weight_sums == 0).unsqueeze(-1)
    centroids = torch.where(empty_mask, old_centroids.float(), centroids)

    return centroids.to(x.dtype)


def main():
    torch.manual_seed(0)

    B, N, D = 32, 74256, 128  # modest sizes for quick correctness test
    K = 1000
    dtype = torch.float16

    x = torch.randn(B, N, D, device="cuda", dtype=dtype)
    x_norm = F.normalize(x, p=2, dim=-1)

    cluster_ids = torch.randint(0, K, (B, N), device="cuda", dtype=torch.int32)

    # Random old centroids for handling empty clusters
    old_centroids = F.normalize(torch.randn(B, K, D, device="cuda", dtype=dtype), p=2, dim=-1)

    # ---------------- Correctness check (compile Triton kernel) ----------------
    ref_centroids = torch_loop_centroid_update_cosine(x_norm, cluster_ids, old_centroids)
    tri_centroids = triton_centroid_update_cosine(x_norm, cluster_ids, old_centroids)  # this call triggers compilation
    tri_sorted_centroids = triton_centroid_update_sorted_cosine(x_norm, cluster_ids, old_centroids)

    # Validate correctness (includes first-run compile)
    if torch.allclose(ref_centroids, tri_centroids, atol=1e-3, rtol=1e-3):
        print("Centroid update: PASS ✅")
    else:
        max_diff = (ref_centroids - tri_centroids).abs().max().item()
        print(f"Centroid update: FAIL ❌ | max diff = {max_diff}")

    # Validate new sorted kernel
    if torch.allclose(ref_centroids, tri_sorted_centroids, atol=1e-3, rtol=1e-3):
        print("Sorted centroid update: PASS ✅")
    else:
        max_diff = (ref_centroids - tri_sorted_centroids).abs().max().item()
        print(f"Sorted centroid update: FAIL ❌ | max diff = {max_diff}")


    # show some examples
    print(f"ref_centroids[0,0:5,0:5]: {ref_centroids[0,0:5,0:5]}")
    print(f"tri_centroids[0,0:5,0:5]: {tri_centroids[0,0:5,0:5]}")
    print(f"tri_sorted_centroids[0,0:5,0:5]: {tri_sorted_centroids[0,0:5,0:5]}")

    # ---------------- Efficiency benchmark (exclude compile) ----------------
    repeats = 20

    # Torch loop timing
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in trange(repeats):
        torch_loop_centroid_update_cosine(x_norm, cluster_ids, old_centroids)
    end.record(); torch.cuda.synchronize()
    torch_time = start.elapsed_time(end) / repeats  # average per run (ms)

    # Triton timing (already compiled)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in trange(repeats):
        triton_centroid_update_cosine(x_norm, cluster_ids, old_centroids)
    end.record(); torch.cuda.synchronize()
    triton_time = start.elapsed_time(end) / repeats  # average per run (ms)

    # Sorted Triton timing (already compiled)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in trange(repeats):
        triton_centroid_update_sorted_cosine(x_norm, cluster_ids, old_centroids)
    end.record(); torch.cuda.synchronize()
    triton_sorted_time = start.elapsed_time(end) / repeats  # average per run (ms)

    print(f"\n=== Efficiency (average over {repeats} runs, exclude compile) ===")
    print(f"Torch loop   : {torch_time:.2f} ms")
    print(f"Triton kernel: {triton_time:.2f} ms (speed-up x{torch_time / triton_time:.2f})")
    print(f"Triton sorted: {triton_sorted_time:.2f} ms (speed-up x{torch_time / triton_sorted_time:.2f})")


if __name__ == "__main__":
    main()
