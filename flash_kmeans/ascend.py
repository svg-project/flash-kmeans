"""Euclidean Lloyd iterations on Ascend with streaming assignment."""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _assign_direct(X, C, Out, N: tl.constexpr, D: tl.constexpr, K: tl.constexpr,
            BN: tl.constexpr, BK: tl.constexpr, BD: tl.constexpr):
    b = tl.program_id(1)
    n = tl.program_id(0) * BN + tl.arange(0, BN)
    d = tl.arange(0, BD)
    x = tl.load(X + (b * N + n[:, None]) * D + d[None, :],
                (n[:, None] < N) & (d[None, :] < D), 0)
    best = tl.full((BN,), float("inf"), tl.float32)
    best_id = tl.zeros((BN,), tl.int32)
    for start in range(tl.cdiv(K, BK)):
        k = start * BK + tl.arange(0, BK)
        c = tl.load(C + (b * K + k[None, :]) * D + d[:, None],
                    (k[None, :] < K) & (d[:, None] < D), 0)
        c32 = c.to(tl.float32)
        scores = tl.sum(c32 * c32, axis=0)[None, :] - 2.0 * tl.dot(x, c)
        scores = tl.where(k[None, :] < K, scores, float("inf"))
        value = tl.min(scores, axis=1)
        index = tl.min(tl.where(scores == value[:, None], k[None, :], 2147483647), axis=1)
        take = value < best
        best = tl.where(take, value, best)
        best_id = tl.where(take, index, best_id)
    tl.store(Out + b * N + n, best_id, n < N)


@triton.jit
def _assign_packed(X, C, Norm, Out, N: tl.constexpr, D: tl.constexpr, K: tl.constexpr,
                   KP: tl.constexpr,
                   BN: tl.constexpr, BK: tl.constexpr, BD: tl.constexpr):
    b = tl.program_id(1)
    d = tl.arange(0, BD)
    for tile in range(tl.program_id(0), tl.cdiv(N, BN), tl.num_programs(0)):
        n = tile * BN + tl.arange(0, BN)
        x = tl.load(X + (b * N + n[:, None]) * D + d[None, :],
                    (n[:, None] < N) & (d[None, :] < D), 0)
        best = tl.full((BN,), float("inf"), tl.float32)
        best_id = tl.zeros((BN,), tl.int32)
        for start in range(tl.cdiv(K, BK)):
            k = start * BK + tl.arange(0, BK)
            c = tl.load(C + b * D * KP + d[:, None] * KP + k[None, :],
                        (k[None, :] < KP) & (d[:, None] < D), 0)
            norms = tl.load(Norm + b * KP + k, k < KP, float("inf"))
            scores = norms[None, :] - 2.0 * tl.dot(x, c)
            value, index = tl.min(scores, axis=1, return_indices=True)
            index += start * BK
            take = value < best
            best = tl.where(take, value, best)
            best_id = tl.where(take, index, best_id)
        tl.store(Out + b * N + n, best_id, n < N)


@triton.jit
def _partial(X, Idx, Labels, Sum, Count, N: tl.constexpr, D: tl.constexpr,
             K: tl.constexpr, S: tl.constexpr, BN: tl.constexpr, BD: tl.constexpr):
    tile = tl.program_id(0)
    b = tl.program_id(1)
    stripe = tile % S
    n = tile * BN + tl.arange(0, BN)
    ids = tl.load(Labels + b * N + n, n < N, -1)
    rows = tl.load(Idx + b * N + n, n < N, 0)
    first = tl.load(Labels + b * N + tile * BN)
    last = tl.load(Labels + b * N + tl.minimum((tile + 1) * BN, N) - 1)
    for ds in range(tl.cdiv(D, BD)):
        d = ds * BD + tl.arange(0, BD)
        values = tl.load(X + (b * N + rows[:, None]) * D + d[None, :],
                         (n[:, None] < N) & (d[None, :] < D), 0).to(tl.float32)
        for cid in range(first, last + 1):
            mask = ids == cid
            total = tl.sum(tl.where(mask[:, None], values, 0), axis=0)
            tl.atomic_add(Sum + ((b * S + stripe) * K + cid) * D + d, total, d < D)
            if ds == 0:
                count = tl.sum(mask.to(tl.int32), axis=0)
                tl.atomic_add(Count + (b * S + stripe) * K + cid, count)


@triton.jit
def _finish(Sum, Count, Old, Out, D: tl.constexpr, K: tl.constexpr,
            S: tl.constexpr, BD: tl.constexpr):
    k = tl.program_id(0)
    b = tl.program_id(1)
    ds = tl.program_id(2)
    d = ds * BD + tl.arange(0, BD)
    s = tl.arange(0, S)
    sums = tl.load(Sum + ((b * S + s[:, None]) * K + k) * D + d[None, :],
                   d[None, :] < D, 0)
    counts = tl.load(Count + (b * S + s) * K + k)
    count = tl.sum(counts, axis=0)
    value = tl.sum(sums, axis=0) / tl.maximum(count, 1)
    old = tl.load(Old + (b * K + k) * D + d, d < D, 0)
    tl.store(Out + (b * K + k) * D + d, tl.where(count > 0, value, old), d < D)


def assign(x, centroids, *, block_n=64, block_k=64, packed=True):
    """Assign nearest centers without a global N-by-K distance tensor.

    Squared point norms are constant across centers and are omitted. Center
    norms and dot products accumulate in FP32; ties choose the first center.
    """
    b, n, d = x.shape
    k = centroids.shape[1]
    x, centroids = x.contiguous(), centroids.contiguous()
    out = torch.empty((b, n), device=x.device, dtype=torch.int32)
    if packed:
        kp = triton.cdiv(k, block_k) * block_k
        norms = F.pad(centroids.float().square().sum(-1), (0, kp - k), value=float("inf"))
        packed_centers = F.pad(centroids.transpose(1, 2), (0, kp - k)).contiguous()
        _assign_packed[(min(64, triton.cdiv(n, block_n)), b)](
            x, packed_centers, norms, out, n, d, k, kp, block_n, block_k,
            max(16, triton.next_power_of_2(d)))
    else:
        batch_chunk = max(1, 65535 // triton.cdiv(n, 16))
        for bs in range(0, b, batch_chunk):
            size = min(batch_chunk, b - bs)
            _assign_direct[(triton.cdiv(n, 16), size)](
                x[bs:bs + size], centroids[bs:bs + size], out[bs:bs + size],
                n, d, k, 16, 32, max(16, triton.next_power_of_2(d)))
    return out


def update(x, labels, old_centroids, *, stripes=1, block_n=128):
    """Reduce sorted token chunks into striped FP32 sums, then combine stripes.

    Fixed token chunks balance large clusters across programs. Stripes reduce
    atomic contention when many chunks contribute to the same cluster.
    """
    b, n, d = x.shape
    k = old_centroids.shape[1]
    x, old_centroids = x.contiguous(), old_centroids.contiguous()
    # Native floating-point sort is substantially faster on Ascend. Every
    # cluster ID in this range is represented exactly by FP32.
    sortable = labels.float() if k <= 2**24 else labels
    sorted_labels, indices = torch.sort(sortable, dim=-1)
    sorted_labels = sorted_labels.to(torch.int32)
    sums = torch.zeros((b, stripes, k, d), device=x.device, dtype=torch.float32)
    counts = torch.zeros((b, stripes, k), device=x.device, dtype=torch.int32)
    out = torch.empty_like(old_centroids)
    _partial[(triton.cdiv(n, block_n), b)](
        x, indices, sorted_labels, sums, counts, n, d, k, stripes, block_n, 32)
    _finish[(k, b, triton.cdiv(d, 128))](sums, counts, old_centroids, out, d, k, stripes, 128)
    return out


def batch_kmeans_Euclid(x, n_clusters, max_iters=100, tol=0.0,
                        init_centroids=None, verbose=False, *, use_heuristic=True):
    """Ascend backend with the upstream initialization and stopping behavior."""
    b, n, d = x.shape
    if x.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("Ascend streaming k-means supports float16 and bfloat16 inputs")
    if init_centroids is None:
        indices = torch.randint(0, n, (b, n_clusters), device=x.device)
        centroids = torch.gather(x, 1, indices[..., None].expand(-1, -1, d))
    else:
        centroids = init_centroids
    centroids = centroids.view(b, n_clusters, d)
    for it in range(max_iters):
        labels = assign(x, centroids)
        new = update(x, labels, centroids)
        shift = (new - centroids).norm(dim=-1).max()
        if verbose:
            print(f"Iter {it}, center shift: {shift.item():.6f}")
        if shift < tol:
            break
        centroids = new.clone()
    return labels, centroids, it + 1
