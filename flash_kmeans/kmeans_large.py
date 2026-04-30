from __future__ import annotations

import time
import warnings
from typing import Optional
import torch

try:
    from flash_kmeans.assign_euclid_triton import euclid_assign_triton
    from flash_kmeans.centroid_update_triton import triton_centroid_update_sorted_euclid
    from flash_kmeans.kmeans_triton_impl import batch_kmeans_Euclid

    _HAS_TRITON_IMPL = True
except Exception:
    _HAS_TRITON_IMPL = False


def _resolve_devices(device):
    """Resolve device argument to a list of torch.device objects.

    - device=None  → all available GPUs
    - device="cuda:2" or torch.device("cuda:2") → single GPU
    """
    if device is None:
        G = torch.cuda.device_count()
        if G == 0:
            raise RuntimeError("No CUDA devices available")
        return [torch.device(f"cuda:{i}") for i in range(G)]
    dev = torch.device(device)
    return [dev]


def kmeans_largeN(
    x: torch.Tensor,
    n_clusters: int,
    max_iters: int = 100,
    tol: float = 0.0,
    verbose: bool = False,
    BLOCK_N=1048576,
    init_centroids: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    flash-kmeans for large n_samples (N), x is on CPU. (too large to fit into GPU memory)
    This function will copy data from CPU to GPU in chunks of size BLOCK_N.
    overlap data transfer and computation using multiple CUDA streams.

    When device=None and multiple GPUs are available, automatically partitions
    work across all GPUs with a gather-reduce-broadcast AllReduce per iteration.

    Returns:
        cluster_ids: (N,) int32 Tensor, cluster assignment for each point.
        centroids: (n_clusters, D) final cluster centers.
    """
    devices = _resolve_devices(device)
    G = len(devices)
    dtype = dtype or x.dtype

    if not x.is_pinned() and G > 1:
        warnings.warn(
            "x is not in pinned memory. H2D transfers will not overlap with compute. "
            "Use x = x.pin_memory() for best multi-GPU performance.",
            stacklevel=2,
        )

    N, D = x.shape
    K = n_clusters
    num_blocks = (N + BLOCK_N - 1) // BLOCK_N

    # --- Data partitioning across GPUs ---
    blocks_per_gpu = [(num_blocks // G) + (1 if g < (num_blocks % G) else 0) for g in range(G)]
    block_start = [0] * G
    for g in range(1, G):
        block_start[g] = block_start[g - 1] + blocks_per_gpu[g - 1]
    point_start = [block_start[g] * BLOCK_N for g in range(G)]
    point_end = [min(point_start[g] + blocks_per_gpu[g] * BLOCK_N, N) for g in range(G)]

    # Filter out GPUs that got 0 blocks
    active_gpus = [g for g in range(G) if blocks_per_gpu[g] > 0]

    # --- Per-GPU resource allocation ---
    buf_size = 2
    work_streams = {}   # work_streams[g][0..1]
    reduce_stream = {}  # reduce_stream[g]
    centroids_g = {}    # replicated centroids on each GPU
    cluster_ids_g = {}  # local cluster_ids on each GPU
    x_sq_cache_g = {}   # cached x_sq blocks per GPU

    for g in active_gpus:
        dev = devices[g]
        with torch.cuda.device(dev):
            work_streams[g] = [torch.cuda.Stream(device=dev) for _ in range(buf_size)]
            reduce_stream[g] = torch.cuda.Stream(device=dev)
            n_points_g = point_end[g] - point_start[g]
            cluster_ids_g[g] = torch.empty((n_points_g,), device=dev, dtype=torch.int32)
            x_sq_cache_g[g] = [None] * blocks_per_gpu[g]

    # Initialize centroids on reduce_stream[active_gpus[0]]
    primary_gpu = active_gpus[0]
    primary_dev = devices[primary_gpu]
    with torch.cuda.device(primary_dev):
        with torch.cuda.stream(reduce_stream[primary_gpu]):
            if init_centroids is not None:
                centroids = init_centroids.to(device=primary_dev, dtype=dtype, non_blocking=True)
            else:
                indices = torch.randint(0, N, (K,), device="cpu", dtype=torch.int32)
                centroids = x[indices].to(device=primary_dev, dtype=dtype, non_blocking=True)

    # CPU-side output for cluster_ids
    cluster_ids_cpu = torch.empty((N,), dtype=torch.int32, pin_memory=True) if G > 1 else None

    # Replicate centroids to all GPUs
    for g in active_gpus:
        dev = devices[g]
        with torch.cuda.device(dev):
            with torch.cuda.stream(reduce_stream[g]):
                if g == primary_gpu:
                    centroids_g[g] = centroids
                else:
                    centroids_g[g] = torch.empty((K, D), device=dev, dtype=centroids.dtype)
                    centroids_g[g].copy_(centroids, non_blocking=True)

    # Allocate staging buffers on primary GPU for gather-reduce (G > 1)
    if G > 1:
        with torch.cuda.device(primary_dev):
            staging_sums = {g: torch.empty((K, D), device=primary_dev, dtype=torch.float32)
                           for g in active_gpus if g != primary_gpu}
            staging_cnts = {g: torch.empty((K,), device=primary_dev, dtype=torch.int32)
                           for g in active_gpus if g != primary_gpu}

    # --- Per-GPU done events (per local block) and init events ---
    done_events = {}
    init_event = {}
    for g in active_gpus:
        dev = devices[g]
        with torch.cuda.device(dev):
            done_events[g] = [torch.cuda.Event() for _ in range(blocks_per_gpu[g])]
            init_event[g] = torch.cuda.Event()

    with torch.cuda.device(primary_dev):
        reduce_done_event = torch.cuda.Event()

    # --- Main iteration loop ---
    for it in range(max_iters):
        start_time = time.time()

        # Phase 1: Init — zero accumulators, compute c_sq on each GPU
        centroid_sums_g = {}
        centroid_cnts_g = {}
        c_sq_g = {}
        for g in active_gpus:
            dev = devices[g]
            with torch.cuda.device(dev):
                with torch.cuda.stream(reduce_stream[g]):
                    centroid_sums_g[g] = torch.zeros((K, D), device=dev, dtype=torch.float32)
                    centroid_cnts_g[g] = torch.zeros((K,), device=dev, dtype=torch.int32)
                    c_sq_g[g] = (centroids_g[g] ** 2).sum(dim=-1)  # (K,)
                    init_event[g].record(reduce_stream[g])

        # Phase 2: Block processing — double-buffered H2D + compute per GPU
        for g in active_gpus:
            dev = devices[g]
            local_blocks = blocks_per_gpu[g]
            ps = point_start[g]

            with torch.cuda.device(dev):
                for local_idx in range(local_blocks):
                    flag = local_idx % buf_size
                    n_start = ps + local_idx * BLOCK_N
                    n_end = min(n_start + BLOCK_N, N)

                    # For single-GPU, last block can finalize centroids directly
                    is_last_block = (G == 1) and (n_end >= N)

                    ws = work_streams[g][flag]
                    with torch.cuda.stream(ws):
                        ws.wait_event(init_event[g])

                        x_block = x[n_start:n_end].to(dev, non_blocking=True, dtype=dtype)

                        # Cache x_sq on first iteration
                        if x_sq_cache_g[g][local_idx] is None:
                            x_sq_cache_g[g][local_idx] = (x_block ** 2).sum(dim=-1)

                        # Sequential dependency within this GPU
                        if local_idx > 0:
                            ws.wait_event(done_events[g][local_idx - 1])

                        # Local offset for cluster_ids
                        local_offset = local_idx * BLOCK_N
                        local_end = min(local_offset + BLOCK_N, point_end[g] - ps)

                        cluster_ids_block = euclid_assign_triton(
                            x_block.unsqueeze(0),
                            centroids_g[g].unsqueeze(0),
                            x_sq_cache_g[g][local_idx].unsqueeze(0),
                            out=cluster_ids_g[g][local_offset:local_end].unsqueeze(0),
                            c_sq=c_sq_g[g].unsqueeze(0),
                        )

                        new_centroids_local = triton_centroid_update_sorted_euclid(
                            x=x_block.unsqueeze(0),
                            cluster_ids=cluster_ids_block,
                            old_centroids=centroids_g[g].unsqueeze(0),
                            centroid_sums=centroid_sums_g[g].unsqueeze(0),
                            centroid_cnts=centroid_cnts_g[g].unsqueeze(0),
                            calculate_new=is_last_block,
                        )

                        done_events[g][local_idx].record(ws)

        # Phase 3: Gather-Reduce-Broadcast
        with torch.cuda.device(primary_dev):
            with torch.cuda.stream(reduce_stream[primary_gpu]):
                # Wait for all GPUs to finish all blocks
                for g in active_gpus:
                    reduce_stream[primary_gpu].wait_event(done_events[g][blocks_per_gpu[g] - 1])

                if G == 1:
                    # Single GPU: new_centroids_local already finalized by last block
                    new_centroids = new_centroids_local.squeeze(0)
                else:
                    # Multi-GPU: gather partial sums/counts to primary GPU, then reduce
                    for g in active_gpus:
                        if g != primary_gpu:
                            staging_sums[g].copy_(centroid_sums_g[g], non_blocking=True)
                            staging_cnts[g].copy_(centroid_cnts_g[g], non_blocking=True)

                    total_sums = centroid_sums_g[primary_gpu]
                    total_cnts = centroid_cnts_g[primary_gpu]
                    for g in active_gpus:
                        if g != primary_gpu:
                            total_sums = total_sums + staging_sums[g]
                            total_cnts = total_cnts + staging_cnts[g]

                    # Finalize: new_centroids = total_sums / clamp(total_cnts, min=1)
                    # Handle empty clusters by keeping old centroid
                    mask = total_cnts > 0  # (K,)
                    new_centroids = torch.where(
                        mask.unsqueeze(-1),
                        total_sums / total_cnts.clamp(min=1).unsqueeze(-1).float(),
                        centroids_g[primary_gpu].float(),
                    ).to(dtype)

                shift = (new_centroids - centroids_g[primary_gpu]).norm(dim=-1).max()

                if verbose:
                    print(f"Iter {it}, center shift: {shift.item():.6f}, time: {time.time() - start_time:.2f}s")

                if shift < tol:
                    # Copy final cluster_ids back before breaking
                    if G > 1:
                        for g in active_gpus:
                            ps_g = point_start[g]
                            pe_g = point_end[g]
                            cluster_ids_cpu[ps_g:pe_g].copy_(cluster_ids_g[g][:pe_g - ps_g])
                    break

                # Broadcast new centroids to all GPUs
                centroids_g[primary_gpu] = new_centroids.clone()
                for g in active_gpus:
                    if g != primary_gpu:
                        with torch.cuda.device(devices[g]):
                            centroids_g[g].copy_(new_centroids, non_blocking=True)

                reduce_done_event.record(reduce_stream[primary_gpu])

        # Ensure all GPUs see the broadcast before next iteration
        for g in active_gpus:
            if g != primary_gpu:
                reduce_stream[g].wait_event(reduce_done_event)

    # Phase 4: Sync and gather results
    reduce_stream[primary_gpu].synchronize()

    if G == 1:
        return cluster_ids_g[primary_gpu], new_centroids
    else:
        # Copy cluster_ids from all GPUs back to CPU (if not already done by convergence break)
        for g in active_gpus:
            ps_g = point_start[g]
            pe_g = point_end[g]
            cluster_ids_cpu[ps_g:pe_g].copy_(cluster_ids_g[g][:pe_g - ps_g])
        return cluster_ids_cpu, new_centroids.cpu()


def kmeans_largeN_assign(
    x: torch.Tensor,
    centroids: torch.Tensor,
    BLOCK_N=1048576,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """
    flash-kmeans assign cluster_ids for each samples in Euclidean distance.
    for large n_samples (N), x is on CPU. (too large to fit into GPU memory)

    When device=None and multiple GPUs are available, automatically partitions
    work across all GPUs.

    Args:
        x: (N, D) data points on CPU
        centroids: (K, D) cluster centers on CPU/GPU

    Returns:
        cluster_ids: (N,) int32 Tensor, cluster assignment for each point.
    """
    devices = _resolve_devices(device)
    G = len(devices)
    dtype = dtype or x.dtype

    N, D = x.shape
    assert centroids.shape[1] == D, "centroids and x must have the same feature dimension"

    num_blocks = (N + BLOCK_N - 1) // BLOCK_N

    # --- Data partitioning across GPUs ---
    blocks_per_gpu = [(num_blocks // G) + (1 if g < (num_blocks % G) else 0) for g in range(G)]
    block_start = [0] * G
    for g in range(1, G):
        block_start[g] = block_start[g - 1] + blocks_per_gpu[g - 1]
    point_start = [block_start[g] * BLOCK_N for g in range(G)]
    point_end = [min(point_start[g] + blocks_per_gpu[g] * BLOCK_N, N) for g in range(G)]

    active_gpus = [g for g in range(G) if blocks_per_gpu[g] > 0]

    # --- Per-GPU resources ---
    buf_size = 2
    work_streams = {}
    centroids_g = {}
    cluster_ids_g = {}
    x_sq_cache_g = {}
    done_events = {}
    init_event = {}

    primary_gpu = active_gpus[0]
    primary_dev = devices[primary_gpu]

    # Set up centroids on primary GPU first
    with torch.cuda.device(primary_dev):
        centroids_on_primary = centroids.to(primary_dev, non_blocking=True)

    for g in active_gpus:
        dev = devices[g]
        with torch.cuda.device(dev):
            work_streams[g] = [torch.cuda.Stream(device=dev) for _ in range(buf_size)]
            n_points_g = point_end[g] - point_start[g]
            cluster_ids_g[g] = torch.empty((n_points_g,), device=dev, dtype=torch.int32)
            x_sq_cache_g[g] = [None] * blocks_per_gpu[g]
            done_events[g] = [torch.cuda.Event() for _ in range(blocks_per_gpu[g])]
            init_event[g] = torch.cuda.Event()

            if g == primary_gpu:
                centroids_g[g] = centroids_on_primary
            else:
                centroids_g[g] = centroids_on_primary.to(dev, non_blocking=True)

    # Compute c_sq on each GPU and record init events
    for g in active_gpus:
        dev = devices[g]
        with torch.cuda.device(dev):
            # Use default stream for init
            c_sq_g = (centroids_g[g] ** 2).sum(dim=-1)
            # Store c_sq for this GPU
            centroids_g[g] = (centroids_g[g], c_sq_g)  # tuple: (centroids, c_sq)
            init_event[g].record(torch.cuda.current_stream(dev))

    # Process blocks per GPU
    for g in active_gpus:
        dev = devices[g]
        local_blocks = blocks_per_gpu[g]
        ps = point_start[g]
        cent_g, c_sq = centroids_g[g]

        with torch.cuda.device(dev):
            for local_idx in range(local_blocks):
                flag = local_idx % buf_size
                n_start = ps + local_idx * BLOCK_N
                n_end = min(n_start + BLOCK_N, N)

                ws = work_streams[g][flag]
                with torch.cuda.stream(ws):
                    ws.wait_event(init_event[g])

                    x_block = x[n_start:n_end].to(dev, non_blocking=True, dtype=dtype)

                    if x_sq_cache_g[g][local_idx] is None:
                        x_sq_cache_g[g][local_idx] = (x_block ** 2).sum(dim=-1)

                    if local_idx > 0:
                        ws.wait_event(done_events[g][local_idx - 1])

                    local_offset = local_idx * BLOCK_N
                    local_end = min(local_offset + BLOCK_N, point_end[g] - ps)

                    euclid_assign_triton(
                        x_block.unsqueeze(0),
                        cent_g.unsqueeze(0),
                        x_sq_cache_g[g][local_idx].unsqueeze(0),
                        out=cluster_ids_g[g][local_offset:local_end].unsqueeze(0),
                        c_sq=c_sq.unsqueeze(0),
                    )

                    done_events[g][local_idx].record(ws)

    # Sync all GPUs
    for g in active_gpus:
        with torch.cuda.device(devices[g]):
            done_events[g][blocks_per_gpu[g] - 1].synchronize()

    if G == 1:
        return cluster_ids_g[primary_gpu]

    # Gather cluster_ids to CPU
    cluster_ids_cpu = torch.empty((N,), dtype=torch.int32, pin_memory=True)
    for g in active_gpus:
        ps_g = point_start[g]
        pe_g = point_end[g]
        cluster_ids_cpu[ps_g:pe_g].copy_(cluster_ids_g[g][:pe_g - ps_g])
    return cluster_ids_cpu


if __name__ == "__main__":
    N, D, K = 10000_0000, 128, 8192
    # N, D, K = 19713_2288, 128, 16384


    x = torch.randn(N, D, device="cpu", pin_memory=True, dtype=torch.float32)
    cent = x[torch.randperm(N)[:K]].to("cuda")
    print("Data prepared")

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    # warm up
    _, _ = kmeans_largeN(
        x,
        n_clusters=K,
        max_iters=10,
        tol=-1,
        verbose=True,
        # BLOCK_N=65536,
        # init_centroids=cent,
    )

    # test
    torch.cuda.synchronize()
    start_time = time.time()
    start.record()
    cluster_ids_v3, centroids_v3 = kmeans_largeN(
        x,
        n_clusters=K,
        max_iters=10,
        tol=-1,
        verbose=True,
        # BLOCK_N=65536,
        # init_centroids=cent,
    )
    end.record()
    torch.cuda.synchronize()
    end_time = time.time()
    esptime = start.elapsed_time(end)
    print(f"Time taken: {esptime:.2f}ms  {esptime/10:2f} ms/pre")
    print(f"Time taken: {(end_time-start_time)*1e3:.2f}ms  {(end_time-start_time)*1e3/10:2f} ms/pre")
