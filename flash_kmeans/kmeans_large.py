from __future__ import annotations

import time
from typing import Optional
import torch

try:
    from flash_kmeans.assign_euclid_triton import euclid_assign_triton
    from flash_kmeans.centroid_update_triton import triton_centroid_update_sorted_euclid
    from flash_kmeans.kmeans_triton_impl import batch_kmeans_Euclid

    _HAS_TRITON_IMPL = True
except Exception:
    _HAS_TRITON_IMPL = False


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

    Returns:
        cluster_ids: (N,) LongTensor, cluster assignment for each point.
        centroids: (n_clusters, D) final cluster centers.
    """
    device = device or torch.device("cuda")
    dtype = dtype or x.dtype


    # x : (N, D)   N ~ 1e8, D ~ 128
    N, D = x.shape
    K = n_clusters

    num_blocks = (N + BLOCK_N - 1) // BLOCK_N
    x_sq_blocks = [None] * num_blocks

    update_stream = torch.cuda.Stream()
    buf_size = 2
    work_streams = [torch.cuda.Stream() for _ in range(buf_size)]
    done_events = [torch.cuda.Event() for _ in range(num_blocks)]
    init_event = torch.cuda.Event()

    with torch.cuda.stream(update_stream):
        if init_centroids is not None:
            centroids = init_centroids.to(device, non_blocking=True)
        else:
            indices = torch.randint(0, N, (K,), device="cpu", dtype=torch.int32)
            centroids = x[indices].to(device, non_blocking=True)
        cluster_ids = torch.empty((N,), device=device, dtype=torch.int32)

    for it in range(max_iters):
        start_time = time.time()

        with torch.cuda.stream(update_stream):
            centroid_sums = torch.zeros((K, D), device=device, dtype=torch.float32)
            centroid_cnts = torch.zeros((K,), device=device, dtype=torch.int32)
            c_sq = (centroids**2).sum(dim=-1) # (K, )
            init_event.record(update_stream)

        for n_start in range(0, N, BLOCK_N):
            idx = n_start // BLOCK_N
            flag = idx % buf_size # flag for buffer
            n_end = min(n_start + BLOCK_N, N)
            is_last_block = n_end == N

            with torch.cuda.stream(work_streams[flag]):
                work_streams[flag].wait_event(init_event)
                
                x_block = x[n_start:n_end].to("cuda", non_blocking=True, dtype=dtype)  # x_block : (n, D)
                n = n_end - n_start

                # pre-compute squared L2 norm of all points (constant during iterations)
                if x_sq_blocks[idx] is None:
                    x_sq_blocks[idx] = (x_block**2).sum(dim=-1)

                # wait for previous block to done all computes, unsure the overlap
                if idx > 0:
                    work_streams[flag].wait_event(done_events[idx - 1])

                # calculate assignments for this block
                cluster_ids_block = euclid_assign_triton(
                    x_block.unsqueeze(0),  # (1, n, D)
                    centroids.unsqueeze(0),  # (1, K, D)
                    x_sq_blocks[idx].unsqueeze(0),  # (1, n)
                    out=cluster_ids[n_start:n_end].unsqueeze(0),  # (1, n)
                    c_sq=c_sq.unsqueeze(0),  # (1, K)
                )  # (1, n)

                new_centroids = triton_centroid_update_sorted_euclid(
                    x=x_block.unsqueeze(0),  # (1, n, D)
                    cluster_ids=cluster_ids_block,  # (1, n)
                    old_centroids=centroids.unsqueeze(0),  # (1, K, D)
                    centroid_sums=centroid_sums.unsqueeze(0),  # (1, K, D)
                    centroid_cnts=centroid_cnts.unsqueeze(0),  # (1, K)
                    calculate_new=is_last_block,
                )

                # This will mark the final block in the stream, record will overwrite previous one
                done_events[idx].record(work_streams[flag])

        with torch.cuda.stream(update_stream):
            # wait for the last blocks to finish
            update_stream.wait_event(done_events[num_blocks - 1])
            # finalize centroid update
            new_centroids.squeeze_(0)
            shift = (new_centroids - centroids).norm(dim=-1).max()
            if verbose:
                print(f"Iter {it}, center shift: {shift.item():.6f}, time: {time.time() - start_time:.2f}s")

            if shift < tol:
                break

            centroids = new_centroids.clone()

    return cluster_ids.squeeze(0), new_centroids.squeeze(0)


def kmeans_largeN_assign(
    x: torch.Tensor,
    centroids: torch.Tensor,
    BLOCK_N=1048576,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    flash-kmeans assign cluster_ids for each samples in Euclidean distance.
    for large n_samples (N), x is on CPU. (too large to fit into GPU memory)
    
    Args:
        x: (N, D) data points on CPU
        centroids: (K, D) cluster centers on CPU/GPU

    Returns:
        cluster_ids: (N,) LongTensor, cluster assignment for each point.
    """
    device = device or torch.device("cuda")
    dtype = dtype or x.dtype

    # x : (N, D)   N ~ 1e8, D ~ 128
    N, D = x.shape
    K = centroids.shape[0]
    assert centroids.shape[1] == D, "centroids and x must have the same feature dimension"

    num_blocks = (N + BLOCK_N - 1) // BLOCK_N
    x_sq_blocks = [None] * num_blocks

    update_stream = torch.cuda.Stream()
    buf_size = 2
    work_streams = [torch.cuda.Stream() for _ in range(buf_size)]
    done_events = [torch.cuda.Event() for _ in range(num_blocks)]
    init_event = torch.cuda.Event()

    with torch.cuda.stream(update_stream):
        centroids.to(device, non_blocking=True)
        cluster_ids = torch.empty((N,), device=device, dtype=torch.int32)

    with torch.cuda.stream(update_stream):
        c_sq = (centroids**2).sum(dim=-1) # (K, )
        init_event.record(update_stream)

    for n_start in range(0, N, BLOCK_N):
        idx = n_start // BLOCK_N
        flag = idx % buf_size # flag for buffer
        n_end = min(n_start + BLOCK_N, N)
        is_last_block = n_end == N

        with torch.cuda.stream(work_streams[flag]):
            work_streams[flag].wait_event(init_event)
            
            x_block = x[n_start:n_end].to("cuda", non_blocking=True, dtype=dtype)  # x_block : (n, D)
            n = n_end - n_start

            # pre-compute squared L2 norm of all points (constant during iterations)
            if x_sq_blocks[idx] is None:
                x_sq_blocks[idx] = (x_block**2).sum(dim=-1)

            # wait for previous block to done all computes, unsure the overlap
            if idx > 0:
                work_streams[flag].wait_event(done_events[idx - 1])

            # calculate assignments for this block
            euclid_assign_triton(
                x_block.unsqueeze(0),  # (1, n, D)
                centroids.unsqueeze(0),  # (1, K, D)
                x_sq_blocks[idx].unsqueeze(0),  # (1, n)
                out=cluster_ids[n_start:n_end].unsqueeze(0),  # (1, n)
                c_sq=c_sq.unsqueeze(0),  # (1, K)
            )  # (1, n)

            # This will mark the final block in the stream, record will overwrite previous one
            done_events[idx].record(work_streams[flag])

    return cluster_ids.squeeze(0)

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


    # # warm up
    # x_gpu = x.to("cuda")
    # _, _, _ = batch_kmeans_Euclid(
    #     x_gpu.unsqueeze(0),
    #     n_clusters=K,
    #     max_iters=10,
    #     tol=-1,
    #     verbose=True,
    #     init_centroids=cent,
    # )

    # torch.cuda.empty_cache()
    # # test
    # start.record()
    # ref_cluster_ids, ref_centroids, _ = batch_kmeans_Euclid(
    #     x.unsqueeze(0).to("cuda", non_blocking=True),
    #     n_clusters=K,
    #     max_iters=10,
    #     tol=-1,
    #     verbose=True,
    #     init_centroids=cent,
    # )
    # end.record()
    # torch.cuda.synchronize()
    # print(f"Time taken (triton): {start.elapsed_time(end):.2f}ms")

    # torch.testing.assert_close(cluster_ids_v3.to(torch.float32), ref_cluster_ids.squeeze(0).to(torch.float32))


    # try:
    #     torch.testing.assert_close(cluster_ids, cluster_ids_v3)
    # except Exception as e:
    #     print("Mismatch between v1 and v3 cluster assignments")
    #     print(e)
