"""SM100 (B200) fused Lloyd iteration in CuTe DSL.

Two kernels per iteration (vs ~25 launches + a radix sort + a host sync in the
Triton flash-kmeans baseline):

1. KMeansAssignSm100 — warp-specialized tcgen05 GEMM with fused epilogue.
   Per CTA = one (128-token m-tile, batch l): X tile [128,128] bf16 is TMA-loaded
   into SMEM once; the kernel loops over the 591 centroids in 128-column blocks:
   MMA (SMEM x SMEM -> TMEM fp32, 2-stage ping-pong) then 4 epilogue warps load
   the accumulator via tcgen05.Ld32x32b (thread t owns accumulator row t) and
   keep a thread-local running argmin of score = c_sq[j] - 2*x.c. After the last
   block each thread writes its token's (argmin id, best score), bumps the
   cluster histogram, and the warps cooperatively scatter the X rows into the
   fp32 centroid-sum buffer with red.global.add.v4.f32 (working set ~24 MB,
   L2-resident). dist^2(token, own centroid) = best score + ||x||^2.

2. FinalizeUpdateSm100 — one small CTA per (cluster, l): mean = sums/count
   (empty cluster keeps its previous centroid, like flash-kmeans), writes bf16
   centroids + fp32 c_sq for the next assign, and zeroes sums/counts unless it
   is the last iteration (counts are consumed by the balanced repair).

Tensor layouts (element order, all created as permuted views of contiguous
[BH, ...] torch tensors):
  mX    (M, K=128, L) bf16   K-major
  mC    (NCLS, K, L)  bf16   K-major
  mCsq  (NCLS, L)     f32
  mIdx  (M, L)        i32    mBest (M, L) f32
  mHist (NCLS, L)     i32    mSums (NCLS, K, L) f32
"""
import contextlib
import operator

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import torch
from cutlass import Boolean, Float32, Int32
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.runtime import from_dlpack

from .cute_utils import (atomic_add_fp32x4, atomic_add_i32, elem_pointer,
                         f32_bits, i32_as_f32)

TILE_M = 128
TILE_N = 128
TILE_K = 128  # head dim, static
C_STAGES = 2  # keep SMEM < 113KB so 2 CTAs fit per SM
ACC_STAGES = 2
EPI_COLS = 32  # columns per TMEM->RMEM subtile


@cute.struct
class _AssignSharedStorage:
    x_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2]
    c_mbar_ptr: cute.struct.MemRange[cutlass.Int64, C_STAGES * 2]
    acc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, ACC_STAGES * 2]
    ring_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 4]
    tmem_dealloc_mbar: cutlass.Int64
    tmem_holding_buffer: cutlass.Int32


class KMeansAssignSm100:
    def __init__(self, ncls: int, fuse_sums: bool = True, topj: int = 1):
        self.ncls = ncls
        self.n_blocks = (ncls + TILE_N - 1) // TILE_N
        self.fuse_sums = fuse_sums
        self.topj = topj  # 1: argmin; 4: sorted top-4 candidates (repair)
        self.num_sms = torch.cuda.get_device_properties(
            torch.cuda.current_device()
        ).multi_processor_count
        assert topj in (1, 4)
        if topj != 1:
            assert not fuse_sums, "top-4 variant is for the repair path only"

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mC: cute.Tensor,
        mCsq: cute.Tensor,
        mIdx: cute.Tensor,
        mBest: cute.Tensor,
        mHist: cute.Tensor,
        mSums: cute.Tensor,
        mXsq: cute.Tensor,
        write_xsq: Int32,
        stream: cuda.CUstream,
    ):
        op = tcgen05.MmaF16BF16Op(
            cutlass.BFloat16,
            cutlass.Float32,
            (TILE_M, TILE_N, 16),
            tcgen05.CtaGroup.ONE,
            tcgen05.OperandSource.SMEM,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K,
        )
        tiled_mma = cute.make_tiled_mma(op)
        mma_tiler_mnk = (TILE_M, TILE_N, TILE_K)

        x_smem_layout = sm100_utils.make_smem_layout_a(
            tiled_mma, mma_tiler_mnk, cutlass.BFloat16, 1
        )
        c_smem_layout = sm100_utils.make_smem_layout_b(
            tiled_mma, mma_tiler_mnk, cutlass.BFloat16, C_STAGES
        )

        cta_layout_mnk = cute.make_layout((1, 1, 1))
        cta_layout_vmnk = cute.tiled_divide(cta_layout_mnk, (tiled_mma.thr_id,))

        tma_op = cpasync.CopyBulkTensorTileG2SOp()
        x_smem_layout_slice = cute.slice_(x_smem_layout, (None, None, None, 0))
        x_tma_atom, x_tma_tensor = cute.nvgpu.make_tiled_tma_atom_A(
            tma_op, mX, x_smem_layout_slice, mma_tiler_mnk, tiled_mma, cta_layout_vmnk.shape
        )
        c_smem_layout_slice = cute.slice_(c_smem_layout, (None, None, None, 0))
        c_tma_atom, c_tma_tensor = cute.nvgpu.make_tiled_tma_atom_B(
            tma_op, mC, c_smem_layout_slice, mma_tiler_mnk, tiled_mma, cta_layout_vmnk.shape
        )

        # persistent grid: 2 CTAs per SM, grid-stride over (m_tile, l) tiles
        # (shapes are compile-time static here, so plain Python arithmetic)
        num_tiles = ((mX.shape[0] + TILE_M - 1) // TILE_M) * mX.shape[2]
        n_persistent = self.num_sms * 2
        grid = (min(num_tiles, n_persistent), 1, 1)
        self.kernel(
            tiled_mma,
            x_tma_atom,
            x_tma_tensor,
            mX,
            c_tma_atom,
            c_tma_tensor,
            mCsq,
            mIdx,
            mBest,
            mHist,
            mSums,
            mXsq,
            write_xsq,
            x_smem_layout,
            c_smem_layout,
            cta_layout_vmnk,
        ).launch(grid=grid, block=[576, 1, 1], cluster=(1, 1, 1), stream=stream)

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        x_tma_atom: cute.CopyAtom,
        mX: cute.Tensor,
        mX_plain: cute.Tensor,
        c_tma_atom: cute.CopyAtom,
        mC: cute.Tensor,
        mCsq: cute.Tensor,
        mIdx: cute.Tensor,
        mBest: cute.Tensor,
        mHist: cute.Tensor,
        mSums: cute.Tensor,
        mXsq: cute.Tensor,
        write_xsq: Int32,
        x_smem_layout: cute.ComposedLayout,
        c_smem_layout: cute.ComposedLayout,
        cta_layout_vmnk: cute.Layout,
    ):
        NCLS = self.ncls
        N_BLOCKS = self.n_blocks
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        grid_x, _, _ = cute.arch.grid_dim()
        # persistent grid-stride over tiles; l-major decode so a CTA's
        # consecutive tiles usually share l (c_sq smem reload amortized)
        num_m_tiles = cute.ceil_div(mIdx.shape[0], TILE_M)
        num_tiles = num_m_tiles * mHist.shape[1]  # mHist is always (K, L)

        # Warp roles: 0-7 argmin epilogue (two groups splitting the
        # accumulator subtiles), 8-15 sums-scatter consumers (decoupled via a
        # 2-slot sIdx ring so tile t's scatter overlaps tile t+1's argmin),
        # 16 MMA, 17 TMA.
        # fuse_sums (Lloyd): 8 argmin warps + 8 scatter warps;
        # topj repair variant: no scatter role, so all 16 warps do argmin
        n_epi_groups = 2 if self.fuse_sums else 4
        argmin_end = 4 * n_epi_groups
        epilogue_warp_ids = tuple(range(argmin_end))
        mma_warp_id = 16
        tma_warp_id = 17
        scatter_warp0 = 8

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(_AssignSharedStorage)
        sX = smem.allocate_tensor(
            element_type=cutlass.BFloat16,
            layout=x_smem_layout.outer,
            byte_alignment=128,
            swizzle=x_smem_layout.inner,
        )
        sC = smem.allocate_tensor(
            element_type=cutlass.BFloat16,
            layout=c_smem_layout.outer,
            byte_alignment=128,
            swizzle=c_smem_layout.inner,
        )
        sIdxR = smem.allocate_tensor(
            element_type=cutlass.Int32,
            layout=cute.make_layout(2 * TILE_M),
            byte_alignment=16,
        )
        sCsq = smem.allocate_tensor(
            element_type=cutlass.Float32,
            layout=cute.make_layout(N_BLOCKS * TILE_N),
            byte_alignment=16,
        )
        n_peers = 1 if self.fuse_sums else 3
        sValB = smem.allocate_tensor(
            element_type=cutlass.Float32,
            layout=cute.make_layout(TILE_M * self.topj * n_peers),
            byte_alignment=16,
        )
        sIdxB = smem.allocate_tensor(
            element_type=cutlass.Int32,
            layout=cute.make_layout(TILE_M * self.topj * n_peers),
            byte_alignment=16,
        )

        if warp_idx == tma_warp_id:
            cpasync.prefetch_descriptor(x_tma_atom)
            cpasync.prefetch_descriptor(c_tma_atom)

        thr_mma = tiled_mma.get_slice(0)

        tCrX = tiled_mma.make_fragment_A(sX)
        tCrC = tiled_mma.make_fragment_B(sC)

        acc_shape = tiled_mma.partition_shape_C((TILE_M, TILE_N))
        tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, ACC_STAGES))

        epilogue_sync_barrier = pipeline.NamedBarrier(barrier_id=1, num_threads=128 * n_epi_groups)
        tmem_alloc_barrier = pipeline.NamedBarrier(barrier_id=2, num_threads=32 * (argmin_end + 1))
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buffer,
            barrier_for_retrieve=tmem_alloc_barrier,
            allocator_warp_id=epilogue_warp_ids[0],
            is_two_cta=False,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar,
        )

        x_bytes = cute.size_in_bytes(cutlass.BFloat16, cute.select(x_smem_layout, mode=[0, 1, 2]))
        c_bytes = cute.size_in_bytes(cutlass.BFloat16, cute.select(c_smem_layout, mode=[0, 1, 2]))

        x_producer, x_consumer = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.x_mbar_ptr.data_ptr(),
            num_stages=1,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            tx_count=x_bytes,
            cta_layout_vmnk=cta_layout_vmnk,
        ).make_participants()
        c_producer, c_consumer = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.c_mbar_ptr.data_ptr(),
            num_stages=C_STAGES,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            tx_count=c_bytes,
            cta_layout_vmnk=cta_layout_vmnk,
        ).make_participants()
        acc_producer, acc_consumer = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_mbar_ptr.data_ptr(),
            num_stages=ACC_STAGES,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, size=len(epilogue_warp_ids)
            ),
            cta_layout_vmnk=cta_layout_vmnk,
        ).make_participants()
        # sIdx ring: argmin warps produce a slot per tile, scatter warps
        # consume it (one elect_one arrival per warp on each side)
        ring = pipeline.PipelineAsync.create(
            barrier_storage=storage.ring_mbar_ptr.data_ptr(),
            num_stages=2,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 8),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 8),
        )
        ring_pstate = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 2)
        ring_cstate = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 2)

        M_total = mIdx.shape[0]

        # ------------------------- TMA warp -------------------------
        if warp_idx == tma_warp_id:
            cute.experimental.iket.range_push("tma_block")
            t = Int32(bidx)
            while t < num_tiles:
                m_block = t % num_m_tiles
                l = t // num_m_tiles
                # per-tile gmem tiling + TMA partition (all-static layouts,
                # dynamic offsets only)
                if cutlass.const_expr(len(mX.shape) == 4):
                    # strided X view (M, K, H, B): no host-side permute copy
                    H_dim = mX.shape[2]
                    mX_l = mX[None, None, l % H_dim, l // H_dim]
                else:
                    mX_l = mX[None, None, l]
                mC_l = mC[None, None, l]
                gX = cute.local_tile(mX_l, (TILE_M, TILE_N, TILE_K),
                                     (m_block, None, None), proj=(1, None, 1))
                gC = cute.local_tile(mC_l, (TILE_M, TILE_N, TILE_K),
                                     (None, None, None), proj=(None, 1, 1))
                tCgX = thr_mma.partition_A(gX)
                tCgC = thr_mma.partition_B(gC)
                tXsX, tXgX = cpasync.tma_partition(
                    x_tma_atom,
                    0,
                    cute.make_layout(1),
                    cute.group_modes(sX, 0, 3),
                    cute.group_modes(tCgX, 0, 3),
                )
                tCsC, tCgC_tma = cpasync.tma_partition(
                    c_tma_atom,
                    0,
                    cute.make_layout(1),
                    cute.group_modes(sC, 0, 3),
                    cute.group_modes(tCgC, 0, 3),
                )
                cute.experimental.iket.range_push("x_issue")
                xh = x_producer.acquire_and_advance()
                cute.copy(x_tma_atom, tXgX[(None, 0)], tXsX[(None, 0)],
                          tma_bar_ptr=xh.barrier)
                cute.experimental.iket.range_pop()
                for n_blk in cutlass.range(N_BLOCKS, unroll=1):
                    cute.experimental.iket.range_push("c_acquire")
                    ch = c_producer.acquire_and_advance()
                    cute.experimental.iket.range_pop()
                    cute.experimental.iket.range_push("c_issue")
                    cute.copy(
                        c_tma_atom,
                        tCgC_tma[(None, n_blk, 0)],
                        tCsC[(None, ch.index)],
                        tma_bar_ptr=ch.barrier,
                    )
                    cute.experimental.iket.range_pop()
                t = t + grid_x
            x_producer.tail()
            c_producer.tail()
            cute.experimental.iket.range_pop()  # tma_block

        # ------------------------- MMA warp -------------------------
        elif warp_idx == mma_warp_id:
            cute.experimental.iket.range_push("mma_block")
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(cutlass.Float32)
            tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)

            t = Int32(bidx)
            while t < num_tiles:
                cute.experimental.iket.range_push("x_wait")
                xh = x_consumer.wait_and_advance()
                cute.experimental.iket.range_pop()
                for n_blk in cutlass.range(N_BLOCKS, unroll=1):
                    cute.experimental.iket.range_push("c_wait")
                    ch = c_consumer.wait_and_advance()
                    cute.experimental.iket.range_pop()
                    cute.experimental.iket.range_push("acc_acquire")
                    acc_empty = acc_producer.acquire_and_advance()
                    cute.experimental.iket.range_pop()
                    tCtAcc = tCtAcc_base[(None, None, None, acc_empty.index)]
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                    cute.experimental.iket.range_push("mma_issue")
                    cute.gemm(
                        tiled_mma,
                        tCtAcc,
                        tCrX[(None, None, None, 0)],
                        tCrC[(None, None, None, ch.index)],
                        tCtAcc,
                    )
                    ch.release()
                    acc_empty.commit()
                    cute.experimental.iket.range_pop()
                xh.release()
                t = t + grid_x
            acc_producer.tail()
            cute.experimental.iket.range_pop()  # mma_block

        # ---------------------- argmin epilogue warps ----------------------
        elif warp_idx < argmin_end:
            tmem.allocate(ACC_STAGES * TILE_N)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(cutlass.Float32)
            tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)

            group = warp_idx // 4          # 0: subtiles 0..1, 1: subtiles 2..3
            tidx_epi = tidx % 128          # thread == accumulator row

            copy_atom_t2r = cute.make_copy_atom(
                tcgen05.Ld32x32bOp(tcgen05.Repetition(EPI_COLS), tcgen05.Pack.NONE),
                cutlass.Float32,
            )
            epi_tile = (TILE_M, EPI_COLS)
            # per-stage accumulator subtiles: (TILE_M, EPI_COLS, 1, N_SUB)
            tAcc0_epi = cute.flat_divide(
                tCtAcc_base[(None, None, None, 0)][((None, None), 0, 0)], epi_tile
            )
            tiled_copy_t2r = tcgen05.make_tmem_copy(copy_atom_t2r, tAcc0_epi[(None, None, 0, 0)])
            thr_copy_t2r = tiled_copy_t2r.get_slice(tidx_epi)

            cC = cute.make_identity_tensor((TILE_M, TILE_N))
            cC_epi = cute.flat_divide(cC, epi_tile)
            tTR_cC = thr_copy_t2r.partition_D(cC_epi)

            tTR_rAcc = cute.make_rmem_tensor(
                tTR_cC[(None, None, None, 0, 0)].shape, cutlass.Float32
            )
            n_sub = cute.size(tTR_cC.shape, mode=[4])
            sub_per_group = n_sub // n_epi_groups

            tTR_cC_0 = tTR_cC[(None, None, None, 0, 0)]
            row_in_tile = tTR_cC_0[0][0]

            cute.experimental.iket.range_push("epilogue_block")
            t = Int32(bidx)
            l_prev = Int32(-1)
            while t < num_tiles:
                m_block = t % num_m_tiles
                l = t // num_m_tiles
                row_g = m_block * TILE_M + row_in_tile
                row_valid = row_g < M_total

                # stage c_sq into SMEM (pad tail = +big so no guard in the
                # argmin loop); 256 threads cooperate. Only reloaded when the
                # batch index changes (l-major tile order amortizes this).
                if l != l_prev:
                    cute.experimental.iket.range_push("csq_stage")
                    for j in cutlass.range_constexpr(
                            (N_BLOCKS * TILE_N + 128 * n_epi_groups - 1) // (128 * n_epi_groups)):
                        jj = j * 128 * n_epi_groups + tidx
                        if jj < N_BLOCKS * TILE_N:
                            sCsq[jj] = mCsq[jj, l] if jj < NCLS else Float32(3.0e38)
                    epilogue_sync_barrier.arrive_and_wait()
                    cute.experimental.iket.range_pop()
                l_prev = l

                best = Float32(3.0e38)
                best_idx = Int32(0)
                b1v = Float32(3.0e38)
                b2v = Float32(3.0e38)
                b3v = Float32(3.0e38)
                b1i = Int32(0)
                b2i = Int32(0)
                b3i = Int32(0)

                for n_blk in cutlass.range_constexpr(N_BLOCKS):
                    cute.experimental.iket.range_push("acc_wait")
                    acc_full = acc_consumer.wait_and_advance()
                    cute.experimental.iket.range_pop()
                    cute.experimental.iket.range_push("epi_compute")
                    tAcc_epi = cute.flat_divide(
                        tCtAcc_base[(None, None, None, acc_full.index)][((None, None), 0, 0)],
                        epi_tile,
                    )
                    tTR_tAcc = thr_copy_t2r.partition_S(tAcc_epi)
                    for s2 in cutlass.range_constexpr(sub_per_group):
                        s = group * sub_per_group + s2
                        cute.copy(tiled_copy_t2r, tTR_tAcc[(None, None, None, 0, s)], tTR_rAcc)
                        for i in cutlass.range_constexpr(cute.size(tTR_rAcc)):
                            col = n_blk * TILE_N + tTR_cC[(None, None, None, 0, s)][i][1]
                            score = sCsq[col] - 2.0 * tTR_rAcc[i]
                            if cutlass.const_expr(self.topj == 1):
                                # packed argmin: col id lives in the low 10
                                # mantissa bits (score coarsened by ~2^-13
                                # relative) — one fmin carries value + index
                                packed = i32_as_f32(
                                    (f32_bits(score) & Int32(-1024)) | Int32(col))
                                best = cute.arch.fmin(packed, best)
                            else:
                                coli = Int32(col)
                                if score < b3v:
                                    # insertion into the sorted 4 (best==b0)
                                    b3i = b2i if score < b2v else coli
                                    b3v = b2v if score < b2v else score
                                    b2i = b1i if score < b1v else (coli if score < b2v else b2i)
                                    b2v = b1v if score < b1v else (score if score < b2v else b2v)
                                    b1i = best_idx if score < best else (coli if score < b1v else b1i)
                                    b1v = best if score < best else (score if score < b1v else b1v)
                                    best_idx = coli if score < best else best_idx
                                    best = score if score < best else best
                    cute.arch.fence_view_async_tmem_load()
                    with cute.arch.elect_one():
                        acc_full.release()
                    cute.experimental.iket.range_pop()  # epi_compute

                # combine the two groups' halves: B publishes, A merges + writes
                cute.experimental.iket.range_push("merge")
                if cutlass.const_expr(self.fuse_sums):
                    # reserve the ring slot before group 0 writes it
                    ring.producer_acquire(ring_pstate)
                if cutlass.const_expr(self.topj == 1):
                    if group == 1:
                        sValB[row_in_tile] = best
                    epilogue_sync_barrier.arrive_and_wait()
                    if group == 0:
                        # packed values merge with fmin (index rides in the
                        # low bits; ties resolve deterministically)
                        best = cute.arch.fmin(sValB[row_in_tile], best)
                        best_idx = f32_bits(best) & Int32(1023)
                        if row_valid:
                            mIdx[row_g, l] = best_idx
                            mBest[row_g, l] = best
                            atomic_add_i32(1, elem_pointer(mHist, (best_idx, l)))
                        if cutlass.const_expr(self.fuse_sums):
                            sIdxR[ring_pstate.index * TILE_M + row_in_tile] = (
                                best_idx if row_valid else Int32(-1))
                else:
                    if group > 0:
                        base_s = (row_in_tile * 3 + (group - 1)) * 4
                        sValB[base_s + 0] = best
                        sIdxB[base_s + 0] = best_idx
                        sValB[base_s + 1] = b1v
                        sIdxB[base_s + 1] = b1i
                        sValB[base_s + 2] = b2v
                        sIdxB[base_s + 2] = b2i
                        sValB[base_s + 3] = b3v
                        sIdxB[base_s + 3] = b3i
                    epilogue_sync_barrier.arrive_and_wait()
                    if group == 0:
                        for j in cutlass.range_constexpr(12):
                            vb = sValB[row_in_tile * 12 + j]
                            ib = sIdxB[row_in_tile * 12 + j]
                            if vb < b3v:
                                b3i = b2i if vb < b2v else ib
                                b3v = b2v if vb < b2v else vb
                                b2i = b1i if vb < b1v else (ib if vb < b2v else b2i)
                                b2v = b1v if vb < b1v else (vb if vb < b2v else b2v)
                                b1i = best_idx if vb < best else (ib if vb < b1v else b1i)
                                b1v = best if vb < best else (vb if vb < b1v else b1v)
                                best_idx = ib if vb < best else best_idx
                                best = vb if vb < best else best
                        if row_valid:
                            mIdx[row_g, 0, l] = best_idx
                            mBest[row_g, 0, l] = best
                            mIdx[row_g, 1, l] = b1i
                            mBest[row_g, 1, l] = b1v
                            mIdx[row_g, 2, l] = b2i
                            mBest[row_g, 2, l] = b2v
                            mIdx[row_g, 3, l] = b3i
                            mBest[row_g, 3, l] = b3v
                cute.experimental.iket.range_pop()  # merge
                if cutlass.const_expr(self.fuse_sums):
                    # hand the slot to the scatter warps
                    with cute.arch.elect_one():
                        ring.producer_commit(ring_pstate)
                    ring_pstate.advance()

                t = t + grid_x

            cute.experimental.iket.range_pop()  # epilogue_block
            tmem.relinquish_alloc_permit()
            tmem.free(tmem_ptr)

        # ---------------------- scatter warps ----------------------
        # consume the sIdx ring: run-aggregated centroid-sum scatter + fused
        # ||x||^2, fully overlapped with the argmin warps' next tile
        elif warp_idx < mma_warp_id:
            if cutlass.const_expr(self.fuse_sums):
                lane = tidx % 32
                d0 = 4 * lane
                swid = warp_idx - scatter_warp0
                rows_per_warp = TILE_M // 8
                copy_ld64 = cute.make_copy_atom(
                    cute.nvgpu.CopyUniversalOp(), cutlass.BFloat16, num_bits_per_copy=64
                )
                t = Int32(bidx)
                while t < num_tiles:
                    m_block = t % num_m_tiles
                    l = t // num_m_tiles
                    cute.experimental.iket.range_push("sums_scatter")
                    ring.consumer_wait(ring_cstate)
                    sbase = ring_cstate.index * TILE_M
                    acc0 = Float32(0.0)
                    acc1 = Float32(0.0)
                    acc2 = Float32(0.0)
                    acc3 = Float32(0.0)
                    cur_c = Int32(-1)
                    BATCH = 8
                    for b in cutlass.range_constexpr(rows_per_warp // BATCH):
                        rxs = []
                        for j in cutlass.range_constexpr(BATCH):
                            r = swid * rows_per_warp + b * BATCH + j
                            rg = m_block * TILE_M + r
                            rg_s = rg if rg < M_total else M_total - 1
                            if cutlass.const_expr(len(mX_plain.shape) == 4):
                                Hp = mX_plain.shape[2]
                                xptr = elem_pointer(
                                    mX_plain, (rg_s, d0, l % Hp, l // Hp))
                            else:
                                xptr = elem_pointer(mX_plain, (rg_s, d0, l))
                            ptr8 = cute.make_ptr(
                                cutlass.BFloat16,
                                xptr.toint(),
                                cute.AddressSpace.gmem,
                                assumed_align=8,
                            )
                            gxv = cute.make_tensor(ptr8, cute.make_layout(4))
                            rx = cute.make_rmem_tensor(cute.make_layout(4), cutlass.BFloat16)
                            cute.copy(copy_ld64, gxv, rx)
                            rxs.append(rx)
                        for j in cutlass.range_constexpr(BATCH):
                            r = swid * rows_per_warp + b * BATCH + j
                            c_r = sIdxR[sbase + r]
                            rx = rxs[j]
                            if write_xsq != 0:
                                q = (Float32(rx[0]) * Float32(rx[0])
                                     + Float32(rx[1]) * Float32(rx[1])
                                     + Float32(rx[2]) * Float32(rx[2])
                                     + Float32(rx[3]) * Float32(rx[3]))
                                qs = cute.arch.warp_reduction(q, operator.add,
                                                              threads_in_group=32)
                                rg2 = m_block * TILE_M + r
                                if lane == 0:
                                    if rg2 < M_total:
                                        mXsq[rg2, l] = qs
                            if c_r != cur_c:
                                if cur_c >= 0:
                                    atomic_add_fp32x4(
                                        acc0, acc1, acc2, acc3,
                                        elem_pointer(mSums, (cur_c, d0, l)),
                                    )
                                acc0 = Float32(0.0)
                                acc1 = Float32(0.0)
                                acc2 = Float32(0.0)
                                acc3 = Float32(0.0)
                                cur_c = c_r
                            if c_r >= 0:
                                acc0 = acc0 + Float32(rx[0])
                                acc1 = acc1 + Float32(rx[1])
                                acc2 = acc2 + Float32(rx[2])
                                acc3 = acc3 + Float32(rx[3])
                    if cur_c >= 0:
                        atomic_add_fp32x4(
                            acc0, acc1, acc2, acc3,
                            elem_pointer(mSums, (cur_c, d0, l)),
                        )
                        cur_c = Int32(-1)
                    with cute.arch.elect_one():
                        ring.consumer_release(ring_cstate)
                    ring_cstate.advance()
                    cute.experimental.iket.range_pop()  # sums_scatter
                    t = t + grid_x


@cute.struct
class _FinalizeSharedStorage:
    sRed: cute.struct.MemRange[cutlass.Float32, 4]


class FinalizeUpdateSm100:
    def __init__(self, ncls: int, d: int = TILE_K):
        self.ncls = ncls
        self.d = d

    @cute.jit
    def __call__(
        self,
        mSums: cute.Tensor,
        mHist: cute.Tensor,
        mCent: cute.Tensor,
        mCsq: cute.Tensor,
        zero_after: Int32,
        stream: cuda.CUstream,
    ):
        grid = (self.ncls * mHist.shape[1], 1, 1)
        self.kernel(mSums, mHist, mCent, mCsq, zero_after).launch(
            grid=grid, block=[self.d, 1, 1], stream=stream
        )

    @cute.kernel
    def kernel(
        self,
        mSums: cute.Tensor,
        mHist: cute.Tensor,
        mCent: cute.Tensor,
        mCsq: cute.Tensor,
        zero_after: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        flat, _, _ = cute.arch.block_idx()
        c = flat % self.ncls
        l = flat // self.ncls

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(_FinalizeSharedStorage)
        sRed = cute.make_tensor(storage.sRed.data_ptr(), cute.make_layout(4))

        cnt = mHist[c, l]
        s = mSums[c, tidx, l]
        old = Float32(mCent[c, tidx, l])
        mean = old if cnt == 0 else s / Float32(cnt)
        mCent[c, tidx, l] = mCent.element_type(mean)

        sq = mean * mean
        wsum = cute.arch.warp_reduction(sq, operator.add, threads_in_group=32)
        lane = tidx % 32
        wid = tidx // 32
        if lane == 0:
            sRed[wid] = wsum
        cute.arch.barrier()
        if tidx == 0:
            mCsq[c, l] = sRed[0] + sRed[1] + sRed[2] + sRed[3]

        if zero_after != 0:
            mSums[c, tidx, l] = Float32(0.0)
            if tidx == 0:
                mHist[c, l] = Int32(0)


# ------------------------------------------------------------------
# host wrapper
# ------------------------------------------------------------------

from .cache_utils import get_jit_cache, tvm_ffi_available  # noqa: E402

_compile_cache = get_jit_cache("lloyd")


@contextlib.contextmanager
def _compile_on_device(device):
    """Compile with the tensors' own device as current: the kernels bake
    device facts at trace time (persistent-grid SM count), so the bake must
    match the launch target and the `num_sms` in the cache key.
    Restores the previous current device on exit."""
    prev = torch.cuda.current_device()
    changed = (device is not None and getattr(device, "index", None) is not None
               and device.index != prev)
    if changed:
        torch.cuda.set_device(device)
    try:
        yield
    finally:
        if changed:
            torch.cuda.set_device(prev)


def _views(x, cents, csq, ids, best, hist, sums, xsq):
    return (
        x.permute(1, 2, 0),
        cents.permute(1, 2, 0),
        csq.permute(1, 0),
        ids.permute(1, 0),
        best.permute(1, 0),
        hist.permute(1, 0),
        sums.permute(1, 2, 0),
        xsq.permute(1, 0),
    )


def _compile_args(device, tensors):
    """(from_dlpack example args, stream, compile kwargs) for this device."""
    tvm_ffi = tvm_ffi_available()
    cute_tensors = [
        from_dlpack(t, assumed_align=16, enable_tvm_ffi=tvm_ffi) for t in tensors
    ]
    if tvm_ffi:
        # AOT-exportable compile: the stream must be a fake placeholder;
        # the real one is supplied per call. Kernels land in the persistent
        # disk cache (see cache_utils) instead of being recompiled per run.
        return cute_tensors, cute.runtime.make_fake_stream(), \
            {"options": "--enable-tvm-ffi"}
    stream = cuda.CUstream(torch.cuda.current_stream(device).cuda_stream)
    return cute_tensors, stream, {}


def _get_compiled(N, K, BH, device, tensors, fuse_sums=True, topj=1, rank4=False):
    # The kernel specializes on the STATIC layout of the example tensors, so
    # BOTH shape and strides of the (permuted) x view go into the key:
    # rank-4 views with the same shape but different layouts (NHD-permuted vs
    # HND-contiguous, or different B/H splits at equal BH) are distinct
    # specializations. The other views are fresh contiguous allocations whose
    # layouts are fully determined by geom+K+topj (see call sites).
    xshape = tuple(tensors[0].shape)  # (N, D, BH) or (N, D, H, B)
    xstrides = tuple(tensors[0].stride())
    num_sms = torch.cuda.get_device_properties(device).multi_processor_count
    akey = ("sm100", xshape, xstrides, K, fuse_sums, topj, num_sms, "assign")
    # FinalizeUpdateSm100 specializes only on K and the (K,D,BH)/(K,BH)
    # contiguous views — key it on exactly that so new token counts N reuse
    # it (it has no device dependence either: grid=(K, BH), block=(D)).
    fkey = ("sm100", "finalize", K, xshape[1], BH)
    need_a = akey not in _compile_cache
    need_f = fuse_sums and fkey not in _compile_cache
    if need_a or need_f:
        with _compile_on_device(device):
            (xv, cv, csqv, iv, bv, hv, sv, qv), stream, ckwargs = \
                _compile_args(device, tensors)
            if need_a:
                _compile_cache[akey] = cute.compile(
                    KMeansAssignSm100(K, fuse_sums=fuse_sums, topj=topj),
                    xv, cv, csqv, iv, bv, hv, sv, qv, Int32(1), stream,
                    **ckwargs,
                )
            if need_f:
                _compile_cache[fkey] = cute.compile(
                    FinalizeUpdateSm100(K), sv, hv, cv, csqv, Int32(1), stream,
                    **ckwargs,
                )
    return _compile_cache[akey], (_compile_cache[fkey] if fuse_sums else None)


@torch.no_grad()
def lloyd_cute(x: torch.Tensor, n_clusters: int, max_iters: int,
               init_centroids: torch.Tensor | None = None):
    """x: [BH, N, D=128] bf16 contiguous CUDA. Returns
    (ids i32 [BH,N], scores f32 [BH,N], counts i32 [BH,K], centroids bf16
    [BH,K,D], x_sq f32 [BH,N]). scores = c_sq - 2 x.c of the assigned cluster
    (dist^2 = scores + x_sq; x_sq is computed for free in the fused scatter)."""
    assert x.dtype == torch.bfloat16
    if x.dim() == 4:
        # [B, H, N, D] view with D innermost-contiguous (e.g. a permuted NHD
        # tensor) — consumed zero-copy via a 4D TMA descriptor
        Bx, Hx, N, D = x.shape
        BH = Bx * Hx
        assert x.stride(-1) == 1
        xv_t = x.permute(2, 3, 1, 0)               # (N, D, H, B)
    else:
        assert x.is_contiguous()
        BH, N, D = x.shape
        xv_t = x.permute(1, 2, 0)                  # (N, D, BH)
    assert D == TILE_K, f"D={D} unsupported (expected {TILE_K})"
    K = n_clusters
    device = x.device

    if init_centroids is None:
        idx = torch.randint(0, N, (BH, K), device=device)
        if x.dim() == 4:
            cents = torch.gather(
                x, 2, idx.view(Bx, Hx, K, 1).expand(-1, -1, -1, D)
            ).reshape(BH, K, D).contiguous()
        else:
            cents = torch.gather(x, 1, idx[..., None].expand(-1, -1, D)).contiguous()
    else:
        cents = init_centroids.to(torch.bfloat16).contiguous().clone()

    csq = (cents.float() ** 2).sum(-1).contiguous()
    ids = torch.empty(BH, N, dtype=torch.int32, device=device)
    best = torch.empty(BH, N, dtype=torch.float32, device=device)
    hist = torch.zeros(BH, K, dtype=torch.int32, device=device)
    sums = torch.zeros(BH, K, D, dtype=torch.float32, device=device)
    xsq = torch.empty(BH, N, dtype=torch.float32, device=device)

    views = (
        xv_t,
        cents.permute(1, 2, 0),
        csq.permute(1, 0),
        ids.permute(1, 0),
        best.permute(1, 0),
        hist.permute(1, 0),
        sums.permute(1, 2, 0),
        xsq.permute(1, 0),
    )
    assign, finalize = _get_compiled(N, K, BH, device, views, rank4=(x.dim() == 4))
    xv, cv, csqv, iv, bv, hv, sv, qv = views

    stream = cuda.CUstream(torch.cuda.current_stream(device).cuda_stream)
    for it in range(max_iters):
        assign(xv, cv, csqv, iv, bv, hv, sv, qv,
               Int32(1 if it == max_iters - 1 else 0), stream)
        finalize(sv, hv, cv, csqv, Int32(1 if it < max_iters - 1 else 0), stream)
    return ids, best, hist, cents, xsq
