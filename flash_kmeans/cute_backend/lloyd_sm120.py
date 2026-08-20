"""SM120 (RTX PRO 6000 Blackwell) fused Lloyd iteration in CuTe DSL.

Same two-kernel structure and host contract as lloyd_sm100/lloyd_sm90, with
the architecture-specific core swapped:

- warp-level mma.sync m16n8k16 (MmaF16BF16Op) with explicit ldmatrix
  SMEM->RMEM copies (sm120 has no wgmma / tcgen05). atom_layout (8,1,1):
  all 8 math warps tile the M dimension, so each row's score fragment is
  owned by a single warp and the packed argmin finishes with the same two
  butterfly shuffles across the owning 4-lane quad as on sm90.
- TMA G2S loads (supported on sm120) with the same X (1 stage) and
  C (C_STAGES ring) PipelineTmaAsync pipelines as sm90.
- X fragments (A operand) are loaded once per tile and reused across all
  centroid blocks; B fragments are ldmatrix'd per stage, software-pipelined
  against the mma.sync stream (quack gemm_sm120 pattern).
- The centroid-sum scatter / fused ||x||^2 reuse the sm100 run-aggregated
  SIMT code verbatim (8 math warps x 16 consecutive rows each).

sm120 is bandwidth-bound (GDDR7): each CTA processes TWO 128-row X slots
against one shared centroid-block ring, halving centroid TMA traffic.
TILE_N=32 / C_STAGES=3 keeps SMEM (~92 KB) under the 99 KB/SM budget.
Kernel grid: non-persistent (m_tiles, L). Tensor views identical to sm100.
"""
import operator

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import torch
from cutlass import Float32, Int32
from cutlass.cute.nvgpu import cpasync, warp

import quack.sm90_utils as qsm90
from quack import layout_utils
from cutlass.utils import LayoutEnum

from .cute_utils import (atomic_add_fp32x4, atomic_add_i32, elem_pointer,
                         f32_bits, i32_as_f32)

TILE_M = 128          # rows per X slot; a CTA processes X_PAIR slots
TILE_N = 32
TILE_K = 128
C_STAGES = 3
NUM_MMA_REGS = 232
NUM_PRODUCER_REGS = 40


@cute.struct
class _AssignSharedStorageSm120:
    x_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 4]
    c_mbar_ptr: cute.struct.MemRange[cutlass.Int64, C_STAGES * 2]


class KMeansAssignSm120:
    def __init__(self, ncls: int, fuse_sums: bool = True, topj: int = 1):
        self.ncls = ncls
        self.n_blocks = (ncls + TILE_N - 1) // TILE_N
        self.fuse_sums = fuse_sums
        self.topj = topj
        assert topj in (1, 4, 8)
        if topj != 1:
            assert not fuse_sums
        # dual X slots only for the main (topj=1) kernel; the top-4 repair
        # kernel runs on small excess sets and needs sTopV smem instead
        self.x_pair = 2 if topj == 1 else 1
        import torch as _torch
        self.num_sms = _torch.cuda.get_device_properties(
            _torch.cuda.current_device()).multi_processor_count

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
        op = warp.MmaF16BF16Op(cutlass.BFloat16, cutlass.Float32, (16, 8, 16))
        tiled_mma = cute.make_tiled_mma(op, cute.make_layout((8, 1, 1)))

        x_smem_layout = qsm90.make_smem_layout(
            cutlass.BFloat16, LayoutEnum.ROW_MAJOR, (TILE_M, TILE_K), self.x_pair
        )
        c_smem_layout = qsm90.make_smem_layout(
            cutlass.BFloat16, LayoutEnum.ROW_MAJOR, (TILE_N, TILE_K), C_STAGES
        )

        tma_op = cpasync.CopyBulkTensorTileG2SOp()
        x_tma_atom, x_tma_tensor = cpasync.make_tiled_tma_atom(
            tma_op, mX, cute.select(x_smem_layout, mode=[0, 1]), (TILE_M, TILE_K), 1
        )
        c_tma_atom, c_tma_tensor = cpasync.make_tiled_tma_atom(
            tma_op, mC, cute.select(c_smem_layout, mode=[0, 1]), (TILE_N, TILE_K), 1
        )

        num_pair_tiles = cute.ceil_div(mIdx.shape[0], TILE_M * self.x_pair)
        total_tiles = num_pair_tiles * mHist.shape[1]
        grid = (min(total_tiles, self.num_sms), 1, 1)
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
        ).launch(grid=grid, block=[384, 1, 1], cluster=(1, 1, 1), stream=stream)

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
    ):
        NCLS = self.ncls
        N_BLOCKS = self.n_blocks
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        grid_x, _, _ = cute.arch.grid_dim()

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(_AssignSharedStorageSm120)
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
        sIdx = smem.allocate_tensor(
            element_type=cutlass.Int32,
            layout=cute.make_layout(2 * TILE_M * self.x_pair),
            byte_alignment=16,
        )
        sCsq = smem.allocate_tensor(
            element_type=cutlass.Float32,
            layout=cute.make_layout(N_BLOCKS * TILE_N),
            byte_alignment=16,
        )
        sTopV = smem.allocate_tensor(
            element_type=cutlass.Float32,
            layout=cute.make_layout(
                TILE_M * 4 * self.topj if self.topj > 1 else 4),
            byte_alignment=16,
        )

        if warp_idx == 0:
            cpasync.prefetch_descriptor(x_tma_atom)
            cpasync.prefetch_descriptor(c_tma_atom)

        # pipelines: X (1 stage) and C (C_STAGES), TMA -> 8 math warps
        X_PAIR = self.x_pair
        x_pipe = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.x_mbar_ptr.data_ptr(),
            num_stages=X_PAIR,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 8),
            tx_count=cute.size_in_bytes(
                cutlass.BFloat16, cute.select(x_smem_layout, mode=[0, 1])),
        )
        c_pipe = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.c_mbar_ptr.data_ptr(),
            num_stages=C_STAGES,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 8),
            tx_count=cute.size_in_bytes(
                cutlass.BFloat16, cute.select(c_smem_layout, mode=[0, 1])),
        )

        M_total = mIdx.shape[0]
        num_pair_tiles = cute.ceil_div(M_total, TILE_M * self.x_pair)
        num_tiles = num_pair_tiles * mHist.shape[1]
        math_barrier = pipeline.NamedBarrier(barrier_id=1, num_threads=256)

        # ------------------------- producer WG -------------------------
        if warp_idx < 4:
            cute.arch.setmaxregister_decrease(NUM_PRODUCER_REGS)
            if warp_idx == 0:
                num_m_slots = cute.ceil_div(mIdx.shape[0], TILE_M)
                xp_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Producer, X_PAIR)
                cp_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Producer, C_STAGES)
                t = Int32(bidx)
                while t < num_tiles:
                    mp = t % num_pair_tiles
                    l = t // num_pair_tiles
                    if cutlass.const_expr(len(mX.shape) == 4):
                        H_dim = mX.shape[2]
                        mX_l = mX[None, None, l % H_dim, l // H_dim]
                    else:
                        mX_l = mX[None, None, l]
                    mC_l = mC[None, None, l]
                    gX = cute.local_tile(mX_l, (TILE_M, TILE_K), (None, 0))
                    gC = cute.local_tile(mC_l, (TILE_N, TILE_K), (None, 0))
                    tXsX, tXgX = cpasync.tma_partition(
                        x_tma_atom, 0, cute.make_layout(1),
                        cute.group_modes(sX, 0, 2), cute.group_modes(gX, 0, 2),
                    )
                    tCsC, tCgC = cpasync.tma_partition(
                        c_tma_atom, 0, cute.make_layout(1),
                        cute.group_modes(sC, 0, 2), cute.group_modes(gC, 0, 2),
                    )
                    for sl in cutlass.range_constexpr(X_PAIR):
                        # clamp the tail slot to the last valid tile;
                        # duplicate rows are masked by row_g < M_total guards
                        m_sl = mp * X_PAIR + sl
                        m_sl = m_sl if m_sl < num_m_slots else num_m_slots - 1
                        x_pipe.producer_acquire(xp_state)
                        cute.copy(x_tma_atom, tXgX[None, m_sl],
                                  tXsX[None, xp_state.index],
                                  tma_bar_ptr=x_pipe.producer_get_barrier(xp_state))
                        xp_state.advance()
                    for n_blk in cutlass.range(N_BLOCKS, unroll=1):
                        c_pipe.producer_acquire(cp_state)
                        cute.copy(c_tma_atom, tCgC[None, n_blk],
                                  tCsC[None, cp_state.index],
                                  tma_bar_ptr=c_pipe.producer_get_barrier(cp_state))
                        cp_state.advance()
                    t = t + grid_x
                x_pipe.producer_tail(xp_state)
                c_pipe.producer_tail(cp_state)
            else:
                # ---------------- scatter warps (1..3) ----------------
                # Tile t's centroid-sum scatter runs here, overlapped with
                # the math warps' tile t+1 (rows re-read from gmem/L2; the
                # X smem slots were released early for TMA prefetch).
                # "ready" = sIdx(t) published (math arrives non-blocking);
                # "done" = scatter(t-1) finished (math waits <= 1 tile).
                if cutlass.const_expr(self.fuse_sums):
                    swarp = warp_idx - 1               # 0..2
                    lane_s = tidx % 32
                    d0s = 4 * lane_s
                    bar_r = pipeline.NamedBarrier(barrier_id=2, num_threads=352)
                    bar_d = pipeline.NamedBarrier(barrier_id=3, num_threads=352)
                    copy_ld64s = cute.make_copy_atom(
                        cute.nvgpu.CopyUniversalOp(), cutlass.BFloat16,
                        num_bits_per_copy=64)
                    SIDX_RING_S = TILE_M * self.x_pair
                    NB_BATCH = (TILE_M * self.x_pair) // 8
                    it_par_s = Int32(0)
                    prev_base = Int32(0)
                    prev_l = Int32(0)
                    prev_off = Int32(0)
                    prev_valid = Int32(0)
                    t = Int32(bidx)
                    while t < num_tiles:
                        mp = t % num_pair_tiles
                        l = t // num_pair_tiles
                        if prev_valid == 1:
                            for bb in cutlass.range_constexpr((NB_BATCH + 2) // 3):
                                b = bb * 3 + swarp
                                if b < NB_BATCH:
                                    rxs = []
                                    for j in cutlass.range_constexpr(8):
                                        r = b * 8 + j
                                        rg = prev_base + r
                                        rg_s = rg if rg < M_total else M_total - 1
                                        if cutlass.const_expr(len(mX_plain.shape) == 4):
                                            Hp = mX_plain.shape[2]
                                            xptr = elem_pointer(
                                                mX_plain,
                                                (rg_s, d0s, prev_l % Hp, prev_l // Hp))
                                        else:
                                            xptr = elem_pointer(
                                                mX_plain, (rg_s, d0s, prev_l))
                                        ptr8 = cute.make_ptr(
                                            cutlass.BFloat16, xptr.toint(),
                                            cute.AddressSpace.gmem, assumed_align=8)
                                        gxv = cute.make_tensor(ptr8, cute.make_layout(4))
                                        rx = cute.make_rmem_tensor(
                                            cute.make_layout(4), cutlass.BFloat16)
                                        cute.copy(copy_ld64s, gxv, rx)
                                        rxs.append(rx)
                                    for j in cutlass.range_constexpr(8):
                                        r = b * 8 + j
                                        c_r = sIdx[prev_off + r]
                                        rx = rxs[j]
                                        if write_xsq != 0:
                                            q = (Float32(rx[0]) * Float32(rx[0])
                                                 + Float32(rx[1]) * Float32(rx[1])
                                                 + Float32(rx[2]) * Float32(rx[2])
                                                 + Float32(rx[3]) * Float32(rx[3]))
                                            qs = cute.arch.warp_reduction(
                                                q, operator.add, threads_in_group=32)
                                            rg2 = prev_base + r
                                            if lane_s == 0:
                                                if rg2 < M_total:
                                                    mXsq[rg2, prev_l] = qs
                                        if c_r >= 0:
                                            atomic_add_fp32x4(
                                                Float32(rx[0]), Float32(rx[1]),
                                                Float32(rx[2]), Float32(rx[3]),
                                                elem_pointer(
                                                    mSums, (c_r, d0s, prev_l)),
                                            )
                        bar_r.arrive_and_wait()
                        bar_d.arrive_unaligned()
                        prev_base = mp * (TILE_M * self.x_pair)
                        prev_l = l
                        prev_off = it_par_s * SIDX_RING_S
                        prev_valid = Int32(1)
                        it_par_s = 1 - it_par_s
                        t = t + grid_x

                    # drain: the last tile's scatter
                    if prev_valid == 1:
                        for bb in cutlass.range_constexpr((NB_BATCH + 2) // 3):
                            b = bb * 3 + swarp
                            if b < NB_BATCH:
                                rxs = []
                                for j in cutlass.range_constexpr(8):
                                    r = b * 8 + j
                                    rg = prev_base + r
                                    rg_s = rg if rg < M_total else M_total - 1
                                    if cutlass.const_expr(len(mX_plain.shape) == 4):
                                        Hp = mX_plain.shape[2]
                                        xptr = elem_pointer(
                                            mX_plain,
                                            (rg_s, d0s, prev_l % Hp, prev_l // Hp))
                                    else:
                                        xptr = elem_pointer(
                                            mX_plain, (rg_s, d0s, prev_l))
                                    ptr8 = cute.make_ptr(
                                        cutlass.BFloat16, xptr.toint(),
                                        cute.AddressSpace.gmem, assumed_align=8)
                                    gxv = cute.make_tensor(ptr8, cute.make_layout(4))
                                    rx = cute.make_rmem_tensor(
                                        cute.make_layout(4), cutlass.BFloat16)
                                    cute.copy(copy_ld64s, gxv, rx)
                                    rxs.append(rx)
                                for j in cutlass.range_constexpr(8):
                                    r = b * 8 + j
                                    c_r = sIdx[prev_off + r]
                                    rx = rxs[j]
                                    if write_xsq != 0:
                                        q = (Float32(rx[0]) * Float32(rx[0])
                                             + Float32(rx[1]) * Float32(rx[1])
                                             + Float32(rx[2]) * Float32(rx[2])
                                             + Float32(rx[3]) * Float32(rx[3]))
                                        qs = cute.arch.warp_reduction(
                                            q, operator.add, threads_in_group=32)
                                        rg2 = prev_base + r
                                        if lane_s == 0:
                                            if rg2 < M_total:
                                                mXsq[rg2, prev_l] = qs
                                    if c_r >= 0:
                                        atomic_add_fp32x4(
                                            Float32(rx[0]), Float32(rx[1]),
                                            Float32(rx[2]), Float32(rx[3]),
                                            elem_pointer(
                                                mSums, (c_r, d0s, prev_l)),
                                        )

        # ------------------------- math warps -------------------------
        else:
            cute.arch.setmaxregister_increase(NUM_MMA_REGS)
            tidx_m = tidx - 128                     # 0..255
            thr_mma = tiled_mma.get_slice(tidx_m)

            # ldmatrix SMEM->RMEM copy atoms (A: 16x16, B: 8x16 x2 per inst)
            atom_ld_A = cute.make_copy_atom(
                warp.LdMatrix8x8x16bOp(False, 4), cutlass.BFloat16)
            atom_ld_B = cute.make_copy_atom(
                warp.LdMatrix8x8x16bOp(False, 4), cutlass.BFloat16)
            tiled_copy_A = cute.make_tiled_copy_A(atom_ld_A, tiled_mma)
            tiled_copy_B = cute.make_tiled_copy_B(atom_ld_B, tiled_mma)
            thr_copy_A = tiled_copy_A.get_slice(tidx_m)
            thr_copy_B = tiled_copy_B.get_slice(tidx_m)
            tCsA_cv = thr_copy_A.partition_S(sX)    # (.., k_blk, stage=1)
            tCsB_cv = thr_copy_B.partition_S(sC)    # (.., k_blk, stage)

            accs = [cute.make_rmem_tensor(
                thr_mma.partition_shape_C((TILE_M, TILE_N)), cutlass.Float32)
                for _ in range(self.x_pair)]
            tCsA = thr_mma.partition_A(sX)
            tCsB = thr_mma.partition_B(sC)
            tCrAs = [thr_mma.make_fragment_A(tCsA[None, None, None, 0])
                     for _ in range(self.x_pair)]
            tCrB = thr_mma.make_fragment_B(tCsB[None, None, None, 0])
            tCrA_cvs = [tiled_copy_A.retile(a) for a in tCrAs]
            tCrB_cv = tiled_copy_B.retile(tCrB)
            num_k_blocks = cute.size(tCrAs[0], mode=[2])

            cS = cute.make_identity_tensor((TILE_M, TILE_N))
            tScS = thr_mma.partition_C(cS)
            tScS_mn = layout_utils.convert_layout_acc_mn(tScS)
            rows_per_thr = cute.size(tScS_mn.shape, mode=[0])
            cols_per_thr = cute.size(tScS_mn.shape, mode=[1])

            xc_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.x_pair)
            xc_rel = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.x_pair)
            cc_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, C_STAGES)

            bar_ready = pipeline.NamedBarrier(barrier_id=2, num_threads=352)
            bar_done = pipeline.NamedBarrier(barrier_id=3, num_threads=352)
            it_par = Int32(0)
            SIDX_RING = TILE_M * self.x_pair
            t = Int32(bidx)
            while t < num_tiles:
                mp = t % num_pair_tiles
                l = t // num_pair_tiles
                sidx_off = it_par * SIDX_RING

                # stage c_sq for this tile's l (pad tail = +big); the block
                # barrier below also orders it against last tile's scatter
                for j in cutlass.range_constexpr((N_BLOCKS * TILE_N + 255) // 256):
                    jj = j * 256 + tidx_m
                    if jj < N_BLOCKS * TILE_N:
                        sCsq[jj] = mCsq[jj, l] if jj < NCLS else Float32(3.0e38)
                math_barrier.arrive_and_wait()

                # X slots: load ALL A fragments once, reuse for every
                # centroid blk. X_STAGES == x_pair, so slot sl always lands
                # in smem stage sl (the scatter below relies on this).
                # Slots are NOT released here — the scatter reads sX.
                for sl in cutlass.range_constexpr(self.x_pair):
                    x_pipe.consumer_wait(xc_state)
                    for k in cutlass.range_constexpr(num_k_blocks):
                        cute.copy(tiled_copy_A, tCsA_cv[None, None, k, sl],
                                  tCrA_cvs[sl][None, None, k])
                    xc_state.advance()
                # A fragments now live in registers; release the slots so the
                # producer can prefetch the NEXT tile's X during this tile
                cute.arch.fence_view_async_shared()
                cute.arch.sync_warp()
                for sl in cutlass.range_constexpr(self.x_pair):
                    with cute.arch.elect_one():
                        x_pipe.consumer_release(xc_rel)
                    xc_rel.advance()

                best0 = Float32(3.0e38)
                best1 = Float32(3.0e38)
                bests0 = [[Float32(3.0e38) for _ in range(4)]
                          for _ in range(self.x_pair)]
                bests1 = [[Float32(3.0e38) for _ in range(4)]
                          for _ in range(self.x_pair)]
                # sorted-J lists (ascending) for the top-J path, one per row
                bq0 = [Float32(3.0e38) for _ in range(self.topj)]
                bq1 = [Float32(3.0e38) for _ in range(self.topj)]

                for n_blk in cutlass.range_constexpr(N_BLOCKS):
                    c_pipe.consumer_wait(cc_state)
                    for sl in cutlass.range_constexpr(self.x_pair):
                        accs[sl].fill(0.0)
                    # software-pipeline ldmatrix (k+1) against mma.sync (k);
                    # B fragments are shared by all X slots
                    cute.copy(tiled_copy_B, tCsB_cv[None, None, 0, cc_state.index],
                              tCrB_cv[None, None, 0])
                    for k in cutlass.range_constexpr(num_k_blocks):
                        if cutlass.const_expr(k + 1 < num_k_blocks):
                            cute.copy(tiled_copy_B,
                                      tCsB_cv[None, None, k + 1, cc_state.index],
                                      tCrB_cv[None, None, k + 1])
                        for sl in cutlass.range_constexpr(self.x_pair):
                            cute.gemm(tiled_mma, accs[sl], tCrAs[sl][None, None, k],
                                      tCrB[None, None, k], accs[sl])
                    cute.arch.fence_view_async_shared()
                    cute.arch.sync_warp()
                    c_pipe.consumer_release(cc_state)
                    cc_state.advance()

                    for sl in cutlass.range_constexpr(self.x_pair):
                        acc_mn = layout_utils.convert_layout_acc_mn(accs[sl])
                        for r in cutlass.range_constexpr(rows_per_thr):
                            for cix in cutlass.range_constexpr(cols_per_thr):
                                col = n_blk * TILE_N + tScS_mn[r, cix][1]
                                score = sCsq[col] - 2.0 * acc_mn[r, cix]
                                if cutlass.const_expr(self.topj == 1):
                                    packed = i32_as_f32(
                                        (f32_bits(score) & Int32(-1024)) | Int32(col))
                                    if r == 0:
                                        bests0[sl][cix % 4] = cute.arch.fmin(
                                            packed, bests0[sl][cix % 4])
                                    else:
                                        bests1[sl][cix % 4] = cute.arch.fmin(
                                            packed, bests1[sl][cix % 4])
                                else:
                                    packed = i32_as_f32(
                                        (f32_bits(score) & Int32(-1024)) | Int32(col))
                                    # branchless sorted-J insert: 2J-op fmin/fmax
                                    # bubble network (the divergent if-chain
                                    # version cost ~3x on this scan)
                                    bq = bq0 if r == 0 else bq1
                                    tt = packed
                                    for jq in cutlass.range_constexpr(self.topj):
                                        lo = cute.arch.fmin(tt, bq[jq])
                                        tt = cute.arch.fmax(tt, bq[jq])
                                        bq[jq] = lo

                xr0 = tScS_mn[0, 0][0]
                xr1 = tScS_mn[1, 0][0] if cutlass.const_expr(rows_per_thr > 1) else xr0

                if cutlass.const_expr(self.topj == 1):
                    for sl in cutlass.range_constexpr(self.x_pair):
                        b0 = cute.arch.fmin(
                            cute.arch.fmin(bests0[sl][0], bests0[sl][1]),
                            cute.arch.fmin(bests0[sl][2], bests0[sl][3]))
                        b1 = cute.arch.fmin(
                            cute.arch.fmin(bests1[sl][0], bests1[sl][1]),
                            cute.arch.fmin(bests1[sl][2], bests1[sl][3]))
                        # quad reduce (packed): 2 butterfly rounds
                        for off in cutlass.range_constexpr(2):
                            o = 2 >> off
                            p0 = cute.arch.shuffle_sync_bfly(b0, offset=o, mask_and_clamp=31)
                            b0 = cute.arch.fmin(p0, b0)
                            p1 = cute.arch.shuffle_sync_bfly(b1, offset=o, mask_and_clamp=31)
                            b1 = cute.arch.fmin(p1, b1)
                        # column-0 lane of each quad writes its two rows
                        if tScS_mn[0, 0][1] == 0:
                            for r in cutlass.range_constexpr(2):
                                row_g = (mp * self.x_pair + sl) * TILE_M \
                                    + (xr0 if r == 0 else xr1)
                                bv = b0 if r == 0 else b1
                                bi = f32_bits(bv) & Int32(1023)
                                if row_g < M_total:
                                    mIdx[row_g, l] = bi
                                    mBest[row_g, l] = bv
                                    atomic_add_i32(1, elem_pointer(mHist, (bi, l)))
                                if cutlass.const_expr(self.fuse_sums):
                                    rloc = sl * TILE_M + (xr0 if r == 0 else xr1)
                                    sIdx[sidx_off + rloc] = \
                                        bi if row_g < M_total else Int32(-1)
                else:
                    # top-4: publish each quad lane's sorted-4 (packed) to smem;
                    # the col-0 lane merges 16 -> 4 and writes
                    lane_q = tidx_m % 4
                    for r in cutlass.range_constexpr(2):
                        rloc = xr0 if r == 0 else xr1
                        base_s = (rloc * 4 + lane_q) * self.topj
                        bq = bq0 if r == 0 else bq1
                        for jq in cutlass.range_constexpr(self.topj):
                            sTopV[base_s + jq] = bq[jq]
                    math_barrier.arrive_and_wait()
                    if tScS_mn[0, 0][1] == 0:
                        for r in cutlass.range_constexpr(2):
                            rloc = xr0 if r == 0 else xr1
                            row_g = mp * TILE_M + rloc
                            mq = [Float32(3.0e38) for _ in range(self.topj)]
                            for j in cutlass.range_constexpr(4 * self.topj):
                                vv = sTopV[rloc * 4 * self.topj + j]
                                tt = vv
                                for jq in cutlass.range_constexpr(self.topj):
                                    lo = cute.arch.fmin(tt, mq[jq])
                                    tt = cute.arch.fmax(tt, mq[jq])
                                    mq[jq] = lo
                            if row_g < M_total:
                                for jq in cutlass.range_constexpr(self.topj):
                                    mIdx[row_g, jq, l] = f32_bits(mq[jq]) & Int32(1023)
                                    mBest[row_g, jq, l] = mq[jq]

                if cutlass.const_expr(self.fuse_sums):
                    # publish sIdx(t) to the scatter warps (producer warps
                    # 1..3 scatter tile t while the math warps run tile t+1)
                    # and gate on scatter(t-1) (sIdx parity ring depth 2).
                    # bar_done is a full 256-thread math rendezvous, so it
                    # also plays the tile-end math_barrier role (sCsq).
                    bar_ready.arrive_unaligned()
                    bar_done.arrive_and_wait()
                else:
                    math_barrier.arrive_and_wait()
                it_par = 1 - it_par
                t = t + grid_x


# host wrapper: same contract as lloyd_sm100
from .lloyd_sm100 import (  # noqa: E402
    FinalizeUpdateSm100,
    _compile_args,
    _compile_cache,
    _compile_on_device,
    _views,
)


def _get_compiled(N, K, BH, device, tensors, fuse_sums=True, topj=1,
                  rank4=False):
    # See lloyd_sm100._get_compiled for the cache-key rationale (static
    # shape+stride specialization, slim finalize key, num_sms, disk cache).
    xshape = tuple(tensors[0].shape)  # (N, D, BH) or (N, D, H, B)
    xstrides = tuple(tensors[0].stride())
    num_sms = torch.cuda.get_device_properties(device).multi_processor_count
    akey = ("sm120", xshape, xstrides, K, fuse_sums, topj, num_sms, "assign")
    fkey = ("sm120", "finalize", K, xshape[1], BH)
    need_a = akey not in _compile_cache
    need_f = fuse_sums and fkey not in _compile_cache
    if need_a or need_f:
        with _compile_on_device(device):
            (xv, cv, csqv, iv, bv, hv, sv, qv), stream, ckwargs = \
                _compile_args(device, tensors)
            if need_a:
                _compile_cache[akey] = cute.compile(
                    KMeansAssignSm120(K, fuse_sums=fuse_sums, topj=topj),
                    xv, cv, csqv, iv, bv, hv, sv, qv, Int32(1), stream,
                    **ckwargs,
                )
            if need_f:
                # (mSums, mHist, mCent, mCsq, zero_after)
                _compile_cache[fkey] = cute.compile(
                    FinalizeUpdateSm100(K), sv, hv, cv, csqv, Int32(1), stream,
                    **ckwargs,
                )
    return _compile_cache[akey], (_compile_cache[fkey] if fuse_sums else None)


@torch.no_grad()
def lloyd_cute(x: torch.Tensor, n_clusters: int, max_iters: int,
               init_centroids: torch.Tensor | None = None):
    assert x.dtype == torch.bfloat16
    if x.dim() == 4:
        Bx, Hx, N, D = x.shape
        BH = Bx * Hx
        assert x.stride(-1) == 1
        xv_t = x.permute(2, 3, 1, 0)
    else:
        assert x.is_contiguous()
        BH, N, D = x.shape
        xv_t = x.permute(1, 2, 0)
    assert D == TILE_K
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
    assign, finalize = _get_compiled(N, K, BH, device, views,
                                     rank4=(x.dim() == 4))
    xv, cv, csqv, iv, bv, hv, sv, qv = views

    stream = cuda.CUstream(torch.cuda.current_stream(device).cuda_stream)
    for it in range(max_iters):
        assign(xv, cv, csqv, iv, bv, hv, sv, qv,
               Int32(1 if it == max_iters - 1 else 0), stream)
        finalize(sv, hv, cv, csqv, Int32(1 if it < max_iters - 1 else 0), stream)
    return ids, best, hist, cents, xsq
