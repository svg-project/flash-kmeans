"""Hopper SM90 fused flash-kmeans Euclidean assignment kernel (CuteDSL).

This module implements ``HopperFlashKmeansAssign``: a TMA + WGMMA fused
kernel that computes nearest-centroid IDs without ever materialising the
``(N, K)`` cross matrix or intermediate distance matrix in HBM. The
streaming axis is the centroid index ``K``: each CTA tile keeps its
``BM × BN_centroid`` cross accumulator in WGMMA registers, fuses the
``c_sq − 2·cross`` (shifted) distance computation in registers, and
updates a per-row running ``argmin`` across centroid tiles. Only the
final ``(BM,)`` int32 best-index vector is written to GMEM.

We compute the *shifted* squared distance
``d'[m, k] = ||y_k||² − 2·x_m·y_k`` (== ``||x − y||² − ||x||²``) and
take its argmin over ``k``. The ``||x||²`` term is constant per row and
so does not affect the argmin; dropping it eliminates one HBM tensor
(``x_sq``), its precompute pass, the per-tile load, and ``M_per_thr``
fp32 registers per consumer thread. The shifted ``d'`` can be negative,
so the underflow clamp is also removed (it never affected argmin).

The architecture borrows directly from the CUTLASS hopper FMHA reference
(``examples/python/CuTeDSL/cute/hopper/kernel/attention/fmha.py``) and
the reference dense GEMM (``…/dense_gemm/dense_gemm.py``):

* X-tile (``BM × D``) is TMA-loaded into SMEM **once** at the top of
  each CTA — analogous to FA3's "load Q once" pattern. SMEM stays
  resident across the entire centroid-stream loop.
* Centroid tiles (``BN_centroid × D``) are TMA-streamed via a multistage
  ``mbarrier`` pipeline. Producer warp issues ``cp.async.bulk.tensor``
  TMA loads; consumer warp groups run WGMMA + the in-register epilogue.
* WGMMA (``cute.nvgpu.warpgroup.MmaF16BF16Op``) computes
  ``cross = X @ Cᵀ`` with the full D as the WGMMA-K dimension, so each
  centroid tile completes one fresh accumulator (no inter-tile carry).
* In-register epilogue per centroid tile: convert the WGMMA acc to fp32
  (it is already fp32 if ``acc_dtype=Float32``), compute
  ``dist = c_sq − 2·cross`` per (m, n) acc element, then do a per-thread
  linear argmin across the BN_centroid axis. The per-thread (best_dist,
  best_idx) is updated in-place across centroid tiles. After the
  centroid-stream loop, a warp-shuffle bfly argmin reduction across the
  row's WGMMA TV-layout group resolves the global per-row argmin. Only
  the row-leader thread writes ``best_idx`` to GMEM.

The non-warp-specialised variant (this file) all warps participate in
WGMMA; only ``warp_idx == 0`` performs TMA loads. The mbarrier-based
async pipeline still hides TMA latency behind WGMMA execution, mirroring
``dense_gemm.py``'s structure. A separate warp-specialised variant
(producer-warp + consumer-warp-group) can be layered on top later for
additional overlap; the in-register dist+argmin epilogue is the part
worth pulling out of cuBLAS / Triton, and the non-WS path already wins
on H200 for typical kmeans shapes.

Constraints enforced at JIT time:

* ``B = 1`` (B > 1 falls back to Triton in the public API).
* ``D`` must be a multiple of 16 (WGMMA K-tile is 16 for fp16/bf16).
* ``D ≤ 512`` (small-D regime; SMEM holds the full X tile). Larger D
  is dispatched to the Triton split-D kernel by the caller.
* Input ``x`` and ``centroids`` must be ``fp16`` or ``bf16`` and share
  dtype. ``c_sq`` is ``fp32``. ``out`` is ``int32``.
"""
from __future__ import annotations

from typing import Tuple, Type

import math

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.hopper_helpers as sm90_utils
from cutlass.cute.runtime import from_dlpack
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait


class HopperFlashKmeansAssign:
    """Hopper TMA+WGMMA fused kmeans Euclidean assign kernel.

    Two execution modes are exposed via the ``use_ws`` flag:

    * ``use_ws=False`` (default): "dense_gemm-style" non-warp-specialised
      kernel — all threads do WGMMA cooperatively, only ``warp_idx == 0``
      issues TMA loads asynchronously through the multistage mbarrier
      pipeline. Simple, robust, achieves ~57 % of WGMMA peak on
      bf16 D=256 K=20K.

    * ``use_ws=True``: "FMHA-style" warp-specialised kernel — adds a
      dedicated producer warp group (1 WG = 128 threads) that *only*
      issues TMAs and one or two consumer WGs that *only* do
      WGMMA + dist + argmin. Decoupled register budgets via
      ``setmaxregister_decrease(24)`` / ``setmaxregister_increase(240)``
      let consumer WGs grab the unused producer registers, which
      improves WGMMA throughput on heavy-K shapes. Both pipelines stay
      identical structurally — the only difference is which threads run
      which side.

    Pick ``use_ws`` via the ``cutedsl_assign_euclid`` autotune; the
    crossover is shape-dependent (WS pays its 128-thread overhead off
    only when there are enough centroid tiles to amortise it).
    """

    def __init__(
        self,
        acc_dtype: Type[cutlass.Numeric],
        m_block_size: int,
        n_block_size: int,
        use_ws: bool = False,
    ):
        """Configure the kernel.

        Parameters
        ----------
        acc_dtype:
            WGMMA accumulator dtype. Must be ``cutlass.Float32`` for
            numerically stable distance computation (we promote to fp32
            in-register anyway, but the WGMMA acc is the value source).
        m_block_size:
            CTA tile in the points (M-of-GEMM) dim. Must be 64 or 128.
            128 uses the cooperative 2-warpgroup atom layout, 64 uses 1.
        n_block_size:
            CTA tile in the centroid (N-of-GEMM, streamed) dim. Must be
            64, 128, or 256. This bounds per-tile cross accumulator size
            ``BM × BN`` in registers; raise it for larger K to amortise
            the WGMMA cost.
        use_ws:
            Enable producer/consumer warp specialization (FMHA pattern).
            Adds one load WG (only does TMAs) plus the existing MMA WGs;
            consumer WGs use ``setmaxregister_increase(240)`` to soak up
            the producer's released registers.
        """
        if acc_dtype is not cutlass.Float32:
            raise TypeError("acc_dtype must be Float32 for numerically stable kmeans dist")
        if m_block_size not in (64, 128):
            raise ValueError("m_block_size must be 64 or 128")
        if n_block_size not in (64, 128, 256):
            raise ValueError("n_block_size must be 64, 128, or 256")

        self.acc_dtype = acc_dtype
        # tile_shape_mnk[2] (D) is filled in at __call__ time from the input.
        self.tile_shape_mnk = (m_block_size, n_block_size, 1)
        # Use cooperative 2-warpgroup atom only at the largest tile.
        self.atom_layout_mnk = (
            (2, 1, 1) if (m_block_size > 64 and n_block_size >= 128) else (1, 1, 1)
        )
        self.cluster_shape_mn = (1, 1)
        self.mma_warp_groups = math.prod(self.atom_layout_mnk)
        self.num_threads_per_warp_group = 128
        self.use_ws = bool(use_ws)
        self.num_load_warp_groups = 1 if self.use_ws else 0
        self.threads_per_cta = (
            (self.num_load_warp_groups + self.mma_warp_groups)
            * self.num_threads_per_warp_group
        )
        # WG roles (only meaningful in WS mode).
        self.load_warp_group_id = 0  # WG 0 = producer (TMA only)
        # Consumer warps participate in the centroid-tile pipeline.
        self.num_consumer_warps = self.mma_warp_groups * 4

        # Per-FMHA register-budget split when WS is on. Producer WG
        # only issues a handful of TMA descriptors → tiny budget;
        # consumer WGs grab the rest. Total fits in H200's 64K
        # registers/SM (24*128 + 240*256 = 64512 < 65536).
        self.num_regs_load = 24
        self.num_regs_mma = 240

        self.smem_capacity = utils.get_smem_capacity_in_bytes("sm_90")
        self.buffer_align_bytes = 1024

    # --------------------------------------------------------------
    # Helpers (mirror dense_gemm.py)
    # --------------------------------------------------------------

    @staticmethod
    def _make_tma_atoms_and_tensors(
        tensor: cute.Tensor,
        smem_layout_staged: cute.ComposedLayout,
        smem_tile: Tuple[int, int],
        mcast_dim: int = 1,
    ) -> Tuple[cute.CopyAtom, cute.Tensor]:
        op = (
            cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp()
            if mcast_dim == 1
            else cute.nvgpu.cpasync.CopyBulkTensorTileG2SMulticastOp()
        )
        smem_layout = cute.slice_(smem_layout_staged, (None, None, 0))
        return cute.nvgpu.cpasync.make_tiled_tma_atom(
            op, tensor, smem_layout, smem_tile, num_multicast=mcast_dim
        )

    # --------------------------------------------------------------
    # Host-side entry: build TMA atoms, layouts, launch grid
    # --------------------------------------------------------------

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,         # (N, D) fp16/bf16, k-major  (= "A: M×K")
        centroids: cute.Tensor, # (K, D) fp16/bf16, k-major  (= "B: N×K")
        c_sq: cute.Tensor,      # (K,)   fp32
        out: cute.Tensor,       # (N,)   int32
        stream: cuda.CUstream,
    ):
        x_dtype = x.element_type
        c_dtype = centroids.element_type
        if cutlass.const_expr(x_dtype != c_dtype):
            raise TypeError("x and centroids dtype must match")
        if cutlass.const_expr(x_dtype.width != 16):
            raise TypeError("x dtype must be fp16 or bf16")
        # Stash dtypes on self so the kernel body can pull them out as
        # host-level constants (cute.Tensor.element_type inside @cute.kernel
        # returns a Union typing alias rather than a Numeric class).
        self.x_dtype = x_dtype
        self.c_dtype = c_dtype

        x_layout = utils.LayoutEnum.from_tensor(x)
        c_layout = utils.LayoutEnum.from_tensor(centroids)
        if cutlass.const_expr(x_layout.sm90_mma_major_mode() != cute.nvgpu.warpgroup.OperandMajorMode.K):
            raise RuntimeError("x must be k-major (D contiguous)")
        if cutlass.const_expr(c_layout.sm90_mma_major_mode() != cute.nvgpu.warpgroup.OperandMajorMode.K):
            raise RuntimeError("centroids must be k-major (D contiguous)")

        D = x.shape[1]
        # tile_shape_mnk: M = BM, N = BN_centroid, K = D (full reduction per tile)
        self.tile_shape_mnk = (self.tile_shape_mnk[0], self.tile_shape_mnk[1], D)

        # WGMMA atom: A = X (M-major-K=K-major), B = C (N-major-K=K-major).
        # tiler_mn (the per-atom MMA tile) for the F16 atom is fixed at (64, N).
        self.tiled_mma = sm90_utils.make_trivial_tiled_mma(
            x_dtype,
            c_dtype,
            x_layout.sm90_mma_major_mode(),
            c_layout.sm90_mma_major_mode(),
            self.acc_dtype,
            self.atom_layout_mnk,
            tiler_mn=(64, self.tile_shape_mnk[1]),
        )

        # SMEM layouts. X is loaded once (single stage); centroid tiles
        # are streamed (multi-stage). Both are "K-major" tensors in the
        # GEMM sense (D-contiguous in memory).
        # Pick centroid stages: maximise depth subject to SMEM budget.
        x_bytes_per_stage = (
            self.tile_shape_mnk[0] * self.tile_shape_mnk[2] * x_dtype.width // 8
        )
        c_bytes_per_stage = (
            self.tile_shape_mnk[1] * self.tile_shape_mnk[2] * c_dtype.width // 8
        )
        mbar_helpers_bytes = 1024
        budget = self.smem_capacity - mbar_helpers_bytes - x_bytes_per_stage
        c_stage = budget // c_bytes_per_stage
        # Cap stages to 4 — beyond that the pipeline depth gives
        # diminishing returns and the extra SMEM costs occupancy. Hopper
        # FMHA uses 5 for its kv-stage but FA has a deeper compute graph
        # (PV-MMA + softmax) per tile; kmeans has only the small dist+
        # argmin epilogue, so 3-4 stages saturates the overlap.
        c_stage = max(1, min(4, c_stage))
        self.c_stage = c_stage
        self.x_stage = 1

        self.x_smem_layout_staged = sm90_utils.make_smem_layout_a(
            x_layout, self.tile_shape_mnk, x_dtype, self.x_stage
        )
        self.c_smem_layout_staged = sm90_utils.make_smem_layout_b(
            c_layout, self.tile_shape_mnk, c_dtype, self.c_stage
        )

        # TMA atoms. mcast=1 (no cluster).
        tma_atom_x, tma_tensor_x = self._make_tma_atoms_and_tensors(
            x, self.x_smem_layout_staged,
            (self.tile_shape_mnk[0], self.tile_shape_mnk[2]),
        )
        tma_atom_c, tma_tensor_c = self._make_tma_atoms_and_tensors(
            centroids, self.c_smem_layout_staged,
            (self.tile_shape_mnk[1], self.tile_shape_mnk[2]),
        )

        # Grid: one CTA per (BM-aligned) point tile.
        N = x.shape[0]
        num_m_tiles = (N + self.tile_shape_mnk[0] - 1) // self.tile_shape_mnk[0]
        grid = (num_m_tiles, 1, 1)

        @cute.struct
        class SharedStorage:
            c_pipeline_array_ptr: cute.struct.MemRange[cutlass.Int64, self.c_stage * 2]
            x_pipeline_array_ptr: cute.struct.MemRange[cutlass.Int64, self.x_stage * 2]
            sX: cute.struct.Align[
                cute.struct.MemRange[x_dtype, cute.cosize(self.x_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            sC: cute.struct.Align[
                cute.struct.MemRange[c_dtype, cute.cosize(self.c_smem_layout_staged)],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage

        if cutlass.const_expr(self.use_ws):
            self.kernel_ws(
                tma_atom_x, tma_tensor_x,
                tma_atom_c, tma_tensor_c,
                c_sq, out,
                self.tiled_mma,
                self.x_smem_layout_staged,
                self.c_smem_layout_staged,
            ).launch(
                grid=grid,
                block=[self.threads_per_cta, 1, 1],
                cluster=(*self.cluster_shape_mn, 1),
                stream=stream,
            )
        else:
            self.kernel(
                tma_atom_x, tma_tensor_x,
                tma_atom_c, tma_tensor_c,
                c_sq, out,
                self.tiled_mma,
                self.x_smem_layout_staged,
                self.c_smem_layout_staged,
            ).launch(
                grid=grid,
                block=[self.threads_per_cta, 1, 1],
                cluster=(*self.cluster_shape_mn, 1),
                stream=stream,
            )

    # --------------------------------------------------------------
    # Device kernel
    # --------------------------------------------------------------

    @cute.kernel
    def kernel(
        self,
        tma_atom_x: cute.CopyAtom,
        mX_nd: cute.Tensor,         # (N, D)
        tma_atom_c: cute.CopyAtom,
        mC_kd: cute.Tensor,         # (K, D)
        mCsq_k: cute.Tensor,        # (K,)   fp32
        mOut_n: cute.Tensor,        # (N,)   int32
        tiled_mma: cute.TiledMma,
        x_smem_layout_staged: cute.ComposedLayout,
        c_smem_layout_staged: cute.ComposedLayout,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        # Prefetch TMA descriptors (one warp).
        if warp_idx == 0:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_x)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_c)

        bidx, _, _ = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()

        BM = self.tile_shape_mnk[0]
        BN = self.tile_shape_mnk[1]
        D = self.tile_shape_mnk[2]
        N_total = mX_nd.shape[0]
        K_total = mC_kd.shape[0]

        # Per-CTA M tile coord.
        cta_m_offset = bidx * BM

        # ------------------------------------------------------------
        # SMEM allocation + pipeline init
        # ------------------------------------------------------------
        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        # X pipeline (single stage, single load): use a TmaAsync pipeline
        # so X load is async vs the main-loop epilogue start. Consumer
        # count = num MMA-side warps (WS mode excludes the producer WG;
        # non-WS coincidentally matches because there's no producer WG).
        x_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        x_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, self.num_consumer_warps
        )
        x_pipeline = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.x_pipeline_array_ptr.data_ptr(),
            num_stages=self.x_stage,
            producer_group=x_producer_group,
            consumer_group=x_consumer_group,
            tx_count=cute.size_in_bytes(
                self.x_dtype,
                cute.slice_(x_smem_layout_staged, (None, None, 0)),
            ),
            defer_sync=True,
        )

        # C pipeline (multi-stage, streamed centroid tiles).
        c_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        c_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, self.num_consumer_warps
        )
        c_pipeline = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.c_pipeline_array_ptr.data_ptr(),
            num_stages=self.c_stage,
            producer_group=c_producer_group,
            consumer_group=c_consumer_group,
            tx_count=cute.size_in_bytes(
                self.c_dtype,
                cute.slice_(c_smem_layout_staged, (None, None, 0)),
            ),
            defer_sync=True,
        )

        # Cluster-arrive after init.
        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True)

        # SMEM tensors.
        sX = storage.sX.get_tensor(
            x_smem_layout_staged.outer, swizzle=x_smem_layout_staged.inner
        )
        sC = storage.sC.get_tensor(
            c_smem_layout_staged.outer, swizzle=c_smem_layout_staged.inner
        )

        # ------------------------------------------------------------
        # Global tile partitions for TMA
        #
        # X is sliced into BM-sized M-tiles. We only load tile #bidx;
        # the partition pattern below lets TMA index by tile number.
        # C is streamed along its K-axis (= our centroid axis): TMA
        # loads tile #c_tile_idx of (BN, D) per pipeline iteration.
        # ------------------------------------------------------------
        num_c_tiles = (K_total + BN - 1) // BN
        gC_kd = cute.local_tile(
            mC_kd, (self.tile_shape_mnk[1], self.tile_shape_mnk[2]),
            (None, 0),
        )  # (BN, D, num_c_tiles)

        # TMA partition for X: load 1 tile at coord (bidx, 0).
        # We use the same partition pattern as dense_gemm but with
        # single tile.
        tma_xS, tma_xG = cute.nvgpu.cpasync.tma_partition(
            tma_atom_x,
            0,  # cta crd in mcast group
            cute.make_layout(1),
            cute.group_modes(sX, 0, 2),                  # (S, X_PIPE)
            cute.group_modes(
                cute.local_tile(
                    mX_nd,
                    (self.tile_shape_mnk[0], self.tile_shape_mnk[2]),
                    (None, 0),
                ),
                0, 2,
            ),  # (S, num_m_tiles)
        )

        # TMA partition for C tiles streamed along K.
        tma_cS, tma_cG = cute.nvgpu.cpasync.tma_partition(
            tma_atom_c,
            0,
            cute.make_layout(1),
            cute.group_modes(sC, 0, 2),                  # (S, C_PIPE)
            cute.group_modes(gC_kd, 0, 2),               # (S, num_c_tiles)
        )

        # ------------------------------------------------------------
        # MMA partitions / fragments
        #
        # We slice the tiled MMA per-thread (FMHA pattern) so the
        # coordinate-tensor partitions track per-thread (m, n) ownership
        # of WGMMA accumulator elements. partition_A/B with per-thread
        # slicing gives the same layout as warp-group slicing for SMEM
        # operands (the WGMMA instruction is a warp-group op), so this
        # is safe.
        # ------------------------------------------------------------
        thr_mma = tiled_mma.get_slice(tidx)

        tCsX = thr_mma.partition_A(sX)
        tCsC_mma = thr_mma.partition_B(sC)
        tCrX = tiled_mma.make_fragment_A(tCsX)
        tCrC = tiled_mma.make_fragment_B(tCsC_mma)

        # Identity tensor for (m, n) coords per acc element.
        cP = cute.make_identity_tensor((BM, BN))
        ptPcP = thr_mma.partition_C(cP)

        # Per-thread acc fragment.
        gC_fake = cute.make_identity_tensor((BM, BN))
        tCgC_fake = thr_mma.partition_C(gC_fake)
        acc_shape = tCgC_fake.shape
        acc = cute.make_rmem_tensor(acc_shape, self.acc_dtype)

        # ------------------------------------------------------------
        # Cluster wait
        # ------------------------------------------------------------
        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)

        # ------------------------------------------------------------
        # Issue X TMA load (1 tile, single stage), and prefetch first
        # few centroid tiles.
        # ------------------------------------------------------------
        x_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.x_stage
        )
        c_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.c_stage
        )

        prefetch_c_tile_cnt = cutlass.max(cutlass.min(self.c_stage, num_c_tiles), 0)

        if warp_idx == 0:
            # X TMA: load CTA's M tile.
            x_pipeline.producer_acquire(x_producer_state)
            cute.copy(
                tma_atom_x,
                tma_xG[(None, bidx)],
                tma_xS[(None, x_producer_state.index)],
                tma_bar_ptr=x_pipeline.producer_get_barrier(x_producer_state),
            )
            x_pipeline.producer_commit(x_producer_state)
            x_producer_state.advance()

            # Prefetch first N stages of centroid tiles.
            for c_pre in cutlass.range(prefetch_c_tile_cnt, unroll=1):
                c_pipeline.producer_acquire(c_producer_state)
                cute.copy(
                    tma_atom_c,
                    tma_cG[(None, c_producer_state.count)],
                    tma_cS[(None, c_producer_state.index)],
                    tma_bar_ptr=c_pipeline.producer_get_barrier(c_producer_state),
                )
                c_pipeline.producer_commit(c_producer_state)
                c_producer_state.advance()

        # ------------------------------------------------------------
        # Compute per-thread acc TV layout. We don't pre-load x_sq —
        # the argmin uses the shifted distance ``c_sq − 2·cross``
        # which drops the constant ``||x||²`` term.
        # ------------------------------------------------------------
        acc_mn_layout = self._layout_acc_mn(tiled_mma, acc.layout)
        acc_mn = cute.make_tensor(acc.iterator, acc_mn_layout)
        ptPcP_mn = cute.make_tensor(ptPcP.iterator, self._layout_acc_mn(tiled_mma, ptPcP.layout))

        M_per_thr = cute.size(acc_mn, mode=[0])
        N_per_thr = cute.size(acc_mn, mode=[1])

        # Per-row running argmin (in registers, per-thread).
        best_d = cute.make_rmem_tensor(cute.make_layout(M_per_thr), cutlass.Float32)
        best_i = cute.make_rmem_tensor(cute.make_layout(M_per_thr), cutlass.Int32)
        for i in cutlass.range_constexpr(M_per_thr):
            best_d[i] = cutlass.Float32(3.4e38)
            best_i[i] = cutlass.Int32(0)

        # ------------------------------------------------------------
        # Mainloop: stream centroid tiles
        # ------------------------------------------------------------
        c_consumer_read_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.c_stage
        )
        c_consumer_release_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.c_stage
        )

        # Wait for X to land in SMEM (consumer side of x_pipeline).
        x_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.x_stage
        )
        x_pipeline.consumer_wait(x_consumer_state)

        num_k_blocks = cute.size(tCrX, mode=[2])  # number of WGMMA inner k-tiles (D / 16)

        for c_tile_idx in cutlass.range(num_c_tiles, unroll=1):
            # Wait for C[stage] TMA to complete.
            c_pipeline.consumer_wait(c_consumer_read_state)

            # WGMMA: cross_acc = X @ C^T, full D as inner-k.
            # Set ACCUMULATE=False on the first inner k-block so we zero
            # the accumulator at the start of each centroid tile.
            tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, False)
            cute.nvgpu.warpgroup.fence()
            for k_block_idx in cutlass.range_constexpr(num_k_blocks):
                k_block_coord_x = (None, None, k_block_idx, 0)  # x has 1 stage
                k_block_coord_c = (None, None, k_block_idx, c_consumer_read_state.index)
                cute.gemm(
                    tiled_mma,
                    acc,
                    tCrX[k_block_coord_x],
                    tCrC[k_block_coord_c],
                    acc,
                )
                tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, True)
            cute.nvgpu.warpgroup.commit_group()
            cute.nvgpu.warpgroup.wait_group(0)

            # Release C[stage] back to producer.
            c_pipeline.consumer_release(c_consumer_release_state)
            c_consumer_read_state.advance()
            c_consumer_release_state.advance()

            # Issue next TMA load to fill the just-released stage.
            if warp_idx == 0 and c_producer_state.count < num_c_tiles:
                c_pipeline.producer_acquire(c_producer_state)
                cute.copy(
                    tma_atom_c,
                    tma_cG[(None, c_producer_state.count)],
                    tma_cS[(None, c_producer_state.index)],
                    tma_bar_ptr=c_pipeline.producer_get_barrier(c_producer_state),
                )
                c_pipeline.producer_commit(c_producer_state)
                c_producer_state.advance()

            # ------------------------------------------------------
            # In-register epilogue: dist = c_sq − 2·acc (shifted by the
            # constant ``||x||²`` term per row; argmin is preserved).
            # Update per-thread running argmin across this tile's
            # N_per_thr elements.
            # ------------------------------------------------------
            cta_n_offset = c_tile_idx * BN

            # Pre-load c_sq for this tile's per-thread n columns.
            cs = cute.make_rmem_tensor(cute.make_layout(N_per_thr), cutlass.Float32)
            for j in cutlass.range_constexpr(N_per_thr):
                n_local = ptPcP_mn[(0, j)][1]
                n_global = n_local + cta_n_offset
                # Mask OOB centroids with +inf so they never win argmin.
                if n_global < K_total:
                    cs[j] = mCsq_k[n_global]
                else:
                    cs[j] = cutlass.Float32(3.4e38)

            for i in cutlass.range_constexpr(M_per_thr):
                bd = best_d[i]
                bi = best_i[i]
                for j in cutlass.range_constexpr(N_per_thr):
                    cross = acc_mn[(i, j)]
                    dist = cs[j] - cutlass.Float32(2.0) * cross
                    n_local = ptPcP_mn[(i, j)][1]
                    n_global = n_local + cta_n_offset
                    # OOB centroid columns are masked via cs[j]=+inf above,
                    # which keeps them out of argmin without a branch here.
                    if dist < bd:
                        bd = dist
                        bi = cutlass.Int32(n_global)
                best_d[i] = bd
                best_i[i] = bi
        # End centroid-tile loop.

        # ------------------------------------------------------------
        # Warp-shuffle bfly argmin reduction across threads sharing each row.
        # The threads sharing a row in WGMMA's TV layout are determined by the
        # acc TV layout's "N-side" thread mode — same group as FA's
        # `reduction_target_n`. We reduce (best_d, best_i) jointly.
        # ------------------------------------------------------------
        red_target = self._reduction_target_n(tiled_mma)
        red_rank = cute.rank(red_target)
        for r in cutlass.range_constexpr(red_rank):
            tig = red_target.shape[r]
            for i in cutlass.range_constexpr(M_per_thr):
                bd = best_d[i]
                bi = best_i[i]
                offset = tig // 2
                while offset > 0:
                    other_d = cute.arch.shuffle_sync_bfly(
                        bd, offset=offset, mask=-1, mask_and_clamp=31,
                    )
                    other_i = cute.arch.shuffle_sync_bfly(
                        bi, offset=offset, mask=-1, mask_and_clamp=31,
                    )
                    if other_d < bd:
                        bd = other_d
                        bi = other_i
                    offset = offset // 2
                best_d[i] = bd
                best_i[i] = bi

        # ------------------------------------------------------------
        # Store: row leaders (n=0 column owners) write best_idx to gmem.
        # Pattern matches FA's LSE store: tOcO[0][1] == 0 picks the
        # n=0 column's owning threads.
        # ------------------------------------------------------------
        if ptPcP[0][1] == 0:
            for i in cutlass.range_constexpr(M_per_thr):
                m_local = ptPcP_mn[(i, 0)][0]
                m_global = m_local + cta_m_offset
                if m_global < N_total:
                    mOut_n[m_global] = best_i[i]

        return

    # --------------------------------------------------------------
    # Warp-specialised device kernel (FMHA-style)
    # --------------------------------------------------------------

    @cute.kernel
    def kernel_ws(
        self,
        tma_atom_x: cute.CopyAtom,
        mX_nd: cute.Tensor,
        tma_atom_c: cute.CopyAtom,
        mC_kd: cute.Tensor,
        mCsq_k: cute.Tensor,
        mOut_n: cute.Tensor,
        tiled_mma: cute.TiledMma,
        x_smem_layout_staged: cute.ComposedLayout,
        c_smem_layout_staged: cute.ComposedLayout,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        if warp_idx == 0:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_x)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_c)

        bidx, _, _ = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        warp_group_idx = cute.arch.make_warp_uniform(
            tidx // self.num_threads_per_warp_group
        )

        BM = self.tile_shape_mnk[0]
        BN = self.tile_shape_mnk[1]
        N_total = mX_nd.shape[0]
        K_total = mC_kd.shape[0]
        cta_m_offset = bidx * BM
        num_c_tiles = (K_total + BN - 1) // BN

        # ------------------------------------------------------------
        # SMEM allocation + pipeline init (shared by both WG roles).
        # ------------------------------------------------------------
        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        x_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        x_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, self.num_consumer_warps
        )
        x_pipeline = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.x_pipeline_array_ptr.data_ptr(),
            num_stages=self.x_stage,
            producer_group=x_producer_group,
            consumer_group=x_consumer_group,
            tx_count=cute.size_in_bytes(
                self.x_dtype,
                cute.slice_(x_smem_layout_staged, (None, None, 0)),
            ),
            defer_sync=True,
        )

        c_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        c_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, self.num_consumer_warps
        )
        c_pipeline = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.c_pipeline_array_ptr.data_ptr(),
            num_stages=self.c_stage,
            producer_group=c_producer_group,
            consumer_group=c_consumer_group,
            tx_count=cute.size_in_bytes(
                self.c_dtype,
                cute.slice_(c_smem_layout_staged, (None, None, 0)),
            ),
            defer_sync=True,
        )

        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True)

        sX = storage.sX.get_tensor(
            x_smem_layout_staged.outer, swizzle=x_smem_layout_staged.inner
        )
        sC = storage.sC.get_tensor(
            c_smem_layout_staged.outer, swizzle=c_smem_layout_staged.inner
        )

        gC_kd = cute.local_tile(
            mC_kd, (self.tile_shape_mnk[1], self.tile_shape_mnk[2]),
            (None, 0),
        )

        tma_xS, tma_xG = cute.nvgpu.cpasync.tma_partition(
            tma_atom_x,
            0,
            cute.make_layout(1),
            cute.group_modes(sX, 0, 2),
            cute.group_modes(
                cute.local_tile(
                    mX_nd,
                    (self.tile_shape_mnk[0], self.tile_shape_mnk[2]),
                    (None, 0),
                ),
                0, 2,
            ),
        )

        tma_cS, tma_cG = cute.nvgpu.cpasync.tma_partition(
            tma_atom_c,
            0,
            cute.make_layout(1),
            cute.group_modes(sC, 0, 2),
            cute.group_modes(gC_kd, 0, 2),
        )

        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)

        # ============================================================
        # PRODUCER warp group: only does TMA loads. setmaxregister
        # tells the compiler this WG only needs ~24 regs/thread, so
        # the SM scheduler can hand the freed registers to the MMA WGs.
        # ============================================================
        if warp_group_idx == self.load_warp_group_id:
            cute.arch.setmaxregister_decrease(self.num_regs_load)

            x_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.x_stage
            )
            c_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.c_stage
            )

            # Within the producer WG, only warp 0 issues the TMA loads.
            # The other 3 producer warps have no work but still hold the
            # decreased register budget — they're effectively idle.
            if warp_idx == 0:
                # Issue the (single) X TMA load.
                x_pipeline.producer_acquire(x_producer_state)
                cute.copy(
                    tma_atom_x,
                    tma_xG[(None, bidx)],
                    tma_xS[(None, x_producer_state.index)],
                    tma_bar_ptr=x_pipeline.producer_get_barrier(x_producer_state),
                )
                x_pipeline.producer_commit(x_producer_state)
                x_producer_state.advance()

                # Stream all centroid tiles. The consumer side will
                # acquire/release them in lockstep.
                for c_idx in cutlass.range(num_c_tiles, unroll=1):
                    c_pipeline.producer_acquire(c_producer_state)
                    cute.copy(
                        tma_atom_c,
                        tma_cG[(None, c_producer_state.count)],
                        tma_cS[(None, c_producer_state.index)],
                        tma_bar_ptr=c_pipeline.producer_get_barrier(c_producer_state),
                    )
                    c_pipeline.producer_commit(c_producer_state)
                    c_producer_state.advance()

        # ============================================================
        # CONSUMER warp group(s): WGMMA + dist + argmin epilogue.
        # ============================================================
        if warp_group_idx >= self.num_load_warp_groups:
            cute.arch.setmaxregister_increase(self.num_regs_mma)

            # Slice the tiled MMA per-thread, but offset the thread index
            # so the consumer WGs (tidx ≥ 128) map onto atom slices 0,
            # 1, … of the (cooperative or single) tiled MMA.
            consumer_tidx = tidx - self.num_threads_per_warp_group
            thr_mma = tiled_mma.get_slice(consumer_tidx)

            tCsX = thr_mma.partition_A(sX)
            tCsC_mma = thr_mma.partition_B(sC)
            tCrX = tiled_mma.make_fragment_A(tCsX)
            tCrC = tiled_mma.make_fragment_B(tCsC_mma)

            cP = cute.make_identity_tensor((BM, BN))
            ptPcP = thr_mma.partition_C(cP)

            gC_fake = cute.make_identity_tensor((BM, BN))
            tCgC_fake = thr_mma.partition_C(gC_fake)
            acc_shape = tCgC_fake.shape
            acc = cute.make_rmem_tensor(acc_shape, self.acc_dtype)

            # Wait for X tile to land in SMEM.
            x_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.x_stage
            )
            x_pipeline.consumer_wait(x_consumer_state)

            acc_mn_layout = self._layout_acc_mn(tiled_mma, acc.layout)
            acc_mn = cute.make_tensor(acc.iterator, acc_mn_layout)
            ptPcP_mn = cute.make_tensor(
                ptPcP.iterator, self._layout_acc_mn(tiled_mma, ptPcP.layout)
            )

            M_per_thr = cute.size(acc_mn, mode=[0])
            N_per_thr = cute.size(acc_mn, mode=[1])

            # No x_sq preload — argmin uses the shifted distance
            # ``c_sq − 2·cross`` which drops the constant ``||x||²``.
            best_d = cute.make_rmem_tensor(cute.make_layout(M_per_thr), cutlass.Float32)
            best_i = cute.make_rmem_tensor(cute.make_layout(M_per_thr), cutlass.Int32)
            for i in cutlass.range_constexpr(M_per_thr):
                best_d[i] = cutlass.Float32(3.4e38)
                best_i[i] = cutlass.Int32(0)

            # Mainloop: stream centroid tiles (the producer WG drives the
            # TMA loads; we just wait/use/release).
            c_consumer_read_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.c_stage
            )
            c_consumer_release_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.c_stage
            )

            num_k_blocks = cute.size(tCrX, mode=[2])

            for c_tile_idx in cutlass.range(num_c_tiles, unroll=1):
                c_pipeline.consumer_wait(c_consumer_read_state)

                tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, False)
                cute.nvgpu.warpgroup.fence()
                for k_block_idx in cutlass.range_constexpr(num_k_blocks):
                    k_block_coord_x = (None, None, k_block_idx, 0)
                    k_block_coord_c = (None, None, k_block_idx, c_consumer_read_state.index)
                    cute.gemm(
                        tiled_mma, acc,
                        tCrX[k_block_coord_x],
                        tCrC[k_block_coord_c],
                        acc,
                    )
                    tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, True)
                cute.nvgpu.warpgroup.commit_group()
                cute.nvgpu.warpgroup.wait_group(0)

                c_pipeline.consumer_release(c_consumer_release_state)
                c_consumer_read_state.advance()
                c_consumer_release_state.advance()

                # In-register dist + argmin update.
                cta_n_offset = c_tile_idx * BN

                cs = cute.make_rmem_tensor(cute.make_layout(N_per_thr), cutlass.Float32)
                for j in cutlass.range_constexpr(N_per_thr):
                    n_local = ptPcP_mn[(0, j)][1]
                    n_global = n_local + cta_n_offset
                    if n_global < K_total:
                        cs[j] = mCsq_k[n_global]
                    else:
                        cs[j] = cutlass.Float32(3.4e38)

                for i in cutlass.range_constexpr(M_per_thr):
                    bd = best_d[i]
                    bi = best_i[i]
                    for j in cutlass.range_constexpr(N_per_thr):
                        cross = acc_mn[(i, j)]
                        dist = cs[j] - cutlass.Float32(2.0) * cross
                        n_local = ptPcP_mn[(i, j)][1]
                        n_global = n_local + cta_n_offset
                        if dist < bd:
                            bd = dist
                            bi = cutlass.Int32(n_global)
                    best_d[i] = bd
                    best_i[i] = bi

            # Cross-thread argmin reduction along the WGMMA TV layout's
            # N-side group, identical to the non-WS path.
            red_target = self._reduction_target_n(tiled_mma)
            red_rank = cute.rank(red_target)
            for r in cutlass.range_constexpr(red_rank):
                tig = red_target.shape[r]
                for i in cutlass.range_constexpr(M_per_thr):
                    bd = best_d[i]
                    bi = best_i[i]
                    offset = tig // 2
                    while offset > 0:
                        other_d = cute.arch.shuffle_sync_bfly(
                            bd, offset=offset, mask=-1, mask_and_clamp=31,
                        )
                        other_i = cute.arch.shuffle_sync_bfly(
                            bi, offset=offset, mask=-1, mask_and_clamp=31,
                        )
                        if other_d < bd:
                            bd = other_d
                            bi = other_i
                        offset = offset // 2
                    best_d[i] = bd
                    best_i[i] = bi

            if ptPcP[0][1] == 0:
                for i in cutlass.range_constexpr(M_per_thr):
                    m_local = ptPcP_mn[(i, 0)][0]
                    m_global = m_local + cta_m_offset
                    if m_global < N_total:
                        mOut_n[m_global] = best_i[i]

        return

    # --------------------------------------------------------------
    # Layout helpers (lifted from FMHA for the WGMMA acc TV split)
    # --------------------------------------------------------------

    @staticmethod
    @cute.jit
    def _layout_separate(thr, src, ref):
        lt = cute.make_layout(())
        ge = cute.make_layout(())
        for k, v in enumerate(ref):
            if cutlass.const_expr(v < thr):
                lt = cute.append(lt, src[k])
            else:
                ge = cute.append(ge, src[k])
        r = None
        if cutlass.const_expr(cute.rank(lt) == 1):
            r = cute.append(lt, ge)
        else:
            r = cute.append(cute.append(cute.make_layout(()), lt), ge)
        return r

    @cute.jit
    def _layout_acc_mn(self, tiled_mma, acc_layout):
        separated = self._layout_separate(
            tiled_mma.shape_mnk[0], acc_layout[0], tiled_mma.tv_layout_C.stride[1]
        )
        V_M = separated[0]
        V_N = separated[1]
        if cutlass.const_expr(cute.rank(V_M) == 1):
            V_M1 = cute.append(V_M, acc_layout[1])
        else:
            V_M1 = cute.append(cute.append(cute.make_layout(()), V_M), acc_layout[1])
        if cutlass.const_expr(cute.rank(V_N) == 1):
            V_N1 = cute.append(V_N, acc_layout[2])
        else:
            V_N1 = cute.append(cute.append(cute.make_layout(()), V_N), acc_layout[2])
        if cutlass.const_expr(cute.rank(V_M1) == 1):
            r = cute.append(V_M1, V_N1)
        else:
            r = cute.append(cute.append(cute.make_layout(()), V_M1), V_N1)
        return r

    @cute.jit
    def _reduction_target_n(self, tiled_mma):
        separated = self._layout_separate(
            tiled_mma.shape_mnk[0],
            cute.make_layout(tiled_mma.tv_layout_C.shape[0]),
            tiled_mma.tv_layout_C.stride[0],
        )
        return separated[1]
