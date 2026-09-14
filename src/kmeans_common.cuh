// Device-side helpers and kernel templates shared by the flash-kmeans CUDA
// translation units.
#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cfloat>
#include <cstdio>
#include <cstdlib>
#include <type_traits>

#define FK_CHECK(expr)                                                          \
  do {                                                                          \
    cudaError_t fk_err__ = (expr);                                              \
    if (fk_err__ != cudaSuccess) {                                              \
      std::fprintf(stderr, "flash-kmeans CUDA error at %s:%d: %s\n", __FILE__,  \
                   __LINE__, cudaGetErrorString(fk_err__));                     \
      std::abort();                                                             \
    }                                                                           \
  } while (0)

namespace fk {

// ---------------------------------------------------------------------------
// small device utilities
// ---------------------------------------------------------------------------

__device__ __forceinline__ std::uint64_t splitmix64(std::uint64_t z) {
  z += 0x9E3779B97F4A7C15ull;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}

__device__ __forceinline__ void atomic_max_float(float* addr, float val) {
  // diff^2 is non-negative, so unsigned integer order matches float order.
  if (val >= 0.f) {
    atomicMax(reinterpret_cast<unsigned int*>(addr), __float_as_uint(val));
  }
}

// Block-wide sum; must be called by all threads in the block.
__device__ __forceinline__ float block_reduce_sum(float v) {
  __shared__ float s_tmp[32];
  const int lane = threadIdx.x & 31;
  const int wid = threadIdx.x >> 5;
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    v += __shfl_down_sync(0xffffffffu, v, off);
  }
  if (lane == 0) s_tmp[wid] = v;
  __syncthreads();
  const int nwarps = (blockDim.x + 31) >> 5;
  v = (threadIdx.x < nwarps) ? s_tmp[threadIdx.x] : 0.f;
  if (wid == 0) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
      v += __shfl_down_sync(0xffffffffu, v, off);
    }
    if (lane == 0) s_tmp[0] = v;
  }
  __syncthreads();
  return s_tmp[0];
}

// ---------------------------------------------------------------------------
// dtype traits / conversion
// ---------------------------------------------------------------------------

template <typename T>
struct TypeTraits;

template <>
struct TypeTraits<float> {
  using Vec2 = float2;
  static constexpr int kCode = 0;
};

template <>
struct TypeTraits<__half> {
  using Vec2 = __half2;
  static constexpr int kCode = 1;
};

template <>
struct TypeTraits<__nv_bfloat16> {
  using Vec2 = __nv_bfloat162;
  static constexpr int kCode = 2;
};

// bf16 conversions are implemented with plain bit arithmetic so that the
// kernels also compile and run on pre-Ampere architectures (sm_61 / sm_75)
// where cvt.rn.bf16.f32 is not available.
__device__ __forceinline__ float bf16_to_float(__nv_bfloat16 v) {
  unsigned short u;
  memcpy(&u, &v, sizeof(u));
  return __uint_as_float((unsigned int)u << 16);
}

__device__ __forceinline__ __nv_bfloat16 float_to_bf16(float f) {
  unsigned int u = __float_as_uint(f);
  u += 0x7fffu + ((u >> 16) & 1u);  // round-to-nearest-even at bit 16
  unsigned short s = (unsigned short)(u >> 16);
  __nv_bfloat16 out;
  memcpy(&out, &s, sizeof(s));
  return out;
}

__device__ __forceinline__ float to_float(float v) { return v; }
__device__ __forceinline__ float to_float(__half v) { return __half2float(v); }
__device__ __forceinline__ float to_float(__nv_bfloat16 v) { return bf16_to_float(v); }

__device__ __forceinline__ float2 to_float2(float2 v) { return v; }
__device__ __forceinline__ float2 to_float2(__half2 v) { return __half22float2(v); }
__device__ __forceinline__ float2 to_float2(__nv_bfloat162 v) {
  float2 r;
  r.x = bf16_to_float(v.x);
  r.y = bf16_to_float(v.y);
  return r;
}

template <typename T>
__device__ __forceinline__ T from_float(float v);
template <>
__device__ __forceinline__ float from_float<float>(float v) {
  return v;
}
template <>
__device__ __forceinline__ __half from_float<__half>(float v) {
  return __float2half_rn(v);
}
template <>
__device__ __forceinline__ __nv_bfloat16 from_float<__nv_bfloat16>(float v) {
  return float_to_bf16(v);
}

template <typename T>
__device__ __forceinline__ float2 load2(const T* __restrict__ p) {
  if constexpr (std::is_same<T, float>::value) {
    return *reinterpret_cast<const float2*>(p);
  } else if constexpr (std::is_same<T, __half>::value) {
    return __half22float2(*reinterpret_cast<const __half2*>(p));
  } else {
    return to_float2(*reinterpret_cast<const __nv_bfloat162*>(p));
  }
}

template <typename T>
__device__ __forceinline__ float4 load4(const T* __restrict__ p) {
  if constexpr (std::is_same<T, float>::value) {
    return *reinterpret_cast<const float4*>(p);
  } else {
    float2 a = load2(p);
    float2 b = load2(p + 2);
    return make_float4(a.x, a.y, b.x, b.y);
  }
}

// Dot product of two rows of length d. VEC requires sizeof(T)==2 and even d.
template <typename T, bool VEC>
__device__ __forceinline__ float dot_row_f32(const T* __restrict__ a,
                                             const float* __restrict__ b, int d) {
  float acc = 0.f;
  if (VEC) {
    const int nv = d >> 1;
    const typename TypeTraits<T>::Vec2* av =
        reinterpret_cast<const typename TypeTraits<T>::Vec2*>(a);
#pragma unroll 4
    for (int i = 0; i < nv; ++i) {
      float2 fa = to_float2(av[i]);
      acc = fmaf(fa.x, b[2 * i], acc);
      acc = fmaf(fa.y, b[2 * i + 1], acc);
    }
  } else {
    // a lives in shared memory with an odd-word row stride (bank-conflict
    // free), so only scalar loads are guaranteed to be aligned here.
#pragma unroll 8
    for (int i = 0; i < d; ++i) acc = fmaf(to_float(a[i]), b[i], acc);
  }
  return acc;
}

// Convert an array of T to fp32 (used to pre-convert centroids per iteration).
template <typename T>
__global__ void to_fp32_kernel(const T* __restrict__ src, float* __restrict__ dst,
                               std::int64_t n) {
  const std::int64_t stride = (std::int64_t)gridDim.x * blockDim.x;
  for (std::int64_t i = (std::int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += stride) {
    dst[i] = to_float(src[i]);
  }
}

// Row stride (in 32-bit words) used by the shared-memory x tile.  An odd word
// count makes concurrent row reads by a warp bank-conflict free for 2-byte
// element types.
template <typename T>
__host__ __device__ __forceinline__ int shared_row_words(int d) {
  int words = (int)(((std::int64_t)d * (std::int64_t)sizeof(T) + 3) / 4);
  if ((words & 1) == 0) ++words;
  return words;
}

// ---------------------------------------------------------------------------
// assignment kernels
// ---------------------------------------------------------------------------

// One thread block handles `blockDim.x` consecutive points of one batch.  The
// x tile is staged in shared memory and the block streams over all centroids.
template <typename T, bool VEC, bool MAXIMIZE>
__global__ void assign_kernel(const T* __restrict__ x,
                              const float* __restrict__ xsq,
                              const float* __restrict__ cent,
                              const float* __restrict__ csq,
                              int n, int d, int k, int row_words,
                              int* __restrict__ labels) {
  extern __shared__ unsigned char smem_raw[];
  T* sx = reinterpret_cast<T*>(smem_raw);
  const int elems = row_words * (int)(4 / sizeof(T));
  const int bn = blockDim.x;
  const int b = blockIdx.y;
  const int n0 = blockIdx.x * bn;
  const int t = threadIdx.x;
  const int n_idx = n0 + t;

  const T* xbase = x + ((std::size_t)b * n + n0) * d;
  const int total = bn * d;
  for (int i = t; i < total; i += bn) {
    const int row = i / d;
    const int col = i - row * d;
    sx[row * elems + col] = (n0 + row < n) ? xbase[i] : from_float<T>(0.f);
  }
  __syncthreads();

  if (n_idx >= n) return;

  const T* srow = sx + (std::size_t)t * elems;
  const float* cbase = cent + (std::size_t)b * k * d;
  const float x2 = xsq ? xsq[(std::size_t)b * n + n_idx] : 0.f;

  float best = MAXIMIZE ? -FLT_MAX : FLT_MAX;
  int best_k = 0;
  for (int kk = 0; kk < k; ++kk) {
    const float dot = dot_row_f32<T, VEC>(srow, cbase + (std::size_t)kk * d, d);
    const float score = MAXIMIZE ? dot : (x2 + csq[(std::size_t)b * k + kk] - 2.f * dot);
    const bool better = MAXIMIZE ? (score > best) : (score < best);
    if (better) {
      best = score;
      best_k = kk;
    }
  }
  labels[(std::size_t)b * n + n_idx] = best_k;
}

// Specialised variant for small compile-time feature dimensions: the x row is
// kept in fp32 registers, shared by SPLIT consecutive lanes of a warp, and the
// centroid rows are streamed through a shared-memory tile so the K loop runs
// from shared memory instead of L2.
template <typename T, int D_COMPILE, int SPLIT, int BK, bool MAXIMIZE>
__global__ void assign_kernel_reg(const T* __restrict__ x,
                                  const float* __restrict__ xsq,
                                  const float* __restrict__ cent,
                                  const float* __restrict__ csq, int n, int k,
                                  int* __restrict__ labels) {
  constexpr int D = D_COMPILE;
  constexpr int DSEG = D / SPLIT;
  static_assert(DSEG == 32 || DSEG == 64, "unsupported register tile");
  static_assert(D % 4 == 0, "float4 loads require D % 4 == 0");
  extern __shared__ float s_ctile[];

  const int seg = threadIdx.x % SPLIT;
  const int points_per_block = blockDim.x / SPLIT;
  const int b = blockIdx.y;
  const int p = blockIdx.x * points_per_block + threadIdx.x / SPLIT;
  const int p_safe = (p < n) ? p : (n - 1);  // keep all lanes active for shfl

  const T* xrow = x + ((std::size_t)b * n + p_safe) * D + seg * DSEG;
  float xr[DSEG];
#pragma unroll
  for (int i = 0; i < DSEG; i += 8) {
    if constexpr (std::is_same<T, float>::value) {
      const float4 lo = *reinterpret_cast<const float4*>(xrow + i);
      const float4 hi = *reinterpret_cast<const float4*>(xrow + i + 4);
      xr[i] = lo.x; xr[i + 1] = lo.y; xr[i + 2] = lo.z; xr[i + 3] = lo.w;
      xr[i + 4] = hi.x; xr[i + 5] = hi.y; xr[i + 6] = hi.z; xr[i + 7] = hi.w;
    } else {
      using V2 = typename TypeTraits<T>::Vec2;
      const uint4 q = *reinterpret_cast<const uint4*>(xrow + i);  // 8 elements
      const float2 f0 = to_float2(*reinterpret_cast<const V2*>(&q.x));
      const float2 f1 = to_float2(*reinterpret_cast<const V2*>(&q.y));
      const float2 f2 = to_float2(*reinterpret_cast<const V2*>(&q.z));
      const float2 f3 = to_float2(*reinterpret_cast<const V2*>(&q.w));
      xr[i] = f0.x; xr[i + 1] = f0.y;
      xr[i + 2] = f1.x; xr[i + 3] = f1.y;
      xr[i + 4] = f2.x; xr[i + 5] = f2.y;
      xr[i + 6] = f3.x; xr[i + 7] = f3.y;
    }
  }

  const float x2 = xsq ? xsq[(std::size_t)b * n + p_safe] : 0.f;
  const float* cbase = cent + ((std::size_t)b * k) * D;
  float best = MAXIMIZE ? -FLT_MAX : FLT_MAX;
  int best_k = 0;

  // With SPLIT > 1 each segment is padded by four words so that the lanes of
  // a point read different shared-memory banks (a power-of-two DSEG would
  // otherwise map every segment onto the same bank).  SPLIT == 1 keeps the
  // contiguous layout and vectorised copies.
  constexpr int SEG_STRIDE = (SPLIT == 1) ? DSEG : (DSEG + 4);
  for (int k0 = 0; k0 < k; k0 += BK) {
    const int kcount = (k0 + BK <= k) ? BK : (k - k0);
    if constexpr (SPLIT == 1) {
      const float4* __restrict__ g4 =
          reinterpret_cast<const float4*>(cbase + (std::size_t)k0 * D);
      float4* s4 = reinterpret_cast<float4*>(s_ctile);
      const int n4 = kcount * (D / 4);
      for (int i = threadIdx.x; i < n4; i += blockDim.x) s4[i] = g4[i];
    } else {
      const float* __restrict__ gbase = cbase + (std::size_t)k0 * D;
      const int n = kcount * D;
      for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const int kk = i / D;
        const int d = i - kk * D;
        const int sg = d / DSEG;
        s_ctile[(kk * SPLIT + sg) * SEG_STRIDE + (d - sg * DSEG)] = gbase[i];
      }
    }
    __syncthreads();

    for (int kk = 0; kk < kcount; ++kk) {
      const float* crow = s_ctile + (kk * SPLIT + seg) * SEG_STRIDE;
      float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;
#pragma unroll
      for (int i = 0; i < DSEG; i += 4) {
        const float4 c4 = *reinterpret_cast<const float4*>(crow + i);
        d0 = fmaf(xr[i], c4.x, d0);
        d1 = fmaf(xr[i + 1], c4.y, d1);
        d2 = fmaf(xr[i + 2], c4.z, d2);
        d3 = fmaf(xr[i + 3], c4.w, d3);
      }
      float dot = (d0 + d1) + (d2 + d3);
      if (SPLIT >= 2) dot += __shfl_xor_sync(0xffffffffu, dot, 1);
      if (SPLIT >= 4) dot += __shfl_xor_sync(0xffffffffu, dot, 2);
      if (SPLIT >= 8) dot += __shfl_xor_sync(0xffffffffu, dot, 4);
      const float score = MAXIMIZE
                              ? dot
                              : (x2 + csq[(std::size_t)b * k + k0 + kk] -
                                 2.f * dot);
      const bool better = MAXIMIZE ? (score > best) : (score < best);
      if (better) {
        best = score;
        best_k = k0 + kk;
      }
    }
    __syncthreads();
  }
  if (seg == 0 && p < n) labels[(std::size_t)b * n + p] = best_k;
}

// Fallback for feature dimensions that do not fit in shared memory: x rows are
// re-read from global memory for every centroid.
template <typename T, bool VEC, bool MAXIMIZE>
__global__ void assign_kernel_global_x(const T* __restrict__ x,
                                       const float* __restrict__ xsq,
                                       const float* __restrict__ cent,
                                       const float* __restrict__ csq,
                                       int n, int d, int k,
                                       int* __restrict__ labels) {
  const int n_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (n_idx >= n) return;
  const int b = blockIdx.y;
  const T* xrow = x + ((std::size_t)b * n + n_idx) * d;
  const float* cbase = cent + (std::size_t)b * k * d;
  const float x2 = xsq ? xsq[(std::size_t)b * n + n_idx] : 0.f;

  float best = MAXIMIZE ? -FLT_MAX : FLT_MAX;
  int best_k = 0;
  for (int kk = 0; kk < k; ++kk) {
    const float dot = dot_row_f32<T, VEC>(xrow, cbase + (std::size_t)kk * d, d);
    const float score = MAXIMIZE ? dot : (x2 + csq[(std::size_t)b * k + kk] - 2.f * dot);
    const bool better = MAXIMIZE ? (score > best) : (score < best);
    if (better) {
      best = score;
      best_k = kk;
    }
  }
  labels[(std::size_t)b * n + n_idx] = best_k;
}

// ---------------------------------------------------------------------------
// row statistics kernels
// ---------------------------------------------------------------------------

template <typename T>
__global__ void row_sq_kernel(const T* __restrict__ x, float* __restrict__ out,
                              std::int64_t rows, int d) {
  const std::int64_t r = blockIdx.x;
  if (r >= rows) return;
  const T* row = x + r * d;
  float ss = 0.f;
  for (int i = threadIdx.x; i < d; i += blockDim.x) {
    const float v = to_float(row[i]);
    ss = fmaf(v, v, ss);
  }
  const float total = block_reduce_sum(ss);
  if (threadIdx.x == 0) out[r] = total;
}

template <typename T>
__global__ void normalize_rows_kernel(const T* __restrict__ x,
                                      T* __restrict__ out, std::int64_t rows,
                                      int d) {
  const std::int64_t r = blockIdx.x;
  if (r >= rows) return;
  const T* row = x + r * d;
  T* orow = out + r * d;
  float ss = 0.f;
  for (int i = threadIdx.x; i < d; i += blockDim.x) {
    const float v = to_float(row[i]);
    ss = fmaf(v, v, ss);
  }
  const float total = block_reduce_sum(ss);
  const float inv = 1.f / fmaxf(sqrtf(total), 1e-12f);
  for (int i = threadIdx.x; i < d; i += blockDim.x) {
    orow[i] = from_float<T>(to_float(row[i]) * inv);
  }
}

// ---------------------------------------------------------------------------
// random centroid initialization (random rows of x, with replacement)
// ---------------------------------------------------------------------------

template <typename T>
__global__ void rng_gather_kernel(const T* __restrict__ x, T* __restrict__ out,
                                  std::int64_t rows_total, std::int64_t n,
                                  int d, std::int64_t k, std::uint64_t seed) {
  const std::int64_t r = blockIdx.x;
  if (r >= rows_total) return;
  const std::int64_t b = r / k;
  const std::uint64_t h = splitmix64(seed + (std::uint64_t)r * 0xD1B54A32D192ED03ull);
  const std::int64_t src_row = (std::int64_t)(h % (std::uint64_t)n);
  const T* src = x + (b * n + src_row) * d;
  T* dst = out + r * d;
  for (int i = threadIdx.x; i < d; i += blockDim.x) dst[i] = src[i];
}

// ---------------------------------------------------------------------------
// centroid accumulation (atomic, fp32 accumulation buffer)
// ---------------------------------------------------------------------------

template <typename T, bool VEC>
__global__ void accumulate_kernel(const T* __restrict__ x,
                                  const int* __restrict__ labels,
                                  float* __restrict__ sums,
                                  int* __restrict__ counts,
                                  std::int64_t total, int n, int d,
                                  std::int64_t k) {
  const std::int64_t stride = (std::int64_t)gridDim.x * blockDim.x;
  for (std::int64_t i = (std::int64_t)blockIdx.x * blockDim.x + threadIdx.x;
       i < total; i += stride) {
    const int lab = labels[i];
    const std::int64_t b = i / n;
    const T* row = x + i * d;
    float* dst = sums + (b * k + lab) * d;

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    if (VEC && (d & 3) == 0) {
      for (int dd = 0; dd < d; dd += 4) {
        atomicAdd(reinterpret_cast<float4*>(dst + dd), load4<T>(row + dd));
      }
      atomicAdd(&counts[b * k + lab], 1);
      continue;
    }
#endif
    for (int dd = 0; dd < d; ++dd) atomicAdd(&dst[dd], to_float(row[dd]));
    atomicAdd(&counts[b * k + lab], 1);
  }
}

// Add `src` into `acc` element-wise (host-side use: cross-GPU partial reduction).
static __global__ void add_inplace_float_kernel(const float* __restrict__ src,
                                                float* __restrict__ acc,
                                                std::int64_t n) {
  const std::int64_t stride = (std::int64_t)gridDim.x * blockDim.x;
  for (std::int64_t i = (std::int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += stride) {
    acc[i] += src[i];
  }
}

static __global__ void add_inplace_int_kernel(const int* __restrict__ src,
                                              int* __restrict__ acc,
                                              std::int64_t n) {
  const std::int64_t stride = (std::int64_t)gridDim.x * blockDim.x;
  for (std::int64_t i = (std::int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += stride) {
    acc[i] += src[i];
  }
}

// ---------------------------------------------------------------------------
// centroid finalize
// ---------------------------------------------------------------------------

// One block per (batch, cluster) row: mean of accumulated sums, empty clusters
// keep the previous centroid, optional L2 renormalization (cosine / dot mode).
// Tracks max squared centroid movement in `max_shift_sq`.
template <typename T, bool NORMALIZE>
__global__ void finalize_kernel(const float* __restrict__ sums,
                                const int* __restrict__ counts,
                                const T* __restrict__ old_cent,
                                T* __restrict__ new_cent,
                                float* __restrict__ max_shift_sq, int k, int d) {
  const int kk = blockIdx.x;
  const int b = blockIdx.y;
  const std::size_t row = (std::size_t)b * k + kk;
  const int cnt = counts[row];
  const float* srow = sums + row * d;
  const T* orow = old_cent + row * d;
  T* nrow = new_cent + row * d;

  float norm_scale = 1.f;
  if (NORMALIZE) {
    float ssq = 0.f;
    if (cnt > 0) {
      for (int i = threadIdx.x; i < d; i += blockDim.x) {
        const float v = srow[i];
        ssq = fmaf(v, v, ssq);
      }
    } else {
      for (int i = threadIdx.x; i < d; i += blockDim.x) {
        const float v = to_float(orow[i]);
        ssq = fmaf(v, v, ssq);
      }
    }
    const float total = block_reduce_sum(ssq);
    const float nm = (cnt > 0) ? (sqrtf(total) / (float)cnt) : sqrtf(total);
    norm_scale = 1.f / fmaxf(nm, 1e-12f);
  }

  const float inv_cnt = (cnt > 0) ? (1.f / (float)cnt) : 0.f;
  float diff2 = 0.f;
  for (int i = threadIdx.x; i < d; i += blockDim.x) {
    const float ov = to_float(orow[i]);
    float val = (cnt > 0) ? (srow[i] * inv_cnt) : ov;
    if (NORMALIZE) val *= norm_scale;
    nrow[i] = from_float<T>(val);
    const float dv = val - ov;
    diff2 = fmaf(dv, dv, diff2);
  }
  const float total = block_reduce_sum(diff2);
  if (threadIdx.x == 0) atomic_max_float(max_shift_sq, total);
}

}  // namespace fk
