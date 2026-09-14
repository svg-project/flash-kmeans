// Host-side kernel launch helpers shared by the batch and large-N code paths.
#pragma once

#include "kmeans_common.cuh"
#include <algorithm>
#include <cstddef>
#include <stdexcept>

namespace fk {

struct DeviceAlloc {
  void* p = nullptr;
  DeviceAlloc() = default;
  explicit DeviceAlloc(std::size_t bytes) {
    FK_CHECK(cudaMalloc(&p, bytes ? bytes : 1));
  }
  DeviceAlloc(const DeviceAlloc&) = delete;
  DeviceAlloc& operator=(const DeviceAlloc&) = delete;
  DeviceAlloc(DeviceAlloc&& o) noexcept : p(o.p) { o.p = nullptr; }
  DeviceAlloc& operator=(DeviceAlloc&& o) noexcept {
    if (this != &o) {
      if (p) cudaFree(p);
      p = o.p;
      o.p = nullptr;
    }
    return *this;
  }
  ~DeviceAlloc() {
    if (p) cudaFree(p);
  }
  template <typename T>
  T* as() const {
    return static_cast<T*>(p);
  }
};

inline int max_dynamic_smem(int device) {
  int v = 48 * 1024;
  FK_CHECK(cudaDeviceGetAttribute(&v, cudaDevAttrMaxSharedMemoryPerBlockOptin,
                                  device));
  return v;
}

template <typename T, bool VEC, bool MAXIMIZE>
inline void launch_assign(const T* x, const float* xsq, const float* cent,
                          const float* csq, std::int64_t B, std::int64_t N,
                          std::int64_t D, std::int64_t K, int* labels,
                          int device, cudaStream_t st) {
  const int d = (int)D;
  const int k = (int)K;
  const int row_bytes = shared_row_words<T>(d) * 4;
  const int max_smem = max_dynamic_smem(device);
  const int budget = std::min(max_smem, 96 * 1024);

  int bn = 256;
  while (bn > 32 && (std::int64_t)bn * row_bytes > budget) bn -= 32;

  if ((std::int64_t)bn * row_bytes <= max_smem) {
    const std::size_t smem = (std::size_t)bn * row_bytes;
    if (smem > 48 * 1024) {
      FK_CHECK(cudaFuncSetAttribute(assign_kernel<T, VEC, MAXIMIZE>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    (int)smem));
    }
    dim3 grid((unsigned)((N + bn - 1) / bn), (unsigned)B);
    assign_kernel<T, VEC, MAXIMIZE>
        <<<grid, bn, smem, st>>>(x, xsq, cent, csq, (int)N, d, k,
                                 shared_row_words<T>(d), labels);
  } else {
    dim3 grid((unsigned)((N + 255) / 256), (unsigned)B);
    assign_kernel_global_x<T, VEC, MAXIMIZE>
        <<<grid, 256, 0, st>>>(x, xsq, cent, csq, (int)N, d, k, labels);
  }
}

// Compile-time D specialisations that keep the x row in fp32 registers
// (shared by SPLIT lanes per point) and stream centroid tiles through shared
// memory.
template <typename T, int Dc, int Split, int Bk, bool Max>
inline void launch_reg_case(const T* x, const float* xsq, const float* cent,
                            const float* csq, std::int64_t B, std::int64_t N,
                            std::int64_t K, int* labels, cudaStream_t st) {
  const int bn = 128;
  const int points_per_block = bn / Split;
  const dim3 grid((unsigned)((N + points_per_block - 1) / points_per_block),
                  (unsigned)B);
  constexpr int kSegStride = (Split == 1) ? (Dc / Split) : ((Dc / Split) + 4);
  const std::size_t smem = (std::size_t)Bk * Split * kSegStride * sizeof(float);
  assign_kernel_reg<T, Dc, Split, Bk, Max>
      <<<grid, bn, smem, st>>>(x, xsq, cent, csq, (int)N, (int)K, labels);
}

template <typename T>
inline bool launch_assign_reg(const T* x, const float* xsq, const float* cent,
                              const float* csq, std::int64_t B, std::int64_t N,
                              std::int64_t D, std::int64_t K, bool maximize,
                              int* labels, cudaStream_t st) {
  constexpr std::int64_t kIntMax = 2147483647;
  if (N > kIntMax || K > kIntMax) return false;
#define FK_REG_CASE(Dc, Split, Bk)                                            \
  do {                                                                        \
    if (maximize)                                                             \
      launch_reg_case<T, Dc, Split, Bk, true>(x, xsq, cent, csq, B, N, K,     \
                                              labels, st);                    \
    else                                                                      \
      launch_reg_case<T, Dc, Split, Bk, false>(x, xsq, cent, csq, B, N, K,    \
                                               labels, st);                   \
    return true;                                                              \
  } while (0)
  switch (D) {
    case 32: FK_REG_CASE(32, 1, 32);
    case 64: FK_REG_CASE(64, 1, 32);
    case 128: FK_REG_CASE(128, 2, 32);
    case 256: FK_REG_CASE(256, 4, 16);
    case 512: FK_REG_CASE(512, 8, 8);
    default: return false;
  }
#undef FK_REG_CASE
}

template <typename T>
inline void launch_assign_dispatch(const T* x, const float* xsq, const float* cent,
                                   const float* csq, std::int64_t B,
                                   std::int64_t N, std::int64_t D,
                                   std::int64_t K, bool maximize, int* labels,
                                   int device, cudaStream_t st) {
  if (launch_assign_reg<T>(x, xsq, cent, csq, B, N, D, K, maximize, labels, st)) {
    return;
  }
  const bool vec = (sizeof(T) == 2) && ((D & 1) == 0);
  if (maximize) {
    if (vec)
      launch_assign<T, true, true>(x, xsq, cent, csq, B, N, D, K, labels, device, st);
    else
      launch_assign<T, false, true>(x, xsq, cent, csq, B, N, D, K, labels, device, st);
  } else {
    if (vec)
      launch_assign<T, true, false>(x, xsq, cent, csq, B, N, D, K, labels, device, st);
    else
      launch_assign<T, false, false>(x, xsq, cent, csq, B, N, D, K, labels, device, st);
  }
}

template <typename T>
inline void launch_accumulate(const T* x, const int* labels, float* sums,
                              int* counts, std::int64_t total, std::int64_t N,
                              std::int64_t D, std::int64_t K,
                              cudaStream_t st) {
  const int threads = 256;
  std::int64_t want = (total + threads - 1) / threads;
  const int blocks = (int)std::min<std::int64_t>(want, 16384);
  if (blocks <= 0) return;
  if (D % 4 == 0) {
    accumulate_kernel<T, true>
        <<<blocks, threads, 0, st>>>(x, labels, sums, counts, total, (int)N,
                                     (int)D, K);
  } else {
    accumulate_kernel<T, false>
        <<<blocks, threads, 0, st>>>(x, labels, sums, counts, total, (int)N,
                                     (int)D, K);
  }
}

template <typename T>
inline void launch_finalize(const float* sums, const int* counts,
                            const T* old_cent, T* new_cent, float* shift,
                            std::int64_t B, std::int64_t K, std::int64_t D,
                            bool normalize, cudaStream_t st) {
  dim3 grid((unsigned)K, (unsigned)B);
  const int threads = 128;
  if (normalize) {
    finalize_kernel<T, true>
        <<<grid, threads, 0, st>>>(sums, counts, old_cent, new_cent, shift,
                                   (int)K, (int)D);
  } else {
    finalize_kernel<T, false>
        <<<grid, threads, 0, st>>>(sums, counts, old_cent, new_cent, shift,
                                   (int)K, (int)D);
  }
}

template <typename T>
inline void launch_to_fp32(const T* src, float* dst, std::int64_t n,
                           cudaStream_t st) {
  if (n <= 0) return;
  const int threads = 256;
  const int blocks =
      (int)std::min<std::int64_t>((n + threads - 1) / threads, 16384);
  to_fp32_kernel<T><<<blocks, threads, 0, st>>>(src, dst, n);
}

template <typename T>
inline void launch_row_sq(const T* x, float* out, std::int64_t rows, int d,
                          cudaStream_t st) {
  if (rows <= 0) return;
  row_sq_kernel<T><<<(unsigned)rows, 128, 0, st>>>(x, out, rows, d);
}

template <typename T>
inline void launch_normalize_rows(const T* x, T* out, std::int64_t rows, int d,
                                  cudaStream_t st) {
  if (rows <= 0) return;
  normalize_rows_kernel<T><<<(unsigned)rows, 128, 0, st>>>(x, out, rows, d);
}

}  // namespace fk
