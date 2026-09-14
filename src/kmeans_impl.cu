// Host-side implementation of the batched k-means loop plus the assignment and
// centroid-update entry points.
#include "flash_kmeans.h"
#include "kmeans_common.cuh"
#include "kmeans_launch.cuh"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>

namespace fk {
namespace {

// ---------------------------------------------------------------------------
// typed batched k-means
// ---------------------------------------------------------------------------

template <typename T>
int batch_kmeans_typed(const T* x, const T* init_centroids, int* out_labels,
                       T* out_centroids, std::int64_t B, std::int64_t N,
                       std::int64_t D, std::int64_t K, int max_iters, float tol,
                       int mode, std::uint64_t seed, bool verbose, int device,
                       cudaStream_t st) {
  if (B <= 0 || N <= 0 || D <= 0 || K <= 0) {
    throw std::invalid_argument("batch_kmeans: B, N, D, K must all be positive");
  }
  constexpr std::int64_t kIntMax = 2147483647;
  if (B > kIntMax || N > kIntMax || D > kIntMax || K > kIntMax) {
    throw std::invalid_argument("batch_kmeans: tensor dimensions exceed int32");
  }

  const bool maximize = (mode != kEuclid);
  // Cosine normalizes the input once; cosine and dot both renormalize the
  // updated centroids (the original dot mode reuses the cosine update).
  const bool normalize_x = (mode == kCosine);
  const bool normalize_centroids = (mode != kEuclid);
  const std::size_t cent_elems = (std::size_t)B * K * D;
  const std::size_t x_elems = (std::size_t)B * N * D;
  const std::size_t rows_x = (std::size_t)B * N;
  const std::size_t rows_c = (std::size_t)B * K;

  DeviceAlloc cur(cent_elems * sizeof(T));
  DeviceAlloc nxt(cent_elems * sizeof(T));
  DeviceAlloc sums(rows_c * D * sizeof(float));
  DeviceAlloc counts(rows_c * sizeof(int));
  DeviceAlloc shift(sizeof(float));
  DeviceAlloc xsq;
  DeviceAlloc csq;
  DeviceAlloc xnorm;
  DeviceAlloc cent_f32(cent_elems * sizeof(float));

  const T* xsrc = x;
  if (normalize_x) {
    xnorm = DeviceAlloc(x_elems * sizeof(T));
    launch_normalize_rows<T>(x, xnorm.as<T>(), rows_x, (int)D, st);
    xsrc = xnorm.as<T>();
  }

  if (init_centroids != nullptr) {
    FK_CHECK(cudaMemcpyAsync(cur.as<T>(), init_centroids, cent_elems * sizeof(T),
                             cudaMemcpyDeviceToDevice, st));
  } else {
    rng_gather_kernel<T><<<(unsigned)rows_c, 128, 0, st>>>(
        xsrc, cur.as<T>(), (std::int64_t)rows_c, N, (int)D, K, seed);
  }
  if (normalize_x) {
    launch_normalize_rows<T>(cur.as<T>(), cur.as<T>(), rows_c, (int)D, st);
  }

  if (!maximize) {
    xsq = DeviceAlloc(rows_x * sizeof(float));
    csq = DeviceAlloc(rows_c * sizeof(float));
    launch_row_sq<T>(x, xsq.as<float>(), rows_x, (int)D, st);
  }

  int it = 0;
  bool converged = false;
  for (; it < max_iters; ++it) {
    launch_to_fp32<T>(cur.as<T>(), cent_f32.as<float>(), rows_c * D, st);
    if (!maximize) {
      launch_row_sq<float>(cent_f32.as<float>(), csq.as<float>(), rows_c, (int)D,
                           st);
    }
    launch_assign_dispatch<T>(xsrc, xsq.as<float>(), cent_f32.as<float>(),
                              csq.as<float>(), B, N, D, K, maximize, out_labels,
                              device, st);
    FK_CHECK(cudaMemsetAsync(sums.as<float>(), 0, rows_c * D * sizeof(float), st));
    FK_CHECK(cudaMemsetAsync(counts.as<int>(), 0, rows_c * sizeof(int), st));
    launch_accumulate<T>(xsrc, out_labels, sums.as<float>(), counts.as<int>(),
                         rows_x, N, D, K, st);
    FK_CHECK(cudaMemsetAsync(shift.as<float>(), 0, sizeof(float), st));
    launch_finalize<T>(sums.as<float>(), counts.as<int>(), cur.as<T>(),
                       nxt.as<T>(), shift.as<float>(), B, K, D, normalize_centroids, st);

    float host_shift_sq = 0.f;
    FK_CHECK(cudaMemcpyAsync(&host_shift_sq, shift.as<float>(), sizeof(float),
                             cudaMemcpyDeviceToHost, st));
    FK_CHECK(cudaStreamSynchronize(st));
    const float shift_v = sqrtf(host_shift_sq);
    if (verbose) {
      std::printf("Iter %d, center shift: %.6f\n", it, shift_v);
    }
    if (shift_v < tol) {
      converged = true;
      break;
    }
    std::swap(cur, nxt);
  }

  if (max_iters > 0) {
    FK_CHECK(cudaMemcpyAsync(out_centroids, cur.as<T>(), cent_elems * sizeof(T),
                             cudaMemcpyDeviceToDevice, st));
  }
  if (max_iters <= 0) return 0;
  return converged ? (it + 1) : max_iters;
}

// ---------------------------------------------------------------------------
// typed assignment only
// ---------------------------------------------------------------------------

template <typename T>
void assign_typed(const T* x, const T* centroids, int* out_labels, std::int64_t B,
                  std::int64_t N, std::int64_t D, std::int64_t K, int mode,
                  int device, cudaStream_t st) {
  const bool maximize = (mode != kEuclid);
  DeviceAlloc xsq, csq;
  DeviceAlloc cent_f32((std::size_t)B * K * D * sizeof(float));
  launch_to_fp32<T>(centroids, cent_f32.as<float>(), (std::size_t)B * K * D, st);
  if (!maximize) {
    xsq = DeviceAlloc((std::size_t)B * N * sizeof(float));
    launch_row_sq<T>(x, xsq.as<float>(), B * N, (int)D, st);
    csq = DeviceAlloc((std::size_t)B * K * sizeof(float));
    launch_row_sq<float>(cent_f32.as<float>(), csq.as<float>(), B * K, (int)D, st);
  }
  launch_assign_dispatch<T>(x, xsq.as<float>(), cent_f32.as<float>(),
                            csq.as<float>(), B, N, D, K, maximize, out_labels,
                            device, st);
}

}  // namespace

// ---------------------------------------------------------------------------
// public API
// ---------------------------------------------------------------------------

int batch_kmeans(const void* x, const void* init_centroids, void* out_labels,
                 void* out_centroids, std::int64_t B, std::int64_t N,
                 std::int64_t D, std::int64_t K, int max_iters, float tol,
                 int mode, int dtype, std::uint64_t seed, bool verbose,
                 int device, void* stream) {
  FK_CHECK(cudaSetDevice(device));
  cudaStream_t st = reinterpret_cast<cudaStream_t>(stream);
  switch (dtype) {
    case kF32:
      return batch_kmeans_typed<float>(
          static_cast<const float*>(x), static_cast<const float*>(init_centroids),
          static_cast<int*>(out_labels), static_cast<float*>(out_centroids), B, N,
          D, K, max_iters, tol, mode, seed, verbose, device, st);
    case kF16:
      return batch_kmeans_typed<__half>(
          static_cast<const __half*>(x), static_cast<const __half*>(init_centroids),
          static_cast<int*>(out_labels), static_cast<__half*>(out_centroids), B,
          N, D, K, max_iters, tol, mode, seed, verbose, device, st);
    case kBF16:
      return batch_kmeans_typed<__nv_bfloat16>(
          static_cast<const __nv_bfloat16*>(x),
          static_cast<const __nv_bfloat16*>(init_centroids),
          static_cast<int*>(out_labels),
          static_cast<__nv_bfloat16*>(out_centroids), B, N, D, K, max_iters, tol,
          mode, seed, verbose, device, st);
    default:
      throw std::invalid_argument("flash-kmeans: unknown dtype code");
  }
}

void euclid_assign(const void* x, const void* centroids, void* out_labels,
                   std::int64_t B, std::int64_t N, std::int64_t D,
                   std::int64_t K, int mode, int dtype, int device,
                   void* stream) {
  FK_CHECK(cudaSetDevice(device));
  cudaStream_t st = reinterpret_cast<cudaStream_t>(stream);
  switch (dtype) {
    case kF32:
      assign_typed<float>(static_cast<const float*>(x),
                          static_cast<const float*>(centroids),
                          static_cast<int*>(out_labels), B, N, D, K, mode,
                          device, st);
      return;
    case kF16:
      assign_typed<__half>(static_cast<const __half*>(x),
                           static_cast<const __half*>(centroids),
                           static_cast<int*>(out_labels), B, N, D, K, mode,
                           device, st);
      return;
    case kBF16:
      assign_typed<__nv_bfloat16>(static_cast<const __nv_bfloat16*>(x),
                                  static_cast<const __nv_bfloat16*>(centroids),
                                  static_cast<int*>(out_labels), B, N, D, K,
                                  mode, device, st);
      return;
    default:
      throw std::invalid_argument("flash-kmeans: unknown dtype code");
  }
}

void centroid_update(const void* x, const void* cluster_ids,
                     const void* old_centroids, void* sums_buf, void* counts_buf,
                     void* out_centroids, std::int64_t B, std::int64_t N,
                     std::int64_t D, std::int64_t K, int mode, int dtype,
                     bool calculate_new, int device, void* stream) {
  FK_CHECK(cudaSetDevice(device));
  cudaStream_t st = reinterpret_cast<cudaStream_t>(stream);
  const bool normalize = (mode != kEuclid);
  const std::size_t rows_c = (std::size_t)B * K;

  DeviceAlloc sums_owned, counts_owned, shift_dummy(sizeof(float));
  float* sums = static_cast<float*>(sums_buf);
  int* counts = static_cast<int*>(counts_buf);
  if (sums == nullptr) {
    sums_owned = DeviceAlloc(rows_c * D * sizeof(float));
    sums = sums_owned.as<float>();
    FK_CHECK(cudaMemsetAsync(sums, 0, rows_c * D * sizeof(float), st));
  }
  if (counts == nullptr) {
    counts_owned = DeviceAlloc(rows_c * sizeof(int));
    counts = counts_owned.as<int>();
    FK_CHECK(cudaMemsetAsync(counts, 0, rows_c * sizeof(int), st));
  }

#define FK_RUN_UPDATE(T)                                                        \
  do {                                                                          \
    launch_accumulate<T>(static_cast<const T*>(x),                              \
                         static_cast<const int*>(cluster_ids), sums, counts,    \
                         (std::int64_t)B * N, N, D, K, st);                     \
    if (calculate_new) {                                                        \
      FK_CHECK(cudaMemsetAsync(shift_dummy.as<float>(), 0, sizeof(float), st)); \
      launch_finalize<T>(sums, counts, static_cast<const T*>(old_centroids),    \
                         static_cast<T*>(out_centroids),                        \
                         shift_dummy.as<float>(), B, K, D, normalize, st);      \
    }                                                                           \
  } while (0)

  switch (dtype) {
    case kF32: FK_RUN_UPDATE(float); break;
    case kF16: FK_RUN_UPDATE(__half); break;
    case kBF16: FK_RUN_UPDATE(__nv_bfloat16); break;
    default:
      throw std::invalid_argument("flash-kmeans: unknown dtype code");
  }
#undef FK_RUN_UPDATE
}

void row_sq(const void* x, void* out_f32, std::int64_t rows, std::int64_t D,
            int dtype, int device, void* stream) {
  FK_CHECK(cudaSetDevice(device));
  cudaStream_t st = reinterpret_cast<cudaStream_t>(stream);
  switch (dtype) {
    case kF32:
      launch_row_sq<float>(static_cast<const float*>(x),
                           static_cast<float*>(out_f32), rows, (int)D, st);
      return;
    case kF16:
      launch_row_sq<__half>(static_cast<const __half*>(x),
                            static_cast<float*>(out_f32), rows, (int)D, st);
      return;
    case kBF16:
      launch_row_sq<__nv_bfloat16>(static_cast<const __nv_bfloat16*>(x),
                                   static_cast<float*>(out_f32), rows, (int)D, st);
      return;
    default:
      throw std::invalid_argument("flash-kmeans: unknown dtype code");
  }
}

}  // namespace fk
