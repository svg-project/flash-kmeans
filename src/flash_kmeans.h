// C++/CUDA core API for flash-kmeans (nanobind rewrite).
//
// All pointers are raw CUDA device pointers unless a name says otherwise.
// The Python layer is a thin marshalling shim: it extracts data pointers,
// shapes, dtypes and the current CUDA stream from torch tensors and calls
// into these entry points.
#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace fk {

enum DTypeCode { kF32 = 0, kF16 = 1, kBF16 = 2 };
enum KMeansMode { kEuclid = 0, kCosine = 1, kDot = 2 };

// Batched k-means on GPU-resident data.
//
//   x              [B, N, D] contiguous, dtype
//   init_centroids [B, K, D] contiguous, dtype, may be nullptr (random init)
//   out_labels     [B, N] int32 (device)
//   out_centroids  [B, K, D] dtype (device)
//
// Returns the number of iterations executed (matches the original Python
// implementation's `it + 1`).
int batch_kmeans(const void* x, const void* init_centroids,
                 void* out_labels, void* out_centroids,
                 std::int64_t B, std::int64_t N, std::int64_t D, std::int64_t K,
                 int max_iters, float tol, int mode, int dtype,
                 std::uint64_t seed, bool verbose, int device, void* stream);

// Assignment only.  mode == kEuclid: nearest centroid by L2 distance;
// mode == kCosine/kDot: argmax dot product (caller normalizes for cosine).
//   out_labels [B, N] int32 (device)
void euclid_assign(const void* x, const void* centroids, void* out_labels,
                   std::int64_t B, std::int64_t N, std::int64_t D, std::int64_t K,
                   int mode, int dtype, int device, void* stream);

// Centroid update (accumulate + finalize) exposed for API compatibility with
// the original `triton_centroid_update_*` helpers.
//   cluster_ids   [B, N] int32
//   old_centroids [B, K, D] dtype
//   sums          [B, K, D] float32, may be nullptr (allocated / zeroed here)
//   counts        [B, K] int32, may be nullptr (allocated / zeroed here)
//   out_centroids [B, K, D] dtype, may be nullptr when calculate_new == false
void centroid_update(const void* x, const void* cluster_ids, const void* old_centroids,
                     void* sums, void* counts, void* out_centroids,
                     std::int64_t B, std::int64_t N, std::int64_t D, std::int64_t K,
                     int mode, int dtype, bool calculate_new,
                     int device, void* stream);

struct LargeNOptions {
  std::int64_t N = 0;
  std::int64_t D = 0;
  std::int64_t K = 0;
  int max_iters = 100;
  float tol = 0.f;
  int dtype = kF32;
  std::int64_t block_n = 1 << 20;   // points per H2D chunk
  const int* devices = nullptr;     // CUDA ordinals, devices[0] is primary
  int num_devices = 1;
  std::uint64_t seed = 0;
  bool verbose = false;
};

// Streaming k-means for data that lives in host memory (too large for one GPU).
//
//   x_cpu           [N, D] host pointer (pinned memory recommended)
//   init_centroids  [K, D] dtype, device pointer on devices[0], or nullptr
//   out_labels      int32 [N]; device pointer when labels_to_cpu == false,
//                   otherwise a pinned host pointer
//   out_centroids   [K, D] dtype; device pointer on devices[0] when
//                   centroids_to_cpu == false, otherwise a pinned host pointer
void kmeans_large_n(const void* x_cpu, const void* init_centroids,
                    void* out_labels, bool labels_to_cpu,
                    void* out_centroids, bool centroids_to_cpu,
                    const LargeNOptions& opt, void* stream);

// Assignment for large N host data.
//   centroids       [K, D] dtype; device pointer on devices[0] when
//                   centroids_on_device == true, otherwise a host pointer
//   out_labels      int32 [N]; device pointer when labels_to_cpu == false,
//                   otherwise a pinned host pointer
void kmeans_large_n_assign(const void* x_cpu, const void* centroids,
                           bool centroids_on_device,
                           void* out_labels, bool labels_to_cpu,
                           std::int64_t N, std::int64_t D, std::int64_t K,
                           int dtype, std::int64_t block_n,
                           const int* devices, int num_devices, void* stream);

// Utility: squared L2 norm of every row, fp32 output (used by the wrappers and
// tests for parity with the Python reference implementation).
void row_sq(const void* x, void* out_f32, std::int64_t rows, std::int64_t D,
            int dtype, int device, void* stream);

}  // namespace fk
