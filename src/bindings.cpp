// nanobind bindings for the flash-kmeans C++/CUDA core.
//
// The Python layer passes raw pointers (``tensor.data_ptr()``), shapes, dtype
// codes and the current CUDA stream; nanobind performs only scalar conversion.
#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>

#include <cstdint>
#include <vector>

#include "flash_kmeans.h"

namespace nb = nanobind;
using namespace fk;

namespace {

const void* as_const_ptr(std::uintptr_t p) {
  return p ? reinterpret_cast<const void*>(p) : nullptr;
}
void* as_ptr(std::uintptr_t p) { return reinterpret_cast<void*>(p); }

}  // namespace

NB_MODULE(_flash_kmeans_cpp, m) {
  m.doc() = "C++/CUDA core of flash-kmeans (nanobind bindings)";

  m.def(
      "batch_kmeans",
      [](std::uintptr_t x, std::uintptr_t init_centroids,
         std::uintptr_t out_labels, std::uintptr_t out_centroids,
         std::int64_t B, std::int64_t N, std::int64_t D, std::int64_t K,
         int max_iters, float tol, int mode, int dtype, std::uint64_t seed,
         bool verbose, int device, std::uintptr_t stream) {
        return batch_kmeans(as_const_ptr(x), as_const_ptr(init_centroids),
                            as_ptr(out_labels), as_ptr(out_centroids), B, N, D, K,
                            max_iters, tol, mode, dtype, seed, verbose, device,
                            as_ptr(stream));
      },
      nb::arg("x"), nb::arg("init_centroids"), nb::arg("out_labels"),
      nb::arg("out_centroids"), nb::arg("B"), nb::arg("N"), nb::arg("D"),
      nb::arg("K"), nb::arg("max_iters"), nb::arg("tol"), nb::arg("mode"),
      nb::arg("dtype"), nb::arg("seed"), nb::arg("verbose"), nb::arg("device"),
      nb::arg("stream"));

  m.def(
      "euclid_assign",
      [](std::uintptr_t x, std::uintptr_t centroids, std::uintptr_t out_labels,
         std::int64_t B, std::int64_t N, std::int64_t D, std::int64_t K,
         int mode, int dtype, int device, std::uintptr_t stream) {
        euclid_assign(as_const_ptr(x), as_const_ptr(centroids),
                      as_ptr(out_labels), B, N, D, K, mode, dtype, device,
                      as_ptr(stream));
      },
      nb::arg("x"), nb::arg("centroids"), nb::arg("out_labels"), nb::arg("B"),
      nb::arg("N"), nb::arg("D"), nb::arg("K"), nb::arg("mode"),
      nb::arg("dtype"), nb::arg("device"), nb::arg("stream"));

  m.def(
      "centroid_update",
      [](std::uintptr_t x, std::uintptr_t cluster_ids,
         std::uintptr_t old_centroids, std::uintptr_t sums, std::uintptr_t counts,
         std::uintptr_t out_centroids, std::int64_t B, std::int64_t N,
         std::int64_t D, std::int64_t K, int mode, int dtype,
         bool calculate_new, int device, std::uintptr_t stream) {
        centroid_update(as_const_ptr(x), as_const_ptr(cluster_ids),
                        as_const_ptr(old_centroids), as_ptr(sums),
                        as_ptr(counts), as_ptr(out_centroids), B, N, D, K, mode,
                        dtype, calculate_new, device, as_ptr(stream));
      },
      nb::arg("x"), nb::arg("cluster_ids"), nb::arg("old_centroids"),
      nb::arg("sums"), nb::arg("counts"), nb::arg("out_centroids"),
      nb::arg("B"), nb::arg("N"), nb::arg("D"), nb::arg("K"), nb::arg("mode"),
      nb::arg("dtype"), nb::arg("calculate_new"), nb::arg("device"),
      nb::arg("stream"));

  m.def(
      "row_sq",
      [](std::uintptr_t x, std::uintptr_t out, std::int64_t rows,
         std::int64_t D, int dtype, int device, std::uintptr_t stream) {
        row_sq(as_const_ptr(x), as_ptr(out), rows, D, dtype, device,
               as_ptr(stream));
      },
      nb::arg("x"), nb::arg("out"), nb::arg("rows"), nb::arg("D"),
      nb::arg("dtype"), nb::arg("device"), nb::arg("stream"));

  m.def(
      "kmeans_large_n",
      [](std::uintptr_t x_cpu, std::uintptr_t init_centroids,
         std::uintptr_t out_labels, bool labels_to_cpu,
         std::uintptr_t out_centroids, bool centroids_to_cpu, std::int64_t N,
         std::int64_t D, std::int64_t K, int max_iters, float tol, int dtype,
         std::int64_t block_n, std::vector<int> devices, std::uint64_t seed,
         bool verbose) {
        LargeNOptions opt;
        opt.N = N;
        opt.D = D;
        opt.K = K;
        opt.max_iters = max_iters;
        opt.tol = tol;
        opt.dtype = dtype;
        opt.block_n = block_n;
        opt.devices = devices.empty() ? nullptr : devices.data();
        opt.num_devices = (int)devices.size();
        opt.seed = seed;
        opt.verbose = verbose;
        kmeans_large_n(as_const_ptr(x_cpu), as_const_ptr(init_centroids),
                       as_ptr(out_labels), labels_to_cpu,
                       as_ptr(out_centroids), centroids_to_cpu, opt, nullptr);
      },
      nb::arg("x_cpu"), nb::arg("init_centroids"), nb::arg("out_labels"),
      nb::arg("labels_to_cpu"), nb::arg("out_centroids"),
      nb::arg("centroids_to_cpu"), nb::arg("N"), nb::arg("D"), nb::arg("K"),
      nb::arg("max_iters"), nb::arg("tol"), nb::arg("dtype"),
      nb::arg("block_n"), nb::arg("devices"), nb::arg("seed"),
      nb::arg("verbose"));

  m.def(
      "kmeans_large_n_assign",
      [](std::uintptr_t x_cpu, std::uintptr_t centroids,
         bool centroids_on_device, std::uintptr_t out_labels, bool labels_to_cpu,
         std::int64_t N, std::int64_t D, std::int64_t K, int dtype,
         std::int64_t block_n, std::vector<int> devices) {
        kmeans_large_n_assign(as_const_ptr(x_cpu), as_const_ptr(centroids),
                              centroids_on_device, as_ptr(out_labels),
                              labels_to_cpu, N, D, K, dtype, block_n,
                              devices.empty() ? nullptr : devices.data(),
                              (int)devices.size(), nullptr);
      },
      nb::arg("x_cpu"), nb::arg("centroids"), nb::arg("centroids_on_device"),
      nb::arg("out_labels"), nb::arg("labels_to_cpu"), nb::arg("N"),
      nb::arg("D"), nb::arg("K"), nb::arg("dtype"), nb::arg("block_n"),
      nb::arg("devices"));
}
