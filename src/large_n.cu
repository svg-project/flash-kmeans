// Streaming (large-N) k-means for host-resident data, with optional multi-GPU
// partitioning.  The CPU tensor is copied to the GPU in `block_n` chunks with
// double buffering so that H2D transfers overlap compute, mirroring the
// original Python implementation.
#include "flash_kmeans.h"
#include "kmeans_common.cuh"
#include "kmeans_launch.cuh"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <vector>

namespace fk {
namespace {

struct DeviceState {
  int ordinal = 0;
  std::int64_t n_points = 0;
  std::int64_t point_start = 0;
  std::int64_t block_start = 0;
  std::int64_t n_blocks = 0;

  cudaStream_t work[2] = {nullptr, nullptr};
  cudaStream_t reduce = nullptr;
  cudaEvent_t ev_init = nullptr;
  cudaEvent_t ev_work_done[2] = {nullptr, nullptr};

  DeviceAlloc cent;   // [K, D] current centroids (replicated)
  DeviceAlloc cent_f32;  // [K, D] fp32 copy for the assignment kernel
  DeviceAlloc sums;   // [K, D] fp32 partial sums
  DeviceAlloc cnts;   // [K] int32 partial counts
  DeviceAlloc csq;    // [K] fp32 cached ||centroid||^2
  DeviceAlloc cbuf[2];  // [block_n, D] double-buffered input chunk
  DeviceAlloc xsq[2];   // fp32 ||x||^2 scratch: persistent cache (index 0) or
                        // one per stream for the assign-only path
  DeviceAlloc labels;   // [n_points] int32
};

struct PrimaryState {
  DeviceAlloc nxt;  // [K, D] updated centroids
  DeviceAlloc shift;  // fp32 scalar
  std::vector<DeviceAlloc> staging_sums;
  std::vector<DeviceAlloc> staging_cnts;
};

int query_devices(const int* devices, int num_devices) {
  int available = 0;
  FK_CHECK(cudaGetDeviceCount(&available));
  int g = num_devices > 0 ? num_devices : available;
  if (g > available) g = available;
  if (g <= 0) throw std::runtime_error("flash-kmeans: no CUDA devices available");
  for (int i = 0; i < g; ++i) {
    if (devices == nullptr) continue;
    if (devices[i] < 0 || devices[i] >= available) {
      throw std::invalid_argument("flash-kmeans: invalid CUDA device ordinal");
    }
  }
  return g;
}

std::vector<int> resolve_device_list(const int* devices, int num_devices) {
  const int g = query_devices(devices, num_devices);
  std::vector<int> out(g);
  for (int i = 0; i < g; ++i) out[i] = devices ? devices[i] : i;
  return out;
}

// Reduce block_n so that the two staging buffers plus the per-point caches fit
// comfortably in device memory (important for 8 GB cards such as the GTX 1070).
std::int64_t cap_block_n(const std::vector<int>& devs, std::int64_t block_n,
                         std::int64_t N, std::int64_t D, std::int64_t K,
                         std::size_t esize, bool with_labels) {
  double cap = (double)block_n;
  for (int ordinal : devs) {
    FK_CHECK(cudaSetDevice(ordinal));
    std::size_t free_bytes = 0, total_bytes = 0;
    FK_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    // persistent allocations for this device (upper bound; labels/xsq scale
    // with N, not block_n)
    const double n_points = (double)N / (double)devs.size();
    const double fixed = (double)K * D * (double)(esize + 4) +  // cent + sums
                         (double)K * 8 +  // counts + csq
                         n_points * (double)(with_labels ? 8 : 4);  // xsq/labels
    const double margin = 128.0 * 1024 * 1024;
    const double avail = (double)free_bytes - fixed - margin;
    const double per_point = 2.0 * (double)D * (double)esize;
    if (avail <= per_point) return 32;
    cap = std::min(cap, avail / per_point);
  }
  cap = std::max(cap, 32.0);
  if (cap >= (double)N) return N;
  return (std::int64_t)cap;
}

template <typename T>
void setup_device(DeviceState& d, std::int64_t block_n, std::int64_t D,
                  std::int64_t K, std::int64_t labels_elems,
                  bool with_accumulators, std::int64_t xsq_elems,
                  bool xsq_per_stream) {
  FK_CHECK(cudaSetDevice(d.ordinal));
  FK_CHECK(cudaStreamCreateWithFlags(&d.work[0], cudaStreamNonBlocking));
  FK_CHECK(cudaStreamCreateWithFlags(&d.work[1], cudaStreamNonBlocking));
  FK_CHECK(cudaStreamCreateWithFlags(&d.reduce, cudaStreamNonBlocking));
  FK_CHECK(cudaEventCreateWithFlags(&d.ev_init, cudaEventDisableTiming));
  FK_CHECK(cudaEventCreateWithFlags(&d.ev_work_done[0], cudaEventDisableTiming));
  FK_CHECK(cudaEventCreateWithFlags(&d.ev_work_done[1], cudaEventDisableTiming));

  const std::size_t esize = sizeof(T);
  d.cent = DeviceAlloc((std::size_t)K * D * esize);
  d.cent_f32 = DeviceAlloc((std::size_t)K * D * sizeof(float));
  d.csq = DeviceAlloc((std::size_t)K * sizeof(float));
  d.cbuf[0] = DeviceAlloc((std::size_t)block_n * D * esize);
  d.cbuf[1] = DeviceAlloc((std::size_t)block_n * D * esize);
  if (with_accumulators) {
    d.sums = DeviceAlloc((std::size_t)K * D * sizeof(float));
    d.cnts = DeviceAlloc((std::size_t)K * sizeof(int));
  }
  if (labels_elems > 0) {
    d.labels = DeviceAlloc((std::size_t)labels_elems * sizeof(int));
  }
  if (xsq_elems > 0) {
    d.xsq[0] = DeviceAlloc((std::size_t)xsq_elems * sizeof(float));
    if (xsq_per_stream) {
      d.xsq[1] = DeviceAlloc((std::size_t)xsq_elems * sizeof(float));
    }
  }
}

inline void destroy_device(DeviceState& d) {
  if (d.work[0]) cudaStreamDestroy(d.work[0]);
  if (d.work[1]) cudaStreamDestroy(d.work[1]);
  if (d.reduce) cudaStreamDestroy(d.reduce);
  if (d.ev_init) cudaEventDestroy(d.ev_init);
  if (d.ev_work_done[0]) cudaEventDestroy(d.ev_work_done[0]);
  if (d.ev_work_done[1]) cudaEventDestroy(d.ev_work_done[1]);
}

template <typename T>
void large_n_typed(const void* x_cpu, const void* init_centroids,
                   void* out_labels, bool labels_to_cpu, void* out_centroids,
                   bool centroids_to_cpu, const LargeNOptions& opt) {
  const std::int64_t N = opt.N;
  const std::int64_t D = opt.D;
  const std::int64_t K = opt.K;
  if (N <= 0 || D <= 0 || K <= 0) {
    throw std::invalid_argument("kmeans_largeN: N, D, K must all be positive");
  }
  const std::vector<int> ordinals = resolve_device_list(opt.devices, opt.num_devices);
  const int G = (int)ordinals.size();
  const std::size_t esize = sizeof(T);

  const std::int64_t block_n = cap_block_n(ordinals, opt.block_n, N, D, K, esize,
                                           /*with_labels=*/true);
  const std::int64_t nb = (N + block_n - 1) / block_n;

  std::vector<DeviceState> devs(G);
  std::vector<std::int64_t> bpg(G, 0);
  for (int g = 0; g < G; ++g) bpg[g] = nb / G + (g < (int)(nb % G) ? 1 : 0);
  std::vector<std::int64_t> bstart(G, 0), pstart(G, 0), pend(G, 0);
  for (int g = 1; g < G; ++g) bstart[g] = bstart[g - 1] + bpg[g - 1];
  for (int g = 0; g < G; ++g) {
    pstart[g] = std::min<std::int64_t>(N, bstart[g] * block_n);
    pend[g] = std::min<std::int64_t>(N, (bstart[g] + bpg[g]) * block_n);
  }

  try {
    for (int g = 0; g < G; ++g) {
      DeviceState& d = devs[g];
      d.ordinal = ordinals[g];
      d.n_points = pend[g] - pstart[g];
      d.point_start = pstart[g];
      d.block_start = bstart[g];
      d.n_blocks = bpg[g];
      setup_device<T>(d, block_n, D, K, d.n_points, /*with_accumulators=*/true,
                      /*xsq_elems=*/d.n_points, /*xsq_per_stream=*/false);
    }
    DeviceState& p = devs[0];

    std::vector<DeviceAlloc> staging_sums, staging_cnts;
    PrimaryState prim;
    {
      FK_CHECK(cudaSetDevice(ordinals[0]));
      prim.nxt = DeviceAlloc((std::size_t)K * D * esize);
      prim.shift = DeviceAlloc(sizeof(float));
    }
    for (int g = 1; g < G; ++g) {
      FK_CHECK(cudaSetDevice(ordinals[0]));
      staging_sums.emplace_back((std::size_t)K * D * sizeof(float));
      staging_cnts.emplace_back((std::size_t)K * sizeof(int));
    }

    // ---- centroid initialization -----------------------------------------
    {
      DeviceState& p = devs[0];
      FK_CHECK(cudaSetDevice(p.ordinal));
      if (init_centroids != nullptr) {
        FK_CHECK(cudaMemcpyAsync(p.cent.as<T>(), init_centroids,
                                 (std::size_t)K * D * esize,
                                 cudaMemcpyDeviceToDevice, p.reduce));
      } else {
        std::vector<T> host((std::size_t)K * D);
        std::mt19937_64 rng(opt.seed);
        const char* xb = static_cast<const char*>(x_cpu);
        for (std::int64_t k = 0; k < K; ++k) {
          const std::int64_t idx = (std::int64_t)(rng() % (std::uint64_t)N);
          std::memcpy(host.data() + (std::size_t)k * D,
                      xb + (std::size_t)idx * D * esize, (std::size_t)D * esize);
        }
        FK_CHECK(cudaMemcpyAsync(p.cent.as<T>(), host.data(),
                                 (std::size_t)K * D * esize,
                                 cudaMemcpyHostToDevice, p.reduce));
      }
      FK_CHECK(cudaStreamSynchronize(p.reduce));
    }
    for (int g = 1; g < G; ++g) {
      FK_CHECK(cudaSetDevice(devs[g].ordinal));
      FK_CHECK(cudaMemcpyAsync(devs[g].cent.as<T>(), devs[0].cent.as<T>(),
                               (std::size_t)K * D * esize, cudaMemcpyDefault,
                               devs[g].reduce));
      FK_CHECK(cudaStreamSynchronize(devs[g].reduce));
    }

    // ---- main iteration loop ----------------------------------------------
    int iters_run = 0;
    int converged = 0;
    for (int it = 0; it < opt.max_iters; ++it) {
      // phase 1: reset accumulators and refresh ||centroid||^2 per device
      for (int g = 0; g < G; ++g) {
        DeviceState& d = devs[g];
        if (d.n_blocks == 0) continue;
        FK_CHECK(cudaSetDevice(d.ordinal));
        FK_CHECK(cudaMemsetAsync(d.sums.as<float>(), 0,
                                 (std::size_t)K * D * sizeof(float), d.reduce));
        FK_CHECK(cudaMemsetAsync(d.cnts.as<int>(), 0,
                                 (std::size_t)K * sizeof(int), d.reduce));
        launch_to_fp32<T>(d.cent.as<T>(), d.cent_f32.as<float>(),
                          (std::size_t)K * D, d.reduce);
        launch_row_sq<float>(d.cent_f32.as<float>(), d.csq.as<float>(), K,
                             (int)D, d.reduce);
        FK_CHECK(cudaEventRecord(d.ev_init, d.reduce));
        FK_CHECK(cudaStreamWaitEvent(d.work[0], d.ev_init, 0));
        FK_CHECK(cudaStreamWaitEvent(d.work[1], d.ev_init, 0));
      }

      // phase 2: stream blocks, assign + accumulate
      for (int g = 0; g < G; ++g) {
        DeviceState& d = devs[g];
        if (d.n_blocks == 0) continue;
        FK_CHECK(cudaSetDevice(d.ordinal));
        for (std::int64_t bi = 0; bi < d.n_blocks; ++bi) {
          const int flag = (int)(bi & 1);
          cudaStream_t ws = d.work[flag];
          const std::int64_t n_start = (d.block_start + bi) * block_n;
          const std::int64_t n_end = std::min<std::int64_t>(n_start + block_n, N);
          const std::int64_t n_this = n_end - n_start;
          const std::int64_t local_off = n_start - d.point_start;
          FK_CHECK(cudaMemcpyAsync(
              d.cbuf[flag].as<T>(),
              static_cast<const char*>(x_cpu) + (std::size_t)n_start * D * esize,
              (std::size_t)n_this * D * esize, cudaMemcpyHostToDevice, ws));
          if (it == 0) {
            launch_row_sq<T>(d.cbuf[flag].as<T>(),
                             d.xsq[0].as<float>() + local_off, n_this, (int)D, ws);
          }
          launch_assign_dispatch<T>(
              d.cbuf[flag].as<T>(), d.xsq[0].as<float>() + local_off,
              d.cent_f32.as<float>(), d.csq.as<float>(), /*B=*/1, n_this, D, K,
              /*maximize=*/false, d.labels.as<int>() + local_off, d.ordinal, ws);
          launch_accumulate<T>(d.cbuf[flag].as<T>(),
                               d.labels.as<int>() + local_off, d.sums.as<float>(),
                               d.cnts.as<int>(), n_this, n_this, D, K, ws);
        }
      }

      // phase 3: join all devices and finalize on the primary device
      for (int g = 0; g < G; ++g) {
        DeviceState& d = devs[g];
        if (d.n_blocks == 0) continue;
        FK_CHECK(cudaSetDevice(d.ordinal));
        FK_CHECK(cudaEventRecord(d.ev_work_done[0], d.work[0]));
        FK_CHECK(cudaEventRecord(d.ev_work_done[1], d.work[1]));
      }
      FK_CHECK(cudaSetDevice(p.ordinal));
      for (int g = 0; g < G; ++g) {
        if (devs[g].n_blocks == 0) continue;
        FK_CHECK(cudaStreamWaitEvent(p.reduce, devs[g].ev_work_done[0], 0));
        FK_CHECK(cudaStreamWaitEvent(p.reduce, devs[g].ev_work_done[1], 0));
      }
      if (G > 1) {        for (int g = 1; g < G; ++g) {
          if (devs[g].n_blocks == 0) continue;
          FK_CHECK(cudaMemcpyAsync(staging_sums[g - 1].as<float>(),
                                   devs[g].sums.as<float>(),
                                   (std::size_t)K * D * sizeof(float),
                                   cudaMemcpyDefault, p.reduce));
          FK_CHECK(cudaMemcpyAsync(staging_cnts[g - 1].as<int>(),
                                   devs[g].cnts.as<int>(),
                                   (std::size_t)K * sizeof(int),
                                   cudaMemcpyDefault, p.reduce));
        }
        const int threads = 256;
        for (int g = 1; g < G; ++g) {
          if (devs[g].n_blocks == 0) continue;
          const std::int64_t nsum = K * D;
          const int blocks =
              (int)std::min<std::int64_t>((nsum + threads - 1) / threads, 16384);
          add_inplace_float_kernel<<<blocks, threads, 0, p.reduce>>>(
              staging_sums[g - 1].as<float>(), p.sums.as<float>(), nsum);
          const int cblocks =
              (int)std::min<std::int64_t>((K + threads - 1) / threads, 16384);
          add_inplace_int_kernel<<<cblocks, threads, 0, p.reduce>>>(
              staging_cnts[g - 1].as<int>(), p.cnts.as<int>(), K);
        }
      }

      FK_CHECK(cudaMemsetAsync(prim.shift.as<float>(), 0, sizeof(float), p.reduce));
      launch_finalize<T>(p.sums.as<float>(), p.cnts.as<int>(), p.cent.as<T>(),
                         prim.nxt.as<T>(), prim.shift.as<float>(), /*B=*/1, K, D,
                         /*normalize=*/false, p.reduce);
      float host_shift_sq = 0.f;
      FK_CHECK(cudaMemcpyAsync(&host_shift_sq, prim.shift.as<float>(),
                               sizeof(float), cudaMemcpyDeviceToHost, p.reduce));
      FK_CHECK(cudaStreamSynchronize(p.reduce));
      const float shift_v = sqrtf(host_shift_sq);
      if (opt.verbose) {
        std::printf("Iter %d, center shift: %.6f\n", it, shift_v);
      }
      iters_run = it + 1;
      if (shift_v < opt.tol) {
        converged = 1;
        break;
      }

      // broadcast the updated centroids to every device
      for (int g = 0; g < G; ++g) {
        FK_CHECK(cudaSetDevice(devs[g].ordinal));
        FK_CHECK(cudaMemcpyAsync(devs[g].cent.as<T>(), prim.nxt.as<T>(),
                                 (std::size_t)K * D * esize, cudaMemcpyDefault,
                                 devs[g].reduce));
      }
    }  // iteration loop

    // ---- outputs -----------------------------------------------------------
    FK_CHECK(cudaSetDevice(p.ordinal));
    FK_CHECK(cudaMemcpyAsync(
        out_centroids, prim.nxt.as<T>(), (std::size_t)K * D * esize,
        centroids_to_cpu ? cudaMemcpyDeviceToHost : cudaMemcpyDeviceToDevice,
        p.reduce));
    FK_CHECK(cudaStreamSynchronize(p.reduce));

    if (!labels_to_cpu) {
      FK_CHECK(cudaMemcpyAsync(out_labels, devs[0].labels.as<int>(),
                               (std::size_t)N * sizeof(int),
                               cudaMemcpyDeviceToDevice, devs[0].reduce));
      FK_CHECK(cudaStreamSynchronize(devs[0].reduce));
    } else {
      for (int g = 0; g < G; ++g) {
        DeviceState& d = devs[g];
        if (d.n_points <= 0) continue;
        FK_CHECK(cudaSetDevice(d.ordinal));
        FK_CHECK(cudaStreamWaitEvent(d.reduce, d.ev_work_done[0], 0));
        FK_CHECK(cudaStreamWaitEvent(d.reduce, d.ev_work_done[1], 0));
        FK_CHECK(cudaMemcpyAsync(static_cast<char*>(out_labels) +
                                     (std::size_t)d.point_start * sizeof(int),
                                 d.labels.as<int>(),
                                 (std::size_t)d.n_points * sizeof(int),
                                 cudaMemcpyDeviceToHost, d.reduce));
        FK_CHECK(cudaStreamSynchronize(d.reduce));
      }
    }
    (void)iters_run;
    (void)converged;

    for (int g = 0; g < G; ++g) destroy_device(devs[g]);
  } catch (...) {
    for (int g = 0; g < G; ++g) destroy_device(devs[g]);
    throw;
  }
}

// ---------------------------------------------------------------------------
// assignment only for large N
// ---------------------------------------------------------------------------

template <typename T>
void large_n_assign_typed(const void* x_cpu, const void* centroids,
                          bool centroids_on_device, void* out_labels,
                          bool labels_to_cpu, std::int64_t N, std::int64_t D,
                          std::int64_t K, std::int64_t block_n_req,
                          const int* devices_in, int num_devices_in) {
  if (N <= 0 || D <= 0 || K <= 0) {
    throw std::invalid_argument("kmeans_largeN_assign: N, D, K must be positive");
  }
  const std::vector<int> ordinals =
      resolve_device_list(devices_in, num_devices_in);
  const int G = (int)ordinals.size();
  const std::size_t esize = sizeof(T);

  const std::int64_t block_n =
      cap_block_n(ordinals, block_n_req, N, D, K, esize, /*with_labels=*/true);
  const std::int64_t nb = (N + block_n - 1) / block_n;

  std::vector<DeviceState> devs(G);
  std::vector<std::int64_t> bpg(G, 0);
  for (int g = 0; g < G; ++g) bpg[g] = nb / G + (g < (int)(nb % G) ? 1 : 0);
  std::vector<std::int64_t> bstart(G, 0), pstart(G, 0), pend(G, 0);
  for (int g = 1; g < G; ++g) bstart[g] = bstart[g - 1] + bpg[g - 1];
  for (int g = 0; g < G; ++g) {
    pstart[g] = std::min<std::int64_t>(N, bstart[g] * block_n);
    pend[g] = std::min<std::int64_t>(N, (bstart[g] + bpg[g]) * block_n);
  }

  try {
    for (int g = 0; g < G; ++g) {
      DeviceState& d = devs[g];
      d.ordinal = ordinals[g];
      d.n_points = pend[g] - pstart[g];
      d.point_start = pstart[g];
      d.block_start = bstart[g];
      d.n_blocks = bpg[g];
      setup_device<T>(d, block_n, D, K, d.n_points, /*with_accumulators=*/false,
                      /*xsq_elems=*/block_n, /*xsq_per_stream=*/true);
    }

    // ---- centroids on every device ----------------------------------------
    for (int g = 0; g < G; ++g) {
      DeviceState& d = devs[g];
      FK_CHECK(cudaSetDevice(d.ordinal));
      if (g == 0 && centroids_on_device) {
        FK_CHECK(cudaMemcpyAsync(d.cent.as<T>(), centroids,
                                 (std::size_t)K * D * esize,
                                 cudaMemcpyDeviceToDevice, d.reduce));
      }
      FK_CHECK(cudaStreamSynchronize(d.reduce));
    }
    if (centroids_on_device) {
      for (int g = 1; g < G; ++g) {
        FK_CHECK(cudaSetDevice(devs[g].ordinal));
        FK_CHECK(cudaMemcpyAsync(devs[g].cent.as<T>(), devs[0].cent.as<T>(),
                                 (std::size_t)K * D * esize, cudaMemcpyDefault,
                                 devs[g].reduce));
        FK_CHECK(cudaStreamSynchronize(devs[g].reduce));
      }
    } else {
      for (int g = 0; g < G; ++g) {
        FK_CHECK(cudaSetDevice(devs[g].ordinal));
        FK_CHECK(cudaMemcpyAsync(devs[g].cent.as<T>(), centroids,
                                 (std::size_t)K * D * esize,
                                 cudaMemcpyHostToDevice, devs[g].reduce));
        FK_CHECK(cudaStreamSynchronize(devs[g].reduce));
      }
    }

    // ---- stream blocks ------------------------------------------------------
    for (int g = 0; g < G; ++g) {
      DeviceState& d = devs[g];
      if (d.n_blocks == 0) continue;
      FK_CHECK(cudaSetDevice(d.ordinal));
      launch_to_fp32<T>(d.cent.as<T>(), d.cent_f32.as<float>(),
                        (std::size_t)K * D, d.reduce);
      launch_row_sq<float>(d.cent_f32.as<float>(), d.csq.as<float>(), K,
                           (int)D, d.reduce);
      FK_CHECK(cudaEventRecord(d.ev_init, d.reduce));
      FK_CHECK(cudaStreamWaitEvent(d.work[0], d.ev_init, 0));
      FK_CHECK(cudaStreamWaitEvent(d.work[1], d.ev_init, 0));

      for (std::int64_t bi = 0; bi < d.n_blocks; ++bi) {
        const int flag = (int)(bi & 1);
        cudaStream_t ws = d.work[flag];
        const std::int64_t n_start = (d.block_start + bi) * block_n;
        const std::int64_t n_end = std::min<std::int64_t>(n_start + block_n, N);
        const std::int64_t n_this = n_end - n_start;
        const std::int64_t local_off = n_start - d.point_start;
        FK_CHECK(cudaMemcpyAsync(
            d.cbuf[flag].as<T>(),
            static_cast<const char*>(x_cpu) + (std::size_t)n_start * D * esize,
            (std::size_t)n_this * D * esize, cudaMemcpyHostToDevice, ws));
        launch_row_sq<T>(d.cbuf[flag].as<T>(), d.xsq[flag].as<float>(), n_this,
                         (int)D, ws);
        launch_assign_dispatch<T>(
            d.cbuf[flag].as<T>(), d.xsq[flag].as<float>(),
            d.cent_f32.as<float>(), d.csq.as<float>(), /*B=*/1, n_this, D, K,
            /*maximize=*/false, d.labels.as<int>() + local_off, d.ordinal, ws);
      }
      FK_CHECK(cudaEventRecord(d.ev_work_done[0], d.work[0]));
      FK_CHECK(cudaEventRecord(d.ev_work_done[1], d.work[1]));
    }

    // ---- collect labels ------------------------------------------------------
    if (!labels_to_cpu) {
      DeviceState& d = devs[0];
      FK_CHECK(cudaSetDevice(d.ordinal));
      FK_CHECK(cudaStreamWaitEvent(d.reduce, d.ev_work_done[0], 0));
      FK_CHECK(cudaStreamWaitEvent(d.reduce, d.ev_work_done[1], 0));
      FK_CHECK(cudaMemcpyAsync(out_labels, d.labels.as<int>(),
                               (std::size_t)N * sizeof(int),
                               cudaMemcpyDeviceToDevice, d.reduce));
      FK_CHECK(cudaStreamSynchronize(d.reduce));
    } else {
      for (int g = 0; g < G; ++g) {
        DeviceState& d = devs[g];
        if (d.n_points <= 0) continue;
        FK_CHECK(cudaSetDevice(d.ordinal));
        FK_CHECK(cudaStreamWaitEvent(d.reduce, d.ev_work_done[0], 0));
        FK_CHECK(cudaStreamWaitEvent(d.reduce, d.ev_work_done[1], 0));
        FK_CHECK(cudaMemcpyAsync(static_cast<char*>(out_labels) +
                                     (std::size_t)d.point_start * sizeof(int),
                                 d.labels.as<int>(),
                                 (std::size_t)d.n_points * sizeof(int),
                                 cudaMemcpyDeviceToHost, d.reduce));
        FK_CHECK(cudaStreamSynchronize(d.reduce));
      }
    }

    for (int g = 0; g < G; ++g) destroy_device(devs[g]);
  } catch (...) {
    for (int g = 0; g < G; ++g) destroy_device(devs[g]);
    throw;
  }
}

}  // namespace

// ---------------------------------------------------------------------------
// public API
// ---------------------------------------------------------------------------

void kmeans_large_n(const void* x_cpu, const void* init_centroids,
                    void* out_labels, bool labels_to_cpu, void* out_centroids,
                    bool centroids_to_cpu, const LargeNOptions& opt,
                    void* stream) {
  (void)stream;
  switch (opt.dtype) {
    case kF32:
      large_n_typed<float>(x_cpu, init_centroids, out_labels, labels_to_cpu,
                           out_centroids, centroids_to_cpu, opt);
      return;
    case kF16:
      large_n_typed<__half>(x_cpu, init_centroids, out_labels, labels_to_cpu,
                            out_centroids, centroids_to_cpu, opt);
      return;
    case kBF16:
      large_n_typed<__nv_bfloat16>(x_cpu, init_centroids, out_labels,
                                   labels_to_cpu, out_centroids,
                                   centroids_to_cpu, opt);
      return;
    default:
      throw std::invalid_argument("flash-kmeans: unknown dtype code");
  }
}

void kmeans_large_n_assign(const void* x_cpu, const void* centroids,
                           bool centroids_on_device, void* out_labels,
                           bool labels_to_cpu, std::int64_t N, std::int64_t D,
                           std::int64_t K, int dtype, std::int64_t block_n,
                           const int* devices, int num_devices, void* stream) {
  (void)stream;
  switch (dtype) {
    case kF32:
      large_n_assign_typed<float>(x_cpu, centroids, centroids_on_device,
                                  out_labels, labels_to_cpu, N, D, K, block_n,
                                  devices, num_devices);
      return;
    case kF16:
      large_n_assign_typed<__half>(x_cpu, centroids, centroids_on_device,
                                   out_labels, labels_to_cpu, N, D, K, block_n,
                                   devices, num_devices);
      return;
    case kBF16:
      large_n_assign_typed<__nv_bfloat16>(
          x_cpu, centroids, centroids_on_device, out_labels, labels_to_cpu, N, D,
          K, block_n, devices, num_devices);
      return;
    default:
      throw std::invalid_argument("flash-kmeans: unknown dtype code");
  }
}

}  // namespace fk
