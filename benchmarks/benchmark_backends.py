"""Compare end-to-end Euclidean k-means backends on a shared input/init."""
import argparse
import statistics

import torch

from flash_kmeans import batch_kmeans_Euclid


def cuda_time(fn, warmup, repeats):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))
    return statistics.median(samples)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--samples", type=int, default=8192)
    parser.add_argument("--clusters", type=int, default=64)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()

    torch.manual_seed(0)
    x = torch.randn(
        args.batch,
        args.samples,
        128,
        device="cuda",
        dtype=torch.bfloat16,
    )
    indices = torch.randint(
        0, args.samples, (args.batch, args.clusters), device=x.device
    )
    initial = torch.gather(
        x, 1, indices[..., None].expand(-1, -1, x.shape[-1])
    ).contiguous()

    def run(backend):
        return batch_kmeans_Euclid(
            x,
            args.clusters,
            max_iters=args.iters,
            tol=0.0,
            init_centroids=initial,
            backend=backend,
        )

    print(
        f"device={torch.cuda.get_device_name()} "
        f"shape=({args.batch}, {args.samples}, 128) K={args.clusters} "
        f"iters={args.iters}"
    )
    results = {}
    for backend in ("triton", "cute"):
        results[backend] = cuda_time(
            lambda backend=backend: run(backend), args.warmup, args.repeats
        )
        print(f"{backend:>7}: {results[backend]:.3f} ms")
    print(f"speedup: {results['triton'] / results['cute']:.2f}x")


if __name__ == "__main__":
    main()
