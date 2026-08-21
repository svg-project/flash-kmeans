"""Compare end-to-end Euclidean k-means backends on a shared input/init.

Correctness is checked before timing so a backend that regressed cannot be
reported as a speedup: both backends run on the same data from the same
initial centroids, and their inertia must agree within ``--rtol``.
"""
import argparse
import statistics
import sys

import torch

from flash_kmeans import batch_kmeans_Euclid


def cuda_time(fn, warmup, repeats, device):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(device)
    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize(device)
        samples.append(start.elapsed_time(end))
    return statistics.median(samples)


def inertia(x, ids, centroids, chunk=1 << 16):
    """Mean squared distance to the assigned centroid, chunked over N.

    Materializing the full (B, N, D) gather would cost more memory than the
    dataset itself at the geometries this script is meant to run.
    """
    dim = x.shape[-1]
    total = 0.0
    count = 0
    for i in range(0, x.shape[1], chunk):
        block = x[:, i:i + chunk].float()
        own = torch.gather(
            centroids.float(),
            1,
            ids[:, i:i + chunk].long()[..., None].expand(-1, -1, dim),
        )
        total += float(((block - own) ** 2).sum(-1).sum())
        count += block.shape[0] * block.shape[1]
    return total / count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--samples", type=int, default=8192)
    parser.add_argument("--clusters", type=int, default=64)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--device", default="cuda", help="CUDA device to run on")
    parser.add_argument(
        "--rtol",
        type=float,
        default=5e-3,
        help="max allowed relative inertia difference between backends",
    )
    parser.add_argument(
        "--skip-check", action="store_true", help="time only, do not verify inertia"
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        parser.error("this benchmark requires CUDA")
    device = torch.device(args.device)
    if device.type != "cuda":
        parser.error(f"--device must be a CUDA device, got {args.device!r}")
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    # The Triton kernels launch on the current device's stream regardless of
    # where their tensors live, so make the target current for the whole run.
    torch.cuda.set_device(device)

    torch.manual_seed(0)
    x = torch.randn(
        args.batch,
        args.samples,
        128,
        device=device,
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
        f"device={torch.cuda.get_device_name(device)} "
        f"cap=sm_{''.join(str(v) for v in torch.cuda.get_device_capability(device))} "
        f"shape=({args.batch}, {args.samples}, 128) K={args.clusters} "
        f"iters={args.iters}"
    )

    if not args.skip_check:
        reference = {}
        for backend in ("triton", "cute"):
            ids, centroids, _ = run(backend)
            torch.cuda.synchronize(device)
            reference[backend] = (ids, inertia(x, ids, centroids))
        triton_inertia = reference["triton"][1]
        cute_inertia = reference["cute"][1]
        relative = abs(cute_inertia - triton_inertia) / triton_inertia
        agreement = (
            (reference["triton"][0] == reference["cute"][0]).float().mean().item()
        )
        print(
            f"  inertia: triton={triton_inertia:.5f} cute={cute_inertia:.5f} "
            f"rel_diff={relative:.3e} (limit {args.rtol:.1e})  "
            f"label_agreement={agreement:.4f}"
        )
        if not relative < args.rtol:
            print(
                f"FAIL: cute inertia differs from triton by {relative:.3e} "
                f"(> {args.rtol:.1e}); not reporting timings.",
                file=sys.stderr,
            )
            return 1
        del reference
        torch.cuda.empty_cache()

    results = {}
    for backend in ("triton", "cute"):
        results[backend] = cuda_time(
            lambda backend=backend: run(backend), args.warmup, args.repeats, device
        )
        print(f"{backend:>7}: {results[backend]:.3f} ms")
    print(f"speedup: {results['triton'] / results['cute']:.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
