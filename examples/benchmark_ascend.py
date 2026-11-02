"""Complete Lloyd comparisons with compilation excluded and allocation tracked."""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import torch_npu
from flash_kmeans.ascend import assign, update


def torch_assign(x, centers, block_n=2048, block_k=256, fp32=False):
    b, n, d = x.shape
    k = centers.shape[1]
    norms = centers.float().square().sum(-1)
    labels = torch.empty((b, n), dtype=torch.int32, device=x.device)
    for ns in range(0, n, block_n):
        points = x[:, ns:ns + block_n]
        best = torch.full(points.shape[:2], float("inf"), device=x.device)
        ids = torch.zeros(points.shape[:2], dtype=torch.int32, device=x.device)
        for ks in range(0, k, block_k):
            right = centers[:, ks:ks + block_k].transpose(1, 2)
            cross = torch.bmm(points.float(), right.float()) if fp32 else torch.bmm(points, right).float()
            scores = norms[:, None, ks:ks + block_k] - 2 * cross
            value, index = scores.min(-1)
            take = value < best
            best = torch.where(take, value, best)
            ids = torch.where(take, (index + ks).int(), ids)
        labels[:, ns:ns + block_n] = ids
    return labels


def torch_update(x, labels, old, batch_chunk=1):
    b, n, d = x.shape
    k = old.shape[1]
    sums = torch.zeros((b, k, d), device=x.device, dtype=torch.float32)
    counts = torch.zeros((b, k), device=x.device, dtype=torch.float32)
    ones = torch.ones(min(b, batch_chunk) * n, device=x.device)
    offsets = torch.arange(min(b, batch_chunk), device=x.device)[:, None] * k if batch_chunk > 1 else None
    for bi in range(0, b, batch_chunk):
        size = min(batch_chunk, b - bi)
        ids = labels[bi:bi + size].long()
        if offsets is not None:
            ids = ids + offsets[:size]
        ids = ids.flatten()
        sums[bi:bi + size].view(-1, d).index_add_(0, ids, x[bi:bi + size].reshape(-1, d).float())
        counts[bi:bi + size].view(-1).index_add_(0, ids, ones[:size * n])
    return torch.where(counts[:, :, None] > 0,
                       sums / counts[:, :, None].clamp_min(1), old).to(x.dtype)


def lloyd(x, initial, variant, iterations, torch_config=(2048, 256, 1)):
    centers = initial
    for _ in range(iterations):
        labels = (torch_assign(x, centers, *torch_config[:2], fp32=variant == "torch_fp32")
                  if variant.startswith("torch") else assign(x, centers, packed=variant != "direct"))
        new = (torch_update(x, labels, centers, batch_chunk=torch_config[2]) if variant.startswith("torch")
               else update(x, labels, centers, stripes=8 if variant == "striped" else 1))
        shift = (new - centers).norm(dim=-1).max()
        if shift < 0:
            break
        centers = new.clone()
    return labels, centers


def objective(x, labels, centers):
    total = 0.0
    for ns in range(0, x.shape[1], 2048):
        target = centers.gather(1, labels[:, ns:ns + 2048].long()[:, :, None].expand(-1, -1, x.shape[-1]))
        total += (x[:, ns:ns + 2048].float() - target.float()).square().sum().item()
    return total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", type=int, nargs=4, default=[2, 8192, 128, 64], metavar=("B", "N", "D", "K"))
    parser.add_argument("--skew", type=float, default=0.9)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=917)
    parser.add_argument("--distribution", choices=["gaussian", "mixture"], default="mixture")
    parser.add_argument("--features", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variants", nargs="+", default=["torch_fp16", "torch_fp32", "direct", "packed_single", "striped"])
    parser.add_argument("--torch-config", type=int, nargs=3, default=[2048, 256, 1], metavar=("BLOCK_N", "BLOCK_K", "BATCH_CHUNK"))
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    b, n, d, k = args.shape
    feature_metadata = None
    if args.features:
        source_features = torch.load(args.features, map_location="cpu", weights_only=True)
        x = source_features.pop("features").to(torch.float16)
        feature_metadata = source_features
        b, n, d = x.shape
        indices = torch.randint(0, n, (b, k))
        initial = x.gather(1, indices[..., None].expand(-1, -1, d)).clone()
    elif args.distribution == "gaussian":
        x = torch.randn(b, n, d, dtype=torch.float16)
        initial = x[:, torch.linspace(0, n - 1, k).long()].clone()
    else:
        source = torch.randn(b, k, d)
        ids = torch.randint(k, (b, n))
        ids[torch.rand(b, n) < args.skew] = 0
        x = (source.gather(1, ids[:, :, None].expand(-1, -1, d)) + 0.25 * torch.randn(b, n, d)).half()
        initial = source.half()
    initial = initial.npu()
    x = x.npu()
    result = {"shape": [b, n, d, k], "skew": args.skew,
              "seed": args.seed, "distribution": "wan" if args.features else args.distribution,
              "features": str(args.features) if args.features else None,
              "feature_metadata": feature_metadata,
              "iterations": args.iterations, "repeats": args.repeats,
              "torch_config": args.torch_config,
              "torch": torch.__version__, "torch_npu": torch_npu.__version__,
              "records": []}
    for variant in args.variants:
        start = time.perf_counter()
        lloyd(x, initial, variant, 1, args.torch_config)
        torch.npu.synchronize()
        setup_s = time.perf_counter() - start
        for repeat in range(args.repeats):
            torch.npu.reset_peak_memory_stats()
            start = time.perf_counter()
            labels, centers = lloyd(x, initial, variant, args.iterations, args.torch_config)
            torch.npu.synchronize()
            elapsed = time.perf_counter() - start
            peak = torch.npu.max_memory_allocated()
            record = {"variant": variant, "repeat": repeat, "seconds": elapsed,
                      "peak_bytes": peak, "objective": objective(x, labels, centers),
                      "compile_and_warmup_seconds": setup_s}
            if args.features:
                counts = torch.stack([torch.bincount(row.long(), minlength=k) for row in labels.cpu()])
                record["largest_cluster_fraction"] = (counts.max(-1).values.float() / n).tolist()
                record["empty_cluster_fraction"] = (counts == 0).float().mean(-1).tolist()
            result["records"].append(record)
            print(json.dumps(record), flush=True)
            del labels, centers
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
