"""Validate Ascend assignments, reductions and complete Lloyd iterations."""

import json
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import torch_npu
from flash_kmeans import batch_kmeans_Euclid, FlashKMeans
from flash_kmeans.ascend import assign, update


def reference_assign(x, centers):
    return (centers.double().square().sum(-1)[:, None, :]
            - 2 * x.double() @ centers.double().transpose(1, 2)).argmin(-1)


def reference_update(x, labels, old):
    out = old.double().clone()
    for b in range(x.shape[0]):
        for k in range(old.shape[1]):
            mask = labels[b] == k
            if mask.any():
                out[b, k] = x[b, mask].double().mean(0)
    return out.to(x.dtype)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--large", action="store_true")
    args = parser.parse_args()
    torch.manual_seed(713)
    records = []
    for dtype in (torch.float16, torch.bfloat16):
        for b, n, d, k in ((2, 257, 31, 7), (1, 1025, 128, 67)):
            x = torch.randn(b, n, d, dtype=dtype)
            centers = torch.randn(b, k, d, dtype=dtype)
            xn, cn = x.npu(), centers.npu()
            labels = assign(xn, cn).cpu().long()
            expected = reference_assign(x, centers)
            torch.testing.assert_close(labels, expected, rtol=0, atol=0)
            skew = torch.zeros((b, n), dtype=torch.int32)
            skew[:, ::17] = k - 2
            for stripes in (1, 8):
                got = update(xn, skew.npu(), cn, stripes=stripes).cpu()
                want = reference_update(x, skew, centers)
                torch.testing.assert_close(got, want, rtol=0, atol=0)
            records.append({"dtype": str(dtype), "shape": [b, n, d, k],
                            "assignment": "pass", "skewed_update": "pass"})
            print(json.dumps(records[-1]), flush=True)

    x = torch.randn(2, 513, 64, dtype=torch.float16)
    centers = x[:, :9].clone()
    expected = centers
    for _ in range(4):
        labels = reference_assign(x, expected)
        expected = reference_update(x, labels, expected)
    got_labels, got_centers, iters = batch_kmeans_Euclid(
        x.npu(), 9, max_iters=4, init_centroids=centers.npu())
    torch.testing.assert_close(got_labels.cpu().long(), labels, rtol=0, atol=0)
    torch.testing.assert_close(got_centers.cpu(), expected, rtol=0.001, atol=0.001)
    assert iters == 4
    got_labels, got_centers, iters = batch_kmeans_Euclid(
        x.npu(), 9, max_iters=4, tol=1000, init_centroids=centers.npu())
    torch.testing.assert_close(got_centers.cpu(), centers, rtol=0, atol=0)
    assert iters == 1
    ties = torch.zeros((1, 19, 31), dtype=torch.float16).npu()
    assert not assign(ties, ties[:, :7]).cpu().any()
    estimator = FlashKMeans(64, 9, niter=4, tol=0, dtype=torch.float16,
                            device="npu", chunk_size_data_cpu=16)
    estimator.train(x[0])
    prediction = estimator.predict(x[0]).cpu()
    expected_prediction = reference_assign(x[:1], estimator.centroids_b.cpu())[0]
    torch.testing.assert_close(prediction.long(), expected_prediction, rtol=0, atol=0)
    if args.large:
        values = torch.randint(-3, 4, (33, 65536, 1))
        large_x = values.expand(-1, -1, 16).half().contiguous()
        large_centers = torch.tensor([-2, 0, 2], dtype=torch.float16)[None, :, None].expand(33, -1, 16).contiguous()
        expected = torch.where(values[..., 0] <= -1, 0, torch.where(values[..., 0] <= 1, 1, 2))
        actual = assign(large_x.npu(), large_centers.npu()).cpu().long()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        records.append({"token_tiles": 67584, "persistent_launch": "pass"})
    torch.npu.synchronize()
    records.append({"lloyd_iterations": 4, "early_stop": "pass", "ties": "pass"})
    print(json.dumps({"status": "pass", "cases": records}, indent=2))


if __name__ == "__main__":
    main()
