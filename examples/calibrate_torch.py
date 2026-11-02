"""Select PyTorch tiles and update batching on disjoint calibration inputs."""
import json
import time
from pathlib import Path

import torch
import torch_npu
from benchmark_ascend import torch_assign, torch_update

torch.manual_seed(733)
b, n, d, k = 8, 16384, 128, 1000
centers = torch.randn(b, k, d, dtype=torch.float16).npu()
labels = torch.randint(k, (b, n), device="npu", dtype=torch.int32)
labels[:, :n * 9 // 10] = 0
x = (centers.gather(1, labels.long()[..., None].expand(-1, -1, d)) +
     0.25 * torch.randn(b, n, d, device="npu")).half()
reference = torch_assign(x, centers, fp32=True)
center_reference = torch_update(x, reference, centers)
records = []


def measure(function):
    function()
    torch.npu.synchronize()
    before = time.perf_counter()
    for _ in range(3):
        function()
    torch.npu.synchronize()
    return (time.perf_counter() - before) / 3


for fp32 in (False, True):
    for block_n, block_k in [(512, 256), (2048, 256), (2048, 1024), (8192, 1024), (16384, 1024)]:
        function = lambda: torch_assign(x, centers, block_n, block_k, fp32)
        torch.testing.assert_close(function(), reference, rtol=0, atol=0)
        row = {"operation": "assignment", "fp32": fp32, "block_n": block_n, "block_k": block_k,
               "seconds": measure(function)}
        records.append(row)
        print(json.dumps(row), flush=True)
for chunk in (1, 2, 4, 8):
    function = lambda: torch_update(x, reference, centers, chunk)
    torch.testing.assert_close(function(), center_reference, rtol=0.001, atol=0.001)
    row = {"operation": "update", "batch_chunk": chunk, "seconds": measure(function)}
    records.append(row)
    print(json.dumps(row), flush=True)
Path("results").mkdir(exist_ok=True)
Path("results/torch_calibration.json").write_text(json.dumps({"seed": 733, "shape": [b, n, d, k], "records": records}, indent=2))
