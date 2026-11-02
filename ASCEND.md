# Ascend Euclidean clustering

The Ascend backend implements batched Lloyd iterations for FP16 and BF16 inputs.
It keeps the upstream initialization, empty-cluster behavior, first-index tie
breaking, and tolerance check before committing the new centers.

Distance assignment streams center tiles through Cube matrix products. Centers
are packed once per assignment into a padded `[batch, dimension, clusters]`
layout; packing and FP32 center norms are included in the measured operation.
A persistent launch bounds the grid size for the upstream video example.

Center updates sort cluster IDs using Ascend's native FP32 sort, divide the
ordered tokens into fixed-size chunks, accumulate FP32 partial sums into
independent stripes, and merge the stripes. Cluster IDs up to `2**24 - 1` are
exactly representable in FP32. Larger cluster counts use integer sorting.
Empty clusters keep their previous centers.

## Environment

The validated environment uses Python 3.11.14, PyTorch 2.9.0,
`torch_npu 2.9.0.post1`, Triton Ascend `3.2.0.dev20260322`, CANN 8.5.1,
and an Ascend 910B2C. Install into an environment that already contains the
matching Ascend packages:

```bash
python -m pip install --no-deps -e .
```

## API

```python
import torch
import torch_npu
from flash_kmeans import FlashKMeans, batch_kmeans_Euclid

x = torch.randn(2, 8192, 128, dtype=torch.float16, device="npu")
labels, centers, iterations = batch_kmeans_Euclid(x, 64, max_iters=10)

estimator = FlashKMeans(d=128, k=64, niter=10, device="npu", dtype=torch.float16)
estimator.fit(x)
labels = estimator.predict(x)
```

## Validation and experiments

Load the matching CANN environment and select an idle NPU before running the
correctness checks or benchmarks.

```bash
source /usr/local/Ascend/cann/set_env.sh
export OMP_NUM_THREADS=8
ASCEND_RT_VISIBLE_DEVICES=0 python examples/ascend_correctness.py --large
ASCEND_RT_VISIBLE_DEVICES=0 python examples/calibrate_torch.py
ASCEND_RT_VISIBLE_DEVICES=0 python examples/benchmark_ascend.py \
    --shape 32 75600 128 1000 --skew 0.9 --seed 917 --iterations 5 --repeats 1 \
    --torch-config 2048 1024 4 \
    --variants torch_fp16 torch_fp32 packed_single striped \
    --output results/ascend_seed917.json
```

Correctness checks compare against independent CPU FP64 assignment and mean
calculations, including FP16/BF16, uneven cluster sizes, empty clusters, early
stopping, ties, the estimator API, and 67,584 logical token tiles.

The benchmark includes assignment, center updates, the convergence check, and
the center copy. Compilation and warmup are recorded separately. Repeat each
configuration in independent processes with seeds 917, 918, and 919. Use source
cluster dominance 0, 0.5, and 0.9 for the official shape, and
`--shape 2 8192 128 64` for the direct-mapping comparison.

## Complete Lloyd results

Five iterations on `[32, 75600, 128]` with 1,000 centers, FP16 input, and three
independent processes per configuration:

| Implementation | Uniform (ms) | 50% dominant (ms) | 90% dominant (ms) | Peak allocation (MiB) |
|---|---:|---:|---:|---:|
| Tuned PyTorch, FP16 matmul | 747.11 | 752.54 | 768.59 | 1635.29 |
| Tuned PyTorch, FP32 matmul | 775.72 | 780.60 | 797.13 | 1635.29 |
| Streaming, one reduction stripe (default) | 274.26 | 274.40 | 274.67 | 706.35 |
| Streaming, eight stripes | 274.72 | 274.45 | 274.77 | 794.37 |
| Smaller-block PyTorch, FP16 matmul | 755.29 | 1042.47 | 1444.44 | 890.37 |

The default backend is 2.72–2.80 times faster than tuned FP16 PyTorch. Its peak
allocation is 706 MiB, compared with 1635 MiB for the faster batched-update
baseline and 890 MiB for the smaller-block baseline. Eight reduction stripes
provide no speed advantage over one stripe in these cases and use 88 MiB more
memory. The largest relative objective difference across individual runs is
`1.31e-8`.

The tuned baseline uses 2048-token by 1024-center tiles and updates four batches
together. These parameters were selected on separate seed-733 inputs of shape
`[8, 16384, 128]` with 1,000 centers. The smaller-block baseline uses 2048-token
by 256-center tiles and updates one batch at a time. Both accumulate center sums
in FP32. Across the three processes, tuned FP16 times range from 742.66 to
749.82 ms for uniform inputs and 765.18 to 773.30 ms for 90% dominance; default
backend times range from 273.74 to 274.60 ms and 274.60 to 274.79 ms respectively.

## Wan attention features

The public Wan2.1-1.3B model supplies 27 post-RoPE query tensors from three
prompts, layers 0/12/24, and sampling steps 0/10/19. Generation uses 20 steps,
81 frames at 480 by 832, and seeds 3100–3102. Four heads per tensor produce
`[4, 32760, 128]` inputs, converted from BF16 to FP16 for every clustering path.
Ten complete iterations with 1,000 centers and three independent initialization
seeds per tensor produce these averages over 81 tensor/seed combinations:

| Implementation | Ten iterations (ms) | Peak allocation (MiB) |
|---|---:|---:|
| Tuned PyTorch FP16 | 65.33 | 164.11 |
| Tuned PyTorch FP32 | 79.10 | 164.11 |
| Streaming, one stripe | 36.08 | 40.95 |
| Streaming, eight stripes | 36.21 | 54.16 |

The paired geometric-mean speedup over FP16 PyTorch is 1.81 times, with a
1.72–1.88 range. The maximum relative objective difference from FP32 PyTorch
is `5.02e-5`. The largest observed cluster occupies 0.443% of tokens.

Feature collection additionally requires diffusers 0.38.0, accelerate,
sentencepiece, and the public checkpoint downloaded locally:

```bash
python examples/collect_wan_features.py --model models/Wan2.1-T2V-1.3B-Diffusers
python examples/benchmark_ascend.py \
    --features features/wan/prompt0_step0_layer0.pt --shape 4 32760 128 1000 \
    --iterations 10 --repeats 1 --seed 917 --torch-config 2048 1024 4 \
    --variants torch_fp16 torch_fp32 packed_single striped \
    --output results/wan_prompt0_step0_layer0_seed917.json
```

Repeat the clustering command for each collected tensor and seeds 917–919.

## Limitations

The backend requires inputs and centers to fit in NPU memory. The CUDA
multi-device and CPU-streaming interfaces retain their existing implementation.
NPU performance depends on the installed Triton Ascend/CANN combination.
The direct mapping baseline is restricted to medium-size experiments.
