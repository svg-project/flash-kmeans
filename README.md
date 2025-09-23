# flash-kmeans

Fast batched K-Means clustering implemented with Triton GPU kernels.

## Installation

Clone the repository and install in editable mode:

```bash
git clone https://github.com/yourname/flash-kmeans.git
cd flash-kmeans
pip install -e .
```

## Usage

```python
import torch
from flash_kmeans import batch_kmeans_Euclid

x = torch.randn(32, 75600, 128, device="cuda", dtype=torch.float16)
cluster_ids, centers, _ = batch_kmeans_Euclid(x, n_clusters=1000, verbose=True)
```
