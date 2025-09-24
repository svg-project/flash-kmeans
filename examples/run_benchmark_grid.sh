#!/usr/bin/env bash
# Benchmark grid runner for flash-kmeans examples.
# Edit the arrays below to adjust the search space.
# The script will iterate over all combinations and launch the benchmark.
# Results are printed to stdout; you may redirect to a log file.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
BENCHMARK_PY="${SCRIPT_DIR}/benchmark_flash_kmeans.py"

# ----------------- parameter grid -----------------
BATCH_SIZES=(8 32)
NUM_POINTS=(16384 32768 65536 131072)
DIMS=(128)
NUM_CLUSTERS=(100 1000)
MAX_ITERS=(100)
TOLS=(-1)
# ---------------------------------------------------

for b in "${BATCH_SIZES[@]}"; do
  for n in "${NUM_POINTS[@]}"; do
    for d in "${DIMS[@]}"; do
      for k in "${NUM_CLUSTERS[@]}"; do
        for it in "${MAX_ITERS[@]}"; do
          for tol in "${TOLS[@]}"; do
            echo "-----------------------------------------------" | tee /dev/stderr
            echo "Running: b=$b n=$n d=$d k=$k max_iters=$it tol=$tol" | tee /dev/stderr
            python "${BENCHMARK_PY}" \
              --batch-size "$b" \
              --num-points "$n" \
              --dim "$d" \
              --num-clusters "$k" \
              --max-iters "$it" \
              --tol "$tol"
            echo "-----------------------------------------------" | tee /dev/stderr
          done
        done
      done
    done
  done
done
