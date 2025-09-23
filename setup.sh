#!/usr/bin/env bash
set -e

REPO_URL="https://github.com/yourname/flash-kmeans.git"
TARGET_DIR="flash-kmeans"

if [ -d "$TARGET_DIR" ]; then
  echo "Directory $TARGET_DIR already exists. Skipping clone."
else
  git clone "$REPO_URL" "$TARGET_DIR"
fi

cd "$TARGET_DIR"

python3 -m pip install -e .
