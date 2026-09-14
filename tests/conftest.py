"""Shared pytest configuration for the flash-kmeans tests."""

from __future__ import annotations

import os

import pytest


def pytest_collection_modifyitems(config, items):
    """Skip performance tests unless explicitly requested."""
    if os.environ.get("FLASH_KMEANS_PERF") == "1":
        return
    skip = pytest.mark.skip(reason="performance test; set FLASH_KMEANS_PERF=1")
    for item in items:
        if "perf" in item.keywords:
            item.add_marker(skip)
