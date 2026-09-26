"""Shared helpers for the dataset loaders. Pure stdlib."""

import hashlib
import json
import os
from pathlib import Path

_STRATUNE_ROOT = Path(os.environ.get("STRATUNE_ROOT") or Path(__file__).resolve().parents[3])
# Root of the benchmark payloads (see the top-level README for the layout).
BENCHMARK_DATA = Path(os.environ.get("STRATUNE_BENCHMARK_DATA") or _STRATUNE_ROOT / "benchmark_data")

VALID_TIERS = ("train", "test")


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def check_tier(tier):
    if tier not in VALID_TIERS:
        raise ValueError(f"tier must be one of {VALID_TIERS}, got {tier!r}")
