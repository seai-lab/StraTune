"""The train and test splits of the paper and the stratified training batches.

Each dataset directory under ${STRATUNE_BENCHMARK_DATA} has train/task_ids.json
and test/task_ids.json. load_split returns the ids in file order together with
the values recorded in run manifests. The split sizes of the paper are enforced
unless the data root is the small sample shipped in data/benchmark_sample.
"""
import hashlib
import json
import os

from method import config
from method.common import util

from .common import BENCHMARK_DATA, check_tier, load_json


def _sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def is_sample_data() -> bool:
    """True when STRATUNE_BENCHMARK_DATA points at the small sample shipped in data/benchmark_sample."""
    return (BENCHMARK_DATA / "SAMPLE.json").exists()


def load_split(dataset: str, tier: str) -> dict:
    check_tier(tier)
    config.check_dataset(dataset)
    ids = [str(i) for i in load_json(BENCHMARK_DATA / config.BENCHMARK_DIRS[dataset] / tier / "task_ids.json")]
    n = len(ids) if is_sample_data() else config.SPLIT_SIZES[dataset][tier]
    if len(ids) != n or len(set(ids)) != n:
        raise RuntimeError(f"{dataset}/{tier}: expected {n} distinct task ids, found {len(ids)}")
    return {
        "dataset": dataset,
        "split": tier,
        "n": n,
        "ids": ids,
        "ids_sha256": _sha256(json.dumps(ids)),
        "strata_fields": config.STRATA_FIELDS[dataset],
    }


# ---------------------------------------------------------------------------
# Deterministic stratified training batches. Within each stratum, tasks are ordered by
# sha256("stratune_batch|<ds>|<id>") and dealt round-robin into T batches, so every
# batch approximates the stratum mix. Depends only on the split and the strata metadata.
T_BATCHES = 16


def batch_hash(dataset: str, task_id: str) -> str:
    return util.sha256_str(f"stratune_batch|{dataset}|{task_id}")


def make_batches(dataset: str, subset: dict) -> list[list[str]]:
    meta = util.read_json(os.path.join(
        util.WORKDIR, "data", "strata", dataset, "strata_meta_train.json"))
    fields = subset["strata_fields"]

    strata: dict[str, list[str]] = {}
    for tid in subset["ids"]:
        key = "||".join(str(meta[tid][f]) for f in fields)
        strata.setdefault(key, []).append(tid)

    batches: list[list[str]] = [[] for _ in range(T_BATCHES)]
    cursor = 0  # global round-robin cursor so small strata spread out too
    for key in sorted(strata):
        members = sorted(strata[key], key=lambda t: batch_hash(dataset, t))
        for tid in members:
            batches[cursor % T_BATCHES].append(tid)
            cursor += 1
    return batches
