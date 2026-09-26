#!/usr/bin/env python3
"""Test evaluation of a skill on the official test split.

Usage:
    python3 -m method.evaluate --run-id <training_run_id> [--workers N]
    python3 -m method.evaluate --skill <skill.txt> --dataset <dataset> [--workers N]

With --run-id the final skill is resolved from the completed training-run
manifest and its hash is re-verified; results are written to the run's output
directory. With --skill any skill file (for example one of data/skills/) is
evaluated and results are written to runtime_data/skill_eval/.
"""
import argparse
import os
import sys


_STRATUNE_ROOT = os.environ.get("STRATUNE_ROOT") or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
WORKDIR = _STRATUNE_ROOT
sys.path.insert(0, WORKDIR)
os.environ.setdefault("STRATUNE_WORKSPACE", WORKDIR)

from method.common import execution  # noqa: E402
from method.common import lineage  # noqa: E402
from method.common import util  # noqa: E402

from method import config  # noqa: E402

TEST_SPLITS = config.TEST_SPLITS
DEFAULT_WORKERS = {"docvqa": 64, "mind2web": 24, "spreadsheetbench": 12, "livemath": 24}


def materialize_docvqa_test(subset):
    """DocVQA test images are materialized on first use."""
    from method.environments.datasets.materialize_docvqa import main as mat_main, materialized_dir
    if not os.path.exists(os.path.join(materialized_dir(subset), "MANIFEST.json")):
        mat_main("test")


def _test_tasks(dataset: str, subset: dict) -> list:
    if dataset == "docvqa":
        materialize_docvqa_test(subset)
        return execution.docvqa_tasks(subset, "test")
    return execution.tasks_for(dataset, subset, "test")


def _evaluate(dataset: str, skill_text: str, skill_sha: str, out_dir: str,
              workers: int | None, extra: dict) -> dict:
    split_name = TEST_SPLITS[dataset]
    subset = execution.load_subset(dataset, split_name)
    workers = workers or DEFAULT_WORKERS[dataset]
    tasks = _test_tasks(dataset, subset)
    meta = execution.run_rollout(
        dataset, tasks, skill_text, workers=workers,
        out_dir=out_dir, name=f"test_{split_name}",
        cache_ns=f"test_{dataset}_{skill_sha[:16]}")
    summary = {
        "dataset": dataset,
        "final_skill_sha256": skill_sha,
        "test_subset": f"{dataset}_{split_name}",
        "test_ids_sha256": subset["ids_sha256"],
        "test_n": subset["n"],
        "mean_primary": meta["mean_primary"],
        "scored": meta["scored"],
        "errors": meta["errors"],
        "ledger": os.path.join(out_dir, f"test_{split_name}.jsonl"),
        "seconds": meta["seconds"],
    }
    summary.update(extra)
    util.atomic_write_json(os.path.join(out_dir, "test_summary.json"), summary)
    print(f"TEST {dataset}: mean_primary={summary['mean_primary']} "
          f"scored={summary['scored']}/{subset['n']} errors={summary['errors']}")
    return summary


def run_test(run_id: str, workers: int | None = None) -> dict:
    """Evaluate the final skill of a completed training run."""
    resolved = lineage.resolve_final_skill(run_id)
    run = resolved["run"]
    dataset = run["dataset"]
    lineage.log_test_access(
        dataset, f"{dataset}_{TEST_SPLITS[dataset]}", purpose="official_test",
        run_id=run_id, n_tasks=execution.load_subset(dataset, TEST_SPLITS[dataset])["n"],
        extra={"skill_sha256": resolved["skill_sha256"]})
    return _evaluate(
        dataset, resolved["skill_text"], resolved["skill_sha256"],
        os.path.join(run["heavy_dir"], "test"), workers,
        {"method": run["method"], "method_version": run["method_version"],
         "train_ids_sha256": run["train_ids_sha256"],
         "training_run_id": run_id,
         "training_manifest_sha256": resolved["training_manifest_sha256"],
         "final_skill_path": resolved["skill_path"],
         "target_model": run["target_model"],
         "scorer_version": run["scorer_version"]})


def run_skill(skill_path: str, dataset: str, workers: int | None = None) -> dict:
    """Evaluate a skill file, for example one of data/skills/, on the test split."""
    from method.config import TARGET_MODEL
    with open(skill_path, encoding="utf-8") as f:
        skill_text = f.read()
    sha = util.sha256_str(skill_text)
    out_dir = os.path.join(util.DATA_ROOT, "skill_eval", f"{dataset}_{sha[:12]}")
    return _evaluate(dataset, skill_text, sha, out_dir, workers,
                     {"skill_path": os.path.abspath(skill_path), "target_model": TARGET_MODEL})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--run-id", help="completed training run to evaluate")
    g.add_argument("--skill", help="skill file to evaluate (requires --dataset)")
    ap.add_argument("--dataset", choices=list(TEST_SPLITS))
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()
    if args.skill:
        if not args.dataset:
            ap.error("--skill requires --dataset")
        run_skill(args.skill, args.dataset, args.workers)
    else:
        run_test(args.run_id, args.workers)
