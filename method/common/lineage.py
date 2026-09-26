"""Training-run registry and test-time skill resolution.

A training run and its test evaluation are recorded together: the fixed
training IDs, a unique training_run_id, the completed run, the final skill
(path and sha256), and the test ledger linked to that run id.

Rules enforced here:
- a test evaluation can only be started from a training_run_id, never from a
  raw skill path;
- resolution fails unless the run status is "completed", the final skill is
  recorded, and the recomputed sha256 matches;
- completing a run makes the skill file read-only; any later change breaks
  the hash and a new training run is required.

Run manifests are small JSON files in workdir/state/runs/ (atomic writes);
large outputs live in data_root/runs/<run_id>/.
"""
import json
import os
import time
from datetime import datetime
from datetime import timezone
import stat

from method.common import util


RUNS_STATE_DIR = os.path.join(util.WORKDIR, "state", "runs")
RUNS_HEAVY_DIR = os.path.join(util.DATA_ROOT, "runs")
TEST_ACCESS_LEDGER = os.path.join(util.WORKDIR, "state", "test_access_ledger.jsonl")

VALID_STATUS = ("created", "running", "completed", "failed", "superseded")


class LineageError(RuntimeError):
    """Raised when a training run cannot be created, completed, or resolved."""


def _run_path(run_id: str) -> str:
    return os.path.join(RUNS_STATE_DIR, f"{run_id}.json")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def create_run(dataset: str, method: str, *, method_version: str,
               train_subset: dict, initial_skill_path: str,
               config: dict, target_model: str, optimizer_model: str,
               scorer_version: str) -> dict:
    """Create a new training run and record its identity (training split hash,
    initial skill hash, config hash) before any model call."""
    subset = train_subset
    ts = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    run_id = f"{dataset}_{method}_{method_version}_{ts}"
    if os.path.exists(_run_path(run_id)):
        raise LineageError(f"run id collision: {run_id}")
    heavy = os.path.join(RUNS_HEAVY_DIR, run_id)
    os.makedirs(heavy, exist_ok=True)
    manifest = {
        "training_run_id": run_id,
        "dataset": dataset,
        "method": method,
        "method_version": method_version,
        "status": "created",
        "train_subset": f"{dataset}_{subset['split']}{subset['n']}",
        "train_ids_sha256": subset["ids_sha256"],
        "train_n": subset["n"],
        "initial_skill_path": initial_skill_path,
        "initial_skill_sha256": util.sha256_file(initial_skill_path),
        "config": config,
        "config_sha256": util.sha256_str(util.canonical_json(config)),
        "target_model": target_model,
        "optimizer_model": optimizer_model,
        "scorer_version": scorer_version,
        "heavy_dir": heavy,
        "created_utc": _now(),
        "final_skill_path": None,
        "final_skill_sha256": None,
        "completed_utc": None,
    }
    util.atomic_write_json(_run_path(run_id), manifest)
    return manifest


def load_run(run_id: str) -> dict:
    p = _run_path(run_id)
    if not os.path.exists(p):
        raise LineageError(f"unknown training_run_id: {run_id}")
    return util.read_json(p)


def update_status(run_id: str, status: str) -> dict:
    assert status in VALID_STATUS, status
    m = load_run(run_id)
    if m["status"] == "completed" and status != "superseded":
        raise LineageError(f"run {run_id} is completed and immutable (only supersede allowed)")
    m["status"] = status
    util.atomic_write_json(_run_path(run_id), m)
    return m


def complete_run(run_id: str, final_skill_path: str) -> dict:
    """Freeze the run's final skill: hash it, make it read-only, mark the run
    completed. After this the manifest is immutable."""
    m = load_run(run_id)
    if m["status"] == "completed":
        raise LineageError(f"run {run_id} already completed")
    if not os.path.isfile(final_skill_path):
        raise LineageError(f"final skill not found: {final_skill_path}")
    sha = util.sha256_file(final_skill_path)
    os.chmod(final_skill_path, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    m["final_skill_path"] = final_skill_path
    m["final_skill_sha256"] = sha
    m["status"] = "completed"
    m["completed_utc"] = _now()
    util.atomic_write_json(_run_path(run_id), m)
    return m


def manifest_sha256(run_id: str) -> str:
    return util.sha256_file(_run_path(run_id))


def resolve_final_skill(run_id: str) -> dict:
    """Resolve the final skill of a completed training run for test evaluation.

    Fails unless the run is completed, the final skill is recorded, the file
    exists, and the recomputed sha256 matches. Returns {skill_text, skill_path,
    skill_sha256, run: <manifest>, training_manifest_sha256}.
    """
    m = load_run(run_id)
    if m["status"] != "completed":
        raise LineageError(
            f"run {run_id} status={m['status']!r}; Test requires a completed run (fail closed)")
    path, recorded = m.get("final_skill_path"), m.get("final_skill_sha256")
    if not path or not recorded:
        raise LineageError(f"run {run_id} has no frozen final skill (fail closed)")
    if not os.path.isfile(path):
        raise LineageError(f"final skill file missing: {path} (fail closed)")
    actual = util.sha256_file(path)
    if actual != recorded:
        raise LineageError(
            f"final skill hash mismatch for {run_id}: manifest {recorded} != actual {actual}; "
            "the skill changed after freezing — create a new training run (fail closed)")
    with open(path, encoding="utf-8") as f:
        text = f.read()
    return {
        "skill_text": text,
        "skill_path": path,
        "skill_sha256": actual,
        "run": m,
        "training_manifest_sha256": manifest_sha256(run_id),
    }


def log_test_access(dataset: str, subset_manifest: str, purpose: str,
                    run_id: str | None, n_tasks: int, extra: dict | None = None):
    """Append a record to the test-access ledger."""
    rec = {
        "ts": _now(),
        "dataset": dataset,
        "subset_manifest": subset_manifest,
        "purpose": purpose,
        "training_run_id": run_id,
        "n_tasks": n_tasks,
    }
    if extra:
        rec.update(extra)
    with open(TEST_ACCESS_LEDGER, "a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
