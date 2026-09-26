"""Executing a skill on tasks: the environment executor, parallel sharded
rollouts, and the task lists of the paper's splits.
"""
import json
import logging
import os
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed

from method.common import util

util.setup_env()


# ---------------------------------------------------------------------------
# Shared environment execution: one place that turns (dataset, task, skill)
# into a scored record, so training rollouts, candidate evaluation, and test
# evaluation score tasks identically.
#
# Cache policy: target-LLM calls are cached under
# data_root/cache/llm/<dataset>/<skill_sha16>/; identical (task, skill, model,
# decoding) pairs are shared across runs and phases.
_pool_lock = threading.Lock()
_clients: dict = {}
_identity_checked = False


def client_for(cache_ns: str):
    """BedrockConverseClient with a JsonlCache rooted at cache/llm/<ns>.
    Identity is verified once per process."""
    global _identity_checked
    from method.environments.bedrock_client import BedrockConverseClient
    with _pool_lock:
        if cache_ns not in _clients:
            _clients[cache_ns] = BedrockConverseClient(
                cache_root=os.path.join(util.DATA_ROOT, "cache", "llm", cache_ns),
                verify_identity=not _identity_checked)
            _identity_checked = True
        return _clients[cache_ns]


class EnvExecutor:
    """Executes tasks for one dataset. Env objects are created per (skill)
    where needed (m2w cache path embeds work_name; ssb work dirs embed it)."""

    def __init__(self, dataset: str, work_name: str, m2w_step_workers: int = 3):
        self.dataset = dataset
        self.work_name = work_name
        self.m2w_step_workers = m2w_step_workers
        self._lock = threading.Lock()
        self._envs: dict = {}

    def _env_for(self, skill_sha: str):
        with self._lock:
            if skill_sha in self._envs:
                return self._envs[skill_sha]
            if self.dataset == "docvqa":
                from method.environments.tasks.docvqa import DocVQAEnv
                env = DocVQAEnv()
            elif self.dataset == "mind2web":
                from method.environments.tasks.mind2web import M2WEnv
                env = M2WEnv(
                    cache_path=os.path.join(
                        util.DATA_ROOT, "cache", "m2w_eval",
                        f"{self.work_name}.{skill_sha[:16]}.jsonl"),
                    max_step_workers=self.m2w_step_workers)
            elif self.dataset == "spreadsheetbench":
                from method.environments.tasks.spreadsheet_codegen import SpreadsheetCodegenEnv
                env = SpreadsheetCodegenEnv(work_root=os.path.join(
                    util.DATA_ROOT, "tmp", "ssb_exec", self.work_name, skill_sha[:16]))
            elif self.dataset == "livemath":
                from method.environments.tasks.livemath import LiveMathEnv
                env = LiveMathEnv()
            else:
                raise ValueError(self.dataset)
            self._envs[skill_sha] = env
            return env

    def run(self, task: dict, skill_text: str) -> dict:
        """task: dict from rollout.{docvqa,mind2web,ssb}_tasks. Returns the
        env record (has 'primary')."""
        skill_sha = util.sha256_str(skill_text)
        client = client_for(f"{self.dataset}/{skill_sha[:16]}")
        env = self._env_for(skill_sha)
        if self.dataset == "docvqa":
            with open(task["image_path"], "rb") as f:
                png = f.read()
            rec = env.evaluate_task(
                {"qid": task["qid"], "question": task["question"],
                 "answers": task["answers"], "image_png": png},
                skill_text, client)
        elif self.dataset == "mind2web":
            rec = env.evaluate_task(task["record"], skill_text, client)
        elif self.dataset == "livemath":
            rec = env.evaluate_task(task, skill_text, client)
        else:
            rec = env.evaluate_task(task["task_obj"], skill_text, client)
        return rec


# ---------------------------------------------------------------------------
# Sharded parallel task execution with deterministic merge.
#
# Each worker appends JSON lines to its own shard file; a single process merges
# shards deterministically (sorted by task_id). Completed task_ids are skipped on
# resume, so an interrupted run continues where it left off.
class CredentialFailure(RuntimeError):
    """Raised by a client when credentials are unusable; aborts the whole run."""

log = logging.getLogger("stratune.parallel")


class ShardedRunner:
    def __init__(self, out_dir: str, name: str, workers: int):
        self.out_dir = out_dir
        self.name = name
        self.workers = workers
        self.shard_dir = os.path.join(out_dir, f"{name}.shards")
        os.makedirs(self.shard_dir, exist_ok=True)
        self._locks = [threading.Lock() for _ in range(workers)]
        self._abort = threading.Event()

    # ------------------------------------------------------------------
    def _done_ids(self) -> set:
        done = set()
        for fn in os.listdir(self.shard_dir):
            if not fn.endswith(".jsonl"):
                continue
            with open(os.path.join(self.shard_dir, fn)) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        done.add(json.loads(line)["task_id"])
                    except (json.JSONDecodeError, KeyError):
                        continue  # torn write from a crash; task will re-run
        return done

    def run(self, tasks: list, fn) -> str:
        """tasks: list of dicts each having 'task_id'. fn(task) -> result dict.

        Returns path of the merged, deterministic JSONL ledger.
        Worker exceptions are recorded per-task as {"error": ...}; a
        CredentialFailure aborts the whole run.
        """
        done = self._done_ids()
        todo = [t for t in tasks if t["task_id"] not in done]
        log.info("%s: %d total, %d done, %d todo, workers=%d",
                 self.name, len(tasks), len(done), len(todo), self.workers)

        def _work(idx_task):
            idx, task = idx_task
            if self._abort.is_set():
                return
            shard = idx % self.workers
            try:
                result = fn(task)
                rec = {"task_id": task["task_id"], **result}
            except CredentialFailure:
                self._abort.set()
                raise
            except Exception as e:  # noqa: BLE001 - task-level failure is data
                rec = {"task_id": task["task_id"], "error": f"{type(e).__name__}: {e}",
                       "traceback": traceback.format_exc()[-2000:]}
            line = json.dumps(rec, ensure_ascii=False)
            path = os.path.join(self.shard_dir, f"shard_{shard:03d}.jsonl")
            with self._locks[shard]:
                with open(path, "a") as f:
                    f.write(line + "\n")
                    f.flush()
                    os.fsync(f.fileno())

        with ThreadPoolExecutor(max_workers=self.workers) as ex:
            futures = [ex.submit(_work, (i, t)) for i, t in enumerate(todo)]
            for fut in as_completed(futures):
                fut.result()  # re-raise CredentialFailure and programming errors

        return self.merge([t["task_id"] for t in tasks])

    # ------------------------------------------------------------------
    def merge(self, expected_ids: list) -> str:
        """Single-process deterministic merge: one record per task_id, ordered
        by expected_ids; later shard lines win over earlier ones (retries)."""
        records = {}
        for fn in sorted(os.listdir(self.shard_dir)):
            if not fn.endswith(".jsonl"):
                continue
            with open(os.path.join(self.shard_dir, fn)) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    records[rec["task_id"]] = rec
        out_path = os.path.join(self.out_dir, f"{self.name}.jsonl")
        missing = [i for i in expected_ids if i not in records]
        with open(out_path + ".tmp", "w") as f:
            for tid in expected_ids:
                if tid in records:
                    f.write(json.dumps(records[tid], ensure_ascii=False) + "\n")
        os.replace(out_path + ".tmp", out_path)
        meta = {"expected": len(expected_ids), "present": len(expected_ids) - len(missing),
                "missing_ids": missing}
        util.atomic_write_json(out_path + ".meta.json", meta)
        if missing:
            log.warning("%s: merged with %d missing tasks", self.name, len(missing))
        return out_path


# ---------------------------------------------------------------------------
# Shared parallel rollout runner: execute one skill over the tasks of a split.
#
# Used by StraTune training and by test evaluation. Task-level parallelism via
# ShardedRunner (resume-safe shards, deterministic merge); model-call caching via
# the BedrockConverseClient JsonlCache, so identical (model, request) pairs are
# not re-billed across runs.
#
# Dataset specifics:
# - docvqa: reads the materialized split dir (tasks.jsonl+images/), see
#   environments/datasets/materialize_docvqa.py.
# - mind2web: in-memory task records; per-task step loop, small step-worker pool.
# - spreadsheetbench: LLM call + sandboxed local execution per case; keep
#   task workers <= 16 so concurrent subprocesses stay bounded.
#!/usr/bin/env python3


def load_subset(dataset: str, split_name: str) -> dict:
    """The paper's split, e.g. ("livemath", "train397"); see environments/datasets/splits.py."""
    from method.environments.datasets.splits import load_split
    tier = "train" if split_name.startswith("train") else "test"
    return load_split(dataset, tier)


# ---------------------------------------------------------------------------
# Task sources (return list of dicts with 'task_id' + payload for the env fn)
# ---------------------------------------------------------------------------

def docvqa_tasks(subset: dict, tier: str) -> list:
    from method.environments.datasets.materialize_docvqa import materialized_dir
    mat_dir = materialized_dir(subset)
    man_path = os.path.join(mat_dir, "MANIFEST.json")
    assert os.path.exists(man_path), \
        f"DocVQA {tier} split not materialized; run: python3 -m method.environments.datasets.materialize_docvqa {tier}"
    man = util.read_json(man_path)
    assert man["ids_sha256"] == subset["ids_sha256"] and man["complete"], \
        f"materialization stale for {mat_dir}"
    recs = {}
    with open(os.path.join(mat_dir, "tasks.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            recs[r["qid"]] = r
    out = []
    for qid in subset["ids"]:
        r = recs[qid]
        out.append({"task_id": qid, "qid": qid, "question": r["question"],
                    "answers": r["answers"],
                    "image_path": os.path.join(mat_dir, "images", f"{qid}.png")})
    return out


def mind2web_tasks(subset: dict, tier: str) -> list:
    from method.environments.datasets.mind2web import Mind2WebDataset
    recs = Mind2WebDataset().records(tier)
    return [{"task_id": i, "record": recs[i]} for i in subset["ids"]]


def ssb_tasks(subset: dict, tier: str) -> list:
    from method.environments.datasets.spreadsheetbench import SpreadsheetBenchDataset
    want = set(subset["ids"])
    by_id = {}
    for t in SpreadsheetBenchDataset().iter_tasks(tier):
        if str(t.task_id) in want:
            by_id[str(t.task_id)] = t
    return [{"task_id": i, "task_obj": by_id[i]} for i in subset["ids"]]


# ---------------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------------

BD = os.environ.get("STRATUNE_BENCHMARK_DATA", os.path.join(util.WORKDIR, "benchmark_data"))


def _read_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f]


def livemath_tasks(subset: dict, tier: str) -> list:
    root = os.path.join(BD, "LiveMathematicianBench", tier)
    labels = {r["id"]: r for r in _read_jsonl(os.path.join(root, "labels.jsonl"))}
    recs = {r["id"]: r for r in _read_jsonl(os.path.join(root, "tasks.jsonl"))}
    out = []
    for tid in subset["ids"]:
        r, lab = recs[tid], labels[tid]
        out.append({"task_id": tid, "question": r["question"],
                    "choices": r["choices"], "month": r["month"],
                    "correct_label": lab["correct_label"]})
    return out


def tasks_for(dataset: str, subset: dict, tier: str) -> list:
    fn = {"docvqa": docvqa_tasks, "mind2web": mind2web_tasks,
          "spreadsheetbench": ssb_tasks, "livemath": livemath_tasks}[dataset]
    return fn(subset, tier)


def run_rollout(dataset: str, tasks: list, skill_text: str, *, workers: int,
                out_dir: str, name: str, cache_ns: str = "",
                m2w_step_workers: int = 3) -> dict:
    """Execute skill over tasks; returns {ledger, seconds, n, errors, scores}.

    cache_ns is unused: caching is keyed in EnvExecutor by
    <dataset>/<skill_sha16>, so identical (task, skill) executions are shared
    across runs and phases.
    """

    os.makedirs(out_dir, exist_ok=True)
    skill_sha = util.sha256_str(skill_text)
    executor = EnvExecutor(dataset, work_name=name,
                           m2w_step_workers=m2w_step_workers)

    def fn(t):
        rec = executor.run(t, skill_text)
        if dataset == "docvqa":
            rec.pop("gold_answers", None)
        elif dataset == "mind2web":
            for s in rec["per_step"]:
                s.pop("raw", None)
        return rec

    t0 = time.time()
    runner = ShardedRunner(out_dir, name, workers=workers)
    ledger = runner.run(tasks, fn)
    dt = time.time() - t0

    n_err, scores = 0, []
    with open(ledger) as f:
        for line in f:
            r = json.loads(line)
            if "error" in r:
                n_err += 1
            elif "primary" in r:
                scores.append(float(r["primary"]))
    meta = {
        "dataset": dataset, "name": name, "n": len(tasks),
        "skill_sha256": skill_sha, "workers": workers,
        "seconds": round(dt, 1),
        "tasks_per_min": round(60 * len(tasks) / dt, 2) if dt > 0 else None,
        "errors": n_err,
        "mean_primary": round(sum(scores) / len(scores), 4) if scores else None,
        "scored": len(scores),
    }
    util.atomic_write_json(os.path.join(out_dir, f"{name}.meta.json"), meta)
    return meta
