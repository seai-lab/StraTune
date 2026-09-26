"""Mind2Web loader: tasks with their steps and 20 re-ranked candidate elements per
step, read from ${STRATUNE_BENCHMARK_DATA}/Mind2Web/.

    train/tasks.jsonl  800 tasks of the official train shards
    test/tasks.json    252 tasks of the official test_task split

Dependencies: stdlib only.
"""
import json

from .common import BENCHMARK_DATA, check_tier, load_json

MIND2WEB_DIR = BENCHMARK_DATA / "Mind2Web"


class Mind2WebTask:
    __slots__ = ("annotation_id", "task", "website", "domain", "subdomain",
                 "action_reprs", "steps")

    def __init__(self, record):
        self.annotation_id = record["annotation_id"]
        self.task = record["task"]
        self.website = record["website"]
        self.domain = record["domain"]
        self.subdomain = record["subdomain"]
        self.action_reprs = record["action_reprs"]
        # steps: [{i, gold_ids, op, value, candidates(20), gold_rank, forced_gold}]
        self.steps = record["steps"]

    @property
    def n_steps(self):
        return len(self.steps)

    def __repr__(self):
        return f"Mind2WebTask(annotation_id={self.annotation_id!r}, n_steps={self.n_steps})"


class Mind2WebDataset:
    def __init__(self, root=None):
        self.root = root or MIND2WEB_DIR
        self._records = {}

    def task_ids(self, tier):
        check_tier(tier)
        return [str(i) for i in load_json(self.root / tier / "task_ids.json")]

    def records(self, tier) -> dict:
        """annotation_id -> raw record. Training steps always have 20 candidates; test
        steps whose gold element is outside the candidates have none and count as failures."""
        check_tier(tier)
        if tier not in self._records:
            if tier == "train":
                recs = []
                with open(self.root / "train" / "tasks.jsonl", "r", encoding="utf-8") as f:
                    for line in f:
                        recs.append(json.loads(line))
            else:
                recs = load_json(self.root / "test" / "tasks.json")
            if tier == "train":
                for r in recs:
                    for s in r["steps"]:
                        if len(s["candidates"]) != 20:
                            raise AssertionError(
                                f"{r['annotation_id']} step {s['i']}: {len(s['candidates'])} candidates != 20")
            self._records[tier] = {r["annotation_id"]: r for r in recs}
        return self._records[tier]

    def iter_tasks(self, tier):
        """Yield Mind2WebTask for every id of `tier`, in task_ids.json order."""
        recs = self.records(tier)
        for aid in self.task_ids(tier):
            yield Mind2WebTask(recs[aid])
