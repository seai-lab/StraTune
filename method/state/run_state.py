"""StraTune run state: version graph, score ledgers, usage caps, budget meter.
Atomically persisted to heavy_dir/state.json after every round (resume-safe).
Skill texts live in heavy_dir/skills/<version_id>.txt (read-only after write).
"""
import os
from method.common import util

from .execution_feedback import has_output


def detect_cached(dataset: str, rec: dict) -> bool:
    if "error" in rec and "primary" not in rec:
        return False
    if dataset == "mind2web":
        steps = rec.get("per_step", [])
        return bool(steps) and all(s.get("cache_hit") for s in steps)
    return bool(rec.get("cached"))


class BudgetMeter:
    def __init__(self, dataset: str, cap: int, state: dict):
        self.dataset = dataset
        self.cap = cap
        self.state = state.setdefault("budget", {"billed": 0, "raw": 0})

    def charge(self, rec: dict):
        self.state["raw"] += 1
        if not detect_cached(self.dataset, rec):
            self.state["billed"] += 1

    @property
    def billed(self):
        return self.state["billed"]

    def exhausted(self, margin: int = 0) -> bool:
        return self.state["billed"] + margin >= self.cap


class RunState:
    def __init__(self, heavy_dir: str, seed_skill: str):
        self.heavy_dir = heavy_dir
        self.path = os.path.join(heavy_dir, "state.json")
        self.skills_dir = os.path.join(heavy_dir, "skills")
        os.makedirs(self.skills_dir, exist_ok=True)
        if os.path.exists(self.path):
            self.d = util.read_json(self.path)
        else:
            self.d = {
                "versions": {}, "champion": None, "provisional": None,
                "archive": [], "scores": {}, "aux": {},
                "usage": {"primary": {}, "guard": {}},
                "next_batch": 0, "updates": {}, "merges_attempted": [],
                "round_log": [],
            }
            v0 = self.add_version(seed_skill, parent=None, patch=None, round_index=-1)
            self.d["champion"] = v0
            self.d["archive"] = [v0]
            self.save()

    # ------------------------------------------------------------------
    def save(self):
        util.atomic_write_json(self.path, self.d)

    def skill_path(self, vid: str) -> str:
        return os.path.join(self.skills_dir, f"{vid}.txt")

    def skill_text(self, vid: str) -> str:
        with open(self.skill_path(vid), encoding="utf-8") as f:
            return f.read()

    def add_version(self, skill_text: str, parent: str | None, patch: dict | None,
                    round_index: int, status: str = "active") -> str:
        sha = util.sha256_str(skill_text)
        vid = f"v{len(self.d['versions']):03d}_{sha[:8]}"
        with open(self.skill_path(vid), "w", encoding="utf-8") as f:
            f.write(skill_text)
        self.d["versions"][vid] = {
            "skill_sha256": sha, "parent": parent, "patch": patch,
            "round": round_index, "status": status,
        }
        self.d["scores"].setdefault(vid, {})
        self.d["aux"].setdefault(vid, {})
        return vid

    # ------------------------------------------------------------------
    def record_scores(self, dataset: str, vid: str, results: dict):
        """results: task_id -> env record. Persists the primary and auxiliary scores."""
        from method.environments.dataset_profiles import PROFILES
        gf = PROFILES[dataset]["guardrail_field"]
        for tid, rec in results.items():
            if "primary" in rec:
                self.d["scores"][vid][tid] = float(rec["primary"])
                v = rec.get(gf)
                if isinstance(v, (int, float, bool)):
                    self.d["aux"][vid][tid] = float(v)
                if has_output(dataset, rec):
                    path = self._execution_path(vid, tid)
                    util.atomic_write_json(path, rec)

    def _execution_path(self, vid, tid):
        sha = self.d["versions"][vid]["skill_sha256"]
        return os.path.join(self.heavy_dir, "execution_records", sha,
                            util.sha256_str(str(tid)) + ".json")

    def known_records(self, vid: str, dataset: str) -> dict:
        """Score ledger rendered as pseudo-records usable by paired.run_version_on
        (primary and auxiliary scores, enough for the statistics)."""
        from method.environments.dataset_profiles import PROFILES
        gf = PROFILES[dataset]["guardrail_field"]
        aux = self.d["aux"].get(vid, {})
        out = {}
        for tid, p in self.d["scores"].get(vid, {}).items():
            path = self._execution_path(vid, tid)
            rec = util.read_json(path) if os.path.isfile(path) else {"score_only": True}
            rec.update(primary=p, cached=True)
            rec["skill_sha256"] = self.d["versions"][vid]["skill_sha256"]
            if tid in aux:
                rec[gf] = aux[tid]
            out[tid] = rec
        return out

    def mean_primary(self, vid: str, task_ids=None) -> float | None:
        s = self.d["scores"].get(vid, {})
        ids = list(s) if task_ids is None else [t for t in task_ids if t in s]
        if not ids:
            return None
        return sum(s[t] for t in ids) / len(ids)
