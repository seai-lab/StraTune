"""Construction of the screening set Q_s for one candidate skill.

The screening set combines samples previously solved by the current skill and a
stratified random sample; a quota of samples related to the failures the
candidate addresses is supported but set to zero in the released configuration.
Selection uses only task inputs, metadata, the candidate's edits, and the recorded
scores of the current skill, never the candidate's own scores.
"""
import re
from method.common import util
from method.environments.dataset_profiles import PROFILES
from method.environments.dataset_profiles import parent_success
from method.environments.dataset_profiles import task_text


MMR_LAMBDA = 0.30
MAX_PRIMARY_USES = 6
MAX_GUARD_USES = 3

_word = re.compile(r"[a-z0-9]+")


def tokens(text: str) -> set:
    return set(_word.findall(str(text).lower())) - {
        "the", "a", "an", "of", "to", "in", "and", "or", "for", "is", "on", "with"}


def overlap(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def patch_tokens(patch: dict, cluster: dict) -> set:
    return (tokens(" ".join(patch.get("scope") or []))
            | tokens(patch.get("new_content") or "")
            | tokens(cluster.get("description") or "")
            | tokens(cluster.get("common_theme") or ""))


def build_impact_set(dataset: str, update_id: str, patch: dict, cluster: dict,
                     pool_ids: list, tasks_by_id: dict, strata_key: dict,
                     parent_scores: dict, usage: dict, round_index: int,
                     exclude_ids: set) -> dict:
    """pool_ids: eligible ids (training samples minus the batch and the samples
    used to generate the candidate), in fixed order. strata_key: id -> stratum.
    parent_scores: recorded scores of the current skill. usage: how often each
    sample has already been used in screening sets. exclude_ids: samples used in
    an earlier screening set of the same candidate.
    Returns the screening set (ids by part, and whether it is large enough)."""
    p = PROFILES[dataset]
    quota = dict(p["impact_quota"])
    ptoks = patch_tokens(patch, cluster)

    def usable_primary(tid):
        return (usage["primary"].get(tid, 0) < MAX_PRIMARY_USES
                and tid not in exclude_ids)

    task_toks = {tid: tokens(task_text(dataset, tasks_by_id[tid])) for tid in pool_ids}

    # Use at most a third of the known solved samples, so that a second, disjoint
    # screening set still has solved samples available (at least 6).
    known_pool = sum(1 for tid in pool_ids
                     if tid in parent_scores and parent_success(dataset, parent_scores[tid])
                     and usage["primary"].get(tid, 0) < MAX_PRIMARY_USES)
    quota["regression"] = min(quota["regression"], max(6, known_pool // 3))

    # ---- related samples: lexical overlap with the candidate's edits, with a diversity penalty ----
    cands = [(tid, overlap(ptoks, task_toks[tid])) for tid in pool_ids
             if usable_primary(tid)]
    cands = [(tid, rel) for tid, rel in cands if rel > 0]
    cands.sort(key=lambda x: (-x[1], util.sha256_str(f"{update_id}|rel|{x[0]}")))
    related, rel_toks = [], []
    for tid, rel in cands:
        if len(related) >= quota["related"]:
            break
        max_sim = max((overlap(task_toks[tid], t) for t in rel_toks), default=0.0)
        if rel - MMR_LAMBDA * max_sim <= 0:
            continue
        related.append(tid)
        rel_toks.append(task_toks[tid])

    # Lexical overlap can be sparse (e.g. web tasks); fill the remaining quota with
    # samples from the same strata as the failures the candidate addresses.
    if len(related) < quota["related"]:
        src_strata = {strata_key.get(t, "?") for t in cluster.get("task_ids", [])}
        backfill = [tid for tid in pool_ids
                    if usable_primary(tid) and tid not in set(related)
                    and strata_key.get(tid, "?") in src_strata]
        backfill.sort(key=lambda t: util.sha256_str(f"{update_id}|relmeta|{t}"))
        related.extend(backfill[: quota["related"] - len(related)])
    related_set = set(related)

    # ---- previously solved samples ----
    anchors = []
    known_success = [tid for tid in pool_ids
                     if tid in parent_scores and parent_success(dataset, parent_scores[tid])
                     and usable_primary(tid) and tid not in related_set]
    scoped = [t for t in known_success if overlap(ptoks, task_toks[t]) > 0]
    rest = [t for t in known_success if t not in set(scoped)]
    for grp in (scoped, rest):
        grp.sort(key=lambda t: util.sha256_str(f"{update_id}|anchor|{t}"))
        # round-robin over strata for the stratified part
        by_stratum: dict = {}
        for t in grp:
            by_stratum.setdefault(strata_key.get(t, "?"), []).append(t)
        keys = sorted(by_stratum)
        while len(anchors) < quota["regression"] and keys:
            for k in list(keys):
                if not by_stratum[k]:
                    keys.remove(k)
                    continue
                anchors.append(by_stratum[k].pop(0))
                if len(anchors) >= quota["regression"]:
                    break
    anchor_set = set(anchors)

    # ---- stratified random samples, ordered by a candidate-specific hash ----
    guard_pool = [tid for tid in pool_ids
                  if usage["guard"].get(tid, 0) < MAX_GUARD_USES
                  and tid not in related_set and tid not in anchor_set
                  and tid not in exclude_ids]
    by_stratum = {}
    for t in sorted(guard_pool, key=lambda t: util.sha256_str(f"{update_id}|guard|{t}")):
        by_stratum.setdefault(strata_key.get(t, "?"), []).append(t)
    guards, keys = [], sorted(by_stratum)
    while len(guards) < quota["random"] and keys:
        for k in list(keys):
            if not by_stratum[k]:
                keys.remove(k)
                continue
            guards.append(by_stratum[k].pop(0))
            if len(guards) >= quota["random"]:
                break

    # ---- is the screening set large and diverse enough? ----
    reasons = []
    for part, got, q in (("related", related, quota["related"]),
                         ("regression", anchors, quota["regression"]),
                         ("random", guards, quota["random"])):
        if len(got) < 0.70 * q:
            reasons.append(f"{part} {len(got)}/{q} < 70% quota")
    rel_strata = [strata_key.get(t, "?") for t in related]
    n_cat = len(set(rel_strata))
    if related and n_cat < 2:
        reasons.append("related covers < 2 strata categories")
    if related:
        top = max(rel_strata.count(c) for c in set(rel_strata))
        if top > 0.70 * len(related) and n_cat > 1:
            reasons.append("one stratum > 70% of related")

    return {
        "update_id": update_id,
        "round_index": round_index,
        "related": related, "regression": anchors, "random": guards,
        "ready": not reasons, "not_ready_reasons": reasons,
        "ids_sha256": util.sha256_ids(related + anchors + guards),
    }


def record_usage(usage: dict, impact: dict):
    for tid in impact["related"] + impact["regression"]:
        usage["primary"][tid] = usage["primary"].get(tid, 0) + 1
    for tid in impact["random"]:
        usage["guard"][tid] = usage["guard"].get(tid, 0) + 1
