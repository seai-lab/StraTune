"""Execution feedback keyed by exact skill text; a record without stored output is treated as unknown.
"""
from method.common import util
from method.evaluation import run_version_on


OUTPUT_KEYS = {
    "docvqa": ("response_text", "predicted_answer"),
    "livemath": ("response_tail", "predicted_label"),
    "mind2web": ("per_step",),
    "spreadsheetbench": ("code", "response_text", "case_results"),
}


def has_output(ds, rec):
    # Presence matters: an observed empty response is different from no record.
    return not rec.get("score_only") and any(k in rec for k in OUTPUT_KEYS.get(ds, ("response_text",)))


def safe_summary(ds, rec):
    from method.operators.full_rewrite import result_summary
    score = rec.get("primary")
    if score is None:
        return "Execution evidence unavailable; no task score was recorded.", None
    if not has_output(ds, rec):
        return ("[Evaluation] Observed task score %.3f. The execution output is not "
                "available in this cache record. Do not infer an empty answer, "
                "missing code, parsing failure, or successful execution steps." % float(score), float(score))
    if ds == "spreadsheetbench" and not rec.get("case_results") and rec.get("hard_pass") is None:
        # Some cache records retain the code but omit per-case evaluation details.
        output = str(rec.get("code", rec.get("response_text", "")))[:2000]
        return (output + f"\n[Evaluation] Observed task score {float(score):.3f}; "
                "per-case pass/fail details are unavailable. Do not infer all cases passed.",
                float(score))
    return result_summary(ds, rec)


def records_for_feedback(ds, st, vid, tasks, executor, workers, budget, n):
    """Records for the first n samples; samples without stored output are executed and charged."""
    text = st.skill_text(vid)
    known = st.known_records(vid, ds)
    selected = tasks[:n]
    complete = {t["task_id"]: known[t["task_id"]] for t in selected
                if t["task_id"] in known and has_output(ds, known[t["task_id"]])}
    missing = [t for t in selected if t["task_id"] not in complete]
    if missing and not budget.exhausted(margin=len(missing)):
        fresh = run_version_on(executor, missing, text, workers, {}, budget)
        st.record_scores(ds, vid, fresh)
        complete.update(fresh)
    for task in selected:
        tid = task["task_id"]
        if tid not in complete:
            complete[tid] = known.get(tid, {"score_only": True})
    return {t["task_id"]: complete[t["task_id"]] for t in selected}


def reference(ds, st, vid, ids, tasks_by_id, executor, workers, budget):
    """Score one skill version on exactly these samples."""
    ids = list(map(str, ids))
    sha = st.d["versions"][vid]["skill_sha256"]
    key = util.sha256_str(util.canonical_json([sha, ids]))
    gen = st.d.setdefault("rw", {})
    cache = gen.setdefault("reference_scores", {})
    records = run_version_on(executor, [tasks_by_id[t] for t in ids],
                             st.skill_text(vid), workers,
                             st.known_records(vid, ds), budget)
    records = {t: records.get(t, {}) for t in ids}
    st.record_scores(ds, vid, records)
    mean = sum(float(records[t].get("primary", 0)) for t in ids) / max(1, len(ids))
    cache[key] = {"skill_sha256": sha, "task_ids": ids, "score": mean}
    return mean, records, key


def comparison(ids, parent, candidate, stage):
    ids = [t for t in ids if parent.get(t, {}).get("primary") is not None]
    deltas = {t: float(candidate.get(t, {}).get("primary", 0)) - float(parent[t]["primary"])
              for t in ids}
    return {"stage": stage, "n_pairs": len(ids), "task_ids": ids,
            "mean_gain": sum(deltas.values()) / len(ids) if ids else None,
            "wins": sum(v > 0 for v in deltas.values()),
            "losses": sum(v < 0 for v in deltas.values()),
            "per_task_delta": deltas}
