"""Paired execution of a parent skill and a candidate, and the gain statistics.

Parent-side results are reused from the version's score ledger when present;
both sides share the (dataset, skill_sha) LLM cache. A candidate-side execution
failure counts as a severe regression and stays in the denominator.
"""
from concurrent.futures import ThreadPoolExecutor
import statistics
from method.environments.dataset_profiles import PROFILES
from method.environments.dataset_profiles import parent_success
from method.environments.dataset_profiles import severe_improvement
from method.environments.dataset_profiles import severe_regression


def run_version_on(executor, tasks, skill_text, workers, known: dict, budget) -> dict:
    """Returns {task_id: record}; skips tasks already in `known` (score ledger).
    budget: BudgetMeter, charged for non-cached executions."""
    todo = [t for t in tasks if t["task_id"] not in known]

    def one(t):
        try:
            rec = executor.run(t, skill_text)
        except Exception as e:  # noqa: BLE001
            rec = {"error": f"{type(e).__name__}: {e}"}
        return t["task_id"], rec

    out = dict(known)
    if todo:
        with ThreadPoolExecutor(max_workers=min(len(todo), workers)) as ex:
            for tid, rec in ex.map(one, todo):
                out[tid] = rec
                budget.charge(rec)
    return out


def retry_failed(executor, tasks, skill_text, results: dict, workers, budget):
    """One retry pass for tasks whose execution failed."""
    failed = [t for t in tasks if "error" in results.get(t["task_id"], {})
              and "primary" not in results.get(t["task_id"], {})]
    if not failed:
        return results
    fresh = run_version_on(executor, failed, skill_text, workers, {}, budget)
    results.update(fresh)
    return results


def paired_stats(dataset: str, impact: dict, parent_res: dict, cand_res: dict) -> dict:
    p = PROFILES[dataset]
    all_ids = impact["related"] + impact["regression"] + impact["random"]

    def primary(res, tid):
        r = res.get(tid) or {}
        return float(r["primary"]) if "primary" in r else None

    pairs, cand_exec_fail = {}, set()
    for tid in all_ids:
        pp, cp = primary(parent_res, tid), primary(cand_res, tid)
        if pp is None:
            continue  # parent-side missing -> not a valid pair
        if cp is None:
            cand_exec_fail.add(tid)
            cp = 0.0  # candidate failure scores 0 but stays in denominator
        pairs[tid] = (pp, cp)

    coverage = len(pairs) / max(1, len(all_ids))
    d = {tid: cp - pp for tid, (pp, cp) in pairs.items()}
    n = len(pairs)
    nonzero = [abs(v) for v in d.values() if abs(v) > 1e-12]
    G = sum(d.values()) / n if n else 0.0
    Gp = sum(max(v, 0.0) for v in d.values()) / n if n else 0.0
    Gm = sum(max(-v, 0.0) for v in d.values()) / n if n else 0.0
    W = sum(1 for v in d.values() if v > 1e-12)
    L = sum(1 for v in d.values() if v < -1e-12)
    eps = max(p["eps0"], (statistics.median(nonzero) / n) if nonzero and n else p["eps0"])

    parent_ok_ids = [tid for tid, (pp, _) in pairs.items() if parent_success(dataset, pp)]
    severe = {tid for tid, (pp, cp) in pairs.items()
              if severe_regression(dataset, pp, cp, tid in cand_exec_fail)
              and parent_success(dataset, pp)}
    severe_up = {tid for tid, (pp, cp) in pairs.items()
                 if tid not in cand_exec_fail and severe_improvement(dataset, pp, cp)}
    R = len(severe) / max(1, len(parent_ok_ids))
    guard_severe = [tid for tid in impact["random"] if tid in severe]

    # auxiliary score (mean change must be >= -eps_guard)
    gf = p["guardrail_field"]

    def aux(res, tid):
        r = res.get(tid) or {}
        v = r.get(gf)
        return float(v) if isinstance(v, (int, float, bool)) else None

    aux_deltas = []
    for tid in pairs:
        pa, ca = aux(parent_res, tid), aux(cand_res, tid)
        if pa is not None and ca is not None:
            aux_deltas.append(ca - pa)
    aux_mean_delta = sum(aux_deltas) / len(aux_deltas) if aux_deltas else 0.0
    eps_guard = min(0.01, eps)

    return {
        "n_pairs": n, "coverage": round(coverage, 4),
        "G": G, "G_plus": Gp, "G_minus": Gm, "W": W, "L": L,
        "n_neq": W + L, "eps": eps, "eps_guard": eps_guard,
        "R": R, "n_parent_success": len(parent_ok_ids),
        "severe_ids": sorted(severe), "severe_up_ids": sorted(severe_up),
        "guard_severe_ids": guard_severe,
        "cand_exec_fail_ids": sorted(cand_exec_fail),
        "aux_mean_delta": aux_mean_delta,
        "aux_guardrail_ok": aux_mean_delta >= -eps_guard,
        "aux_guardrail_catastrophic": aux_mean_delta < -2 * eps_guard,
        "per_task_delta": {tid: round(v, 6) for tid, v in d.items()},
    }
