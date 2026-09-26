#!/usr/bin/env python3
"""StraTune training driver.

Each round: execute the current skill on a training batch, build the optimization state, let the
optimizer LLM select the revision operator and generate candidate skills, evaluate the candidates
(initial screening, further validation), keep saved candidates, and after the budget is spent run
final skill selection.

Usage: python3 -m method.train <dataset> [--dry-run]
"""
import argparse
import os
import sys


_STRATUNE_ROOT = os.environ.get("STRATUNE_ROOT") or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
WORKDIR = _STRATUNE_ROOT
sys.path.insert(0, WORKDIR)

from method import config  # noqa: E402
from method.common import execution  # noqa: E402
from method.common import lineage  # noqa: E402
from method.common import util  # noqa: E402
from method.common.execution import EnvExecutor  # noqa: E402
from method.environments.datasets import splits  # noqa: E402
from method import evaluation  # noqa: E402
from method.evaluation import acceptance_rules  # noqa: E402
from method.operators import optimizer_llm  # noqa: E402
from method.common import skill_edits  # noqa: E402
from method.evaluation import paired_stats  # noqa: E402
from method.evaluation import retry_failed  # noqa: E402
from method.evaluation import run_version_on  # noqa: E402
from method.state import BudgetMeter  # noqa: E402
from method.state import RunState  # noqa: E402
from method.environments import dataset_profiles as profiles  # noqa: E402
from method import state  # noqa: E402
from method.operators import direct_revision  # noqa: E402
from method.operators import iterative_refinement  # noqa: E402
from method.operators import selection  # noqa: E402
from method.environments.dataset_profiles import PROFILES  # noqa: E402
from method.environments.dataset_profiles import parent_success  # noqa: E402

METHOD_VERSION = os.environ.get("STRATUNE_METHOD_TAG", "paper")
TARGET_MODEL = os.environ.get("STRATUNE_TARGET_MODEL",
                              "global.anthropic.claude-haiku-4-5-20251001-v1:0")
OPTIMIZER_MODEL = os.environ.get("STRATUNE_OPTIMIZER_MODEL", "global.anthropic.claude-sonnet-4-6")
MAX_EPOCHS = 4

# Length limits on candidate skills (characters of edited content and of full rewrites).
skill_edits.EDIT_CAP = 2400
skill_edits.GROWTH_FRAC = 0.60
skill_edits.REWRITE_MULT = 4
skill_edits.REWRITE_SLACK = 2500
# Screened candidates with a positive gain below the screening threshold are saved
# for final skill selection.
acceptance_rules.ARCHIVE_POSITIVE = True
# The content guidance of the generation prompts is also used by the full-rewrite form.
optimizer_llm.CONTENT_V25 = True
# Minimum gain for further validation, minimum gain for final skill selection,
# and the size of the pool of saved candidate skills and of the final comparison.
# The dataset configuration (method/configs/<dataset>.json) may override these.
CONFIRM_FLOOR = 0.010
TOURN_FLOOR_MIN = 0.010
ARCHIVE_CAP = 4
TOURN_SEATS = 2


def log(run_id, msg):
    print(f"[stratune/{run_id}] {msg}", flush=True)


def _live(st, key, n=1):
    lv = st.d.setdefault("liveness", {})
    lv[key] = lv.get(key, 0) + n


# ---------------------------------------------------------------------------
@state.screening
def evaluate_update(ds, st, executor, uid, train_ids, tasks_by_id, strata_key,
                    budget, workers, rnd, all_batches, shared_impact=None):
    """Initial screening of one candidate skill against the current skill.

    Builds the screening set (previously solved samples plus a stratified random
    sample, excluding the training batch and the samples used to generate the
    candidate), executes both skills on it, and returns the acceptance decision.
    When the evidence is inconclusive, the candidate is screened again in the
    next round on a disjoint screening set and the two sets are combined."""
    urec = st.d["updates"][uid]
    cand, parent = urec["candidate"], urec["parent"]
    patch = st.d["versions"][cand]["patch"]
    src = set(map(str, all_batches[urec["source_batch"]]))
    src |= set(map(str, (patch.get("source_task_ids") or [])))
    src |= set(map(str, (urec.get("cluster", {}).get("task_ids") or [])))
    pool = [t for t in train_ids if t not in src]
    prev_ids = set()
    for s in urec["impact_sets"]:
        prev_ids |= set(s["related"] + s["regression"] + s["random"])

    if shared_impact is not None:
        imp = dict(shared_impact, update_id=uid, shared=True)
    else:
        imp = evaluation.build_impact_set(
            ds, uid, patch, urec["cluster"], pool, tasks_by_id, strata_key,
            st.d["scores"].get(parent, {}), st.d["usage"],
            round_index=len(urec["impact_sets"]) + 1, exclude_ids=prev_ids)
    urec["readiness_attempts"] += 1
    if not imp["ready"]:
        if urec["readiness_attempts"] >= 2:
            urec["status"] = "insufficient_evidence"
            log(st.run_id, f"{uid}: exam not ready twice ({imp['not_ready_reasons']})")
        else:
            log(st.run_id, f"{uid}: exam not ready ({imp['not_ready_reasons']}); retry next round")
        return None
    if not imp.get("shared"):
        evaluation.record_usage(st.d["usage"], imp)
    urec["impact_sets"].append(imp)

    all_ids = imp["related"] + imp["regression"] + imp["random"]

    # Preliminary check on eight previously solved samples: a candidate that fails
    # nearly all of them is rejected before the full screening set is executed.
    smoke_ids = imp["regression"][:8]
    if len(smoke_ids) == 8:
        s_tasks = [tasks_by_id[t] for t in smoke_ids]
        s_res = run_version_on(executor, s_tasks, st.skill_text(cand), workers,
                               st.known_records(cand, ds), budget)
        st.record_scores(ds, cand, s_res)
        par_known = st.d["scores"].get(parent, {})
        broke = ok = 0
        for t in smoke_ids:
            r = s_res.get(t) or {}
            if "primary" not in r:
                continue
            if parent_success(ds, float(par_known.get(t, 0.0))):
                if parent_success(ds, float(r["primary"])):
                    ok += 1
                else:
                    broke += 1
        if broke >= 6 and ok == 0:
            log(st.run_id, f"{uid}: smoke {broke}/8 anchors broken, 0 kept -> reject")
            urec["stats"].append({"smoke": True, "broke": broke, "kept": ok})
            _live(st, "smoke_rejects")
            return ("reject", {"G": -1.0, "R": 1.0, "W": 0, "L": broke,
                               "severe_ids": smoke_ids[:4]}, imp)

    # The screening set is executed in two stages. The first stage interleaves
    # previously solved and random samples; a candidate that is already clearly
    # losing after it is rejected early. The screening set itself is fixed; only
    # the execution order is staged.
    reg_ids, rnd_ids = imp["regression"], imp["random"]
    inter = []
    for i in range(max(len(reg_ids), len(rnd_ids))):
        if i < len(reg_ids):
            inter.append(reg_ids[i])
        if i < len(rnd_ids):
            inter.append(rnd_ids[i])
    stage1 = inter[: max(12, len(inter) // 2)]
    s_tasks = [tasks_by_id[t] for t in stage1]
    par_res = run_version_on(executor, s_tasks, st.skill_text(parent), workers,
                             st.known_records(parent, ds), budget)
    par_res = retry_failed(executor, s_tasks, st.skill_text(parent), par_res, workers, budget)
    st.record_scores(ds, parent, par_res)
    cand_res = run_version_on(executor, s_tasks, st.skill_text(cand), workers,
                              st.known_records(cand, ds), budget)
    cand_res = retry_failed(executor, s_tasks, st.skill_text(cand), cand_res, workers, budget)
    st.record_scores(ds, cand, cand_res)
    imp1 = {"related": [], "regression": [t for t in stage1 if t in set(reg_ids)],
            "random": [t for t in stage1 if t in set(rnd_ids)]}
    s1 = paired_stats(ds, imp1, par_res, cand_res)
    if (s1["W"] - s1["L"]) <= -2 and s1["G"] <= -s1["eps"]:
        log(st.run_id, f"{uid}: FUTILITY stop at stage-1 (n={s1['n_pairs']} "
                       f"G={s1['G']:.4f} W/L={s1['W']}/{s1['L']}) -> reject")
        urec["stats"].append({k: v for k, v in s1.items() if k != "per_task_delta"})
        _live(st, "futility_stops")
        return ("reject", s1, imp)

    rest = [t for t in all_ids if t not in set(stage1)]
    r_tasks = [tasks_by_id[t] for t in rest]
    par_res = run_version_on(executor, r_tasks, st.skill_text(parent), workers,
                             dict(par_res, **st.known_records(parent, ds)), budget)
    par_res = retry_failed(executor, [tasks_by_id[t] for t in all_ids],
                           st.skill_text(parent), par_res, workers, budget)
    st.record_scores(ds, parent, par_res)
    cand_res = run_version_on(executor, r_tasks, st.skill_text(cand), workers,
                              dict(cand_res, **st.known_records(cand, ds)), budget)
    cand_res = retry_failed(executor, [tasks_by_id[t] for t in all_ids],
                            st.skill_text(cand), cand_res, workers, budget)
    st.record_scores(ds, cand, cand_res)

    stats = paired_stats(ds, imp, par_res, cand_res)
    urec["stats"].append({k: v for k, v in stats.items() if k != "per_task_delta"})
    if len(urec["stats"]) == 2:
        stats = evaluation.combine_rounds({**urec["stats"][0], "per_task_delta": {}}, stats)
    deltas = stats.get("per_task_delta", {})
    win_cats = {strata_key.get(t, "?") for t, v in deltas.items() if v > 1e-12}
    loss_cats = {strata_key.get(t, "?") for t, v in deltas.items() if v < -1e-12}
    verdict = evaluation.decide(stats, win_categories=max(1, len(win_cats)),
                            loss_categories=max(1, len(loss_cats)),
                            patch_applied_ok=True,
                            final_round=len(urec["stats"]) >= 2)
    urec["last_verdict"] = verdict
    log(st.run_id, f"{uid}: round{len(urec['stats'])} G={stats['G']:.4f} "
                   f"eps={stats['eps']:.4f} W/L={stats['W']}/{stats['L']} "
                   f"R={stats['R']:.3f} -> {verdict['decision']}")
    return verdict["decision"], stats, imp


@state.confirmation
def confirm_commit(ds, st, executor, uid, rnd, train_ids, tasks_by_id, budget,
                   workers, all_batches, n_confirm):
    """Further validation of an accepted candidate on a larger validation set.

    The validation set is drawn from training samples not used in screening or
    generation. The candidate is kept if its mean gain over the previous current
    skill reaches the validation threshold, the regression check passes, and
    enough paired executions succeeded; otherwise it is withdrawn."""
    from method.evaluation import flip_balance_lb
    urec = st.d["updates"][uid]
    new_champ, old_champ = st.d["champion"], urec["parent"]
    used = set()
    for imp_ in urec.get("impact_sets", []):
        used |= set(imp_["related"] + imp_["regression"] + imp_["random"])
    used |= set(map(str, all_batches[urec["source_batch"]]))
    p_ = st.d["versions"][new_champ].get("patch") or {}
    used |= set(map(str, p_.get("source_task_ids") or []))
    used |= set(map(str, (urec.get("cluster", {}) or {}).get("task_ids") or []))
    pool = [t for t in train_ids if t not in used]
    pool.sort(key=lambda t: util.sha256_str(f"confirm|{uid}|{t}"))
    ids = pool[:n_confirm]
    # Validation is executed in two stages; a candidate with a negative mean gain
    # after the first 80 samples is withdrawn without executing the rest.
    c1 = ids[: min(80, len(ids))]
    t1 = [tasks_by_id[t] for t in c1]
    res_old = run_version_on(executor, t1, st.skill_text(old_champ), workers,
                             st.known_records(old_champ, ds), budget)
    st.record_scores(ds, old_champ, res_old)
    res_new = run_version_on(executor, t1, st.skill_text(new_champ), workers,
                             st.known_records(new_champ, ds), budget)
    st.record_scores(ds, new_champ, res_new)
    s80 = paired_stats(ds, {"related": c1, "regression": [], "random": []},
                       res_old, res_new)
    if s80["G"] < 0:
        log(st.run_id, f"{uid}: CONFIRMATION futility at n={s80['n_pairs']} "
                       f"G={s80['G']:+.4f} -> early ROLLBACK")
        _live(st, "confirm_futility_stops")
        stats, early = s80, True
    else:
        rest = ids[len(c1):]
        rt = [tasks_by_id[t] for t in rest]
        res_old2 = run_version_on(executor, rt, st.skill_text(old_champ), workers,
                                  dict(res_old, **st.known_records(old_champ, ds)), budget)
        st.record_scores(ds, old_champ, res_old2)
        res_new2 = run_version_on(executor, rt, st.skill_text(new_champ), workers,
                                  dict(res_new, **st.known_records(new_champ, ds)), budget)
        st.record_scores(ds, new_champ, res_new2)
        imp = {"related": ids, "regression": [], "random": []}
        stats = paired_stats(ds, imp, res_old2, res_new2)
        early = False
    ok = (not early and stats["G"] >= max(PROFILES[ds]["eps0"], CONFIRM_FLOOR)
          and flip_balance_lb(stats) <= 0.5 and stats["coverage"] >= 0.9)
    log(st.run_id, f"{uid}: CONFIRMATION n={stats['n_pairs']} G={stats['G']:+.4f} "
                   f"flipLB={flip_balance_lb(stats):.3f} -> {'CONFIRMED' if ok else 'ROLLBACK'}")
    if ok:
        st.d["versions"][new_champ]["status"] = "champion"
        st.d["provisional"] = None
        _live(st, "confirms")
    else:
        st.d["champion"] = old_champ
        st.d["versions"][new_champ]["status"] = "rolled_back"
        if new_champ in st.d["archive"]:
            st.d["archive"].remove(new_champ)
        st.d["provisional"] = None
        st.d["confirm_rollbacks"] = st.d.get("confirm_rollbacks", 0) + 1
        _live(st, "confirm_rollbacks")
        _postmortem(st, ds, uid, stats, tasks_by_id)
        ec = st.d.setdefault("epoch_commits", {})
        T2 = st.d.get("_T", 16)
        k = str(rnd // T2)
        ec[k] = max(0, ec.get(k, 1) - 1)
    urec["confirm_G"] = stats["G"]
    urec["confirmation_stats"] = stats
    return ok


def _postmortem(st, ds, uid, stats, tasks_by_id):
    urec = st.d["updates"].get(uid, {})
    patch = st.d["versions"].get(urec.get("candidate", ""), {}).get("patch", {})
    gist = (patch.get("new_content")
            or " ".join((e.get("new_content") or "") for e in (patch.get("edits") or [])))[:220]
    theme = urec.get("cluster", {}).get("common_theme", "?")
    sev = stats.get("severe_ids", [])[:4]
    examples = "; ".join(profiles.task_text(ds, tasks_by_id[t])[:110]
                         for t in sev if t in tasks_by_id)
    state.record_postmortem(st, urec.get("form", "?"), theme,
                             gist.replace("\n", " "), stats, examples)


def _apply_decision(st, uid, decision, rnd):
    urec = st.d["updates"][uid]
    cand = urec["candidate"]
    if decision == "commit":
        ec = st.d.setdefault("epoch_commits", {})
        T2 = st.d.get("_T", 16)
        ec[str(rnd // T2)] = ec.get(str(rnd // T2), 0) + 1
        st.d["versions"][cand]["status"] = "provisional_champion"
        st.d["provisional"] = {"version": cand, "old_champion": st.d["champion"],
                               "rounds": 0, "pairs": {}}
        st.d["champion"] = cand
        if cand not in st.d["archive"]:
            st.d["archive"].append(cand)
        urec["status"] = "committed"
    elif decision == "branch":
        st.d["versions"][cand]["status"] = "provisional_branch"
        if cand not in st.d["archive"]:
            st.d["archive"].append(cand)
        urec["status"] = "branched"
        _live(st, "branches")
        entry = st.d.get("adaptive_case_history", {}).get(uid)
        if entry is not None:
            entry["result"] = "saved_without_acceptance"
    elif decision in ("reject", "invalid"):
        st.d["versions"][cand]["status"] = "rejected"
        urec["status"] = decision
    elif decision == "insufficient_evidence":
        st.d["versions"][cand]["status"] = "insufficient_evidence"
        urec["status"] = "insufficient_evidence"
    elif decision == "probation":
        urec["status"] = "pending_impact"
        _live(st, "probations")


def _prune_archive(st):
    d = st.d
    protected = {d["champion"]}
    if d.get("provisional"):
        protected.add(d["provisional"]["old_champion"])
    archive = [v for v in d["archive"]
               if d["versions"][v]["status"] not in ("rejected", "rolled_back", "pruned")]
    if len(archive) <= ARCHIVE_CAP:
        d["archive"] = archive
        return
    champ_scores = d["scores"].get(d["champion"], {})

    def value(v):
        if v in protected:
            return float("inf")
        s = d["scores"].get(v, {})
        common = [t for t in s if t in champ_scores]
        rescues = sum(1 for t in common if s[t] > champ_scores[t] + 1e-12)
        mean = sum(s[t] for t in common) / len(common) if common else 0.0
        return rescues * 10 + mean
    archive.sort(key=value, reverse=True)
    for v in archive[ARCHIVE_CAP:]:
        d["versions"][v]["status"] = "pruned"
    d["archive"] = archive[:ARCHIVE_CAP]


def _donor_salvage(ds, st, executor, tasks_by_id, train_ids, strata_key, budget,
                   workers, rnd, all_batches, n_confirm):
    """Before final skill selection, merge content from saved candidate skills.

    Up to two saved skills that solve at least three samples the current skill
    fails are used as sources. The optimizer LLM proposes a few edits that carry
    the responsible content into the current skill, and the result goes through
    the same candidate evaluation as any other candidate."""
    champ = st.d["champion"]
    champ_scores = st.d["scores"].get(champ, {})
    if not champ_scores:
        return
    donors = []
    for vid, v in st.d["versions"].items():
        if vid == champ or v.get("status") == "champion":
            continue
        s = st.d["scores"].get(vid, {})
        if len(s) < 20:
            continue
        rescues = [t for t, val in s.items()
                   if t in champ_scores and t in tasks_by_id
                   and parent_success(ds, val)
                   and not parent_success(ds, champ_scores[t])]
        if len(rescues) >= 3:
            donors.append((len(rescues), vid, rescues))
    donors.sort(key=lambda x: (-x[0], x[1]))
    seen_sha = {st.d["versions"][champ]["skill_sha256"]}
    taken = 0
    for n_resc, vid, rescues in donors:
        if taken >= 2 or budget.exhausted(margin=300):
            break
        sha = st.d["versions"][vid]["skill_sha256"]
        if sha in seen_sha:
            continue
        seen_sha.add(sha)
        taken += 1
        rescue_lines = [f"[{t}] {profiles.task_text(ds, tasks_by_id[t])[:130]}"
                        for t in sorted(rescues)[:10]]
        pm = ""
        uid_of = st.d["versions"][vid].get("update_id")
        if uid_of:
            lv = (st.d["updates"].get(uid_of) or {}).get("last_verdict") or {}
            pm = "; ".join(lv.get("reasons") or [])[:300]
        skill = st.skill_text(champ)
        try:
            patch = optimizer_llm.propose_semantic_merge(
                ds, skill, st.skill_text(vid), rescue_lines, pm,
                min(skill_edits.EDIT_CAP, 1600), salt=f"{st.sampling_id[-6:]}|salvage{taken}")
            cand_text = skill_edits.apply_patch(skill, patch)
        except (optimizer_llm.LLMOpError, skill_edits.PatchError) as e:
            log(st.run_id, f"salvage donor {vid}: unusable ({e})")
            continue
        patch["source_task_ids"] = sorted(rescues)[:10]
        nv = st.add_version(cand_text, parent=champ, patch=patch,
                            round_index=rnd, status="salvage_candidate")
        uid = f"sv{taken}_{st.d['versions'][nv]['skill_sha256'][:8]}"
        st.d["versions"][nv]["update_id"] = uid
        st.d["updates"][uid] = {
            "update_id": uid, "candidate": nv, "parent": champ, "form": "salvage",
            "cluster": {"description": f"donor salvage from {vid} ({n_resc} rescues)",
                        "common_theme": "surgical transplant of rescue-responsible content",
                        "task_ids": sorted(rescues)[:10]},
            "source_batch": rnd % len(all_batches), "impact_sets": [], "stats": [],
            "readiness_attempts": 0, "status": "pending_impact"}
        _live(st, "salvage_tried")
        out = evaluate_update(ds, st, executor, uid, train_ids, tasks_by_id,
                              strata_key, budget, workers, rnd, all_batches)
        if not out:
            continue
        _apply_decision(st, uid, out[0], rnd)
        if out[0] == "commit":
            confirm_commit(ds, st, executor, uid, rnd, train_ids, tasks_by_id,
                           budget, workers, all_batches, n_confirm)
        st.save()


def _final_tournament(ds, st, executor, tasks_by_id, train_ids, budget, workers,
                      floor_min):
    """Final skill selection.

    The saved candidate skills with the highest mean gain over the current skill
    on the recorded scores are compared with it on a common set of training
    samples, split into a selection subset (60%) and a confirmation subset (40%).
    A candidate replaces the selected skill only if it passes the gain and
    regression checks on both subsets."""
    import statistics
    p = PROFILES[ds]
    champ = st.d["champion"]
    alts = [v for v in st.d["archive"] if v != champ
            and st.d["versions"][v]["status"] not in ("rejected", "rolled_back", "pruned")]
    champ_scores = st.d["scores"].get(champ, {})
    ranked = []
    for v in alts:
        s = st.d["scores"].get(v, {})
        common = [t for t in s if t in champ_scores]
        if len(common) < 20:
            continue
        m = sum(s[t] - champ_scores[t] for t in common) / len(common)
        ranked.append((m, v))
    ranked.sort(key=lambda x: -x[0])
    contenders = [v for _, v in ranked[:TOURN_SEATS]]

    def fit_to_budget(vids):
        ordered = sorted(train_ids, key=lambda t: util.sha256_str(f"tourn|{t}"))
        chosen, est = [], 0
        room = max(0, budget.cap - budget.billed)
        for t in ordered:
            need = sum(1 for v in vids if t not in st.d["scores"].get(v, {}))
            if est + need > room:
                continue
            chosen.append(t)
            est += need
        if len(chosen) < len(train_ids):
            log(st.run_id, f"tournament budget-trimmed to {len(chosen)}/{len(train_ids)}")
        return chosen

    vids = [champ] + contenders
    tourn_ids = fit_to_budget(vids)
    # Disjoint selection (60%) and confirmation (40%) subsets, so that a candidate
    # chosen for its gain on one subset is confirmed on samples that did not
    # influence the choice.
    n_sel = max(1, int(0.6 * len(tourn_ids)))
    sel_ids, conf_ids = tourn_ids[:n_sel], tourn_ids[n_sel:]
    winner = champ
    res_w_all = run_version_on(executor, [tasks_by_id[t] for t in tourn_ids],
                               st.skill_text(champ), workers,
                               st.known_records(champ, ds), budget)
    st.record_scores(ds, champ, res_w_all)
    if not contenders:
        st.d.setdefault("tournament", {})["winner"] = champ
        return champ
    from method.environments.dataset_profiles import severe_improvement

    def paired_seg(res_a, res_c, ids):
        d = []
        for t in ids:
            a, c = res_a.get(t, {}), res_c.get(t, {})
            if "primary" in a and "primary" in c:
                d.append((t, float(a["primary"]) - float(c["primary"])))
        n = len(d)
        vals = [x for _, x in d]
        nz = [abs(x) for x in vals if abs(x) > 1e-12]
        eps_f = max(p["eps0"], (statistics.median(nz) / n) if nz and n else p["eps0"])
        mean = sum(vals) / n if n else 0.0
        W = sum(1 for x in vals if x > 1e-12)
        L = sum(1 for x in vals if x < -1e-12)
        ok = [t for t in ids if "primary" in res_c.get(t, {})
              and parent_success(ds, float(res_c[t]["primary"]))]
        sev = sum(1 for t in ok if "primary" in res_a.get(t, {})
                  and not parent_success(ds, float(res_a[t]["primary"])))
        ups = sum(1 for t in ids
                  if "primary" in res_a.get(t, {}) and "primary" in res_c.get(t, {})
                  and severe_improvement(ds, float(res_c[t]["primary"]), float(res_a[t]["primary"])))
        R_lb = evaluation.wilson_lb(sev, max(1, len(ok)))
        flip_lb = evaluation.wilson_lb(sev, sev + ups) if sev + ups else 0.0
        return {"mean": mean, "eps_f": eps_f, "W": W, "L": L,
                "R_lb": R_lb, "flip_lb": flip_lb, "n": n}

    rounds = []
    for alt in contenders:
        if budget.exhausted(margin=20):
            break
        res_a = run_version_on(executor, [tasks_by_id[t] for t in tourn_ids],
                               st.skill_text(alt), workers,
                               st.known_records(alt, ds), budget)
        st.record_scores(ds, alt, res_a)
        s1 = paired_seg(res_a, res_w_all, sel_ids)
        floor = max(s1["eps_f"], min(floor_min, TOURN_FLOOR_MIN) if TOURN_FLOOR_MIN < 0.010 else floor_min)
        sel_pass = (s1["mean"] >= floor and s1["W"] >= s1["L"]
                    and s1["flip_lb"] <= 0.50 and s1["R_lb"] <= 0.50)
        conf_pass = None
        if sel_pass:
            s2 = paired_seg(res_a, res_w_all, conf_ids)
            conf_pass = (s2["mean"] > 0 and s2["W"] >= s2["L"]
                         and s2["flip_lb"] <= 0.50 and s2["R_lb"] <= 0.50)
            log(st.run_id, f"tournament: {alt} vs {winner}: SELECT diff={s1['mean']:.4f} "
                           f"floor={floor:.4f} PASS; CONFIRM diff={s2['mean']:.4f} "
                           f"W/L={s2['W']}/{s2['L']} -> "
                           f"{'REPLICATED' if conf_pass else 'NOT REPLICATED (upper tail)'}")
        else:
            log(st.run_id, f"tournament: {alt} vs {winner}: SELECT diff={s1['mean']:.4f} "
                           f"floor={floor:.4f} W/L={s1['W']}/{s1['L']} -> kept {winner}")
        rounds.append({"challenger": alt, "incumbent": winner,
                       "select": s1, "floor": floor, "sel_pass": sel_pass,
                       "conf_pass": conf_pass})
        if sel_pass and conf_pass:
            winner, res_w_all = alt, res_a
            _live(st, "tourn_replacements")
        elif sel_pass and not conf_pass:
            _live(st, "tourn_upper_tail_rejects")
    st.d.setdefault("tournament", {}).update(
        {"champion": champ, "contenders": contenders, "rounds": rounds, "winner": winner})
    return winner


# ---------------------------------------------------------------------------
def train(dataset: str, dry_run: bool = False):
    p = PROFILES[dataset]
    p["impact_quota"] = dict(profiles.EXAM_QUOTA)  # screening-set composition
    subset = execution.load_subset(dataset, config.TRAIN_SPLITS[dataset])
    tasks = execution.tasks_for(dataset, subset, "train")
    tasks_by_id = {t["task_id"]: t for t in tasks}
    train_ids = [t["task_id"] for t in tasks]
    meta = util.read_json(os.path.join(util.WORKDIR, "data", "strata", dataset, "strata_meta_train.json"))
    strata_key = {tid: "||".join(str(meta[tid][f]) for f in subset["strata_fields"])
                  for tid in train_ids}
    all_batches = splits.make_batches(dataset, subset)
    config.check_dataset(dataset)
    n_train = len(train_ids)
    cap = config.BUDGET_MULTIPLIER * n_train
    workers = p["eval_workers"]
    n_confirm = profiles.confirm_n(n_train)
    n_slice = profiles.refine_val_n(n_train)
    reserve = profiles.reserve(n_train)
    floor_min = profiles.tourn_floor(n_train)

    # Dataset configuration from method/configs/<dataset>.json (or the file named by
    # STRATUNE_HP_JSON). The values are recorded in the run configuration.
    global MAX_EPOCHS
    hp = {}
    hp_path = os.environ.get("STRATUNE_HP_JSON") or os.path.join(
        WORKDIR, "method", "configs", f"{dataset}.json")
    if os.path.exists(hp_path):
        hp = util.read_json(hp_path)
        if "max_epochs" in hp:
            MAX_EPOCHS = hp["max_epochs"]
        if "exam_regression" in hp:
            p["impact_quota"]["regression"] = hp["exam_regression"]
        if "exam_random" in hp:
            p["impact_quota"]["random"] = hp["exam_random"]
        if "n_confirm" in hp:
            n_confirm = hp["n_confirm"]
        if "refine_val_n" in hp:
            n_slice = hp["refine_val_n"]
        if "w_size" in hp:
            iterative_refinement.W_SIZE = hp["w_size"]
        if "reserve" in hp:
            reserve = hp["reserve"]
        if "eval_workers" in hp:
            workers = p["eval_workers"] = hp["eval_workers"]
        if "archive_cap" in hp:
            global ARCHIVE_CAP
            ARCHIVE_CAP = hp["archive_cap"]
        if "tourn_seats" in hp:
            global TOURN_SEATS
            TOURN_SEATS = hp["tourn_seats"]
        if "budget" in hp:
            cap = hp["budget"]
        if "i3_k" in hp:
            selection.I3_K = int(hp["i3_k"])
        if "n_cands" in hp:
            direct_revision.N_CANDS = int(hp["n_cands"])
        if hp.get("global_stats"):
            direct_revision.GLOBAL_STATS = True
        if "confirm_floor" in hp:
            global CONFIRM_FLOOR
            CONFIRM_FLOOR = float(hp["confirm_floor"])
        if "tourn_floor_min" in hp:
            global TOURN_FLOOR_MIN
            TOURN_FLOOR_MIN = float(hp["tourn_floor_min"])

    seed_path = os.path.join(WORKDIR, "data", "initial_skills", f"{dataset}.txt")
    with open(seed_path) as f:
        seed_text = f.read()

    run_config = {"T": len(all_batches), "budget": cap, "method_spec": "stratune " + METHOD_VERSION,
              "hp_overrides": hp,
              "exam_quota": profiles.EXAM_QUOTA, "n_confirm": n_confirm,
              "refine_val_n": n_slice, "reserve": reserve,
              "tourn_floor": floor_min, "eps0": p["eps0"],
              "success_threshold": p["success_threshold"]}
    m_prop, m_rw = 100, 300  # budget margins required to start I1 and I2
    small_data = splits.is_sample_data()
    if dry_run or small_data:
        # Reduced set sizes and margins for a shortened run or for the small sample data.
        reserve, n_confirm, n_slice = 40, 20, 16
        m_prop, m_rw = 20, 30
        iterative_refinement.SUBMIT_MARGIN, iterative_refinement.SEAT_MARGIN = 50, 40
        iterative_refinement.W_SIZE, iterative_refinement.SMOKE_N = 3, 4
        selection.I3_K, selection.I3_SMOKE_N, selection.GUARD_MULT = 2, 4, 2
        selection.FLOOR_I2_FRAC, selection.FLOOR_I3_FRAC = 0.25, 0.45
        PROFILES[dataset] = dict(p, impact_quota={"related": 0, "regression": 4, "random": 3})
        p = PROFILES[dataset]
        run_config.update({"sample_data": small_data, "reserve": reserve, "n_confirm": n_confirm,
                           "refine_val_n": n_slice, "exam_quota": p["impact_quota"]})
    if dry_run:
        all_batches = [b[:6] for b in all_batches[:4]]
        cap = 250
        run_config.update({"dry_run": True, "T": 4, "budget": cap})
        in_batches = [t for b in all_batches for t in b]
        extra = [t for t in train_ids if t not in set(in_batches)][:60]
        train_ids = in_batches + extra

    run = lineage.create_run(
        dataset, "stratune", method_version=(METHOD_VERSION if not dry_run else "dryrun"),
        train_subset=subset,
        initial_skill_path=seed_path, config=run_config,
        target_model=TARGET_MODEL, optimizer_model=OPTIMIZER_MODEL,
        scorer_version="stratune-scorers-1.0")
    run_id = run["training_run_id"]
    lineage.update_status(run_id, "running")
    st = RunState(run["heavy_dir"], seed_text)
    st.run_id = run_id
    st.sampling_id = hp.get("sampling_id", run_id)
    st.d["_ds"] = dataset
    st.d["_T"] = len(all_batches)
    budget = BudgetMeter(dataset, cap, st.d)
    executor = EnvExecutor(dataset, work_name=run_id)
    log(run_id, f"start: T={len(all_batches)} cap={cap} n_confirm={n_confirm} "
                f"slice={n_slice} reserve={reserve} floor={floor_min} billed={budget.billed}")

    def _rw_submit(text, rnd, note):
        champ = st.d["champion"]
        vid = st.add_version(text, parent=champ,
                             patch={"operation": "rewrite", "old_content": "",
                                    "new_content": text, "scope": ["rewriter"],
                                    "rationale": note},
                             round_index=rnd, status="rw_candidate")
        uid = f"rw{rnd:02d}_{st.d['versions'][vid]['skill_sha256'][:8]}"
        st.d["versions"][vid]["update_id"] = uid
        st.d["updates"][uid] = {
            "update_id": uid, "candidate": vid, "parent": champ, "form": "rw",
            "cluster": {"description": note,
                        "common_theme": "iterative rewriter window-best"},
            "source_batch": rnd % len(all_batches), "impact_sets": [], "stats": [],
            "readiness_attempts": 0, "status": "pending_impact"}
        _live(st, "rw_submissions")
        out = evaluate_update(dataset, st, executor, uid, train_ids, tasks_by_id,
                              strata_key, budget, workers, rnd, all_batches)
        if not out:
            return None
        _apply_decision(st, uid, out[0], rnd)
        promoted = False
        if out[0] == "commit":
            promoted = confirm_commit(dataset, st, executor, uid, rnd, train_ids,
                                      tasks_by_id, budget, workers, all_batches,
                                      n_confirm)
            if promoted:
                _live(st, "rw_commits")
        elif out[0] in ("reject", "invalid"):
            _postmortem(st, dataset, uid, out[1], tasks_by_id)
        return (out[0], out[1], promoted, vid)

    def _rw_submit_e(text, rnd, note):
        """Submit an I2 candidate to candidate evaluation and record the outcome in the strategy history."""
        out = _rw_submit(text, rnd, note)
        if out:
            selection.record_i(st, "I2", out[0], (out[1] or {}).get("G"))
            if out[0] == "commit":
                uid = st.d["versions"][out[3]].get("update_id")
                g_conf = (st.d["updates"].get(uid, {}) or {}).get("confirm_G", 0.0)
                selection.record_i_confirm(st, "I2", bool(out[2]), g_conf)
        return out

    try:
        # ---- warm-up: execute the initial skill on batches until enough solved samples are known ----
        if "warmup_batches" not in st.d:
            champ = st.d["champion"]
            need = p["impact_quota"]["regression"]
            w = 0
            while w < min(6, len(all_batches)):
                known_ok = sum(1 for tid, s in st.d["scores"].get(champ, {}).items()
                               if parent_success(dataset, s))
                if known_ok >= need and w >= 1:
                    break
                b_tasks = [tasks_by_id[t] for t in all_batches[w]]
                res = run_version_on(executor, b_tasks, st.skill_text(champ),
                                     workers, st.known_records(champ, dataset), budget)
                st.record_scores(dataset, champ, res)
                w += 1
            st.d["warmup_batches"] = w
            st.d["next_batch"] = max(st.d["next_batch"], 1)
            known_ok = sum(1 for tid, s in st.d["scores"].get(champ, {}).items()
                           if parent_success(dataset, s))
            log(run_id, f"warm-up: {w} batches, known successes={known_ok} "
                        f"(need {need}), billed={budget.billed}")
            st.save()

        T = len(all_batches)
        total_rounds = T * MAX_EPOCHS
        epoch_commits = st.d.setdefault("epoch_commits", {})
        while st.d["next_batch"] < total_rounds:
            rnd = st.d["next_batch"]
            epoch = rnd // T
            st.d.setdefault("epoch_champs", {}).setdefault(str(epoch), st.d["champion"])
            # Early stopping: no candidate was accepted in the previous epoch and I2 was idle.
            if rnd % T == 0 and epoch > 0:
                prev = str(epoch - 1)
                rw_last = st.d.get("rw", {}).get("last_accept_round", -1)
                if (epoch_commits.get(prev, 0) == 0 and epoch >= 2
                        and rw_last < (epoch - 1) * T):
                    log(run_id, f"epoch {epoch-1} dry (0 commits, rewriter idle); stopping")
                    break
            if (budget.cap - budget.billed) < reserve:
                log(run_id, f"reserving {budget.cap - budget.billed} evals; stopping schedule")
                break
            batch_ids = [str(t) for t in all_batches[rnd % T]]
            if budget.exhausted(margin=len(batch_ids)):
                log(run_id, f"budget cap approaching ({budget.billed}/{cap}); stopping")
                break
            champ = st.d["champion"]
            batch_tasks = [tasks_by_id[t] for t in batch_ids]
            champ_res = run_version_on(executor, batch_tasks, st.skill_text(champ),
                                       workers, st.known_records(champ, dataset), budget)
            st.record_scores(dataset, champ, champ_res)
            mp = st.mean_primary(champ, batch_ids)
            mp_str = "n/a" if mp is None else f"{mp:.4f}"  # None only in --dry-run
            log(run_id, f"round {rnd}: champion {champ} batch mean={mp_str} "
                        f"billed={budget.billed}/{cap}")

            # ---- An accepted candidate is compared with the skill it replaced on the
            # next two training batches and withdrawn if it loses clearly. ----
            prov = st.d.get("provisional")
            if prov and prov["version"] == champ:
                old = prov["old_champion"]
                old_res = run_version_on(executor, batch_tasks, st.skill_text(old),
                                         workers, st.known_records(old, dataset), budget)
                st.record_scores(dataset, old, old_res)
                for tid in batch_ids:
                    a, b = old_res.get(tid, {}), champ_res.get(tid, {})
                    if "primary" in a and "primary" in b:
                        prov["pairs"][tid] = [float(a["primary"]), float(b["primary"])]
                prov["rounds"] += 1
                import statistics
                d = [nb - ob for ob, nb in prov["pairs"].values()]
                nz = [abs(x) for x in d if abs(x) > 1e-12]
                eps_m = max(p["eps0"], (statistics.median(nz) / len(d)) if nz and d else p["eps0"])
                mean_gain = sum(d) / len(d) if d else 0.0
                old_ok = [tid for tid, (ob, nb) in prov["pairs"].items()
                          if parent_success(dataset, ob)]
                sev = sum(1 for tid in old_ok
                          if not parent_success(dataset, prov["pairs"][tid][1])
                          or prov["pairs"][tid][0] - prov["pairs"][tid][1] >= 0.10)
                ups = sum(1 for tid, (ob, nb) in prov["pairs"].items()
                          if profiles.severe_improvement(dataset, ob, nb))
                sev_rate = (evaluation.wilson_lb(sev, sev + ups) if sev + ups else 0.0)
                if mean_gain < -eps_m or sev_rate > 0.65:
                    log(run_id, f"ROLLBACK provisional {champ} -> {old} "
                                f"(gain={mean_gain:.4f}, sev={sev_rate:.3f})")
                    st.d["champion"] = old
                    st.d["versions"][champ]["status"] = "rolled_back"
                    if champ in st.d["archive"]:
                        st.d["archive"].remove(champ)
                    st.d["provisional"] = None
                    _live(st, "window_rollbacks")
                elif prov["rounds"] >= 2:
                    log(run_id, f"provisional {champ} CONFIRMED (gain={mean_gain:.4f})")
                    st.d["versions"][champ]["status"] = "champion"
                    st.d["provisional"] = None

            # ---- Candidates whose screening was inconclusive are screened again ----
            for uid in [u for u, r in st.d["updates"].items()
                        if r["status"] == "pending_impact"]:
                out = evaluate_update(dataset, st, executor, uid, train_ids,
                                      tasks_by_id, strata_key, budget, workers,
                                      rnd, all_batches)
                if out:
                    _apply_decision(st, uid, out[0], rnd)
                    urec = st.d["updates"][uid]
                    if urec.get("form") in state.FORMS:
                        state.record_trial(st, urec["form"], uid, out[0],
                                            (out[1] or {}).get("G"))
                    if out[0] == "commit":
                        ok = confirm_commit(dataset, st, executor, uid, rnd,
                                            train_ids, tasks_by_id, budget,
                                            workers, all_batches, n_confirm)
                        if urec.get("form") in state.FORMS:
                            state.record_confirm(st, urec["form"], uid, ok,
                                                  urec.get("confirm_G", 0.0))
                    elif out[0] in ("reject", "invalid"):
                        _postmortem(st, dataset, uid, out[1], tasks_by_id)

            # ---- One round: execution feedback -> strategy selection (I1, I2, or I3)
            # -> candidate generation under that strategy -> candidate evaluation. ----
            still_pending = any(r["status"] == "pending_impact"
                                for r in st.d["updates"].values())
            if rnd >= 1 and not still_pending and not budget.exhausted(margin=m_prop):
                ev = direct_revision.build_evidence(
                    dataset, st, executor, batch_ids, champ_res, tasks_by_id,
                    budget, workers, rnd, epoch, lambda m: log(run_id, m))

                def _submit_cands(cands, ev_, itag, _rnd=rnd):
                    shared_imp = None
                    parent_at_call = st.d["champion"]
                    for ci, c in enumerate(cands):
                        if budget.exhausted(margin=100):
                            break
                        if st.d["champion"] != parent_at_call:
                            log(run_id, f"round {_rnd}: champion moved; skipping "
                                        f"remaining {len(cands)-ci} stale candidates")
                            break
                        patch = c["patch"]
                        src_ids = sorted(set(c["source_task_ids"])
                                         | set(ev_["offered_gold_ids"] if c["form"] == "b" else []))
                        patch["source_task_ids"] = src_ids
                        patch["update_id"] = None
                        try:
                            cand_vid = st.add_version(c["cand_text"], parent=parent_at_call,
                                                      patch=patch, round_index=_rnd,
                                                      status="probation")
                        except Exception as e:  # noqa: BLE001
                            log(run_id, f"round {_rnd}: candidate versioning failed ({e})")
                            continue
                        uid = f"p{_rnd:02d}{c['form']}_{st.d['versions'][cand_vid]['skill_sha256'][:8]}"
                        st.d["versions"][cand_vid]["update_id"] = uid
                        st.d["updates"][uid] = {
                            "update_id": uid, "candidate": cand_vid,
                            "parent": parent_at_call, "form": c["form"],
                            "theme": c["theme"], "rationale": c["rationale"],
                            "cluster": {"description": c["theme"],
                                        "common_theme": c["theme"],
                                        "task_ids": ev_["lesson_ids"]},
                            "source_batch": _rnd % T, "impact_sets": [],
                            "stats": [], "readiness_attempts": 0,
                            "status": "pending_impact"}
                        _live(st, f"form_{c['form']}_tries")
                        out = evaluate_update(dataset, st, executor, uid, train_ids,
                                              tasks_by_id, strata_key, budget,
                                              workers, _rnd, all_batches,
                                              shared_impact=shared_imp)
                        if not out:
                            continue
                        if shared_imp is None and len(out) > 2 and not out[2].get("shared"):
                            shared_imp = out[2]
                        _apply_decision(st, uid, out[0], _rnd)
                        state.record_trial(st, c["form"], uid, out[0],
                                            (out[1] or {}).get("G"))
                        selection.record_i(st, itag, out[0], (out[1] or {}).get("G"))
                        if out[0] == "commit":
                            _live(st, f"form_{c['form']}_commits")
                            ok = confirm_commit(dataset, st, executor, uid, _rnd,
                                                train_ids, tasks_by_id, budget,
                                                workers, all_batches, n_confirm)
                            state.record_confirm(st, c["form"], uid, ok,
                                                  st.d["updates"][uid].get("confirm_G", 0.0))
                            selection.record_i_confirm(st, itag, ok,
                                                     st.d["updates"][uid].get("confirm_G", 0.0))
                        elif out[0] in ("reject", "invalid"):
                            _postmortem(st, dataset, uid, out[1], tasks_by_id)

                def _rw_step(_rnd=rnd, _bi=batch_ids, _cr=champ_res):
                    if budget.exhausted(margin=m_rw):
                        return
                    iterative_refinement.rewriter_round(dataset, st, executor, _bi, _cr,
                                            tasks_by_id, train_ids, budget, workers,
                                            _rnd, n_slice, _rw_submit_e,
                                            lambda m: log(run_id, m))

                selection.run_round(dataset, st, ev, {
                    "batch_ids": batch_ids, "tasks_by_id": tasks_by_id,
                    "train_ids": train_ids, "executor": executor,
                    "budget": budget, "workers": workers, "rnd": rnd,
                    "epoch": epoch, "T": T, "n_slice": n_slice,
                    "m_prop": m_prop, "m_rw": m_rw,
                    "submit_cands": _submit_cands, "rw_step": _rw_step,
                }, lambda m: log(run_id, m))

            _prune_archive(st)
            st.d["next_batch"] = rnd + 1
            st.d["round_log"].append({"round": rnd, "champion": st.d["champion"],
                                      "billed": budget.billed})
            st.save()

        # ---- Submit the best unsubmitted I2 candidate, if any ----
        if not budget.exhausted(margin=iterative_refinement.SEAT_MARGIN):
            iterative_refinement.flush_window(st, budget, st.d["next_batch"], _rw_submit_e,
                                  lambda m: log(run_id, m))
            st.save()

        # ---- Pending candidates with a positive gain that pass the regression
        # checks are saved for final skill selection ----
        from method.evaluation import flip_balance_lb
        from method.evaluation import r_lb
        for uid, r in list(st.d["updates"].items()):
            if r.get("status") != "pending_impact" or not r.get("stats"):
                continue
            s_last = r["stats"][-1]
            if s_last.get("smoke"):
                continue
            if (float(s_last.get("G") or 0) > 0
                    and s_last.get("W", 0) >= s_last.get("L", 0)
                    and flip_balance_lb(s_last) <= 0.50
                    and r_lb(s_last) <= 0.50
                    and len(s_last.get("guard_severe_ids") or []) <= 1):
                _apply_decision(st, uid, "branch", st.d["next_batch"])
                log(run_id, f"pending rescue: {uid} archived for tournament "
                            f"(G={float(s_last.get('G') or 0):+.4f})")
        st.save()

        # ---- Merge content from saved skills, then final skill selection ----
        if not budget.exhausted(margin=m_rw):
            _donor_salvage(dataset, st, executor, tasks_by_id, train_ids,
                           strata_key, budget, workers, st.d["next_batch"],
                           all_batches, n_confirm)
        final_vid = _final_tournament(dataset, st, executor, tasks_by_id,
                                      train_ids, budget, workers, floor_min)
        final_path = os.path.join(run["heavy_dir"], "final_skill.txt")
        with open(final_path, "w") as f:
            f.write(st.skill_text(final_vid))
        st.d["final_version"] = final_vid
        rw_c = st.d.get("rw", {}).get("counters", {})
        log(run_id, f"liveness: {st.d.get('liveness', {})} | rw={rw_c} | "
                    f"form_mem={ {k: (v['tries'], v['adopted'], v['confirmed']) for k, v in st.d.get('form_mem', {}).items()} }")
        st.save()
        lineage.complete_run(run_id, final_path)
        log(run_id, f"COMPLETED: final={final_vid} billed={budget.billed}/{cap}")
        return run_id
    except BaseException:
        st.save()
        lineage.update_status(run_id, "failed")
        raise


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=list(config.TRAIN_SPLITS))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    train(args.dataset, dry_run=args.dry_run)
