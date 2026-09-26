"""The evaluation history H_t and refinement progress p_t.

Form history: outcomes of candidate evaluation grouped by revision form.
Strategy history: outcomes grouped by search strategy (I1-I3) and the I2 refinement journal.
Past revision cases shown to the optimizer LLM, and the decorators that record
every screening and validation outcome in these histories.
"""
from collections import Counter
import copy
import difflib
import functools
import hashlib
import inspect
import json
import os


# ---------------------------------------------------------------------------
# Form history: pooled per-form outcomes of candidate evaluation, part of the
# evaluation history H_t shown to the optimizer LLM for strategy selection.
#
# The rendered history contains only pooled per-form tallies, confirmed gain
# magnitudes, and records of rejected candidates' regressions with representative
# sample text. It never contains per-sample screening results or acceptance
# thresholds.
# Descriptions of the revision forms shown to the optimizer LLM.
FORMS = {
    "a": """LOCAL RULE EDITS (SkillOpt / Trace2Skill / ExpeL lineage) — add or escalate
   conditional rules. CRAFT (these forms measurably beat abstract advice):
   - express rules as concrete WRONG -> RIGHT contrast pairs generalized from the
     failures (never memorize a task answer);
   - state the ANSWER SHAPE for each input kind (bare value vs sentence, units,
     capitalization, verbatim copying), with a one-line reason;
   - write "When <observable condition in the input or your draft answer>, do
     <specific action>" — never "be careful about X";
   - if the same mistake keeps recurring, ESCALATE: make the rule a MANDATORY
     first-step check with an explicit default direction and a short veto list of
     reasoning phrases that signal the mistake;
   - if failures reveal input artifacts (truncated options, ambiguous separators,
     OCR confusions), add a short handling protocol for the artifact itself;
   - any direction-pushing rule (shorter/longer, act/abstain) MUST state its
     boundary condition and protect the given contrast cases.""",
    "b": """WORKED-EXAMPLES SECTION (Synapse / bootstrapped-demos lineage) — verbatim
   solved (input -> gold) pairs. CRAFT: choose a COVERING subset across distinct
   input kinds and answer shapes; copy gold answers CHARACTER-FOR-CHARACTER; one
   line per example; replace the whole section if it already exists. Known
   failure: examples can induce off-input pattern matching — prefer examples
   whose answer is derivable from the shown input.""",
    "c": """MULTI-ROUTE DERIVATION PROTOCOL (internalized self-consistency) — a short
   MECHANICAL procedure (<= 15 steps) the agent runs inside one response.
   CRAFT — adapt one of these measured patterns to the task shape:
   - CHOICE tasks: derive the decision procedure FROM the failure evidence —
     compare what the GOLD choices' texts have in common versus the predicted
     ones (option families, hedged/collective options, strength ordering) and
     write a mechanical rule for exactly that observed pattern, including an
     explicit DEFAULT when derivation is inconclusive; do NOT prescribe generic
     exhaustive option-checking unless the evidence shows that failure shape;
   - EXTRACTION tasks: derive the answer twice by INDEPENDENT routes (direct
     read vs re-locate-then-reread character-by-character); if they disagree,
     re-read once more and take the char-exact one;
   - SHORT-ANSWER tasks: ask "would this exact string appear in an answer key?"
     — strip anything that would not;
   The protocol must NOT touch the output-format instructions; end it with a
   mandatory 3-6 item final checklist.""",
    "d": """WORKFLOW / SUBROUTINE SECTION (AWM / Trace2Skill lineage) — mined from the
   real solved traces given. CRAFT — three-part structure, all grounded in the
   traces: (1) 3-5 INTENT -> FIRST-ACTION rules (map task phrasing to the correct
   opening move, including what NOT to do first); (2) 3-8 NAMED step flows quoting
   EXACT literal anchors from the traces — literal anchors are what transfers,
   do NOT abstract or genericize them; (3) 2-4 implicit-constraint rules
   (orderings/filters the task text implies without saying). Consolidate, keep lean.""",
    "e": """ONE-SHOT FULL REWRITE (GEPA lineage) — an integrated reorganization.
   CRAFT: restructure toward an INPUT-TYPE TAXONOMY (2-4 kinds among the observed
   failures, each with recognition cue -> procedure -> answer shape); keep every
   rule that plausibly drives current successes; add a MANDATORY 3-6 item final
   checklist; state one explicit DEFAULT DIRECTION with a veto list for the
   dominant failure; PRESERVE the output-format specification character-for-character.
   Known failure: silently dropping unrelated correct rules.""",
}


def _mem(st):
    return st.d.setdefault("form_mem", {})


def record_trial(st, form: str, uid: str, decision: str, G: float | None):
    m = _mem(st).setdefault(form, {"tries": 0, "adopted": 0, "confirmed": 0,
                                   "g_sum": 0.0, "g_n": 0, "confirmed_gains": []})
    m["tries"] += 1
    if G is not None:
        m["g_sum"] += float(G)
        m["g_n"] += 1
    if decision == "commit":
        m["adopted"] += 1


def record_confirm(st, form: str, uid: str, confirmed: bool, g_confirm: float):
    m = _mem(st).setdefault(form, {"tries": 0, "adopted": 0, "confirmed": 0,
                                   "g_sum": 0.0, "g_n": 0, "confirmed_gains": []})
    if confirmed:
        m["confirmed"] += 1
        m["confirmed_gains"].append(round(float(g_confirm), 4))
        del m["confirmed_gains"][:-6]
    else:
        m["adopted"] = max(0, m["adopted"] - 1)  # accepted candidate withdrawn


def record_postmortem(st, form: str, theme: str, gist: str, stats: dict, examples: str):
    entry = (f"- [form {form}] rejected update on theme \"{theme}\" "
             f"(paired mean {float(stats.get('G') or 0):+.3f}, "
             f"{stats.get('W', 0)} wins / {stats.get('L', 0)} losses): \"{gist}\" — "
             f"it REGRESSED tasks such as: {examples or '(guard tasks)'}")
    pms = st.d.setdefault("postmortems", [])
    pms.append(entry)
    del pms[:-6]


def render_form_history(st) -> str:
    """Pooled per-form outcomes and recent records of rejected candidates'
    regressions, followed by the strategy history and past revision cases."""
    mem = _mem(st)
    lines = ["I1/I3 FORM STATISTICS (formal candidate evaluations only; excludes I2 internal attempts):"]
    for f in sorted(FORMS):
        m = mem.get(f)
        if not m or not m["tries"]:
            lines.append(f"  {f}: no I1/I3 candidate evaluations recorded")
            continue
        mean_g = (m["g_sum"] / m["g_n"]) if m["g_n"] else 0.0
        conf = (f", confirmed gains {m['confirmed_gains']}"
                if m.get("confirmed_gains") else "")
        lines.append(f"  {f}: {m['tries']} tried, {m['adopted']} adopted, "
                     f"{m['confirmed']} survived confirmation, "
                     f"mean paired gain {mean_g:+.3f}{conf}")
    out = "\n".join(lines)
    pms = st.d.get("postmortems", [])
    if pms:
        out += "\n\nRECENT POSTMORTEMS (what rejected updates regressed):\n" + "\n".join(pms)
    out += "\n\n" + render_strategy_history(st)
    out += render_cases(st)
    return out


# ---------------------------------------------------------------------------
# Strategy history: per-attempt records of Iterative Refinement (I2), part of the
# evaluation history H_t shown to the optimizer LLM. Attempt-level records are kept
# separate from the evaluation outcomes of whole refinement paths.
I2_FORMS = ("F1", "F2", "F3", "F4")
LABELS = {"a": "F1", "b": "F2", "c": "F3", "d": "F3", "e": "F4", "rw": "F4"}
NATIVE = {"F1": "a", "F2": "b", "F3": "c", "F4": "e"}


def canonical(form):
    value = str(form or "").strip()
    return LABELS.get(value.lower(), value.upper())


def journal(st, event, **data):
    row = {"event": event, **copy.deepcopy(data)}
    if getattr(st, "heavy_dir", None):
        with open(os.path.join(st.heavy_dir, "adaptive_events.jsonl"), "a") as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def begin(st, form, rnd, base_vid, base_sha, batch_ids):
    book = st.d.setdefault("i2_attempts", {})
    key = f"i2_{len(book):05d}"
    entry = {
        "attempt_id": key, "round": rnd, "form": canonical(form),
        "generation_base": base_vid, "generation_base_sha256": base_sha,
        "feedback_skill_sha256": base_sha, "feedback_task_ids": list(batch_ids),
        "mechanically_valid": None, "source_valid": None,
        "valid_generation": False, "internal": None, "base_advanced": False,
        "submissions": {},
    }
    book[key] = entry
    journal(st, "i2_attempt", **entry)
    return entry


def update(st, entry, **fields):
    entry.update(copy.deepcopy(fields))
    journal(st, "i2_attempt_updated", attempt_id=entry["attempt_id"], **fields)


def submitted(st, uid, origin):
    ids = origin.get("attempt_path") or []
    if not ids:
        return
    # The evaluation outcome of a refinement path is attached to its last attempt;
    # it is not a gain estimate for that form alone.
    last = ids[-1]
    entry = st.d.get("i2_attempts", {}).get(last)
    if entry is not None:
        entry["submissions"].setdefault(uid, {
            "attempt_path": list(ids), "screening": None, "confirmation": None})
        journal(st, "i2_submitted", attempt_id=last, uid=uid, attempt_path=ids)


def outcome(st, uid, stage, value):
    for entry in st.d.get("i2_attempts", {}).values():
        sub = entry["submissions"].get(uid)
        if sub is not None:
            sub[stage] = copy.deepcopy(value)
            journal(st, "i2_outcome", attempt_id=entry["attempt_id"], uid=uid,
                    stage=stage, value=value)


def render_strategy_history(st):
    entries = list(st.d.get("i2_attempts", {}).values())
    lines = [
        "I2 FORM HISTORY (attempts include invalid drafts and internally rejected drafts):",
        "Internal gains compare a draft with its exact generation base on common samples.",
        "Submission/acceptance outcomes concern whole paths, not isolated form effects.",
    ]
    for form in I2_FORMS:
        rows = [e for e in entries if e["form"] == form]
        internal = [e["internal"] for e in rows
                    if e.get("internal") and e["internal"].get("stage") == "slice"]
        gains = [v["mean_gain"] for v in internal if v.get("mean_gain") is not None]
        submissions = [s for e in rows for s in e["submissions"].values()]
        screen = Counter(s["screening"]["decision"] for s in submissions if s.get("screening"))
        accepted = sum(bool(s.get("confirmation", {}).get("passed"))
                       for s in submissions if s.get("confirmation") is not None)
        failed = Counter(e.get("failure_stage") for e in rows if e.get("failure_stage"))
        lines.append(
            f"  {form}: attempts={len(rows)}, valid_generation={sum(e['valid_generation'] for e in rows)}, "
            f"internal_smoke={sum(bool(e.get('smoke')) for e in rows)}, internal_slice={len(internal)}, "
            f"base_advances={sum(e['base_advanced'] for e in rows)}, submissions={len(submissions)}, "
            f"screening={dict(screen)}, accepted_after_validation={accepted}, failures={dict(failed)}; "
            f"mean_internal_gain={sum(gains)/len(gains):+.4f}" if gains else
            f"  {form}: attempts={len(rows)}, valid_generation={sum(e['valid_generation'] for e in rows)}, "
            f"internal_smoke={sum(bool(e.get('smoke')) for e in rows)}, internal_slice={len(internal)}, "
            f"base_advances={sum(e['base_advanced'] for e in rows)}, submissions={len(submissions)}, "
            f"screening={dict(screen)}, accepted_after_validation={accepted}, failures={dict(failed)}; "
            "mean_internal_gain=unavailable")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Past revision cases: a bounded set of candidate evaluations per strategy and
# revision form, drawn from training comparisons only; part of the evaluation
# history H_t shown to the optimizer LLM.
FORM_LABELS = {"a": "F1", "b": "F2", "c": "F3", "d": "F3", "e": "F4", "rw": "F4"}
CASE_LIMIT = 6


def short(value, limit=180):
    return " ".join(str(value or "").split())[:limit]


def change_excerpt(parent, candidate, limit=280):
    before, after = parent.splitlines(), candidate.splitlines()
    chunks = []
    for kind, a, b, c, d in difflib.SequenceMatcher(a=before, b=after, autojunk=False).get_opcodes():
        if kind != "equal":
            chunks.append(" / ".join(after[c:d]) if c != d else "Removed: " + " / ".join(before[a:b]))
    return short(" | ".join(chunks), limit)


def context_summary(ev):
    return {
        "failure_examples": [short(line, 160) for line in ev.get("fail_lines", [])[:2]],
        "failure_count": ev.get("n_fail"),
        "batch_size": ev.get("n_batch"),
    }


def outcome_samples(ds, stats, tasks_by_id):
    """Improved and regressed samples from the observed per-sample changes;
    execution failures are excluded."""
    from method.environments import dataset_profiles as profiles
    excluded = set(stats.get("cand_exec_fail_ids", []))
    deltas = {str(k): float(v) for k, v in stats.get("per_task_delta", {}).items()
              if str(k) in tasks_by_id and str(k) not in excluded}
    positive = sorted((k for k, v in deltas.items() if v > 1e-12),
                      key=lambda k: (-deltas[k], k))
    negative = sorted((k for k, v in deltas.items() if v < -1e-12),
                      key=lambda k: (deltas[k], k))
    # A preliminary check that stops early reports only severe regressions;
    # do not fabricate per-sample changes.
    if not negative and "per_task_delta" not in stats:
        negative = [str(k) for k in stats.get("severe_ids", [])
                    if str(k) in tasks_by_id and str(k) not in excluded]

    def render(ids):
        return [{"task_id": k, "input": short(profiles.task_text(ds, tasks_by_id[k]), 150)}
                for k in ids[:2]]

    return {"improved": render(positive), "regressed": render(negative),
            "execution_failure_count": len(excluded)}


def record_screening(st, ds, uid, decision, stats, tasks_by_id, origin):
    if origin.get("strategy") not in ("I1", "I2", "I3"):
        return None
    update = st.d["updates"][uid]
    if "per_task_delta" not in stats:
        # After an early stop, severe_ids may name the previously solved samples
        # that were checked rather than the exact failures; recover the actual
        # paired changes from the score ledgers.
        parent = st.d.get("scores", {}).get(update["parent"], {})
        candidate = st.d.get("scores", {}).get(update["candidate"], {})
        stats = dict(stats, per_task_delta={
            str(t): candidate[str(t)] - parent[str(t)]
            for t in stats.get("severe_ids", [])
            if str(t) in candidate and str(t) in parent
        })
    book = st.d.setdefault("adaptive_case_history", {})
    entry = book.setdefault(uid, {
        "uid": uid, "strategy": origin["strategy"],
        "source_round": origin.get("source_round"),
        "origin": copy.deepcopy(origin), "candidate": update["candidate"],
        "comparison_parent": update["parent"],
        "form": origin.get("form", update.get("form")),
        "form_path": copy.deepcopy(origin.get("form_path", [])),
        "form_explicitly_selected": bool(origin.get("form_explicitly_selected")),
        "problem": short(origin.get("theme") or update.get("theme")
                         or update.get("cluster", {}).get("common_theme"), 180),
        "change": change_excerpt(st.skill_text(update["parent"]),
                                 st.skill_text(update["candidate"])),
        "screening_updates": 0,
    })
    entry["screening_updates"] += 1
    # Store one case per candidate, updating it as more evidence arrives.
    entry["screening"] = {
        "decision": decision, "mean_gain": stats.get("G"),
        "wins": stats.get("W"), "losses": stats.get("L"),
        "n_pairs": stats.get("n_pairs"), **outcome_samples(ds, stats, tasks_by_id),
    }
    entry["last_round"] = st.d.get("next_batch", entry["source_round"] or 0)
    entry["result"] = {
        "commit": "awaiting_further_validation", "reject": "rejected",
        "invalid": "invalid", "branch": "saved_without_acceptance",
        "probation": "pending_more_evidence",
        "insufficient_evidence": "insufficient_evidence",
    }.get(decision, decision)
    return entry


def record_confirmation(st, ds, uid, passed, stats, tasks_by_id):
    entry = st.d.get("adaptive_case_history", {}).get(uid)
    if entry is None:
        return None
    entry["confirmation"] = {
        "passed": bool(passed), "mean_gain": stats.get("G"),
        "n_pairs": stats.get("n_pairs"), **outcome_samples(ds, stats, tasks_by_id),
    }
    entry["result"] = "accepted_after_validation" if passed else "validation_failed"
    return entry


def select_cases(st, limit=CASE_LIMIT):
    """At most two cases per strategy; training evidence only."""
    entries = list(st.d.get("adaptive_case_history", {}).values())
    chosen = []
    for strategy in ("I1", "I2", "I3"):
        candidates = sorted(
            [e for e in entries if e["strategy"] == strategy],
            key=lambda e: (e.get("source_round") or 0, e["uid"]), reverse=True)
        successes = [e for e in candidates if e["result"] == "accepted_after_validation"]
        saved = [e for e in candidates
                 if e["result"] == "saved_without_acceptance"
                 and (e.get("screening", {}).get("n_pairs") or 0) > 0
                 and e["screening"].get("mean_gain") is not None]
        failures = [e for e in candidates
                    if e["result"] == "validation_failed"
                    or (e["result"] == "rejected" and e.get("screening", {}).get("regressed"))]
        useful = successes or saved
        if useful:
            chosen.append(useful[0])
        if failures:
            # Favor another theme/form when several recent failures are available.
            if useful:
                first = useful[0]
                different = [e for e in failures[:4]
                             if (e["problem"], e.get("form")) != (first["problem"], first.get("form"))]
                failures = different or failures
            chosen.append(failures[0])
    assert len({e["uid"] for e in chosen}) == len(chosen)
    return chosen[:limit]


def render_cases(st):
    entries = select_cases(st)
    if not entries:
        return ""
    lines = [
        "\n\nPAST REVISIONS (shared strategy/form cases, grouped by strategy):",
        "These are earlier training comparisons, not guaranteed effects on the current skill.",
        "Form labels: a=F1, b=F2, c/d=F3 (reasoning/workflow), e/rw=F4 (full rewrite).",
        "For I2, final outcomes concern the entire refinement path; do not credit only its last form.",
    ]
    for strategy in ("I1", "I2", "I3"):
        group = [e for e in entries if e["strategy"] == strategy]
        if not group:
            continue
        lines.append(f"{strategy} past revision cases:")
        for e in group:
            path = e.get("form_path") or [e.get("form")]
            forms = " -> ".join(
                f"{FORM_LABELS.get(f, f)}[{f}]" for f in path[-6:] if f)
            mode = "selected form" if e.get("form_explicitly_selected") else "fixed rewrite"
            screen, conf = e["screening"], e.get("confirmation")
            observed = conf or screen
            good = "; ".join(x["input"] for x in observed.get("improved", []))
            bad = "; ".join(x["input"] for x in observed.get("regressed", []))
            lines += [
                f"- Round {e.get('source_round')}, {forms} ({mode}). Problem: {e['problem']}.",
                f"  Status: {e['result']} (saved_without_acceptance means saved, not confirmed).",
                f"  Changed instructions (excerpt): {e['change']}",
                f"  Improved cases: {good or 'none recorded in this comparison'}.",
                f"  Regressed cases: {bad or 'none recorded in this comparison'}.",
                f"  Initial screening: {screen['decision']}; mean gain {screen.get('mean_gain')}; "
                f"improvements={screen.get('wins')}, regressions={screen.get('losses')}.",
                (f"  Further validation: {'passed' if conf['passed'] else 'failed'}; "
                 f"mean gain {conf.get('mean_gain')}.") if conf
                else "  Further validation: not completed.",
            ]
    return "\n".join(lines)


def fingerprint(text):
    return hashlib.sha256(text.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Record screening and validation outcomes in the evaluation history.
def origin_for(st, uid):
    update = st.d["updates"][uid]
    if update.get("adaptive_origin"):
        return update["adaptive_origin"]
    origin = getattr(st, "_submission_origin", None)
    if not origin:
        row = (st.d.get("episode_log") or [{}])[-1]
        strategy = row.get("i")
        if strategy not in ("I1", "I3") or update.get("form") not in ("a", "b", "c", "d", "e"):
            return {}
        form = update.get("form")
        origin = {
            "strategy": strategy, "form": form, "form_path": [form],
            "form_explicitly_selected": True, "source_round": row.get("round"),
            "theme": update.get("theme"), **context_summary(getattr(st, "_round_evidence", {})),
        }
    origin = copy.deepcopy(origin)
    update["adaptive_origin"] = origin
    version = st.d["versions"][update["candidate"]]
    version["adaptive_origin"] = copy.deepcopy(origin)
    if origin["strategy"] == "I2":
        patch = version["patch"]
        patch["source_task_ids"] = sorted(set(patch.get("source_task_ids", []))
                                           | set(origin.get("source_task_ids", [])))
        submitted(st, uid, origin)
    return origin


def screening(function):
    signature = inspect.signature(function)

    @functools.wraps(function)
    def wrapped(*args, **kwargs):
        bound = signature.bind(*args, **kwargs).arguments
        st, uid = bound["st"], bound["uid"]
        origin = origin_for(st, uid)
        result = function(*args, **kwargs)
        if result and origin:
            record_screening(st, bound["ds"], uid, result[0], result[1],
                                   bound["tasks_by_id"], origin)
            outcome(st, uid, "screening", {
                "decision": result[0], "mean_gain": result[1].get("G"),
                "wins": result[1].get("W"), "losses": result[1].get("L"),
                "comparison_parent": st.d["updates"][uid]["parent"]})
        return result
    return wrapped


def confirmation(function):
    signature = inspect.signature(function)

    @functools.wraps(function)
    def wrapped(*args, **kwargs):
        bound = signature.bind(*args, **kwargs).arguments
        st, uid = bound["st"], bound["uid"]
        result = function(*args, **kwargs)
        stats = st.d["updates"][uid].get("confirmation_stats", {})
        record_confirmation(st, bound["ds"], uid, result, stats, bound["tasks_by_id"])
        outcome(st, uid, "confirmation", {
            "passed": bool(result), "mean_gain": stats.get("G"), "n_pairs": stats.get("n_pairs")})
        return result
    return wrapped


