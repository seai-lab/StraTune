"""Iterative refinement (I2).

I2 keeps an intermediate skill, initialized from the current skill when
refinement starts. Each step critiques the executions of this round's batch and
applies the selected revision form to the intermediate skill. The resulting
candidate passes a preliminary check on a few samples and is then scored on the
common training subset, which is used only to rank candidates within a window.
After every W_SIZE steps, the two best distinct candidates of the window are
submitted to candidate evaluation through the submit_fn provided by
method.train. An evaluated candidate may become the next intermediate skill even
if it does not replace the current skill, provided its measured gain is not
clearly below that of the previous intermediate skill.
"""
from method.operators import direct_revision
from method.operators import full_rewrite
from method import state
from method.common import skill_edits
from method.common import util
from method.evaluation import run_version_on


W_SIZE = 4                 # refinement steps per window (dataset configuration: w_size)
SMOKE_N = 12               # samples of the preliminary check
SMOKE_KILL_FRAC = 0.25     # a candidate scoring below this fraction of the intermediate skill's score is dropped
SLICE_SUBMIT_SLACK = 0.02  # submit only candidates scoring at least (intermediate skill - this) on the common subset
SUBMIT_MARGIN = 500        # budget margin required to submit a window
SEAT_MARGIN = 400          # budget margin required per submitted candidate


def _gen(st):
    return st.d.setdefault("rw", {"base_vid": None, "val_ids": None,
                                  "val_score": None, "steps": 0, "window": [],
                                  "base_G": None, "last_accept_round": -1,
                                  "counters": {"drafts": 0, "smoke_killed": 0,
                                               "submitted": 0, "base_adopted": 0,
                                               "committed": 0}})


def rewriter_round(ds, st, executor, batch_ids, batch_res, tasks_by_id,
                   train_ids, budget, workers, rnd, refine_val_n, submit_fn, log):
    """One refinement step. submit_fn(text, rnd, note) -> None or
    (decision, stats, promoted: bool, vid: str); candidate evaluation is owned by
    method.train. Updates the refinement state in st.d['rw']."""
    gen = _gen(st)
    champ = st.d["champion"]
    base_vid = gen["base_vid"] or champ
    base_text = st.skill_text(base_vid)
    base_sha = util.sha256_str(base_text)

    # common training subset for ranking, fixed once per run and disjoint from this batch
    if gen["val_ids"] is None:
        used = set(map(str, batch_ids))
        full_rewrite.REFINE_VAL_N = refine_val_n
        gen["val_ids"] = full_rewrite.refine_val_ids(train_ids, used, f"{st.sampling_id}|rw")
        gen["val_score"] = None
    val_tasks = [tasks_by_id[t] for t in gen["val_ids"]]

    batch_tasks = [tasks_by_id[t] for t in batch_ids]
    run_salt = f"|{st.sampling_id[-6:]}|s{gen['steps']}"  # a new generation sample per step
    form = getattr(st, "_i2_form", "F4")
    ev = getattr(st, "_round_evidence", {})
    n_crit = 3 + gen["steps"] % 3
    actual_res = state.records_for_feedback(
        ds, st, base_vid, batch_tasks, executor, workers, budget, n_crit)
    entry = state.begin(st, form, rnd, base_vid, base_sha, list(actual_res))
    info = {}
    new_text = full_rewrite.refine_step(ds, base_text, batch_tasks[:n_crit], actual_res,
                              variant=gen["steps"], run_salt=run_salt,
                              form=form, ev=ev, history=state.render_form_history(st), generation=info)
    gen["counters"]["drafts"] += 1
    state.update(st, entry, **info, valid_generation=bool(new_text))
    if not new_text:
        gen["steps"] += 1
        return

    base_score, res_a, ref_key = state.reference(
        ds, st, base_vid, gen["val_ids"], tasks_by_id, executor, workers, budget)
    gen.update(val_score=base_score, val_reference=ref_key)
    previous = st.d["versions"].get(gen["base_vid"], {}).get("adaptive_origin", {})
    source_ids = (set(previous.get("source_task_ids", [])) | set(actual_res)
                  | set(info.get("source_task_ids", [])))
    if form != "F4":
        source_ids |= set(map(str, ev.get("lesson_ids", [])))
        source_ids |= set(map(str, ev.get("offered_gold_ids", [])))
    origin = {
        "strategy": "I2", "form": form, "form_explicitly_selected": True,
        "form_path": previous.get("form_path", []) + [form],
        "attempt_path": previous.get("attempt_path", []) + [entry["attempt_id"]],
        "source_task_ids": sorted(source_ids), "source_round": rnd,
        "generation_base": base_vid, "feedback_skill": base_vid,
        "generation_base_sha256": base_sha,
        "theme": info.get("theme") or "Refine the intermediate skill",
        **state.context_summary(ev),
    }

    # Preliminary check on a few samples; a clearly broken candidate is dropped before the full subset is executed.
    known_smoke = {}
    if gen["val_score"] and gen["val_score"] > 0.05:
        smoke_tasks = val_tasks[:SMOKE_N]
        res_s = run_version_on(executor, smoke_tasks, new_text, workers, {}, budget)
        s_mean = (sum(float(r.get("primary", 0)) for r in res_s.values())
                  / max(1, len(smoke_tasks)))
        smoke = state.comparison([t["task_id"] for t in smoke_tasks], res_a, res_s, "smoke")
        state.update(st, entry, smoke=smoke)
        if s_mean < SMOKE_KILL_FRAC * gen["val_score"]:
            gen["steps"] += 1
            gen["counters"]["smoke_killed"] += 1
            state.update(st, entry, internal=smoke, failure_stage="internal_smoke")
            log(f"rw step {gen['steps']}: SMOKE-KILLED draft "
                f"({SMOKE_N}-task mean {s_mean:.3f} vs base {gen['val_score']:.3f})")
            return
        known_smoke = res_s

    res_n = run_version_on(executor, val_tasks, new_text, workers, known_smoke, budget)
    new_score = (sum(float(r.get("primary", 0)) for r in res_n.values())
                 / max(1, len(val_tasks)))
    internal = state.comparison(gen["val_ids"], res_a, res_n, "slice")
    internal.update(reference_key=ref_key, base_score=base_score, candidate_score=new_score,
                    generation_base_sha256=base_sha)
    state.update(st, entry, internal=internal, candidate_sha256=util.sha256_str(new_text))
    gen["steps"] += 1
    w = gen["window"]
    w.append({"score": new_score, "vtext": new_text, "base_score": base_score,
              "reference_key": ref_key, "adaptive_origin": origin})
    log(f"rw step {gen['steps']}: slice={new_score:.4f} (window {len(w)}/{W_SIZE})")
    if len(w) < W_SIZE or budget.exhausted(margin=SUBMIT_MARGIN):
        return
    _submit_window(gen, st, budget, rnd, submit_fn, log, seats=2)


def flush_window(st, budget, rnd, submit_fn, log):
    """At the end of training, a partial window with at least two candidates
    submits its best one to candidate evaluation."""
    gen = _gen(st)
    if len(gen["window"]) < 2 or budget.exhausted(margin=SEAT_MARGIN):
        return
    log(f"rw flush: submitting top-1 of partial window ({len(gen['window'])} drafts)")
    _submit_window(gen, st, budget, rnd, submit_fn, log, seats=1)


def _submit_window(gen, st, budget, rnd, submit_fn, log, seats):
    ranked = sorted(gen["window"], key=lambda x: -x["score"])
    gen["window"] = []
    seen_sha = set()
    for best in ranked[:seats]:
        if budget.exhausted(margin=SEAT_MARGIN):
            break
        bsha = util.sha256_str(best["vtext"])[:16]
        if bsha in seen_sha:
            continue
        seen_sha.add(bsha)
        reference_score = best.get("base_score", gen["val_score"])
        if best["score"] < reference_score - SLICE_SUBMIT_SLACK:
            log(f"rw window: best {best['score']:.4f} catastrophic vs "
                f"{reference_score:.4f}; discarded")
            path = best.get("adaptive_origin", {}).get("attempt_path", [])
            if path:
                state.update(st, st.d["i2_attempts"][path[-1]], failure_stage="window_cutoff")
            continue
        gen["counters"]["submitted"] += 1
        st._submission_origin = best.get("adaptive_origin", {})
        try:
            out = submit_fn(best["vtext"], rnd,
                            f"rewriter window-best slice={best['score']:.4f}")
        finally:
            st._submission_origin = None
        gen["last_accept_round"] = rnd  # a submission counts as I2 activity for early stopping
        if not out:
            continue
        decision, stats, promoted, sub_vid = out
        if promoted:
            gen["counters"]["committed"] += 1
            gen["val_score"] = None
            gen["base_vid"] = None
            gen["base_G"] = None
            gen["val_reference"] = None
            log(f"rw window: draft PROMOTED at round {rnd}")
            break  # the current skill changed; the remaining candidates revise a stale skill
        if stats:
            g_meas = float(stats["G"]) if stats.get("G") is not None else -1.0
            band = max(0.010, 1.5 * float(stats.get("eps") or 0.0))
            if gen.get("base_G_parent") != st.d["champion"]:
                gen["base_G"] = None
                gen["base_G_parent"] = st.d["champion"]
            prev = gen.get("base_G")
            tol = -band if prev is None else max(-band, prev + 1e-9)
            if g_meas >= tol:
                # the submitted candidate becomes the next intermediate skill
                gen["base_vid"] = sub_vid
                gen["base_G"] = g_meas
                gen["val_score"] = None
                gen["val_reference"] = None
                gen["counters"]["base_adopted"] += 1
                path = best.get("adaptive_origin", {}).get("attempt_path", [])
                if path:
                    state.update(st, st.d["i2_attempts"][path[-1]], base_advanced=True)
                log(f"rw window: base ADOPTED (impact G={g_meas:+.4f})")
            else:
                log(f"rw window: base kept (impact G={g_meas:+.4f} below ratchet)")


# ---------------------------------------------------------------------------
# Apply one selected Iterative Refinement (I2) revision form; F4 (full rewrite) is handled by full_rewrite.
FORM_DESCRIPTIONS = {
    "F1": "Conditional rules: write when a concrete condition holds and what action to take.",
    "F2": "Worked examples: insert or replace a section containing only supplied solved examples.",
    "F3": "Reasoning procedure: write a sequence of reasoning/checking steps or task workflow. "
          "Only quote execution-specific tool actions or literal anchors when supplied in real traces.",
    "F4": "Full rewrite: rewrite the whole skill from critiques of the failed samples.",
}


def examples(ev):
    """Only the explicit verified-example block provides demonstration sources."""
    offered = set(map(str, ev.get("offered_gold_ids", [])))
    return {tid: line for tid in sorted(offered) for line in ev.get("gold_lines", [])
            if line.startswith(f"[{tid}] ")}


def available_forms(ev):
    return ["F1", *(["F2"] if examples(ev) else []), "F3", "F4"]


LOCAL_PROMPT = """Improve the intermediate skill using the form already chosen by the strategy selector.
The selected form is {form}. Do not choose another form.
{description}

INTERMEDIATE SKILL (the execution feedback below belongs to this exact skill):
<skill>{skill}</skill>

EXECUTION FEEDBACK AND CRITIQUES:
{feedback}

PAST OPTIMIZATION OUTCOMES:
{history}

SUPPLIED SOLVED EXAMPLES (the only allowed sources for F2):
{examples}
SUCCESSFUL TRAJECTORIES (only these support claims about actual execution traces):
{trajectories}

Return complete edits. Local edits have no separate token-length limit and will
never be silently shortened. Return 1-6 edits, applied in order.
For replace or insert_after, old_content must be an exact, unique substring of
the intermediate skill after preceding edits. For append, old_content is empty.
The allowed operations are replace, insert_after, and append; no full rewrite.
Preserve exact output-format instructions.
For F2, new_content must contain only a heading and complete lines copied VERBATIM
from SUPPLIED SOLVED EXAMPLES, including their [source_id]. Select only those IDs.
Do not invent examples or use history cases as demonstration sources.
{guard}
{spec}

Return ONLY JSON:
{{"source_task_ids": ["ids of supplied examples used, or empty"],
  "theme": "failure addressed",
  "patch": {{"edits": [{{"operation": "append", "old_content": "", "new_content": "..."}}]}}}}
"""


def local_update(skill, gradient, form, ev, history, spec, salt, generation=None):
    info = generation if generation is not None else {}
    offered = examples(ev)
    if form not in available_forms(ev) or form == "F4":
        info.update(mechanically_valid=False, source_valid=False,
                    failure_stage="unavailable_form")
        return None
    prompt = LOCAL_PROMPT.format(
        form=form, description=FORM_DESCRIPTIONS[form], skill=skill,
        feedback=gradient, history=history,
        examples="\n".join(offered.values()) or "(none available; F2 is disabled)",
        trajectories="\n".join(ev.get("traj_lines", [])) or "(none)",
        guard=full_rewrite.FORMAT_GUARD, spec=spec)
    raw_text = full_rewrite.optimizer_call(
        "Execute the specified skill revision form. Return only the requested JSON.",
        prompt, "i2_local_update" + salt)
    try:
        raw = direct_revision.optimizer_llm._extract_json(raw_text)
        patch = raw["patch"]
        if not isinstance(patch, dict) or "edits" not in patch:
            raise ValueError("Expected local edits")
        # Structural validation never trims or changes the proposed patch.
        text = skill_edits.apply_multi(skill, patch)
        info["mechanically_valid"] = True
    except (ValueError, KeyError, TypeError, AttributeError):
        info.update(mechanically_valid=False, source_valid=None, failure_stage="patch_validation")
        return None
    sources = raw.get("source_task_ids", [])
    if not isinstance(sources, list) or any(not isinstance(t, str) for t in sources):
        info.update(source_valid=False, failure_stage="source_validation")
        return None
    source_ids = set(sources)
    if form == "F2":
        copied = []
        for edit in patch["edits"]:
            for line in edit.get("new_content", "").splitlines():
                if not line.strip() or line.strip() == "### Worked examples":
                    continue
                copied.append(line)
        expected = [offered[t] for t in sources if t in offered]
        valid = bool(source_ids) and source_ids <= offered.keys() and (
            len(sources) == len(source_ids) and sorted(copied) == sorted(expected))
        if not valid:
            info.update(source_valid=False, failure_stage="example_grounding")
            return None
    else:
        allowed = set(map(str, ev.get("lesson_ids", []))) | offered.keys()
        if not source_ids <= allowed:
            info.update(source_valid=False, failure_stage="source_validation")
            return None
    info.update(source_valid=True, source_task_ids=sorted(source_ids),
                theme=str(raw.get("theme", ""))[:180], patch=patch,
                generator="specified_local_form")
    return text
