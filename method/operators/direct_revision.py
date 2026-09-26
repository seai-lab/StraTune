"""Direct revision (I1) and construction of the execution feedback.

One optimizer-LLM call per round receives the current skill, the execution
feedback, the menu of revision forms, and the form history, and returns two
candidate skills in two different forms, both revising the same current skill.

The generation prompt never sees per-sample screening scores or acceptance
thresholds. The optimizer LLM's own attempts at failed samples (verified fixes)
only annotate the failure descriptions; they never block a candidate.
"""
import os
import re

from method.environments import dataset_profiles as profiles
from method import state
from method.operators import optimizer_llm
from method.common import skill_edits
from method.common import util
from method.evaluation import run_version_on


MAX_FAIL_LINES = 12     # failed samples described in the execution feedback
N_CANDS = 2             # candidate skills per I1 call (dataset configuration: n_cands)
GLOBAL_STATS = False    # include training-distribution statistics (dataset configuration: global_stats)
STATS_MIN_SUPPORT = 12  # a recurring answer option must appear in at least this many tasks
STATS_MAX_ROWS = 5      # at most this many statistic rows are shown
_STATS_CACHE = {}       # (ds, n_tasks) -> lines; computed once per run


def form_text(k):
    """Menu text of revision form k."""
    return state.FORMS[k]
VERIFY_K = 8            # failed samples the optimizer LLM attempts itself per round
CONTRAST_HASH_K = 3     # solved samples shown as contrast, chosen by hash order
CONTRAST_LONG_K = 2     # solved samples shown as contrast, chosen by longest gold answer
RESIDUAL_MIN_FAIL = 6   # below this many batch failures, known failures from earlier batches are added
RESIDUAL_WIDTH = 36
GOLD_EXAMPLE_K = 10     # solved (input -> gold) pairs offered as example material
TRAJ_K = 6              # successful trajectories shown
LONGITUDINAL_K = (6, 6, 8)  # improved / regressed / persistently failed samples shown


def build_evidence(ds, st, executor, batch_ids, batch_res, tasks_by_id,
                   budget, workers, rnd, epoch, log):
    """Returns {blocks: {name: text|None}, lesson_ids, offered_gold_ids}."""
    champ = st.d["champion"]
    failures, successes = [], []
    for tid in batch_ids:
        rec = batch_res.get(tid) or {}
        prim = rec.get("primary")
        if prim is None or not profiles.parent_success(ds, float(prim)):
            failures.append(tid)
        else:
            successes.append(tid)

    # ---- When the batch has few failures, add known failures from earlier batches ----
    pool_res = dict(batch_res)
    if len(failures) < RESIDUAL_MIN_FAIL:
        known = st.d["scores"].get(champ, {})
        resid = [t for t, v in known.items()
                 if t in tasks_by_id and not profiles.parent_success(ds, v)
                 and t not in set(batch_ids)]
        if not profiles.PROFILES[ds]["binary"]:
            resid.sort(key=lambda t: (-known[t],
                                      util.sha256_str(f"resid|{st.run_id}|{rnd}|{t}")))
        else:
            resid.sort(key=lambda t: util.sha256_str(f"resid|{st.run_id}|{rnd}|{t}"))
        resid = resid[:RESIDUAL_WIDTH]
        if resid and not budget.exhausted(margin=len(resid)):
            r_tasks = [tasks_by_id[t] for t in resid]
            r_res = run_version_on(executor, r_tasks, st.skill_text(champ),
                                   workers, {}, budget)
            st.record_scores(ds, champ, r_res)
            pool_res.update(r_res)
            failures = failures + resid
            log(f"round {rnd}: residual pooling +{len(resid)} known failures")

    lesson_ids = sorted(failures,
                        key=lambda t: util.sha256_str(f"lesson|{t}"))[:MAX_FAIL_LINES]

    # Samples whose scores come from the score ledger have no execution record;
    # fetch the full record through the execution cache.
    stale = [t for t in lesson_ids
             if "predicted_answer" not in (pool_res.get(t) or {})
             and "per_step" not in (pool_res.get(t) or {})
             and "case_results" not in (pool_res.get(t) or {})
             and "predicted_label" not in (pool_res.get(t) or {})]
    if stale and not budget.exhausted(margin=len(stale)):
        fresh = run_version_on(executor, [tasks_by_id[t] for t in stale],
                               st.skill_text(champ), workers, {}, budget)
        pool_res.update(fresh)
        st.record_scores(ds, champ, fresh)

    # ---- The optimizer LLM attempts some failed samples itself; verified fixes annotate the feedback ----
    fix_by_id, vstats = {}, {"tried": 0, "verified": 0}
    try:
        from method.operators.verified_fixes import verified_lessons
        import os
        work_root = os.path.join(util.DATA_ROOT, "tmp", "ssb_exec",
                                 f"{st.run_id}_fixverify")
        fixes, vstats = verified_lessons(ds, lesson_ids[:VERIFY_K], tasks_by_id,
                                         pool_res, work_root)
        fix_by_id = {f["task_id"]: f["fix_summary"] for f in fixes}
        log(f"round {rnd}: blind-solve annotation {vstats['verified']}/{vstats['tried']}")
    except Exception as e:  # noqa: BLE001 — the annotation must never block generation
        log(f"round {rnd}: blind-solve annotation unavailable ({type(e).__name__}: {e})")

    fail_lines = []
    for t in lesson_ids:
        line = profiles.failure_summary(ds, tasks_by_id[t], pool_res.get(t) or {})
        if t in fix_by_id:
            line += "\n  VERIFIED FIX (optimizer solved this blind; mechanically checked): " + fix_by_id[t]
        elif t in set(lesson_ids[:VERIFY_K]):
            line += "\n  (fix attempt unverified — diagnosis below may be wrong)"
        fail_lines.append(line)

    # ---- Solved samples shown as contrast to the failures ----
    contrast_ids, seen = [], set()
    hs = sorted(successes, key=lambda t: util.sha256_str(f"ctr|{st.run_id}|{rnd}|{t}"))
    for t in hs[:CONTRAST_HASH_K]:
        contrast_ids.append(t)
        seen.add(t)
    scored = [(profiles.gold_len(ds, tasks_by_id[t]), t) for t in successes]
    scored = [(n, t) for n, t in scored if n and t not in seen]
    scored.sort(key=lambda x: (-x[0], util.sha256_str(f"ctrl|{x[1]}")))
    for _, t in scored[:CONTRAST_LONG_K]:
        contrast_ids.append(t)
    stale_c = [t for t in contrast_ids
               if "predicted_answer" not in (pool_res.get(t) or batch_res.get(t) or {})
               and "per_step" not in (pool_res.get(t) or batch_res.get(t) or {})
               and "case_results" not in (pool_res.get(t) or batch_res.get(t) or {})
               and "predicted_label" not in (pool_res.get(t) or batch_res.get(t) or {})]
    if stale_c and not budget.exhausted(margin=len(stale_c)):
        fresh_c = run_version_on(executor, [tasks_by_id[t] for t in stale_c],
                                 st.skill_text(champ), workers, {}, budget)
        pool_res.update(fresh_c)
        st.record_scores(ds, champ, fresh_c)
    contrast_lines = [profiles.success_summary(ds, tasks_by_id[t],
                                               pool_res.get(t) or batch_res.get(t) or {})
                      for t in contrast_ids]
    if scored:
        contrast_lines.append(
            "(the last cases above are the LONGEST-gold correct answers in this batch — "
            "any rule that shortens/trims answers MUST leave them intact; state the "
            "boundary condition, a blanket direction rule will be rejected)")

    # ---- Samples improved, regressed, or persistently failed since the initial skill ----
    longitudinal = None
    snap = st.d.get("epoch_champs", {}).get("0")
    if epoch >= 1 and snap and snap != champ:
        s0 = st.d["scores"].get(snap, {})
        s1 = st.d["scores"].get(champ, {})
        common = [t for t in s1 if t in s0 and t in tasks_by_id]
        improved = [t for t in common
                    if profiles.parent_success(ds, s1[t]) and not profiles.parent_success(ds, s0[t])]
        regressed = [t for t in common
                     if profiles.parent_success(ds, s0[t]) and not profiles.parent_success(ds, s1[t])]
        persistent = [t for t in common
                      if not profiles.parent_success(ds, s0[t]) and not profiles.parent_success(ds, s1[t])]
        ki, kr, kp = LONGITUDINAL_K

        def render(ids, k, tag):
            ids = sorted(ids, key=lambda t: util.sha256_str(f"long|{tag}|{t}"))[:k]
            return [f"[{t}] {profiles.task_text(ds, tasks_by_id[t])[:110]} | "
                    f"{s0[t]:.2f} -> {s1[t]:.2f}" for t in ids]
        longitudinal = (
            f"SINCE THE RUN'S FIRST CHAMPION (real per-task ledger, {len(common)} common tasks):\n"
            f"IMPROVED ({len(improved)}):\n" + "\n".join(render(improved, ki, "i")) +
            f"\nREGRESSED ({len(regressed)}):\n" + "\n".join(render(regressed, kr, "r")) +
            f"\nPERSISTENT FAILURES ({len(persistent)}):\n" + "\n".join(render(persistent, kp, "p")))

    # ---- Successful trajectories (for tasks with step structure) ----
    traj_lines = []
    for t in sorted(successes, key=lambda t: util.sha256_str(f"traj|{rnd}|{t}")):
        r = profiles.traj_repr(ds, tasks_by_id[t], batch_res.get(t) or {})
        if r:
            traj_lines.append(r)
        if len(traj_lines) >= TRAJ_K:
            break

    # ---- Solved (input -> gold) pairs (when the gold answer can be shown) ----
    gold_lines, offered_gold = [], []
    known = st.d["scores"].get(champ, {})
    solved = [t for t, v in known.items()
              if t in tasks_by_id and profiles.parent_success(ds, v)]
    solved.sort(key=lambda t: util.sha256_str(f"gold|{st.sampling_id}|{rnd}|{t}"))
    for t in solved:
        p_ = profiles.demo_pair(ds, tasks_by_id[t])
        if p_:
            gold_lines.append(f"[{t}] {p_}")
            offered_gold.append(t)
        if len(gold_lines) >= GOLD_EXAMPLE_K:
            break

    return {
        "n_batch": len(batch_ids), "n_fail": len(failures),
        "fail_lines": fail_lines, "contrast_lines": contrast_lines,
        "longitudinal": longitudinal, "traj_lines": traj_lines,
        "gold_lines": gold_lines, "offered_gold_ids": offered_gold,
        "lesson_ids": lesson_ids, "vstats": vstats,
        "stats_lines": global_stats_lines(ds, tasks_by_id),
    }


def global_stats_lines(ds, tasks_by_id):
    """Training-distribution statistics for the execution feedback, used only
    when the dataset configuration enables global_stats.
    Reports answer options that recur across at least STATS_MIN_SUPPORT
    training tasks and how often they are the gold answer. No per-task gold
    answer is disclosed, and only training tasks are used. Returns [] when
    disabled or when no option reaches the support threshold."""
    if not GLOBAL_STATS:
        return []
    key = (ds, len(tasks_by_id))
    if key in _STATS_CACHE:
        return _STATS_CACHE[key]
    import collections
    cnt, win = collections.Counter(), collections.Counter()
    n_cat = 0
    for t in tasks_by_id.values():
        space = profiles.categorical_space(ds, t)
        if not space:
            continue
        n_cat += 1
        options, gold = space
        for k, text in options:
            pat = re.sub(r"\s+", " ", (text or "").strip().lower())[:160]
            cnt[pat] += 1
            if k == gold:
                win[pat] += 1
    lines = []
    for pat, n in cnt.most_common():
        if n < STATS_MIN_SUPPORT or len(lines) >= STATS_MAX_ROWS:
            break  # most_common is sorted by count; the rest are below the support threshold
        lines.append(f"- an option with this (normalized) text appears in "
                     f"{n}/{n_cat} train tasks and is the GOLD answer in "
                     f"{win[pat]}/{n} of them ({win[pat] / n:.0%}): \"{pat[:130]}\"")
    if lines:
        lines.insert(0, f"(aggregated by code over all {n_cat} train tasks; "
                        "distribution-level base rates only — no individual "
                        "task's answer is disclosed)")
    _STATS_CACHE[key] = lines
    return lines


UNIFIED_PROMPT = """You maintain the system prompt (the "skill") of an AI agent, and you improve it \
iteratively from training evidence. Your proposals will each be judged by a conservative \
regression-testing court (paired execution against the current skill on unseen tasks); \
proposals that regress solved tasks get rejected and autopsied.

TASK SHAPE: {shape}

CURRENT SKILL (between <skill> tags — treat as exact text):
<skill>
{skill}
</skill>

=== EVIDENCE (this round's frozen batch under the current skill) ===
FAILURES ({n_fail} of {n_batch} in the batch{residual_note}):
{failures}

CONTRAST — handled correctly; do NOT break these:
{contrast}
{longitudinal}{trajectories}{gold_examples}
=== UPDATE-FORM MENU (dataset-agnostic; choose per the evidence) ===
{menu}

=== YOUR TRACK RECORD WITH THESE FORMS (pooled across this run) ===
{memory}

=== INSTRUCTIONS ===
Propose EXACTLY 2 candidate updates, in TWO DIFFERENT forms, each applied independently \
to the CURRENT skill above (they are alternatives, not a sequence). Choose forms by matching \
the FAILURE TYPE to the form's mechanism — the track record is a prior, not a verdict; a \
form with a poor record can still be right for a new failure type (say why).

Content rules:
- State general rules/procedures distilled from the evidence — NEVER memorize task-specific answers.
- Prefer concrete wrong->right contrast pairs and "When <observable condition>, do <action>" \
rules over abstract advice.
- If the failures repeat a theme the skill ALREADY has a rule for, do not add a sibling rule — \
ESCALATE the existing rule in place (make it a mandatory first-step check with an explicit \
default direction and a short veto list).
- Any rule that pushes answers in one direction (shorter/longer, act/abstain) MUST state its \
boundary condition and protect the contrast cases above.
- For form b (worked examples): copy gold answers VERBATIM from the solved pairs given above, \
one line each, covering distinct input kinds; list the task ids you embedded in source_task_ids.
- For form d (workflows): quote EXACT literal anchors from the trajectories; do not abstract \
away names the traces show.
- For form e (rewrite): reproduce the skill's output-format specification character-for-character.

Patch mechanics:
- Forms a-d: {{"edits": [{{"operation": "replace|insert_after|append", "old_content": \
"<exact unique substring or empty>", "new_content": "<text>"}}, ...]}} — 1-6 edits, applied in \
order. Local edits have no separate token-length limit; return complete instructions.
- Form e: {{"operation": "rewrite", "new_content": "<the complete new skill text>"}} — at most \
{rewrite_budget} tokens.

Reply with ONLY JSON:
{{"candidates": [
   {{"form": "a|b|c|d|e",
     "theme": "<the failure theme this targets, 3-8 words>",
     "rationale": "<one sentence: why this form for this evidence>",
     "source_task_ids": ["<ids of tasks whose gold/trace content is EMBEDDED in the patch, else empty>"],
     "patch": {{...}} }},
   ...]}}"""


def validate_candidate(c, skill, log):
    """Validate one raw candidate dict and apply its edits to the skill.
    Returns the canonical candidate or None. Shared by I1 and I3."""
    if not isinstance(c, dict):
        return None
    form = str(c.get("form", "")).strip().lower()[:1]
    patch = c.get("patch")
    if form not in state.FORMS or not isinstance(patch, dict):
        return None
    try:
        if form == "e":
            if patch.get("operation") != "rewrite":
                patch = {"operation": "rewrite",
                         "new_content": patch.get("new_content") or ""}
            cand_text = skill_edits.apply_patch(skill, patch)
        else:
            if "edits" not in patch:
                patch = {"edits": [patch]}
            # Apply the entire proposal. Invalid edits are rejected rather
            # than silently dropping edits or truncating their content.
            cand_text = skill_edits.apply_patch(skill, patch)
    except skill_edits.PatchError as e:
        log(f"candidate (form {form}) unusable: {e}")
        return None
    patch["scope"] = [w for w in str(c.get("theme", "")).split()][:10]
    patch["rationale"] = str(c.get("rationale", ""))[:300]
    return {
        "form": form, "theme": str(c.get("theme", ""))[:120],
        "rationale": str(c.get("rationale", ""))[:300],
        "source_task_ids": [str(t) for t in (c.get("source_task_ids") or [])][:40],
        "patch": patch, "cand_text": cand_text,
    }


def propose(ds, st, skill, ev, salt, log):
    """One I1 generation call -> list of validated candidate skills
    [{form, theme, rationale, source_task_ids, patch, cand_text}]."""
    menu = "\n".join(f"{k}. {form_text(k)}" for k in sorted(state.FORMS))
    rewrite_budget = max(skill_edits.REWRITE_MULT * skill_edits.approx_tokens(skill),
                         skill_edits.approx_tokens(skill) + skill_edits.REWRITE_SLACK)

    def block(title, lines):
        return f"\n{title}:\n" + "\n".join(lines) + "\n" if lines else ""

    # A form tried at least six times without an accepted candidate is treated as
    # saturated; one candidate must then use a form that is not saturated.
    mem = st.d.get("form_mem", {})
    saturated = sorted(f for f, m in mem.items()
                       if m.get("tries", 0) >= 6 and m.get("adopted", 0) == 0)
    unsaturated = sorted(f for f in state.FORMS if f not in saturated)
    explore_note = ""
    if saturated and unsaturated:
        explore_note = (f"\n\nEXPLORATION FLOOR: forms {{{', '.join(saturated)}}} have "
                        f">= 6 tries and 0 adoptions this run — at least ONE of your "
                        f"candidates MUST use a form from {{{', '.join(unsaturated)}}}.")

    prompt = UNIFIED_PROMPT.format(
        shape=profiles.task_shape(ds), skill=skill,
        n_fail=ev["n_fail"], n_batch=ev["n_batch"],
        residual_note=("; includes pooled known residual failures"
                       if ev["n_fail"] > ev["n_batch"] else ""),
        failures="\n".join(ev["fail_lines"]) or "(none)",
        contrast="\n".join(ev["contrast_lines"]) or "(none available)",
        longitudinal=block("LONGITUDINAL (per-task ledger since the first champion)",
                           [ev["longitudinal"]] if ev["longitudinal"] else []),
        trajectories=block("SUCCESSFUL TRAJECTORIES (real solved traces)", ev["traj_lines"]),
        gold_examples=block("SOLVED (INPUT -> GOLD) PAIRS available as example material",
                            ev["gold_lines"])
        + block("GLOBAL TRAIN-DISTRIBUTION STATISTICS (aggregated by code, "
                "Test-blind; where a base rate is strong, write an explicit "
                "DEFAULT/PRIOR rule that cites it)",
                ev.get("stats_lines") or []),
        menu=menu, memory=state.render_form_history(st),
        rewrite_budget=rewrite_budget,
    ) + explore_note + f"\n\n<!-- {salt} -->"
    if ev.get("stats_lines"):
        log(f"global-stats block active ({len(ev['stats_lines']) - 1} rows)")
    if N_CANDS != 2:  # the dataset configuration asks for a different number of candidates
        prompt = prompt.replace(
            "Propose EXACTLY 2 candidate updates, in TWO DIFFERENT forms,",
            f"Propose EXACTLY {N_CANDS} candidate updates, covering AT LEAST TWO "
            f"DIFFERENT forms,", 1)

    try:
        out = optimizer_llm._call_json(prompt, tag="direct_revision_propose")
    except optimizer_llm.LLMOpError as e:
        log(f"proposal call unparseable: {e}")
        return []
    raw = out.get("candidates") or []
    cands = []
    for c in raw[:N_CANDS]:
        cand = validate_candidate(c, skill, log)
        if cand:
            cands.append(cand)
    forms = {c["form"] for c in cands}
    if len(cands) >= 2 and len(forms) < 2:
        cands = cands[:1]  # the candidates must differ in form; keep only the first
        log("proposal diversity floor violated (all one form); keeping first candidate only")
    return cands
