"""Adaptive selection of the revision operator.

Each round, the optimizer LLM chooses a search strategy (I1 direct revision, I2
iterative refinement, I3 parallel sampling) from the optimization state, that is,
from the execution feedback and the strategy and form histories. Revision forms are
chosen under the selected strategy, candidate skills are generated, and every
candidate is submitted to the same candidate evaluation (callbacks owned by
method.train).

The selection and generation prompts never see per-sample screening scores,
acceptance thresholds, or budget figures. Whether I2 or I3 is affordable is
decided in run_round before the prompt is built.
"""
from method.environments import dataset_profiles as profiles
from method.operators import direct_revision
from method.operators import full_rewrite
from method.operators import iterative_refinement
from method import state
from method.operators import optimizer_llm
from method.common import skill_edits
from method.common import util
from method.evaluation import run_version_on


# Parallel sampling (I3). Its candidates are ranked on the same common training
# subset that I2 uses.
I3_K = 3                 # candidate skills generated per I3 round
I3_SMOKE_N = 12          # samples of the preliminary check before ranking
I3_SUBMIT_SLACK = 0.02   # the runner-up is also submitted if its score is within this of the current skill
GUARD_MULT = 6           # I2/I3 are offered only if the remaining budget exceeds GUARD_MULT x their cost
STRATEGIES = ["I1", "I2", "I3"]
DECL_MEMORY = True          # the optimizer LLM sees the strategy and form histories
I2_FORM_MODE = "free"        # I2 chooses its revision form (F1-F4) at each step
# Strategy schedule: a strategy not yet tried when this fraction of the budget is
# spent is forced in the next round, so that the histories cover every strategy.
FLOOR_I2_FRAC = 0.35
FLOOR_I3_FRAC = 0.50

I_CARDS = {
    "I1": """SINGLE-SHOT (cost: none) — one proposal call, exactly 2 candidates in two
   different forms against the frozen current skill. USE WHEN the evidence points at a
   specific, nameable fix (a clear failure theme with an obvious rule/example/protocol
   answer). KNOWN FAILURE: cannot escape a local basin — if the same theme keeps getting
   proposed and rejected across rounds, single shots are exhausted.""",
    "I2": """CHAIN REFINEMENT — continue the private intermediate skill.
   Critique executions from that exact skill, then generate one update in the specified
   revision form. Screen it and compare drafts on common training samples. A window's
   strongest drafts are submitted to formal evaluation. Picking I2 resumes the
   intermediate skill; attempts are not necessarily successful base advances.
   Use when multi-step refinement or integrated reorganization may help. Consult
   attempts, valid drafts, internal gains, and formal evaluation outcomes.""",
    "I3": """PARALLEL SAMPLING (cost: a few smokes + slice scores) — the SAME evidence,
   ONE declared form, 3 independent salted draws; obviously-broken drafts die on a
   12-task smoke; survivors are scored on the frozen slice and only the best 1-2 face
   the court. USE WHEN candidate quality feels like a lottery (plausible candidates keep
   losing narrowly, or the winning content shape is known but each single draw is noisy)
   — buying more tickets and pre-screening them raises the acceptance rate. KNOWN
   FAILURE: with too small a slice, picking the best draw is itself a lottery.""",
}

DECLARE_PROMPT = """You are the strategy controller of a skill-optimization loop. Each round you
see training evidence and must pick the ITERATION STRUCTURE used to generate this
round's candidate skill-updates. The candidates themselves are judged later by a
conservative regression court; your job here is only to spend this round's shot well.

THIS ROUND'S EVIDENCE (digest):
{digest}

ITERATION MENU:
I1. {i1}
I2. {i2}
I3. {i3}

LINEAGE STATE: {lineage_state}
{i2_state}

YOUR TRACK RECORD THIS RUN (pooled; forms a-e are the update forms of the
generation call; I1/I2/I3 are iteration structures):
{memory}
{saturation}
Reply with ONLY JSON:
{{"iteration": "I1|I2|I3",
  "form": "<I2: one available F1|F2|F3|F4; I3: a|b|c|d|e; I1: null>",
  "why": "<one sentence tying the choice to the evidence and track record>"}}"""


# ---------------------------------------------------------------------------
# Strategy history: outcomes grouped by search strategy (the form history is in form_history).
def _imem(st):
    return st.d.setdefault("iter_mem", {})


def record_i(st, i: str, decision: str, G):
    m = _imem(st).setdefault(i, {"tries": 0, "adopted": 0, "confirmed": 0,
                                 "g_sum": 0.0, "g_n": 0, "confirmed_gains": []})
    m["tries"] += 1
    if G is not None:
        m["g_sum"] += float(G)
        m["g_n"] += 1
    if decision == "commit":
        m["adopted"] += 1


def record_i_confirm(st, i: str, confirmed: bool, g_confirm: float):
    m = _imem(st).setdefault(i, {"tries": 0, "adopted": 0, "confirmed": 0,
                                 "g_sum": 0.0, "g_n": 0, "confirmed_gains": []})
    if confirmed:
        m["confirmed"] += 1
        m["confirmed_gains"].append(round(float(g_confirm), 4))
        del m["confirmed_gains"][:-6]
    else:
        m["adopted"] = max(0, m["adopted"] - 1)


def render_i_memory(st) -> str:
    """Render the strategy history for the selection prompt; I2 refinement steps are listed separately."""
    mem = _imem(st)
    lines = []
    for i in ("I1", "I2", "I3"):
        m = mem.get(i)
        if i == "I2":
            attempts = list(st.d.get("i2_attempts", {}).values())
            m = m or {}
            lines.append(
                f"  I2: {len(attempts)} generation attempts, "
                f"{sum(e['valid_generation'] for e in attempts)} valid drafts, "
                f"{sum(e['base_advanced'] for e in attempts)} base advances, "
                f"{m.get('tries', 0)} formal evaluations, "
                f"{m.get('confirmed', 0)} passed further validation; "
                f"mean formal gain "
                + (f"{m['g_sum']/m['g_n']:+.3f}" if m.get("g_n") else "unavailable"))
            continue
        if not m or not m["tries"]:
            lines.append(f"  {i}: not tried yet this run")
            continue
        conf = (f", confirmed gains {m['confirmed_gains']}"
                if m.get("confirmed_gains") else "")
        if i == "I2":
            lines.append(f"  {i}: {m['tries']} submissions, {m['adopted']} adopted, "
                         f"{m['confirmed']} survived confirmation{conf} "
                         f"(heavy-tailed by design; judge by jackpots, not averages)")
        else:
            mean_g = (m["g_sum"] / m["g_n"]) if m["g_n"] else 0.0
            lines.append(f"  {i}: {m['tries']} tried, {m['adopted']} adopted, "
                         f"{m['confirmed']} survived confirmation, "
                         f"mean paired gain {mean_g:+.3f}{conf}")
    return "\n".join(lines)


def _saturated_is(st):
    return sorted(i for i, m in _imem(st).items()
                  if m.get("tries", 0) >= 6 and m.get("adopted", 0) == 0)


# ---------------------------------------------------------------------------
def _ensure_val(st, ctx):
    """The common training subset on which I2 and I3 rank their candidates, fixed
    once per run, together with the current skill's score on it."""
    gen = iterative_refinement._gen(st)
    if gen["val_ids"] is None:
        used = set(map(str, ctx["batch_ids"]))
        full_rewrite.REFINE_VAL_N = ctx["n_slice"]
        gen["val_ids"] = full_rewrite.refine_val_ids(ctx["train_ids"], used,
                                           f"{st.sampling_id}|rw")
        gen["val_score"] = None
    score, _, key = state.reference(
        st.d["_ds"], st, st.d["champion"], gen["val_ids"], ctx["tasks_by_id"],
        ctx["executor"], ctx["workers"], ctx["budget"])
    # I3 compares with the current skill, which need not be I2's intermediate skill.
    return dict(gen, val_score=score, reference_key=key)


def _digest(ev, st) -> str:
    themes = []
    for ln in ev["fail_lines"][:6]:
        themes.append("- " + ln.splitlines()[0][:140])
    pms = st.d.get("postmortems", [])
    return (f"{ev['n_fail']} failures of {ev['n_batch']} in the frozen batch"
            f"{' (incl. pooled residuals)' if ev['n_fail'] > ev['n_batch'] else ''};"
            f" evidence blocks available: contrast={bool(ev['contrast_lines'])},"
            f" longitudinal={bool(ev['longitudinal'])}, trajectories={bool(ev['traj_lines'])},"
            f" gold pairs={bool(ev['gold_lines'])}.\nSample failure heads:\n"
            + "\n".join(themes)
            + (f"\nMost recent rejection: {pms[-1][:200]}" if pms else ""))


def declare(ds, st, ev, ctx, allowed, log):
    """Ask the optimizer LLM to select a strategy from `allowed` (after the budget
    check and the strategy schedule). A forced I3 still reaches the LLM with
    allowed=["I3"] so that the revision form is chosen by the LLM."""
    if allowed == ["I1"]:
        return {"iteration": "I1", "form": None, "why": "narrowed (baseline/budget guard)"}
    gen = st.d.get("rw") or {}
    lineage_state = (
        f"a draft lineage EXISTS ({gen.get('steps', 0)} attempts so far, "
        f"{gen.get('counters', {}).get('base_adopted', 0)} base advances, window "
        f"{len(gen.get('window', []))} drafts pending)" if gen.get("steps")
        else "no draft lineage yet (picking I2 starts one from the champion)")
    sat = _saturated_is(st)
    saturation = ""
    if sat:
        rest = [i for i in allowed if i not in sat]
        if rest:
            saturation = (f"\nSATURATION: {{{', '.join(sat)}}} have >=6 tries and 0 "
                          f"adoptions this run — pick from {{{', '.join(rest)}}} unless "
                          f"the evidence shows a genuinely NEW failure type (say why).\n")
    mem_block = (state.render_form_history(st) + "\n" + render_i_memory(st)) if DECL_MEMORY \
        else "(track record withheld for this run — choose from the evidence and the menu alone)"
    available = iterative_refinement.available_forms(ev)
    base_vid = gen.get("base_vid") or st.d["champion"]
    i2_state = (
        "\nI2 INTERMEDIATE SKILL:\n" + st.skill_text(base_vid)
        + "\nI2 available revision forms:\n"
        + "\n".join(f"{f}: {iterative_refinement.FORM_DESCRIPTIONS[f]}" for f in available)
        + "\nF2 is available only with supplied solved examples. "
        "I3 uses a=F1, b=F2, c/d=F3, e=F4."
        "\nThe digest concerns the current skill. I2 critiques will use executions "
        "from the intermediate skill shown above.")
    prompt = DECLARE_PROMPT.format(
        digest=_digest(ev, st), i1=I_CARDS["I1"], i2=I_CARDS["I2"],
        i3=I_CARDS["I3"], lineage_state=lineage_state, i2_state=i2_state,
        memory=mem_block,
        saturation=saturation if DECL_MEMORY else "",
    ) + f"\n<!-- {st.sampling_id[-6:]}|r{ctx['rnd']} -->"
    if len(allowed) < 3:
        prompt += (f"\n\nNOTE: only {{{', '.join(allowed)}}} are available this round.")
    try:
        out = optimizer_llm._call_json(prompt, tag="declare_strategy")
    except optimizer_llm.LLMOpError as e:
        log(f"episode: declaration unparseable ({e}); defaulting to {allowed[0]}")
        out = {"iteration": allowed[0]}
    i = str(out.get("iteration", "")).strip().upper()
    if i not in allowed:
        log(f"episode: declared {i!r} not allowed; defaulting to {allowed[0]}")
        i = allowed[0]
    selected = out.get("form")
    if i == "I2":
        form = state.canonical(selected)
        if form not in available:
            log(f"episode: unavailable I2 form {form!r}; using preserved F4 action")
            form = "F4"
        return {"iteration": i, "form": form, "why": str(out.get("why", ""))[:300]}
    form = state.NATIVE.get(state.canonical(selected),
                                 str(selected or "").strip().lower()[:1])
    if i == "I3" and form not in state.FORMS:
        form = "a"
    return {"iteration": i, "form": form if i == "I3" else None,
            "why": str(out.get("why", ""))[:300]}


# ---------------------------------------------------------------------------
def propose_one(ds, st, skill, ev, form, salt, log):
    """One I3 generation call producing one candidate skill of the given form.
    Uses the I1 prompt with the instruction line replaced; the candidate is
    validated by the same helper as I1 candidates."""
    menu = f"{form}. {direct_revision.form_text(form)}"
    rewrite_budget = max(skill_edits.REWRITE_MULT * skill_edits.approx_tokens(skill),
                         skill_edits.approx_tokens(skill) + skill_edits.REWRITE_SLACK)

    def block(title, lines):
        return f"\n{title}:\n" + "\n".join(lines) + "\n" if lines else ""

    base = direct_revision.UNIFIED_PROMPT
    anchor = "Propose EXACTLY 2 candidate updates, in TWO DIFFERENT forms, each applied independently"
    assert anchor in base, "direct_revision.UNIFIED_PROMPT drifted; update selection.propose_one"
    base = base.replace(
        anchor,
        f"Propose EXACTLY 1 candidate update, in form {form} ONLY, applied", 1)
    prompt = base.format(
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
    ) + f"\n\n<!-- {salt} -->"
    try:
        out = optimizer_llm._call_json(prompt, tag="parallel_sample")
    except optimizer_llm.LLMOpError as e:
        log(f"i3 sample call unparseable: {e}")
        return None
    raw = out.get("candidates") or []
    for c in raw[:2]:
        cand = direct_revision.validate_candidate(c, skill, log)
        if cand and cand["form"] == form:
            return cand
    return None


def run_i3(ds, st, ev, form, ctx, log):
    """Parallel sampling: generate I3_K candidates under one form, drop those that
    fail the preliminary check, rank the rest on the common training subset, and
    submit the best one or two to candidate evaluation (ctx['submit_cands'])."""
    budget = ctx["budget"]
    gen = _ensure_val(st, ctx)
    val_ids = gen["val_ids"]
    val_tasks = [ctx["tasks_by_id"][t] for t in val_ids]
    skill = st.skill_text(st.d["champion"])
    scored = []
    for k in range(I3_K):
        if budget.exhausted(margin=I3_SMOKE_N + ctx["m_prop"]):
            break
        cand = propose_one(ds, st, skill, ev, form,
                           salt=f"{st.sampling_id[-6:]}|r{ctx['rnd']}|i3k{k}", log=log)
        if not cand:
            continue
        smoke_tasks = val_tasks[:I3_SMOKE_N]
        res_s = run_version_on(ctx["executor"], smoke_tasks, cand["cand_text"],
                               ctx["workers"], {}, budget)
        s_mean = (sum(float(r.get("primary", 0)) for r in res_s.values())
                  / max(1, len(smoke_tasks)))
        if gen["val_score"] and gen["val_score"] > 0.05 \
                and s_mean < iterative_refinement.SMOKE_KILL_FRAC * gen["val_score"]:
            log(f"i3 draw {k}: SMOKE-KILLED ({s_mean:.3f} vs base {gen['val_score']:.3f})")
            _live(st, "i3_smoke_killed")
            continue
        if budget.exhausted(margin=len(val_tasks)):
            scored.append((s_mean, cand))  # budget too low for the full subset; rank on the preliminary check
            continue
        res_v = run_version_on(ctx["executor"], val_tasks, cand["cand_text"],
                               ctx["workers"], res_s, budget)
        v_mean = (sum(float(r.get("primary", 0)) for r in res_v.values())
                  / max(1, len(val_tasks)))
        log(f"i3 draw {k}: slice={v_mean:.4f} (base {gen['val_score']:.4f})")
        scored.append((v_mean, cand))
    if not scored:
        log("i3: no usable draws")
        return
    scored.sort(key=lambda x: -x[0])
    picked = [scored[0][1]]
    if (len(scored) > 1
            and scored[1][0] >= (gen["val_score"] or 0) - I3_SUBMIT_SLACK):
        picked.append(scored[1][1])
    seen = set()
    picked = [c for c in picked
              if not (util.sha256_str(c["cand_text"]) in seen
                      or seen.add(util.sha256_str(c["cand_text"])))]
    _live(st, "i3_submissions")
    ctx["submit_cands"](picked, ev, "I3")


def _live(st, key):
    st.d.setdefault("liveness", {})
    st.d["liveness"][key] = st.d["liveness"].get(key, 0) + 1


# ---------------------------------------------------------------------------
def run_round(ds, st, ev, ctx, log):
    """One round of operator selection and candidate generation. `ev` is the
    execution feedback built by method.train.
    ctx: batch_ids, tasks_by_id, train_ids, executor, budget, workers, rnd,
    epoch, T, n_slice, m_prop, m_rw, submit_cands(cands, ev, itag), rw_step()."""
    budget, rnd, T = ctx["budget"], ctx["rnd"], ctx["T"]
    st._round_evidence = ev
    im = _imem(st)

    # Budget check: I2 and I3 are offered only if affordable (the LLM never sees these numbers).
    i2_cost = iterative_refinement.SMOKE_N + ctx["n_slice"]
    i3_cost = I3_K * I3_SMOKE_N + 2 * ctx["n_slice"]
    allowed = [i for i in STRATEGIES if i == "I1"
               or (i == "I2" and not budget.exhausted(margin=GUARD_MULT * i2_cost))
               or (i == "I3" and not budget.exhausted(margin=GUARD_MULT * i3_cost))]
    if not allowed:
        log(f"round {rnd}: no affordable search strategy; no candidates generated")
        return

    # The first rounds use I1. Afterwards a strategy that has never been selected
    # is forced once the corresponding fraction of the budget is spent.
    frac = budget.billed / max(1, budget.cap)
    lv = st.d.get("liveness", {})
    if rnd < 2 and "I1" in allowed:
        allowed = ["I1"]
    else:
        if "I2" in allowed and not lv.get("i_i2_declared") and frac >= FLOOR_I2_FRAC:
            allowed = ["I2"]
            log(f"episode: exploration floor -> I2 (billed {frac:.0%}, undeclared)")
        elif "I3" in allowed and not lv.get("i_i3_declared") and frac >= FLOOR_I3_FRAC:
            allowed = ["I3"]
            log(f"episode: exploration floor -> I3 (billed {frac:.0%}, undeclared)")

    if not ev["fail_lines"]:
        # No failures in this batch: only continuing an existing I2 refinement is meaningful.
        if "I2" in allowed and (st.d.get("rw") or {}).get("steps"):
            decl = declare(ds, st, ev, ctx, ["I2"], log)
        else:
            log(f"round {rnd}: no failures anywhere; no candidates generated")
            return
    else:
        decl = declare(ds, st, ev, ctx, allowed, log)
    i = decl["iteration"]
    _live(st, f"i_{i.lower()}_declared")
    log(f"episode r{rnd}: declared {i}"
        + (f" (form {decl['form']})" if decl.get("form") else "")
        + (f" — {decl['why'][:120]}" if decl.get("why") else ""))
    st.d.setdefault("episode_log", []).append(
        {"round": rnd, "i": i, "form": decl.get("form"), "why": decl.get("why", "")[:200]})

    if i == "I1":
        skill = st.skill_text(st.d["champion"])
        cands = direct_revision.propose(ds, st, skill, ev,
                                 salt=f"{st.sampling_id[-6:]}|r{rnd}", log=log)
        _live(st, "proposal_calls")
        ctx["submit_cands"](cands, ev, "I1")
    elif i == "I2":
        st._i2_form = decl.get("form") or "F4"
        ctx["rw_step"]()
    elif i == "I3":
        run_i3(ds, st, ev, decl["form"], ctx, log)
