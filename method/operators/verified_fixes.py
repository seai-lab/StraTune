"""Verified fixes (part of execution feedback preparation): a failure becomes a
lesson only if the optimizer LLM demonstrably solves the sample, checked
mechanically.

- docvqa: the optimizer LLM answers the question from the document image
  without seeing the gold answer; pass iff ANLS >= 0.9 against gold.
- mind2web: for up to 2 wrong-element steps, the optimizer LLM picks the
  element from the same 20-candidate step prompt the target LLM saw, without
  the gold answer; pass iff the element id is among the gold ids.
- spreadsheetbench: the optimizer LLM writes corrected code given the failed
  code and the case errors; the code is executed in the sandbox; pass iff
  hard_pass == 1. Up to 2 attempts (the second sees the first's errors).
- livemath: the optimizer LLM solves the multiple-choice question; pass iff its
  label matches the gold label.

Verification uses optimizer-LLM calls and sandbox CPU only; it does not consume
target-LLM executions from the training budget.
"""
import re
from pathlib import Path

from method.config import OPTIMIZER_MODEL
from method.config import _truncate
from method.common.execution import client_for


FIX_TAG = "optimizer/verified_fix"


def _sonnet(messages, tag, max_tokens=8192):
    client = client_for(FIX_TAG)
    resp = client.converse(OPTIMIZER_MODEL, messages, system=None,
                           max_tokens=max_tokens, temperature=0.0, seed=42,
                           tags={"stage": tag})
    return resp["text"]


# ---------------------------------------------------------------------------
def verify_docvqa(task, rec) -> dict | None:
    from method.environments.tasks.docvqa import score_answer
    from method.environments.tasks.docvqa import fit_png_to_limits
    with open(task["image_path"], "rb") as f:
        png = fit_png_to_limits(f.read())
    text = _sonnet([{
        "role": "user",
        "content": [
            {"text": f"{task['question']}\n\nRead the document image carefully and "
                     f"return ONLY the final answer inside <answer>...</answer> — the exact "
                     f"minimal string as it appears in the document."},
            {"image": {"format": "png", "source": {"bytes": png}}},
        ]}], tag="fix_docvqa")
    scores = score_answer(text, task["answers"])
    if scores["anls"] < 0.9:
        return None
    return {"task_id": task["task_id"],
            "fix_summary": (f"verified correct answer: {scores['predicted_answer']!r} "
                            f"(agent had answered {rec.get('predicted_answer')!r})")}


def verify_mind2web(task, rec) -> dict | None:
    """Uses the same SYSTEM_BASE_M2W system prompt as the target LLM; it carries
    the action format specification the parser expects."""
    from method.environments.mind2web_prompting import SYSTEM_BASE_M2W
    from method.environments.mind2web_prompting import build_step_user
    record = task["record"]
    steps = record.get("steps", [])
    per_step = rec.get("per_step", [])
    wrong = [i for i, s in enumerate(per_step) if not s.get("element_correct")][:2]
    if not wrong:
        return None
    client = client_for(FIX_TAG)
    fixes = []
    for i in wrong:
        user = build_step_user(record, i)
        text = client.converse(
            OPTIMIZER_MODEL, [{"role": "user", "content": [{"text": user}]}],
            system=SYSTEM_BASE_M2W, max_tokens=2048, temperature=0.0, seed=42,
            tags={"stage": "fix_m2w"})["text"]
        matches = re.findall(r"(CLICK|TYPE|SELECT)\s*\[\s*([A-Za-z0-9_-]+)\s*\]\s*(?:\[(.*?)\])?",
                             text, re.IGNORECASE | re.DOTALL)
        m = matches[-1] if matches else None  # the environment parser takes the last action line
        if not m:  # fallback: last bracketed numeric token
            nums = re.findall(r"\[\s*(\d+)\s*\]", text)
            m = (None, nums[-1], None) if nums else None
        if not m:
            continue
        elem = m[1]
        gold = [str(g) for g in steps[i].get("gold_ids", [])]
        if str(elem) in gold:
            chosen = per_step[i]
            fixes.append(f"step {i}: correct element id {elem} "
                         f"(agent chose {chosen.get('element_id')}, op {chosen.get('operation')})")
    if not fixes:
        return None
    return {"task_id": task["task_id"], "fix_summary": "verified fixes — " + "; ".join(fixes)}


def verify_ssb(task, rec, work_root: str) -> dict | None:
    from method.environments.tasks.spreadsheet_codegen import SpreadsheetCodegenEnv
    from method.environments.tasks.spreadsheet_codegen import _task_fields
    from method.environments.tasks.spreadsheet_codegen import extract_code
    from method.environments.tasks.spreadsheet_codegen import preview_workbook
    t = task["task_obj"]
    fields = _task_fields(t)
    try:
        preview = preview_workbook(fields["cases"][0]["input_path"])
    except Exception as exc:  # noqa: BLE001
        preview = f"(preview failed: {exc})"
    errs = []
    for cr in rec.get("case_results", []):
        for e in (cr.get("score_errors") or [])[:3]:
            errs.append(_truncate(e, 200))
        if cr.get("exec_error"):
            errs.append("EXEC: " + _truncate(cr["exec_error"], 200))
    env = SpreadsheetCodegenEnv(work_root=work_root)
    base_prompt = (
        f"A Python script must transform an Excel workbook per this instruction, reading the "
        f"workbook at the path in variable INPUT_PATH and writing the result to OUTPUT_PATH, "
        f"preserving all other cells. Formulas are NOT recalculated by the scorer — write "
        f"literal computed values unless the instruction explicitly demands formulas.\n\n"
        f"# Instruction ({fields['instruction_type']})\n{fields['instruction']}\n"
        f"Expected answer position: {fields['answer_position']}\n\n"
        f"# Input workbook preview\n{preview}\n\n"
        f"# Previous FAILED attempt\n```python\n{_truncate(rec.get('code', ''), 3000)}\n```\n"
        f"Scoring errors: {' | '.join(errs) or 'output values mismatched'}\n\n"
        f"Write a CORRECTED python script in a single ```python``` block.")
    prompt = base_prompt
    for attempt in range(2):
        code = extract_code(_sonnet([{"role": "user", "content": [{"text": prompt}]}],
                                    tag="fix_ssb", max_tokens=16384))
        if not code.strip():
            return None
        wd = Path(work_root) / f"verify_{task['task_id']}_a{attempt}"
        out = env.apply_code_to_task(code, fields, wd)
        if out.get("hard_pass") == 1.0 or out.get("hard_pass") is True:
            return {"task_id": task["task_id"],
                    "fix_summary": (f"verified working fix (all {len(fields['cases'])} cases pass). "
                                    f"Corrected code:\n```python\n{_truncate(code, 1500)}\n```")}
        fail = []
        for cr in out.get("case_results", []):
            for e in (cr.get("score_errors") or [])[:2]:
                fail.append(_truncate(e, 150))
            if cr.get("exec_error"):
                fail.append("EXEC: " + _truncate(cr["exec_error"], 150))
        prompt = base_prompt + (f"\n\nYour previous correction ALSO failed: "
                                f"{' | '.join(fail) or 'wrong values'}\n"
                                f"```python\n{_truncate(code, 2500)}\n```\nTry again.")
    return None


def verify_livemath(task, rec) -> dict | None:
    """The optimizer LLM solves the multiple-choice question itself; the fix is a
    lesson only if its label matches the gold label."""
    from method.environments.tasks.livemath import _LABEL_RE
    choices = "\n".join(f"{c['label']}. {c['text']}" for c in task["choices"])
    text = _sonnet([{"role": "user", "content": [{
        "text": (f"{task['question']}\n\nChoices:\n{choices}\n\n"
                 "Work the mathematics carefully, then end your reply with a "
                 "line of the exact form 'ANSWER: <letter>'.")}]}],
        tag="fix_livemath")
    m = _LABEL_RE.findall(text or "")
    if not m or m[-1].upper() != task["correct_label"]:
        return None
    reason = (text or "").strip().splitlines()
    gist = " ".join(reason[-6:])[:400]
    return {"task_id": task["task_id"],
            "fix_summary": (f"verified correct choice {task['correct_label']!r} "
                            f"(agent chose {rec.get('predicted_label')!r}); "
                            f"key reasoning: {gist}")}


def verified_lessons(dataset: str, lesson_ids: list, tasks_by_id: dict,
                     batch_res: dict, work_root: str) -> tuple[list, dict]:
    """Returns (verified fix records, stats). Only verified lessons enter the
    execution feedback given to the revision prompts."""
    out, tried = [], 0
    for tid in lesson_ids:
        task, rec = tasks_by_id[tid], batch_res.get(tid) or {}
        if "primary" not in rec:
            continue
        tried += 1
        try:
            if dataset == "docvqa":
                fix = verify_docvqa(task, rec)
            elif dataset == "mind2web":
                fix = verify_mind2web(task, rec)
            elif dataset == "livemath":
                fix = verify_livemath(task, rec)
            else:
                fix = verify_ssb(task, rec, work_root)
        except Exception as e:  # noqa: BLE001 — verification failure = no lesson
            fix = None
        if fix:
            out.append(fix)
    return out, {"tried": tried, "verified": len(out)}
