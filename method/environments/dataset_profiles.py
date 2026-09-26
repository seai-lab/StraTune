"""Dataset-specific constants and task renderings.

Success thresholds and screening-set quotas per dataset, and how tasks,
failures, solved samples, and trajectories are rendered in the optimizer-LLM
prompts. These are the only dataset-specific parts of StraTune.
"""
import math
import re

from method.config import _truncate


# ---------------------------------------------------------------------------
# Per-dataset constants (success thresholds, screening-set quotas) and the
# task renderings used in optimizer-LLM prompts. These are the only
# dataset-specific settings of StraTune.
# impact_quota: screening-set quotas for samples related to the failures,
# previously solved samples, and randomly sampled samples.
PROFILES = {
    "docvqa": {
        "eps0": 0.001,
        "success_threshold": 0.90,   # a sample counts as solved when ANLS >= 0.90
        "binary": False,
        "impact_quota": {"related": 128, "regression": 96, "random": 32},
        "guardrail_field": "exact_match",
        "eval_workers": 128,
    },
    "mind2web": {
        "eps0": 0.0025,
        "success_threshold": 0.80,   # task-macro EA >= 0.80
        "binary": False,
        "impact_quota": {"related": 64, "regression": 48, "random": 16},
        "guardrail_field": "af1",
        "eval_workers": 24,
    },
    "spreadsheetbench": {
        "eps0": 0.0025,
        "success_threshold": 1.0,    # hard_pass == 1
        "binary": True,
        "impact_quota": {"related": 32, "regression": 32, "random": 16},
        "guardrail_field": "cell_accuracy",
        "eval_workers": 12,
    },
    "livemath": {
        "eps0": 0.0025,
        "success_threshold": 1.0,    # label accuracy, binary
        "binary": True,
        "impact_quota": {"related": 24, "regression": 24, "random": 12},
        "guardrail_field": "parsed",
        "eval_workers": 24,
    },
}


def parent_success(dataset: str, primary: float) -> bool:
    p = PROFILES[dataset]
    return primary >= p["success_threshold"] - 1e-9


def severe_regression(dataset: str, parent_primary: float, cand_primary: float,
                      cand_error: bool) -> bool:
    """Severe regression: on a binary metric the parent solves the sample and the
    candidate does not; on a continuous metric the score drops by at least 0.10;
    or the candidate fails to execute where the parent executed."""
    if cand_error:
        return True
    p = PROFILES[dataset]
    if p["binary"] or parent_success(dataset, parent_primary):
        if parent_success(dataset, parent_primary) and not parent_success(dataset, cand_primary):
            return True
    if not p["binary"] and (parent_primary - cand_primary) >= 0.10 - 1e-9:
        return True
    return False


def severe_improvement(dataset: str, parent_primary: float, cand_primary: float) -> bool:
    """Mirror of severe_regression: the candidate solves a sample the parent did
    not, or gains at least 0.10 on a continuous metric."""
    p = PROFILES[dataset]
    if not parent_success(dataset, parent_primary) and parent_success(dataset, cand_primary):
        return True
    if not p["binary"] and (cand_primary - parent_primary) >= 0.10 - 1e-9:
        return True
    return False


def gold_len(dataset: str, task: dict) -> int | None:
    """Length of the sample's gold answer, or None where the notion does not
    apply. Used to choose contrasting solved samples with long gold answers for
    failure clusters about over-shortened answers."""
    if dataset == "docvqa":
        golds = task.get("answers") or []
        if golds:
            return max(len(str(g)) for g in golds)
        return None
    if dataset == "mind2web":
        reprs = (task.get("record") or {}).get("action_reprs") or []
        return len(reprs) or None
    return None


TRIM_THEME_WORDS = frozenset((
    "trim", "trimming", "shorter", "short", "shorten", "minimal", "minimality",
    "concise", "brevity", "bare", "strip", "stripping", "remove", "removing",
    "drop", "dropping", "omit", "omitting", "truncate", "truncation",
    "extraneous", "redundant", "verbose", "verbosity", "wrapper", "wrapping",
    "qualifier", "qualifiers", "prefix", "suffix", "over-specification"))


def is_trim_theme(cluster: dict) -> bool:
    """True when a failure cluster is about answers being too long; such clusters
    are shown together with contrasting solved samples whose gold answers are
    long, so that the proposed rule keeps its boundary."""
    text = ((cluster.get("common_theme") or "") + " "
            + (cluster.get("description") or "")).lower()
    words = set(re.findall(r"[a-z-]+", text))
    return bool(words & TRIM_THEME_WORDS)


def demo_pair(dataset: str, task: dict) -> str | None:
    """Render a training (input -> gold) pair as a one-line worked example for
    revision form F2, or None when the dataset's gold answer is not a short
    string (Mind2Web, SpreadsheetBench)."""
    if dataset == "docvqa":
        golds = task.get("answers") or []
        if golds:
            return f"Q: {str(task['question'])[:160]} -> A: {str(golds[0])[:80]}"
    elif dataset == "livemath":
        ch = {c.get("label"): c.get("text") for c in task.get("choices", [])}
        gl = task["correct_label"]
        return (f"Q: {str(task['question'])[:180]}... -> "
                f"ANSWER: {gl}: {str(ch.get(gl))[:80]}")
    return None


def protocol_hint(dataset: str) -> str | None:
    """Dataset description used when generating an answer-derivation procedure
    (revision form F3). None for the step-structured environments (Mind2Web,
    SpreadsheetBench), where this form is not generated."""
    if dataset == "docvqa":
        return ("extraction task over a document image; ANLS-scored short answer "
                "inside <answer></answer> tags")
    if dataset == "livemath":
        return ("research-math five-choice MCQ; label-accuracy-scored "
                "'ANSWER: <letter>' line")
    return None


def traj_repr(dataset: str, task: dict, rec: dict) -> str | None:
    """Compact rendering of a solved Mind2Web trajectory for the workflow form
    (revision form F3). None for datasets without step structure, where the
    form is not generated."""
    if dataset != "mind2web":
        return None
    steps = rec.get("per_step") or []
    if not steps:
        return None
    r = task.get("record") or {}
    # Gold action_reprs carry the exact UI text of each step; fall back to
    # operation names when they are absent.
    reprs = r.get("action_reprs") or []
    if reprs:
        lines = []
        for i, ar in enumerate(reprs[:10]):
            ok = "+" if (i < len(steps) and steps[i].get("element_correct")) else "-"
            lines.append(f"{ok}{str(ar)[:70]}")
        return (f"[{r.get('website', '?')}/{r.get('domain', '?')}] "
                f"{str(r.get('task', ''))[:110]}\n    " + " | ".join(lines))
    ops = []
    for s in steps[:12]:
        mark = "+" if s.get("element_correct") else "-"
        val = str(s.get("value") or "")[:20]
        ops.append(f"{mark}{s.get('operation')}" + (f"({val})" if val else ""))
    return (f"[{r.get('website', '?')}/{r.get('domain', '?')}] "
            f"{str(r.get('task', ''))[:110]} | steps: " + " -> ".join(ops))


def task_text(dataset: str, task: dict) -> str:
    """Input-side text used to retrieve samples related to the failures for the
    screening set; scores are never used."""
    if dataset == "docvqa":
        return task["question"]
    if dataset == "mind2web":
        r = task["record"]
        return f"{r.get('website','')} {r.get('domain','')} {r.get('task','')}"
    if dataset == "livemath":
        return task["question"]
    return task["task_obj"].instruction


def failure_summary(dataset: str, task: dict, rec: dict) -> str:
    """Compact per-sample failure record for the revision
    prompts."""
    if "error" in rec and "primary" not in rec:
        return f"[{task['task_id']}] EXECUTION ERROR: {_truncate(rec['error'], 200)}"
    if dataset == "docvqa":
        # The tail of the response is the optimizer LLM's only view of how the
        # target LLM read the document image.
        return (f"[{task['task_id']}] Q: {_truncate(task['question'], 150)} | "
                f"predicted: {_truncate(rec.get('predicted_answer'), 80)!r} | "
                f"gold: {_truncate(task.get('answers'), 80)} | "
                f"ANLS={rec.get('anls', 0):.2f} raw={rec.get('anls_raw', 0):.2f} | "
                f"model reading tail: {_truncate((rec.get('response_text') or '')[-260:], 200)!r}")
    if dataset == "mind2web":
        # Show the chosen and gold element texts for each wrong step. Steps whose
        # gold element is absent from the candidates cannot be fixed by the skill
        # and are only counted.
        steps = (task.get("record") or {}).get("steps") or []
        wrong, unfix = [], 0
        for i, ps in enumerate(rec.get("per_step", [])):
            if ps.get("element_correct"):
                continue
            if ps.get("fixed_failure"):
                unfix += 1
                continue
            gold_t = chose_t = None
            if i < len(steps):
                cands = {str(c.get("id") or c.get("backend_node_id")): c
                         for c in (steps[i].get("candidates") or [])}
                gids = [str(g) for g in (steps[i].get("gold_ids") or [])]
                gc = next((cands[g] for g in gids if g in cands), None)
                cc = cands.get(str(ps.get("element_id")))
                gold_t = _truncate((gc or {}).get("repr") or (gc or {}).get("text"), 70)
                chose_t = _truncate((cc or {}).get("repr") or (cc or {}).get("text"), 70)
            wrong.append(f"s{i} {ps.get('operation')}: chose {chose_t!r} vs gold {gold_t!r}")
        return (f"[{task['task_id']}] {task['record'].get('website')}: "
                f"{_truncate(task['record'].get('task'), 110)} | EA={rec.get('ea', 0):.2f} | "
                + ("; ".join(wrong[:4]) or "(no fixable wrong-element steps)")
                + (f" | +{unfix} forced-failure steps (gold absent, unfixable)" if unfix else ""))
    if dataset == "livemath":
        # Show the texts of the predicted and gold choices, not only their labels.
        ch = {c.get("label"): c.get("text") for c in task.get("choices", [])}
        pl = rec.get("predicted_label")
        return (f"[{task['task_id']}] Q: {_truncate(task['question'], 180)} | "
                f"predicted {pl!r}: {_truncate(ch.get(pl), 110)!r} | "
                f"gold {task['correct_label']!r}: "
                f"{_truncate(ch.get(task['correct_label']), 110)!r} | "
                f"reasoning tail: {_truncate(rec.get('response_tail'), 110)}")
    errs = []
    for cr in rec.get("case_results", []):
        for e in (cr.get("score_errors") or [])[:2]:
            errs.append(_truncate(e, 130))
        if cr.get("exec_error"):
            errs.append("EXEC: " + _truncate(cr["exec_error"], 130))
    code = _truncate(rec.get("code"), 260)
    ca = rec.get("cell_accuracy")
    return (f"[{task['task_id']}] ({task['task_obj'].instruction_type}) "
            f"{_truncate(task['task_obj'].instruction, 180)} | "
            f"answer_position: {getattr(task['task_obj'], 'answer_position', '')} | "
            f"hard_pass=0 cell_acc={ca if ca is not None else '?'} | "
            f"errors: {' | '.join(errs[:4]) or 'output values mismatched'}"
            + (f" | failing code tail: ```{code}```" if code else ""))


def success_summary(dataset: str, task: dict, rec: dict) -> str:
    if dataset == "docvqa":
        return (f"[{task['task_id']}] Q: {_truncate(task['question'], 160)} | "
                f"answered {_truncate(rec.get('predicted_answer'), 90)!r} correctly (ANLS={rec.get('anls',0):.2f})")
    if dataset == "mind2web":
        return (f"[{task['task_id']}] {task['record'].get('website')}: "
                f"{_truncate(task['record'].get('task'), 120)} | EA={rec.get('ea', 0):.2f}")
    if dataset == "livemath":
        ch = {c.get("label"): c.get("text") for c in task.get("choices", [])}
        pl = rec.get("predicted_label")
        return (f"[{task['task_id']}] Q: {_truncate(task['question'], 160)} | "
                f"answered {pl!r}: {_truncate(ch.get(pl), 100)!r} correctly")
    return (f"[{task['task_id']}] ({task['task_obj'].instruction_type}) "
            f"{_truncate(task['task_obj'].instruction, 160)} | hard_pass=1")


# ---------------------------------------------------------------------------
# Task representation: everything dataset-specific the method consumes is
# declared here or in the re-exported dataset_profiles renderers. The task shape
# below is built from the declared representation fields only.
# Declared task-representation facts (input kind / answer form), same status
# as the failure_summary renderers: representation, not behavior.
SHAPE = {
    "docvqa": {"input": "a document image plus a question about it",
               "answer_form": "a short string inside <answer></answer> tags"},
    "mind2web": {"input": "a website task solved step by step; each step ranks "
                          "candidate page elements",
                 "answer_form": "per step: one element choice + operation (+ value)"},
    "spreadsheetbench": {"input": "a spreadsheet-manipulation instruction plus "
                                  "workbook file(s)",
                         "answer_form": "python code that writes LITERAL computed "
                                        "values (formulas evaluate to None and fail)"},
    "livemath": {"input": "a research-mathematics five-choice question (A-E)",
                 "answer_form": "an 'ANSWER: <letter>' line"},
}


def task_shape(ds: str) -> str:
    p, s = PROFILES[ds], SHAPE[ds]
    kind = "binary" if p["binary"] else "continuous"
    return (f"input: {s['input']}; answer form: {s['answer_form']}; "
            f"scored 0..1 ({kind} primary metric, task counts as solved at "
            f">= {p['success_threshold']})")


def categorical_space(ds: str, task: dict):
    """For tasks whose answer space is an enumerated per-task choice set,
    return (options [(key, text)], gold_key); None otherwise. Used only when
    the dataset configuration enables the training-distribution statistics
    block (global_stats)."""
    if ds == "livemath":
        return ([(c["label"], c["text"]) for c in task["choices"]],
                task["correct_label"])
    return None


# ---- set sizes as functions of |D_train| (no per-dataset tables) ----------------
# screening set Q_s: 18 previously solved samples + 18 randomly sampled (stratified)
EXAM_QUOTA = {"related": 0, "regression": 18, "random": 18}


def confirm_n(n_train: int) -> int:
    return max(100, min(300, round(0.3 * n_train)))


def refine_val_n(n_train: int) -> int:
    return min(64, max(32, n_train // 12))


def reserve(n_train: int) -> int:
    return min(1200, math.ceil(1.5 * n_train))


def tourn_floor(n_train: int) -> float:
    return max(0.010, 2.0 / n_train)
