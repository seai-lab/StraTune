"""LiveMathematicianBench environment. evaluate_task(task, skill_text, client) returns a record with a
float "primary" (answer accuracy), like the other task environments, so training and test share one scorer.

Data is read exclusively from the fixed benchmark payloads under
<benchmark-data-root> (read-only source of truth).
"""
from __future__ import annotations

import os
import re

TARGET_MODEL = os.environ.get("STRATUNE_TARGET_MODEL",
                              "global.anthropic.claude-haiku-4-5-20251001-v1:0")
_STRATUNE_ROOT = os.environ.get("STRATUNE_ROOT") or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
BD = os.environ.get("STRATUNE_BENCHMARK_DATA", os.path.join(_STRATUNE_ROOT, "benchmark_data"))
DATA_ROOT = os.path.join(_STRATUNE_ROOT, "runtime_data")

_ANS_RE = re.compile(r"ANSWER\s*[:：]\s*(.+)", re.IGNORECASE)
_LABEL_RE = re.compile(r"ANSWER\s*[:：]\s*\(?\**([A-E])\**\)?", re.IGNORECASE)


def _extract_answer_line(text: str) -> str | None:
    hits = _ANS_RE.findall(text or "")
    if not hits:
        return None
    return hits[-1].strip().strip("*").strip()


# ---------------------------------------------------------------------------
class LiveMathEnv:
    """Research-math MCQ: five choices, score = label accuracy."""

    def evaluate_task(self, task: dict, skill_text: str, client) -> dict:
        choices = "\n".join(f"{c['label']}. {c['text']}" for c in task["choices"])
        user = (f"{task['question']}\n\nChoices:\n{choices}\n\n"
                "Select the single correct choice.")
        resp = client.converse(
            TARGET_MODEL, [{"role": "user", "content": [{"text": user}]}],
            system=skill_text, max_tokens=6000, temperature=0.0, seed=42,
            tags={"stage": "livemath_eval"})
        text = resp["text"] or ""
        m = _LABEL_RE.findall(text)
        label = m[-1].upper() if m else None
        if label is None:  # fallback: last standalone A-E token near the end
            tail = re.findall(r"\b([A-E])\b", text[-200:])
            label = tail[-1] if tail else None
        parsed = 1.0 if label else 0.0
        primary = 1.0 if (label and label == task["correct_label"]) else 0.0
        return {"task_id": task["task_id"], "primary": primary,
                "predicted_label": label, "parsed": parsed,
                "response_tail": text[-400:]}


# ---------------------------------------------------------------------------
