"""DocVQA benchmark environment (the single evaluator shared by training and test).

The scoring semantics and prompt construction in this file are fixed; the
environment's version string (``VERSION``) is recorded in every result.

Protocol
--------
One target-LLM multimodal call per task: the supplied ``system_prompt`` is
the system message; the user message is the question text (with the fixed
``<answer>...</answer>`` output instruction, identical wording to the
SkillOpt reference rollout) followed by the document image as a PNG content
block. The response is scored with the official ANLS metric.

ANLS semantics (matches the SkillOpt reference evaluator, which itself
matches the official DocVQA evaluation kit):

* prediction and every gold answer are normalized: ``str().strip().lower()``
  and internal whitespace collapsed to single spaces;
* per gold answer, NLD = levenshtein / max(len(pred), len(gold));
  score = 0.0 if NLD >= 0.5 (threshold edge INCLUSIVE: similarity of exactly
  0.5 scores 0.0), else 1 - NLD;
* both empty -> 1.0; exactly one empty -> 0.0;
* per-task ``anls`` = max over the gold-answer list.

Additional fields not in the reference (reported alongside, never primary):
``anls_raw`` = max over golds of (1 - NLD) with NO threshold, and
``exact_match`` = normalized prediction equals some normalized gold.

Differences vs. the SkillOpt reference harness:

* transport is Bedrock Converse with a raw PNG image block instead of an
  OpenAI-style base64 data URI (no ``detail`` parameter exists);
* the system prompt is entirely caller-supplied; the reference wraps an
  optimized skill inside its own ``rollout_system.md`` template;
* answer extraction uses the same logic as the reference
  (``<answer>`` rfind window, else last non-empty line).
"""

from __future__ import annotations

import ast
import io
import json
from collections.abc import Iterable
from typing import Any

from method.environments.bedrock_client import TARGET_MODEL
from method.environments.bedrock_client import BedrockConverseClient
from method.environments.bedrock_client import image_block_png
from method.environments.bedrock_client import text_block

VERSION = "1.0.0"

ANLS_THRESHOLD = 0.5

# Fixed user-message suffix, identical wording to the SkillOpt rollout.
ANSWER_INSTRUCTION = "Return the final answer inside <answer>...</answer>."

# Bedrock Converse hard limits for Anthropic image blocks (with margin).
_MAX_IMAGE_BYTES = 3_500_000
_MAX_IMAGE_DIM = 7_500


# ---------------------------------------------------------------------------
# ANLS scoring (reference-faithful)
# ---------------------------------------------------------------------------

def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip().lower()
    return " ".join(text.split())


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    if len(a) > len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, start=1):
        current = [i]
        for j, char_b in enumerate(b, start=1):
            insert_cost = current[j - 1] + 1
            delete_cost = previous[j] + 1
            replace_cost = previous[j - 1] + (char_a != char_b)
            current.append(min(insert_cost, delete_cost, replace_cost))
        previous = current
    return previous[-1]


def anls_pair_raw(predicted: Any, target: Any) -> float:
    """Similarity 1 - NLD with NO threshold (empty-handling as reference)."""
    predicted_norm = normalize_text(predicted)
    target_norm = normalize_text(target)
    if not predicted_norm and not target_norm:
        return 1.0
    if not predicted_norm or not target_norm:
        return 0.0
    distance = levenshtein(predicted_norm, target_norm)
    return 1.0 - distance / max(len(predicted_norm), len(target_norm))


def anls_pair(predicted: Any, target: Any, threshold: float = ANLS_THRESHOLD) -> float:
    """Official thresholded ANLS for one (prediction, gold) pair.

    score = 0.0 when NLD >= threshold (similarity <= 1 - threshold), else
    1 - NLD. With the default threshold 0.5 a similarity of exactly 0.5
    scores 0.0 (matches the reference ``normalized_distance >= threshold``).
    """
    raw = anls_pair_raw(predicted, target)
    # raw == 1 - NLD for non-empty pairs; empty-pair conventions (1.0 / 0.0)
    # pass through unchanged because 1.0 > threshold-complement and 0.0 fails.
    if (1.0 - raw) >= threshold:
        return 0.0
    return raw


def extract_answer_strings(raw: Any) -> list[str]:
    """Gold-answer list extraction, same logic as the reference."""
    if raw is None:
        return [""]
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return [""]
        parsed = None
        if text[0] in "[{":
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                try:
                    parsed = ast.literal_eval(text)
                except (ValueError, SyntaxError):
                    parsed = None
        if parsed is None:
            return [text]
        return extract_answer_strings(parsed)
    if isinstance(raw, dict):
        for key in ("answers", "ground_truth", "answer"):
            if key in raw:
                return extract_answer_strings(raw[key])
        return [str(raw)]
    if isinstance(raw, Iterable) and not isinstance(raw, (bytes, bytearray)):
        answers: list[str] = []
        for item in raw:
            if isinstance(item, dict):
                for key in ("text", "answer", "value"):
                    if key in item:
                        answers.extend(extract_answer_strings(item[key]))
                        break
                else:
                    answers.append(str(item))
                continue
            answers.append(str(item))
        return answers or [""]
    return [str(raw)]


def extract_answer(text: str) -> str:
    """Prediction extraction, same logic as the reference."""
    lower = text.lower()
    start = lower.rfind("<answer>")
    end = lower.rfind("</answer>")
    if start != -1 and end != -1 and end > start:
        return text[start + len("<answer>"):end].strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else text.strip()


def score_answer(prediction_text: str, gold_answers: Any) -> dict:
    """Score a raw model response against the gold answer list.

    Returns {anls, anls_raw, exact_match, predicted_answer, gold_answers}.
    ``anls`` is the official thresholded per-task score (primary metric).
    """
    answer = extract_answer(prediction_text)
    answers = extract_answer_strings(gold_answers)
    anls = 0.0
    anls_raw = 0.0
    exact = False
    answer_norm = normalize_text(answer)
    for target in answers:
        anls = max(anls, anls_pair(answer, target))
        anls_raw = max(anls_raw, anls_pair_raw(answer, target))
        if answer_norm == normalize_text(target) and answer_norm:
            exact = True
        elif not answer_norm and not normalize_text(target):
            exact = True
    return {
        "anls": anls,
        "anls_raw": anls_raw,
        "exact_match": exact,
        "predicted_answer": answer,
        "gold_answers": answers,
    }


# ---------------------------------------------------------------------------
# Image guard (deterministic downscale to fit Bedrock Converse limits)
# ---------------------------------------------------------------------------

def fit_png_to_limits(
    png_bytes: bytes,
    max_bytes: int = _MAX_IMAGE_BYTES,
    max_dim: int = _MAX_IMAGE_DIM,
) -> bytes:
    """Deterministically shrink an image until it fits API limits.

    Rule: decode with PIL; while encoded PNG exceeds ``max_bytes``
    or either dimension exceeds ``max_dim``, resize by a factor of 0.75
    (LANCZOS) and re-encode as PNG. Returns the original bytes untouched
    when already within limits and already PNG.
    """
    from PIL import Image

    image = Image.open(io.BytesIO(png_bytes))
    is_png = (image.format or "").upper() == "PNG"
    if (
        is_png
        and len(png_bytes) <= max_bytes
        and image.width <= max_dim
        and image.height <= max_dim
    ):
        return png_bytes

    current = image.convert("RGB") if image.mode not in ("RGB", "L", "RGBA") else image
    while True:
        if current.width > max_dim or current.height > max_dim:
            scale = max_dim / max(current.width, current.height)
            current = current.resize(
                (max(1, int(current.width * scale)), max(1, int(current.height * scale))),
                Image.LANCZOS,
            )
        buffer = io.BytesIO()
        current.save(buffer, format="PNG")
        encoded = buffer.getvalue()
        if len(encoded) <= max_bytes:
            return encoded
        current = current.resize(
            (max(1, int(current.width * 0.75)), max(1, int(current.height * 0.75))),
            Image.LANCZOS,
        )


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class DocVQAEnv:
    """Single-call DocVQA evaluation environment.

    ``evaluate_task(task, system_prompt, client)`` performs exactly one
    multimodal Converse call with the fixed decoding configuration and
    returns the per-task score record (primary metric: ``anls``).
    """

    def __init__(
        self,
        model: str = TARGET_MODEL,
        max_tokens: int = 16_384,
        seed: int = 42,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.seed = seed

    def build_user_text(self, question: str) -> str:
        return f"{question}\n\n{ANSWER_INSTRUCTION}"

    def evaluate_task(
        self,
        task: Any,
        system_prompt: str,
        client: BedrockConverseClient,
    ) -> dict:
        """Evaluate one DocVQA task under ``system_prompt``.

        ``task`` is a ``method.environments.datasets.docvqa.DocVQATask`` (or any object /
        mapping exposing ``qid``, ``question``, ``answers`` and PNG bytes via
        ``image_bytes()`` or an ``image_png`` mapping key).
        """
        if isinstance(task, dict):
            qid = task["qid"]
            question = task["question"]
            answers = task["answers"]
            png = task["image_png"]
        else:
            qid = task.qid
            question = task.question
            answers = task.answers
            png = task.image_bytes()

        png = fit_png_to_limits(bytes(png))
        user_text = self.build_user_text(str(question))
        message = {
            "role": "user",
            "content": [text_block(user_text), image_block_png(png)],
        }
        response = client.converse(
            self.model,
            [message],
            system=system_prompt,
            max_tokens=self.max_tokens,
            temperature=0.0,
            seed=self.seed,
            tags={"env": "docvqa", "env_version": VERSION, "qid": str(qid)},
        )
        scores = score_answer(response["text"], answers)
        usage = response.get("usage", {})
        return {
            "qid": qid,
            "primary": scores["anls"],
            "anls": scores["anls"],
            "anls_raw": scores["anls_raw"],
            "exact_match": scores["exact_match"],
            "predicted_answer": scores["predicted_answer"],
            "gold_answers": scores["gold_answers"],
            "response_text": response["text"],
            "stop_reason": response.get("stop_reason"),
            "cached": response.get("cached", False),
            "usage": {
                "input_tokens": int(usage.get("inputTokens", 0)),
                "output_tokens": int(usage.get("outputTokens", 0)),
            },
            "model": self.model,
            "env_version": VERSION,
        }
