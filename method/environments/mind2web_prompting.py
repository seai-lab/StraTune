"""Exact-prompt task-macro Mind2Web evaluator."""

from __future__ import annotations

import re
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Protocol

from method.environments.runtime import CacheRequest
from method.environments.runtime import ExactJSONLCache
from method.environments.types import Deployment
from method.environments.types import Outcome

SYSTEM_BASE_M2W = (
    "You are an expert web-navigation agent. Given a task, the actions taken so "
    "far, and a numbered list of candidate page elements, choose the single next "
    "action.\nAction space:\n"
    "1. CLICK [id]\n2. TYPE [id] [value]\n3. SELECT [id] [value]\n"
    "End your response with a line of the exact form:\n"
    "ACTION: `OP [id] [value]`   (value only for TYPE/SELECT)"
)
_ACTION = re.compile(
    r"(CLICK|TYPE|SELECT)\s*\[\s*([A-Za-z0-9_-]+)\s*\]\s*(?:\[(.*?)\])?",
    re.IGNORECASE | re.DOTALL,
)


class LLM(Protocol):
    def complete(self, system: str, user: str) -> Any: ...


@dataclass(frozen=True)
class Prompt:
    system: str
    user: str


@dataclass(frozen=True)
class StepPrediction:
    operation: str | None
    element_id: str | None
    value: str | None
    raw: str
    element_correct: bool
    action_f1: float
    exact_correct: bool
    fixed_failure: bool
    input_tokens: int
    output_tokens: int
    prompt_characters: int
    cache_hit: bool


@dataclass(frozen=True)
class TaskEvaluation:
    task_id: str
    deployment_hash: str
    outcome: Outcome
    steps: tuple[StepPrediction, ...]


PromptBuilder = Callable[[dict[str, Any], int, Deployment], Prompt]


class Mind2WebTaskEvaluator:
    def __init__(
        self,
        *,
        llm: LLM,
        prompt_builder: PromptBuilder,
        cache: ExactJSONLCache,
        artifact_hash: str,
        model: str,
        decoding_config: dict[str, Any],
        seed: int,
        max_step_workers: int = 1,
    ) -> None:
        if max_step_workers < 1:
            raise ValueError("max_step_workers must be positive")
        self.llm = llm
        self.prompt_builder = prompt_builder
        self.cache = cache
        self.artifact_hash = artifact_hash
        self.model = model
        self.decoding_config = dict(decoding_config)
        self.seed = seed
        self.max_step_workers = max_step_workers

    def __call__(self, task: dict[str, Any], deployment: Deployment) -> Outcome:
        return self.evaluate(task, deployment).outcome

    def evaluate(self, task: dict[str, Any], deployment: Deployment) -> TaskEvaluation:
        indices = list(range(len(task.get("steps", []))))
        if self.max_step_workers == 1:
            predictions = [self._step(task, index, deployment) for index in indices]
        else:
            by_index = {}
            with ThreadPoolExecutor(
                max_workers=min(self.max_step_workers, max(1, len(indices)))
            ) as pool:
                futures = {
                    pool.submit(self._step, task, index, deployment): index
                    for index in indices
                }
                for future in as_completed(futures):
                    by_index[futures[future]] = future.result()
            predictions = [by_index[index] for index in indices]

        count = len(predictions)
        ea = sum(row.element_correct for row in predictions) / count if count else 0.0
        af1 = sum(row.action_f1 for row in predictions) / count if count else 0.0
        ssr = sum(row.exact_correct for row in predictions) / count if count else 0.0
        sr = float(bool(predictions) and all(row.exact_correct for row in predictions))
        scoreable_calls = sum(not row.fixed_failure for row in predictions)
        outcome = Outcome(
            primary=ea,
            guardrails={"af1": af1, "ssr": ssr, "sr": sr},
            costs={
                "target_calls": float(scoreable_calls),
                "input_tokens": float(sum(row.input_tokens for row in predictions)),
                "output_tokens": float(sum(row.output_tokens for row in predictions)),
                "prompt_characters": float(
                    sum(row.prompt_characters for row in predictions)
                ),
            },
        )
        return TaskEvaluation(
            str(task["annotation_id"]),
            deployment.fingerprint,
            outcome,
            tuple(predictions),
        )

    def _step(
        self,
        task: dict[str, Any],
        step_index: int,
        deployment: Deployment,
    ) -> StepPrediction:
        step = task["steps"][step_index]
        if step.get("forced_gold") or step.get("scoreable") is False:
            return StepPrediction(
                None,
                None,
                None,
                "CANDIDATE_RECALL_FAILURE",
                False,
                0.0,
                False,
                True,
                0,
                0,
                0,
                False,
            )
        prompt = self.prompt_builder(task, step_index, deployment)
        request = CacheRequest(
            system_bytes=prompt.system.encode(),
            user_bytes=prompt.user.encode(),
            model=self.model,
            decoding_config=self.decoding_config,
            artifact_hash=self.artifact_hash,
            deployment_hash=deployment.fingerprint,
            seed=self.seed,
        )
        cached = self.cache.lookup(request)
        if cached is None:
            response = self.llm.complete(prompt.system, prompt.user)
            payload = {
                "text": str(response.text),
                "input_tokens": int(getattr(response, "input_tokens", 0)),
                "output_tokens": int(getattr(response, "output_tokens", 0)),
            }
            self.cache.put(request, payload)
            cache_hit = False
        else:
            payload = cached.response
            cache_hit = True
        prediction = parse_action(str(payload["text"]))
        element_ok = prediction[1] in set(step.get("gold_ids", []))
        exact_ok = step_correct(prediction, step)
        return StepPrediction(
            prediction[0],
            prediction[1],
            prediction[2],
            str(payload["text"]),
            element_ok,
            action_f1(prediction[0], prediction[2], step),
            exact_ok,
            False,
            int(payload.get("input_tokens", 0)),
            int(payload.get("output_tokens", 0)),
            len(prompt.system) + len(prompt.user),
            cache_hit,
        )


def render_candidates(step: dict[str, Any]) -> str:
    lines = []
    for candidate in step.get("candidates", []):
        text = str(candidate.get("text") or "").strip()
        extra = f" {candidate['extra']}" if candidate.get("extra") else ""
        lines.append(
            f"[{candidate['id']}] <{candidate['tag']}>{extra} {text}".rstrip()
        )
    return "\n".join(lines)


def build_step_user(task: dict[str, Any], step_index: int) -> str:
    step = task["steps"][step_index]
    previous = task.get("action_reprs", [])[: step.get("i", step_index)][-5:]
    previous_block = "\n".join(
        f"{index + 1}. {action}" for index, action in enumerate(previous)
    ) or "None"
    return (
        f"Task: {task['task']}\n"
        f"Website: {task['website']} ({task['domain']})\n\n"
        f"Previous actions:\n{previous_block}\n\n"
        f"Candidate elements:\n{render_candidates(step)}\n\n"
        "Next action:"
    )


def parse_action(text: str) -> tuple[str | None, str | None, str | None]:
    scopes = []
    scopes.extend(reversed(re.findall(r"ACTION:\s*(.+)", text, flags=re.IGNORECASE)))
    scopes.extend(reversed(re.findall(r"`([^`]*)`", text)))
    scopes.append(text)
    for scope in scopes:
        matches = _ACTION.findall(scope)
        if matches:
            operation, element_id, value = matches[-1]
            return operation.upper(), element_id, (value or "").strip().strip('"\'')
    return None, None, None


def step_correct(
    prediction: tuple[str | None, str | None, str | None],
    step: dict[str, Any],
) -> bool:
    operation, element_id, value = prediction
    if operation is None or element_id not in set(step.get("gold_ids", [])):
        return False
    if operation != str(step.get("op", "")).upper():
        return False
    if operation in {"TYPE", "SELECT"}:
        return _normalize(value) == _normalize(str(step.get("value", "")))
    return True


def action_f1(operation: str | None, value: str | None, step: dict[str, Any]) -> float:
    predicted = _action_text(operation, value)
    gold = _action_text(str(step.get("op", "")), str(step.get("value", "")))
    predicted_tokens = set(predicted.strip().split())
    gold_tokens = set(gold.strip().split())
    if not predicted_tokens and not gold_tokens:
        return 1.0
    if not predicted_tokens or not gold_tokens:
        return 0.0
    overlap = len(predicted_tokens & gold_tokens)
    precision = overlap / len(predicted_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall) if overlap else 0.0


def _action_text(operation: str | None, value: str | None) -> str:
    if operation is None:
        return " " if value is None else f" {value}"
    operation = operation.upper()
    return f"{operation} " if operation == "CLICK" or value is None else f"{operation} {value}"


def _normalize(value: str | None) -> str:
    return " ".join((value or "").lower().split())
