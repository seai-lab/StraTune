"""Mind2Web benchmark environment (the single evaluator shared by training and test).

The evaluation path in this file is fixed; the environment's version string
(``VERSION``) is recorded in every result.

Design
------
Thin glue over the ``method.environments.mind2web_prompting`` evaluator -- the
step-level logic (teacher forcing, action parsing, EA / AF1 / SSR / SR,
forced_gold fixed failures without an LLM call) lives there and is invoked
through ``Mind2WebTaskEvaluator``.

The prompting module hard-codes its baseline system prompt only inside its
prompt *builders*; the evaluator itself takes an arbitrary
``prompt_builder``. This env supplies a builder that uses the caller's
``system_prompt`` verbatim while keeping the user message identical to the
``build_step_user`` output (task header, previous-actions window of 5,
candidate rendering, "Next action:" suffix). Optimized system prompts can
therefore be substituted without changing the fixed user-message format.

Metrics (task level, teacher-forced): ``ea`` (element accuracy, primary),
``af1`` (action F1), ``ssr`` (step success rate), ``sr`` (task success),
plus ``per_step`` records. Task-macro aggregation across tasks is the
caller's responsibility (mean of per-task values).

Caching: two layers -- the ``ExactJSONLCache`` keyed on
(system, user, model, decoding, artifact_hash=sha256(system_prompt),
deployment fingerprint, seed), and the ``BedrockConverseClient``'s own
request cache when the client was constructed with one.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from method.environments.mind2web_prompting import SYSTEM_BASE_M2W
from method.environments.mind2web_prompting import Mind2WebTaskEvaluator
from method.environments.mind2web_prompting import Prompt
from method.environments.mind2web_prompting import build_step_user
from method.environments.bedrock_client import TARGET_MODEL
from method.environments.bedrock_client import BedrockConverseClient
from method.environments.runtime import ExactJSONLCache
from method.environments.types import Deployment

VERSION = "1.0.0"

_GROUP_ID = "m2w-env::system-prompt"

__all__ = ["M2WEnv", "SYSTEM_BASE_M2W", "VERSION", "task_to_dict"]


def task_to_dict(task: Any) -> dict[str, Any]:
    """Normalize a ``method.environments.datasets.mind2web.Mind2WebTask`` (or dict)
    into the mapping shape the evaluator expects."""
    if isinstance(task, dict):
        return task
    return {
        "annotation_id": task.annotation_id,
        "task": task.task,
        "website": task.website,
        "domain": task.domain,
        "subdomain": task.subdomain,
        "action_reprs": task.action_reprs,
        "steps": task.steps,
    }


class _ConverseStepLLM:
    """Adapter: the evaluator's ``LLM.complete(system, user)`` protocol over the
    Bedrock Converse client with the fixed decoding configuration."""

    class _Response:
        __slots__ = ("text", "input_tokens", "output_tokens", "cached")

        def __init__(self, text: str, input_tokens: int, output_tokens: int, cached: bool):
            self.text = text
            self.input_tokens = input_tokens
            self.output_tokens = output_tokens
            self.cached = cached

    def __init__(
        self,
        client: BedrockConverseClient,
        model: str,
        max_tokens: int,
        seed: int,
    ) -> None:
        self.client = client
        self.model = model
        self.max_tokens = max_tokens
        self.seed = seed

    def complete(self, system: str, user: str) -> "_ConverseStepLLM._Response":
        response = self.client.converse(
            self.model,
            [{"role": "user", "content": [{"text": user}]}],
            system=system,
            max_tokens=self.max_tokens,
            temperature=0.0,
            seed=self.seed,
            tags={"env": "mind2web", "env_version": VERSION},
        )
        usage = response.get("usage", {})
        return self._Response(
            str(response["text"]),
            int(usage.get("inputTokens", 0)),
            int(usage.get("outputTokens", 0)),
            bool(response.get("cached", False)),
        )


class M2WEnv:
    """Teacher-forced Mind2Web evaluation under a caller-supplied system
    prompt (user-message format identical to the baseline evaluator)."""

    def __init__(
        self,
        cache_path: str | Path,
        model: str = TARGET_MODEL,
        max_tokens: int = 16_384,
        seed: int = 42,
        max_step_workers: int = 1,
    ) -> None:
        self.cache_path = str(cache_path)
        self.model = model
        self.max_tokens = max_tokens
        self.seed = seed
        self.max_step_workers = max_step_workers
        self._cache = ExactJSONLCache(self.cache_path)

    @staticmethod
    def system_prompt_hash(system_prompt: str) -> str:
        return hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()

    def _evaluator(
        self,
        system_prompt: str,
        client: BedrockConverseClient,
    ) -> tuple[Mind2WebTaskEvaluator, Deployment]:
        artifact_hash = self.system_prompt_hash(system_prompt)

        def prompt_builder(
            task: dict[str, Any],
            step_index: int,
            deployment: Deployment,
        ) -> Prompt:
            # User message keeps the baseline format; only the system prompt is
            # substituted.
            return Prompt(system_prompt, build_step_user(task, step_index))

        evaluator = Mind2WebTaskEvaluator(
            llm=_ConverseStepLLM(client, self.model, self.max_tokens, self.seed),
            prompt_builder=prompt_builder,
            cache=self._cache,
            artifact_hash=artifact_hash,
            model=self.model,
            decoding_config={
                "max_tokens": self.max_tokens,
                "temperature": 0.0,
            },
            seed=self.seed,
            max_step_workers=self.max_step_workers,
        )
        deployment = Deployment(artifact_hash, _GROUP_ID, ())
        return evaluator, deployment

    def evaluate_task(
        self,
        task: Any,
        system_prompt: str,
        client: BedrockConverseClient,
    ) -> dict:
        """Evaluate one Mind2Web task (all steps, teacher-forced).

        Returns ``{annotation_id, primary(=ea), ea, af1, ssr, sr, per_step,
        costs, env_version}``.
        """
        record = task_to_dict(task)
        evaluator, deployment = self._evaluator(system_prompt, client)
        evaluation = evaluator.evaluate(record, deployment)
        outcome = evaluation.outcome
        per_step = [
            {
                "operation": step.operation,
                "element_id": step.element_id,
                "value": step.value,
                "element_correct": step.element_correct,
                "action_f1": step.action_f1,
                "exact_correct": step.exact_correct,
                "fixed_failure": step.fixed_failure,
                "cache_hit": step.cache_hit,
                "input_tokens": step.input_tokens,
                "output_tokens": step.output_tokens,
                "raw": step.raw,
            }
            for step in evaluation.steps
        ]
        return {
            "annotation_id": evaluation.task_id,
            "primary": outcome.primary,
            "ea": outcome.primary,
            "af1": outcome.guardrails["af1"],
            "ssr": outcome.guardrails["ssr"],
            "sr": outcome.guardrails["sr"],
            "per_step": per_step,
            "costs": dict(outcome.costs),
            "system_prompt_sha256": self.system_prompt_hash(system_prompt),
            "model": self.model,
            "env_version": VERSION,
        }
