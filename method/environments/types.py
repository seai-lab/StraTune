"""Small value objects shared by the task environments."""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class Deployment:
    """Identity of the skill under evaluation (its hash) plus a grouping label."""
    artifact_hash: str
    group_id: str
    component_keys: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.artifact_hash or not self.group_id:
            raise ValueError("deployment identity is required")

    @property
    def fingerprint(self) -> str:
        return canonical_hash({"artifact_hash": self.artifact_hash, "group_id": self.group_id,
                               "component_keys": list(self.component_keys)})


@dataclass(frozen=True)
class Outcome:
    """Primary score plus optional guardrail and cost measurements of one task."""
    primary: float
    guardrails: Mapping[str, float] = field(default_factory=dict)
    costs: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        values = [self.primary, *self.guardrails.values(), *self.costs.values()]
        if not all(math.isfinite(float(v)) for v in values):
            raise ValueError("outcome values must be finite")

    def to_dict(self) -> dict[str, Any]:
        return {"primary": self.primary, "guardrails": dict(sorted(self.guardrails.items())),
                "costs": dict(sorted(self.costs.items()))}
