from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4


class ExperimentKind(str, Enum):
    POLICY_CHANGE = "policy_change"
    MODEL_UPGRADE = "model_upgrade"


class ExperimentStatus(str, Enum):
    PROPOSED = "proposed"
    BLOCKED = "blocked"
    SELECTED = "selected"
    MATERIALIZED = "materialized"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


def new_experiment_proposal_id(prefix: str = "exp") -> str:
    return f"{prefix}_{uuid4().hex[:12]}"


@dataclass(slots=True)
class ExperimentProposal:
    proposal_id: str
    experiment_kind: ExperimentKind
    source_signature: str
    title: str
    description: str
    expected_utility: float
    estimated_risk: float
    estimated_cost: float
    confidence: float
    curriculum_signals: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    required_capabilities: list[str] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)
    status: ExperimentStatus = ExperimentStatus.PROPOSED
    cooldown_until_cycle: int | None = None
    selected_in_cycle: int | None = None
    materialized_goal_id: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        for field_name in ("expected_utility", "estimated_risk", "estimated_cost", "confidence"):
            value = float(getattr(self, field_name))
            if not (0.0 <= value <= 1.0):
                raise ValueError(f"{field_name} must be in [0, 1], got {value!r}")

    def touch(self) -> None:
        self.updated_at = datetime.now(timezone.utc)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["experiment_kind"] = self.experiment_kind.value
        payload["status"] = self.status.value
        payload["created_at"] = self.created_at.isoformat()
        payload["updated_at"] = self.updated_at.isoformat()
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExperimentProposal":
        raw = dict(data)
        raw["experiment_kind"] = ExperimentKind(raw["experiment_kind"])
        raw["status"] = ExperimentStatus(raw["status"])
        raw["created_at"] = datetime.fromisoformat(raw["created_at"])
        raw["updated_at"] = datetime.fromisoformat(raw["updated_at"])
        return cls(**raw)


@dataclass(slots=True)
class ExperimentAssessment:
    assessment_id: str
    proposal_id: str
    experiment_kind: ExperimentKind
    source_signature: str
    curriculum_alignment: float
    strategic_fit: float
    urgency: float
    expected_utility: float
    estimated_risk: float
    estimated_cost: float
    confidence: float
    composite_score: float
    admissible: bool
    rationale: str
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["experiment_kind"] = self.experiment_kind.value
        payload["created_at"] = self.created_at.isoformat()
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExperimentAssessment":
        raw = dict(data)
        raw["experiment_kind"] = ExperimentKind(raw["experiment_kind"])
        raw["created_at"] = datetime.fromisoformat(raw["created_at"])
        return cls(**raw)
