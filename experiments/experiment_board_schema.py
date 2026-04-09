from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

from experiments.experiment_schema import ExperimentKind, ExperimentProposal


class ExperimentCampaignStatus(str, Enum):
    DISCOVERED = "discovered"
    QUEUED = "queued"
    DEFERRED = "deferred"
    SELECTED = "selected"
    MATERIALIZED = "materialized"
    COMPLETED = "completed"
    FAILED = "failed"
    BUDGET_EXHAUSTED = "budget_exhausted"


@dataclass(slots=True)
class ExperimentCampaign:
    campaign_id: str
    experiment_kind: ExperimentKind
    source_signature: str
    title: str
    description: str
    tags: list[str]
    required_capabilities: list[str]
    status: ExperimentCampaignStatus = ExperimentCampaignStatus.DISCOVERED
    proposal_ids: list[str] = field(default_factory=list)
    assessment_ids: list[str] = field(default_factory=list)
    latest_proposal: dict[str, Any] = field(default_factory=dict)
    max_total_cost: float = 1.0
    max_total_risk: float = 0.30
    max_attempts: int = 3
    spent_cost: float = 0.0
    spent_risk: float = 0.0
    attempt_count: int = 0
    selection_count: int = 0
    defer_until_cycle: int | None = None
    defer_reason: str | None = None
    last_selected_cycle: int | None = None
    materialized_goal_ids: list[str] = field(default_factory=list)
    last_composite_score: float | None = None
    last_rationale: str | None = None
    evidence: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        if self.max_attempts <= 0:
            raise ValueError("max_attempts must be positive")
        if self.max_total_cost < 0.0 or self.max_total_risk < 0.0:
            raise ValueError("Campaign budgets must be non-negative")
        if self.spent_cost < 0.0 or self.spent_risk < 0.0:
            raise ValueError("Spent budgets must be non-negative")
        if self.attempt_count < 0 or self.selection_count < 0:
            raise ValueError("Counters must be non-negative")

    def touch(self) -> None:
        self.updated_at = datetime.now(timezone.utc)

    @property
    def remaining_cost_budget(self) -> float:
        return round(max(0.0, self.max_total_cost - self.spent_cost), 3)

    @property
    def remaining_risk_budget(self) -> float:
        return round(max(0.0, self.max_total_risk - self.spent_risk), 3)

    def latest_proposal_object(self) -> ExperimentProposal | None:
        if not self.latest_proposal:
            return None
        return ExperimentProposal.from_dict(dict(self.latest_proposal))

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["experiment_kind"] = self.experiment_kind.value
        payload["status"] = self.status.value
        payload["created_at"] = self.created_at.isoformat()
        payload["updated_at"] = self.updated_at.isoformat()
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExperimentCampaign":
        raw = dict(data)
        raw["experiment_kind"] = ExperimentKind(raw["experiment_kind"])
        raw["status"] = ExperimentCampaignStatus(raw["status"])
        raw["created_at"] = datetime.fromisoformat(raw["created_at"])
        raw["updated_at"] = datetime.fromisoformat(raw["updated_at"])
        return cls(**raw)


@dataclass(slots=True)
class ExperimentCycleBudgetSnapshot:
    cycle: int
    reserved_cost: float = 0.0
    reserved_risk: float = 0.0
    selected_campaign_ids: list[str] = field(default_factory=list)
    deferred_campaign_ids: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        if self.cycle <= 0:
            raise ValueError("cycle must be positive")
        if self.reserved_cost < 0.0 or self.reserved_risk < 0.0:
            raise ValueError("reserved budgets must be non-negative")

    def touch(self) -> None:
        self.updated_at = datetime.now(timezone.utc)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["created_at"] = self.created_at.isoformat()
        payload["updated_at"] = self.updated_at.isoformat()
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExperimentCycleBudgetSnapshot":
        raw = dict(data)
        raw["created_at"] = datetime.fromisoformat(raw["created_at"])
        raw["updated_at"] = datetime.fromisoformat(raw["updated_at"])
        return cls(**raw)


def new_experiment_campaign_id(prefix: str = "campaign") -> str:
    return f"{prefix}_{uuid4().hex[:12]}"
