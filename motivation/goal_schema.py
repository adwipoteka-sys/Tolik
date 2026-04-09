from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal
from uuid import uuid4


class GoalSource(str, Enum):
    USER = "user"
    CURIOSITY = "curiosity"
    DRIFT_ALARM = "drift_alarm"
    MEMORY_GAP = "memory_gap"
    REGRESSION_FAILURE = "regression_failure"
    METACOGNITION = "metacognition"
    CURRICULUM = "curriculum"
    TOOLING_GAP = "tooling_gap"
    SCHEDULER = "scheduler"


class GoalStatus(str, Enum):
    PENDING = "pending"
    ACTIVE = "active"
    BLOCKED = "blocked"
    DONE = "done"
    FAILED = "failed"
    REJECTED = "rejected"
    DEFERRED = "deferred"
    CANCELLED = "cancelled"


class GoalKind(str, Enum):
    USER_TASK = "user_task"
    LEARNING = "learning"
    MAINTENANCE = "maintenance"
    EXPLORATION = "exploration"
    REGRESSION_RECOVERY = "regression_recovery"
    TOOL_CREATION = "tool_creation"


Comparator = Literal[">=", "<=", "==", "contains"]
_ALLOWED_COMPARATORS = {">=", "<=", "==", "contains"}


def new_goal_id(prefix: str = "goal") -> str:
    return f"{prefix}_{uuid4().hex[:12]}"


@dataclass(slots=True)
class SuccessCriterion:
    metric: str
    comparator: Comparator
    target: float | int | str
    window: int = 1

    def __post_init__(self) -> None:
        if self.comparator not in _ALLOWED_COMPARATORS:
            raise ValueError(f"Unsupported comparator: {self.comparator}")
        if self.window <= 0:
            raise ValueError("window must be positive")

    def is_satisfied(self, observed_value: Any) -> bool:
        if self.comparator == "contains":
            return str(self.target) in str(observed_value)
        if self.comparator == "==":
            return observed_value == self.target
        try:
            lhs = float(observed_value)
            rhs = float(self.target)
        except (TypeError, ValueError) as exc:
            raise ValueError("Numeric comparator requires numeric values.") from exc
        if self.comparator == ">=":
            return lhs >= rhs
        if self.comparator == "<=":
            return lhs <= rhs
        raise ValueError(f"Unsupported comparator: {self.comparator}")


@dataclass(slots=True)
class GoalBudget:
    max_steps: int
    max_seconds: float
    max_tokens: int | None = None
    max_tool_calls: int = 0
    max_api_calls: int = 0

    def __post_init__(self) -> None:
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if self.max_seconds <= 0:
            raise ValueError("max_seconds must be positive")
        if self.max_tokens is not None and self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive when provided")
        if self.max_tool_calls < 0:
            raise ValueError("max_tool_calls cannot be negative")
        if self.max_api_calls < 0:
            raise ValueError("max_api_calls cannot be negative")


@dataclass(slots=True)
class Goal:
    goal_id: str
    title: str
    description: str
    source: GoalSource
    kind: GoalKind

    expected_gain: float
    novelty: float
    uncertainty_reduction: float
    strategic_fit: float
    risk_estimate: float

    priority: float
    risk_budget: float
    resource_budget: GoalBudget

    success_criteria: list[SuccessCriterion]
    parent_goal_id: str | None = None
    required_capabilities: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)

    status: GoalStatus = GoalStatus.PENDING
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: datetime | None = None

    def __post_init__(self) -> None:
        if not self.goal_id:
            raise ValueError("goal_id is required")
        if not self.success_criteria:
            raise ValueError("At least one success criterion is required.")
        if self.status == GoalStatus.ACTIVE:
            raise ValueError("Goals must not be created directly in ACTIVE state.")
        for metric_name in (
            "expected_gain",
            "novelty",
            "uncertainty_reduction",
            "strategic_fit",
            "risk_estimate",
            "risk_budget",
        ):
            value = getattr(self, metric_name)
            if not (0.0 <= value <= 1.0):
                raise ValueError(f"{metric_name} must be in [0, 1], got {value!r}.")
        if self.priority < 0.0:
            raise ValueError("priority must be non-negative")

    def touch(self) -> None:
        self.updated_at = datetime.now(timezone.utc)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["source"] = self.source.value
        payload["kind"] = self.kind.value
        payload["status"] = self.status.value
        payload["created_at"] = self.created_at.isoformat()
        payload["updated_at"] = self.updated_at.isoformat()
        payload["expires_at"] = self.expires_at.isoformat() if self.expires_at else None
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Goal":
        raw = dict(data)
        raw["source"] = GoalSource(raw["source"])
        raw["kind"] = GoalKind(raw["kind"])
        restored_status = GoalStatus(raw["status"])
        raw["status"] = GoalStatus.PENDING if restored_status == GoalStatus.ACTIVE else restored_status
        raw["resource_budget"] = GoalBudget(**raw["resource_budget"])
        raw["success_criteria"] = [SuccessCriterion(**criterion) for criterion in raw["success_criteria"]]
        raw["created_at"] = datetime.fromisoformat(raw["created_at"])
        raw["updated_at"] = datetime.fromisoformat(raw["updated_at"])
        raw["expires_at"] = datetime.fromisoformat(raw["expires_at"]) if raw.get("expires_at") else None
        goal = cls(**raw)
        goal.status = restored_status
        return goal
