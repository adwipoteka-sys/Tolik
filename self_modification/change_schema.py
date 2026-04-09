from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from benchmarks.skill_arena import SkillArenaCase
from benchmarks.transfer_suite import TransferCase


def new_change_id(prefix: str = "selfmod") -> str:
    return f"{prefix}_{uuid4().hex[:12]}"


@dataclass(slots=True)
class SelfModificationSpec:
    change_id: str
    goal_id: str
    title: str
    target_component: str
    capability: str
    parameter_name: str
    baseline_value: Any
    candidate_value: Any
    rationale: str
    anchor_cases: list[SkillArenaCase]
    transfer_cases: list[TransferCase]
    canary_cases: list[SkillArenaCase]
    threshold: float = 0.99
    allowed_regression_delta: float = 0.0
    status: str = "staged"
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["created_at"] = self.created_at.isoformat()
        payload["anchor_cases"] = [case.to_dict() for case in self.anchor_cases]
        payload["transfer_cases"] = [case.to_dict() for case in self.transfer_cases]
        payload["canary_cases"] = [case.to_dict() for case in self.canary_cases]
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SelfModificationSpec":
        raw = dict(data)
        raw["created_at"] = datetime.fromisoformat(raw["created_at"])
        raw["anchor_cases"] = [SkillArenaCase.from_dict(item) for item in raw.get("anchor_cases", [])]
        raw["transfer_cases"] = [TransferCase.from_dict(item) for item in raw.get("transfer_cases", [])]
        raw["canary_cases"] = [SkillArenaCase.from_dict(item) for item in raw.get("canary_cases", [])]
        return cls(**raw)


@dataclass(slots=True)
class RegressionGateReport:
    change_id: str
    capability: str
    baseline_anchor_run: dict[str, Any]
    candidate_anchor_run: dict[str, Any]
    candidate_transfer_run: dict[str, Any]
    passed: bool
    regression_case_ids: list[str] = field(default_factory=list)
    failure_reasons: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["created_at"] = self.created_at.isoformat()
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RegressionGateReport":
        raw = dict(data)
        raw["created_at"] = datetime.fromisoformat(raw["created_at"])
        return cls(**raw)


@dataclass(slots=True)
class SelfModificationCanaryReport:
    change_id: str
    capability: str
    canary_run: dict[str, Any]
    passed: bool
    rolled_back: bool
    active_before: Any
    restored_value: Any
    active_value_after: Any
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["created_at"] = self.created_at.isoformat()
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SelfModificationCanaryReport":
        raw = dict(data)
        raw["created_at"] = datetime.fromisoformat(raw["created_at"])
        return cls(**raw)
