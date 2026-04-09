from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from automl.response_risk_model import ResponseRiskModel, ResponseRiskTrainingExample
from benchmarks.skill_arena import SkillArenaCase
from benchmarks.transfer_suite import TransferCase
from self_modification.change_schema import new_change_id


@dataclass(slots=True)
class AutoMLSpec:
    change_id: str
    goal_id: str
    title: str
    model_family: str
    target_component: str
    target_attribute: str
    baseline_model: dict[str, Any]
    training_examples: list[ResponseRiskTrainingExample]
    search_space: dict[str, list[float]]
    anchor_cases: list[SkillArenaCase]
    transfer_cases: list[TransferCase]
    canary_cases: list[SkillArenaCase]
    rationale: str
    threshold: float = 0.99
    allowed_regression_delta: float = 0.0
    status: str = "staged"
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["created_at"] = self.created_at.isoformat()
        payload["training_examples"] = [item.to_dict() for item in self.training_examples]
        payload["anchor_cases"] = [case.to_dict() for case in self.anchor_cases]
        payload["transfer_cases"] = [case.to_dict() for case in self.transfer_cases]
        payload["canary_cases"] = [case.to_dict() for case in self.canary_cases]
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AutoMLSpec":
        raw = dict(data)
        raw["created_at"] = datetime.fromisoformat(raw["created_at"])
        raw["training_examples"] = [ResponseRiskTrainingExample.from_dict(item) for item in raw.get("training_examples", [])]
        raw["anchor_cases"] = [SkillArenaCase.from_dict(item) for item in raw.get("anchor_cases", [])]
        raw["transfer_cases"] = [TransferCase.from_dict(item) for item in raw.get("transfer_cases", [])]
        raw["canary_cases"] = [SkillArenaCase.from_dict(item) for item in raw.get("canary_cases", [])]
        return cls(**raw)


@dataclass(slots=True)
class AutoMLTrainingReport:
    change_id: str
    model_family: str
    baseline_model_id: str
    candidate_model: dict[str, Any]
    train_metrics: dict[str, float]
    best_config: dict[str, float]
    leaderboard: list[dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["created_at"] = self.created_at.isoformat()
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AutoMLTrainingReport":
        raw = dict(data)
        raw["created_at"] = datetime.fromisoformat(raw["created_at"])
        return cls(**raw)


@dataclass(slots=True)
class AutoMLRegressionReport:
    change_id: str
    model_family: str
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
    def from_dict(cls, data: dict[str, Any]) -> "AutoMLRegressionReport":
        raw = dict(data)
        raw["created_at"] = datetime.fromisoformat(raw["created_at"])
        return cls(**raw)


@dataclass(slots=True)
class AutoMLCanaryReport:
    change_id: str
    model_family: str
    canary_run: dict[str, Any]
    passed: bool
    rolled_back: bool
    active_before_model_id: str
    active_after_model_id: str
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["created_at"] = self.created_at.isoformat()
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AutoMLCanaryReport":
        raw = dict(data)
        raw["created_at"] = datetime.fromisoformat(raw["created_at"])
        return cls(**raw)


def new_automl_change_id() -> str:
    return new_change_id(prefix="automl")


def model_from_payload(payload: dict[str, Any]) -> ResponseRiskModel:
    family = payload.get("family")
    if family != "response_risk_model":
        raise ValueError(f"Unsupported model family: {family!r}")
    return ResponseRiskModel.from_dict(payload)
