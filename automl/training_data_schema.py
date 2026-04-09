from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from automl.response_risk_model import ResponseRiskTrainingExample
from benchmarks.skill_arena import SkillArenaCase
from benchmarks.transfer_suite import TransferCase

_ALLOWED_SPLITS = {"train", "anchor", "transfer", "canary"}


def new_dataset_example_id(prefix: str = "dsex") -> str:
    return f"{prefix}_{uuid4().hex[:12]}"


def new_dataset_snapshot_id(prefix: str = "dssnap") -> str:
    return f"{prefix}_{uuid4().hex[:12]}"


@dataclass(slots=True)
class CurriculumDatasetExample:
    example_id: str
    model_family: str
    split: str
    source_type: str
    source_signature: str
    description: str
    payload: dict[str, Any]
    target: dict[str, Any]
    tags: list[str] = field(default_factory=list)
    difficulty: float = 0.5
    quality_score: float = 1.0
    lineage: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        if self.split not in _ALLOWED_SPLITS:
            raise ValueError(f"Unsupported dataset split: {self.split}")
        for field_name in ("difficulty", "quality_score"):
            value = float(getattr(self, field_name))
            if not (0.0 <= value <= 1.0):
                raise ValueError(f"{field_name} must be in [0, 1], got {value!r}")

    def fingerprint(self) -> str:
        return json.dumps(
            {
                "model_family": self.model_family,
                "split": self.split,
                "payload": self.payload,
                "target": self.target,
                "tags": sorted(self.tags),
            },
            sort_keys=True,
            ensure_ascii=False,
        )

    def to_training_example(self) -> ResponseRiskTrainingExample:
        if self.split != "train":
            raise ValueError("Only train split examples can be converted into training examples.")
        return ResponseRiskTrainingExample(
            goal=dict(self.payload),
            label=bool(self.target.get("label", False)),
            description=self.description,
        )

    def to_skill_case(self) -> SkillArenaCase:
        if self.split not in {"anchor", "canary"}:
            raise ValueError("Only anchor/canary split examples can be converted into skill-arena cases.")
        return SkillArenaCase(
            case_id=self.example_id,
            payload=dict(self.payload),
            expected=dict(self.target),
            description=self.description,
        )

    def to_transfer_case(self) -> TransferCase:
        if self.split != "transfer":
            raise ValueError("Only transfer split examples can be converted into transfer cases.")
        return TransferCase(
            case_id=self.example_id,
            payload=dict(self.payload),
            expected=dict(self.target),
            description=self.description,
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["created_at"] = self.created_at.isoformat()
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CurriculumDatasetExample":
        raw = dict(data)
        raw["created_at"] = datetime.fromisoformat(raw["created_at"])
        return cls(**raw)


@dataclass(slots=True)
class CurriculumDatasetSnapshot:
    snapshot_id: str
    model_family: str
    title: str
    description: str
    example_ids: list[str]
    status: str = "approved"
    stats: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["created_at"] = self.created_at.isoformat()
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CurriculumDatasetSnapshot":
        raw = dict(data)
        raw["created_at"] = datetime.fromisoformat(raw["created_at"])
        return cls(**raw)
