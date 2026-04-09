from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from benchmarks.skill_arena import SkillArenaCase
from benchmarks.transfer_suite import TransferCase


def new_proposal_id(prefix: str = "selfmodprop") -> str:
    return f"{prefix}_{uuid4().hex[:12]}"


@dataclass(slots=True)
class SelfModificationProposal:
    proposal_id: str
    signature: str
    capability: str
    title: str
    description: str
    target_component: str
    parameter_name: str
    baseline_value: Any
    candidate_value: Any
    rationale: str
    confidence: float
    failure_support: int
    supporting_episode_ids: list[str] = field(default_factory=list)
    supporting_goal_ids: list[str] = field(default_factory=list)
    supporting_root_causes: list[str] = field(default_factory=list)
    supporting_postmortem_ids: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    anchor_cases: list[SkillArenaCase] = field(default_factory=list)
    transfer_cases: list[TransferCase] = field(default_factory=list)
    canary_cases: list[SkillArenaCase] = field(default_factory=list)
    threshold: float = 0.99
    allowed_regression_delta: float = 0.0
    status: str = "proposed"
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def touch(self) -> None:
        self.updated_at = datetime.now(timezone.utc)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["created_at"] = self.created_at.isoformat()
        payload["updated_at"] = self.updated_at.isoformat()
        payload["anchor_cases"] = [case.to_dict() for case in self.anchor_cases]
        payload["transfer_cases"] = [case.to_dict() for case in self.transfer_cases]
        payload["canary_cases"] = [case.to_dict() for case in self.canary_cases]
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SelfModificationProposal":
        raw = dict(data)
        raw["created_at"] = datetime.fromisoformat(raw["created_at"])
        raw["updated_at"] = datetime.fromisoformat(raw["updated_at"])
        raw["anchor_cases"] = [SkillArenaCase.from_dict(item) for item in raw.get("anchor_cases", [])]
        raw["transfer_cases"] = [TransferCase.from_dict(item) for item in raw.get("transfer_cases", [])]
        raw["canary_cases"] = [SkillArenaCase.from_dict(item) for item in raw.get("canary_cases", [])]
        return cls(**raw)
