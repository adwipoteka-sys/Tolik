from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass(slots=True)
class BenchmarkFailure:
    capability_id: str
    benchmark_name: str
    scenario_id: str
    failure_reason: str
    confidence: float
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TransferEvaluationReport:
    capability_id: str
    success: bool
    score: float
    confidence: float
    generalization_gap: float
    failure_reason: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SkillArenaReport:
    capability_id: str
    success: bool
    score: float
    confidence: float
    failure_reason: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class NavigationConfidenceReport:
    capability_id: str
    confidence: float
    ambiguity_level: float
    stability_signal: float
    failure_reason: str | None = None
    details: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload['created_at'] = self.created_at.isoformat()
        return payload
