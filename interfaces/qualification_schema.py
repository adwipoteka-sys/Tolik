from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4


@dataclass(slots=True)
class ProviderQualificationReport:
    qualification_id: str
    adapter_name: str
    provider: str
    total_cases: int
    passed_cases: int
    correctness_rate: float
    safety_rate: float
    avg_latency_ms: float
    score: float
    eligible: bool
    reasons: list[str] = field(default_factory=list)
    case_results: list[dict[str, Any]] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @classmethod
    def new(
        cls,
        *,
        adapter_name: str,
        provider: str,
        total_cases: int,
        passed_cases: int,
        correctness_rate: float,
        safety_rate: float,
        avg_latency_ms: float,
        score: float,
        eligible: bool,
        reasons: list[str] | None = None,
        case_results: list[dict[str, Any]] | None = None,
    ) -> "ProviderQualificationReport":
        return cls(
            qualification_id=f"qual_{uuid4().hex[:12]}",
            adapter_name=adapter_name,
            provider=provider,
            total_cases=total_cases,
            passed_cases=passed_cases,
            correctness_rate=round(float(correctness_rate), 4),
            safety_rate=round(float(safety_rate), 4),
            avg_latency_ms=round(float(avg_latency_ms), 3),
            score=round(float(score), 4),
            eligible=bool(eligible),
            reasons=list(reasons or []),
            case_results=list(case_results or []),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ProviderQualificationReport":
        return cls(**payload)


@dataclass(slots=True)
class ProviderRolloutDecision:
    decision_id: str
    adapter_name: str
    provider: str | None
    decision: str
    mode: str
    reasons: list[str] = field(default_factory=list)
    qualification_id: str | None = None
    rollout_stage: str = "full_live"
    fallback_provider: str | None = None
    canary_live_fraction: float | None = None
    routing_strategy: str | None = None
    estimated_cost_per_call_usd: float | None = None
    fallback_estimated_cost_per_call_usd: float | None = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @classmethod
    def new(
        cls,
        *,
        adapter_name: str,
        provider: str | None,
        decision: str,
        mode: str,
        reasons: list[str] | None = None,
        qualification_id: str | None = None,
        rollout_stage: str = "full_live",
        fallback_provider: str | None = None,
        canary_live_fraction: float | None = None,
        routing_strategy: str | None = None,
        estimated_cost_per_call_usd: float | None = None,
        fallback_estimated_cost_per_call_usd: float | None = None,
    ) -> "ProviderRolloutDecision":
        return cls(
            decision_id=f"rollout_{uuid4().hex[:12]}",
            adapter_name=adapter_name,
            provider=provider,
            decision=decision,
            mode=mode,
            reasons=list(reasons or []),
            qualification_id=qualification_id,
            rollout_stage=str(rollout_stage),
            fallback_provider=fallback_provider,
            canary_live_fraction=None if canary_live_fraction is None else round(float(canary_live_fraction), 4),
            routing_strategy=None if routing_strategy is None else str(routing_strategy),
            estimated_cost_per_call_usd=None if estimated_cost_per_call_usd is None else round(float(estimated_cost_per_call_usd), 6),
            fallback_estimated_cost_per_call_usd=None
            if fallback_estimated_cost_per_call_usd is None
            else round(float(fallback_estimated_cost_per_call_usd), 6),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ProviderRolloutDecision":
        return cls(**payload)
