from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4


@dataclass(slots=True)
class InterfaceCanarySample:
    sample_id: str
    adapter_name: str
    candidate_provider: str
    routed_provider: str
    reference_provider: str | None
    rollout_stage: str
    candidate_status: str
    reference_status: str | None
    candidate_latency_ms: float
    reference_latency_ms: float | None
    correctness_pass: bool
    safety_pass: bool
    agreement_score: float | None
    comparison_profile: str | None = None
    consensus_provider: str | None = None
    consensus_support: int = 0
    consensus_strength: float | None = None
    reasons: list[str] = field(default_factory=list)
    request_summary: dict[str, Any] = field(default_factory=dict)
    candidate_summary: dict[str, Any] = field(default_factory=dict)
    reference_summary: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @classmethod
    def new(
        cls,
        *,
        adapter_name: str,
        candidate_provider: str,
        routed_provider: str,
        reference_provider: str | None,
        rollout_stage: str,
        candidate_status: str,
        reference_status: str | None,
        candidate_latency_ms: float,
        reference_latency_ms: float | None,
        correctness_pass: bool,
        safety_pass: bool,
        agreement_score: float | None,
        comparison_profile: str | None = None,
        consensus_provider: str | None = None,
        consensus_support: int = 0,
        consensus_strength: float | None = None,
        reasons: list[str] | None = None,
        request_summary: dict[str, Any] | None = None,
        candidate_summary: dict[str, Any] | None = None,
        reference_summary: dict[str, Any] | None = None,
    ) -> "InterfaceCanarySample":
        return cls(
            sample_id=f"canarysample_{uuid4().hex[:12]}",
            adapter_name=adapter_name,
            candidate_provider=candidate_provider,
            routed_provider=routed_provider,
            reference_provider=reference_provider,
            rollout_stage=rollout_stage,
            candidate_status=candidate_status,
            reference_status=reference_status,
            candidate_latency_ms=round(float(candidate_latency_ms), 3),
            reference_latency_ms=None if reference_latency_ms is None else round(float(reference_latency_ms), 3),
            correctness_pass=bool(correctness_pass),
            safety_pass=bool(safety_pass),
            agreement_score=None if agreement_score is None else round(float(agreement_score), 4),
            comparison_profile=comparison_profile,
            consensus_provider=consensus_provider,
            consensus_support=int(consensus_support),
            consensus_strength=None if consensus_strength is None else round(float(consensus_strength), 4),
            reasons=list(reasons or []),
            request_summary=dict(request_summary or {}),
            candidate_summary=dict(candidate_summary or {}),
            reference_summary=dict(reference_summary or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "InterfaceCanarySample":
        return cls(**payload)


@dataclass(slots=True)
class ContinuousQualificationReport:
    report_id: str
    adapter_name: str
    candidate_provider: str
    fallback_provider: str | None
    window_size: int
    sample_count: int
    live_canary_count: int
    shadow_only_count: int
    correctness_rate: float
    safety_rate: float
    avg_candidate_latency_ms: float
    baseline_latency_ms: float | None
    avg_agreement_score: float | None
    eligible: bool
    reasons: list[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @classmethod
    def new(
        cls,
        *,
        adapter_name: str,
        candidate_provider: str,
        fallback_provider: str | None,
        window_size: int,
        sample_count: int,
        live_canary_count: int,
        shadow_only_count: int,
        correctness_rate: float,
        safety_rate: float,
        avg_candidate_latency_ms: float,
        baseline_latency_ms: float | None,
        avg_agreement_score: float | None,
        eligible: bool,
        reasons: list[str] | None = None,
    ) -> "ContinuousQualificationReport":
        return cls(
            report_id=f"requal_{uuid4().hex[:12]}",
            adapter_name=adapter_name,
            candidate_provider=candidate_provider,
            fallback_provider=fallback_provider,
            window_size=int(window_size),
            sample_count=int(sample_count),
            live_canary_count=int(live_canary_count),
            shadow_only_count=int(shadow_only_count),
            correctness_rate=round(float(correctness_rate), 4),
            safety_rate=round(float(safety_rate), 4),
            avg_candidate_latency_ms=round(float(avg_candidate_latency_ms), 3),
            baseline_latency_ms=None if baseline_latency_ms is None else round(float(baseline_latency_ms), 3),
            avg_agreement_score=None if avg_agreement_score is None else round(float(avg_agreement_score), 4),
            eligible=bool(eligible),
            reasons=list(reasons or []),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ContinuousQualificationReport":
        return cls(**payload)


@dataclass(slots=True)
class CanaryPromotionDecision:
    decision_id: str
    adapter_name: str
    candidate_provider: str
    fallback_provider: str | None
    action: str
    sample_count: int
    reasons: list[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @classmethod
    def new(
        cls,
        *,
        adapter_name: str,
        candidate_provider: str,
        fallback_provider: str | None,
        action: str,
        sample_count: int,
        reasons: list[str] | None = None,
    ) -> "CanaryPromotionDecision":
        return cls(
            decision_id=f"canarydecision_{uuid4().hex[:12]}",
            adapter_name=adapter_name,
            candidate_provider=candidate_provider,
            fallback_provider=fallback_provider,
            action=action,
            sample_count=int(sample_count),
            reasons=list(reasons or []),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CanaryPromotionDecision":
        return cls(**payload)
