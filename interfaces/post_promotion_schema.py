from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4


@dataclass(slots=True)
class InterfaceShadowRun:
    shadow_run_id: str
    adapter_name: str
    operation: str
    primary_provider: str | None
    shadow_provider: str
    live_status: str
    shadow_status: str
    live_latency_ms: float
    shadow_latency_ms: float
    correctness_pass: bool
    safety_pass: bool
    agreement_score: float | None = None
    comparison_profile: str | None = None
    reasons: list[str] = field(default_factory=list)
    request_summary: dict[str, Any] = field(default_factory=dict)
    live_summary: dict[str, Any] = field(default_factory=dict)
    shadow_summary: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @classmethod
    def new(
        cls,
        *,
        adapter_name: str,
        operation: str,
        primary_provider: str | None,
        shadow_provider: str,
        live_status: str,
        shadow_status: str,
        live_latency_ms: float,
        shadow_latency_ms: float,
        correctness_pass: bool,
        safety_pass: bool,
        agreement_score: float | None,
        comparison_profile: str | None = None,
        reasons: list[str] | None = None,
        request_summary: dict[str, Any] | None = None,
        live_summary: dict[str, Any] | None = None,
        shadow_summary: dict[str, Any] | None = None,
    ) -> "InterfaceShadowRun":
        return cls(
            shadow_run_id=f"shadow_{uuid4().hex[:12]}",
            adapter_name=adapter_name,
            operation=operation,
            primary_provider=primary_provider,
            shadow_provider=shadow_provider,
            live_status=live_status,
            shadow_status=shadow_status,
            live_latency_ms=round(float(live_latency_ms), 3),
            shadow_latency_ms=round(float(shadow_latency_ms), 3),
            correctness_pass=bool(correctness_pass),
            safety_pass=bool(safety_pass),
            agreement_score=None if agreement_score is None else round(float(agreement_score), 4),
            comparison_profile=comparison_profile,
            reasons=list(reasons or []),
            request_summary=dict(request_summary or {}),
            live_summary=dict(live_summary or {}),
            shadow_summary=dict(shadow_summary or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class InterfaceDriftReport:
    drift_report_id: str
    adapter_name: str
    provider: str | None
    sample_count: int
    window_size: int
    correctness_rate: float
    safety_rate: float
    avg_live_latency_ms: float
    baseline_latency_ms: float | None
    avg_agreement_score: float | None
    shadow_comparisons: int
    demotion_triggered: bool
    reasons: list[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @classmethod
    def new(
        cls,
        *,
        adapter_name: str,
        provider: str | None,
        sample_count: int,
        window_size: int,
        correctness_rate: float,
        safety_rate: float,
        avg_live_latency_ms: float,
        baseline_latency_ms: float | None,
        avg_agreement_score: float | None,
        shadow_comparisons: int,
        demotion_triggered: bool,
        reasons: list[str] | None = None,
    ) -> "InterfaceDriftReport":
        return cls(
            drift_report_id=f"drift_{uuid4().hex[:12]}",
            adapter_name=adapter_name,
            provider=provider,
            sample_count=int(sample_count),
            window_size=int(window_size),
            correctness_rate=round(float(correctness_rate), 4),
            safety_rate=round(float(safety_rate), 4),
            avg_live_latency_ms=round(float(avg_live_latency_ms), 3),
            baseline_latency_ms=None if baseline_latency_ms is None else round(float(baseline_latency_ms), 3),
            avg_agreement_score=None if avg_agreement_score is None else round(float(avg_agreement_score), 4),
            shadow_comparisons=int(shadow_comparisons),
            demotion_triggered=bool(demotion_triggered),
            reasons=list(reasons or []),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ProviderDemotionDecision:
    demotion_id: str
    adapter_name: str
    previous_provider: str | None
    fallback_provider: str | None
    action: str
    reasons: list[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @classmethod
    def new(
        cls,
        *,
        adapter_name: str,
        previous_provider: str | None,
        fallback_provider: str | None,
        action: str,
        reasons: list[str] | None = None,
    ) -> "ProviderDemotionDecision":
        return cls(
            demotion_id=f"demote_{uuid4().hex[:12]}",
            adapter_name=adapter_name,
            previous_provider=previous_provider,
            fallback_provider=fallback_provider,
            action=action,
            reasons=list(reasons or []),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
