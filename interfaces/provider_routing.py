from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable
from uuid import uuid4

from interfaces.adapter_schema import InterfaceRuntimeSpec
from interfaces.qualification_schema import ProviderQualificationReport
from motivation.operator_charter import OperatorCharter


@dataclass(slots=True)
class RoutedProviderCandidate:
    provider: str
    report: ProviderQualificationReport | None
    spec: InterfaceRuntimeSpec
    transport: Callable[..., dict[str, Any]] | None = None

    @property
    def qualification_score(self) -> float:
        if self.report is None:
            return 0.0
        return float(self.report.score)

    @property
    def correctness_rate(self) -> float:
        if self.report is None:
            return 0.0
        return float(self.report.correctness_rate)

    @property
    def safety_rate(self) -> float:
        if self.report is None:
            return 0.0
        return float(self.report.safety_rate)

    @property
    def avg_latency_ms(self) -> float:
        if self.report is None:
            return 0.0
        return float(self.report.avg_latency_ms)

    @property
    def estimated_cost_per_call_usd(self) -> float | None:
        value = self.spec.estimated_cost_per_call_usd
        if value is None:
            return None
        return round(float(value), 6)


@dataclass(slots=True)
class ProviderRoutingDecisionRecord:
    routing_id: str
    adapter_name: str
    role: str
    strategy: str
    primary_provider: str | None
    selected_provider: str | None
    selected_routing_score: float | None
    selected_estimated_cost_per_call_usd: float | None
    quality_anchor_score: float | None
    provider_routing_max_score_gap: float
    quality_weight: float
    cost_weight: float
    candidate_rankings: list[dict[str, Any]] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @classmethod
    def new(
        cls,
        *,
        adapter_name: str,
        role: str,
        strategy: str,
        primary_provider: str | None,
        selected_provider: str | None,
        selected_routing_score: float | None,
        selected_estimated_cost_per_call_usd: float | None,
        quality_anchor_score: float | None,
        provider_routing_max_score_gap: float,
        quality_weight: float,
        cost_weight: float,
        candidate_rankings: list[dict[str, Any]] | None = None,
    ) -> "ProviderRoutingDecisionRecord":
        return cls(
            routing_id=f"routing_{uuid4().hex[:12]}",
            adapter_name=adapter_name,
            role=role,
            strategy=str(strategy),
            primary_provider=primary_provider,
            selected_provider=selected_provider,
            selected_routing_score=None if selected_routing_score is None else round(float(selected_routing_score), 4),
            selected_estimated_cost_per_call_usd=None
            if selected_estimated_cost_per_call_usd is None
            else round(float(selected_estimated_cost_per_call_usd), 6),
            quality_anchor_score=None if quality_anchor_score is None else round(float(quality_anchor_score), 4),
            provider_routing_max_score_gap=round(float(provider_routing_max_score_gap), 4),
            quality_weight=round(float(quality_weight), 4),
            cost_weight=round(float(cost_weight), 4),
            candidate_rankings=list(candidate_rankings or []),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ProviderRoutingSelection:
    selected: RoutedProviderCandidate | None
    decision: ProviderRoutingDecisionRecord


class CostAwareFallbackRouter:
    """Select a fallback provider by balancing qualification score and per-call cost."""

    def __init__(self, *, adapter_name: str, charter: OperatorCharter, ledger: Any | None = None) -> None:
        self.adapter_name = adapter_name
        self.charter = charter
        self.ledger = ledger

    def select(
        self,
        *,
        role: str,
        primary_provider: str | None,
        candidates: list[RoutedProviderCandidate],
        quality_anchor_score: float | None = None,
    ) -> ProviderRoutingSelection:
        if not candidates:
            decision = ProviderRoutingDecisionRecord.new(
                adapter_name=self.adapter_name,
                role=role,
                strategy="cost_aware_balance",
                primary_provider=primary_provider,
                selected_provider=None,
                selected_routing_score=None,
                selected_estimated_cost_per_call_usd=None,
                quality_anchor_score=quality_anchor_score,
                provider_routing_max_score_gap=self.charter.provider_routing_max_score_gap,
                quality_weight=self.charter.provider_routing_quality_weight,
                cost_weight=self.charter.provider_routing_cost_weight,
                candidate_rankings=[],
            )
            self._persist(decision)
            return ProviderRoutingSelection(selected=None, decision=decision)

        if quality_anchor_score is None:
            quality_anchor_score = max(candidate.qualification_score for candidate in candidates)
        candidates = sorted(
            candidates,
            key=lambda item: (item.qualification_score, item.correctness_rate, item.safety_rate, -item.avg_latency_ms),
            reverse=True,
        )
        known_costs = [candidate.estimated_cost_per_call_usd for candidate in candidates if candidate.estimated_cost_per_call_usd is not None]
        max_known_cost = max(known_costs) if known_costs else None

        pool: list[RoutedProviderCandidate] = []
        rankings: list[dict[str, Any]] = []
        for candidate in candidates:
            score_gap = max(0.0, float(quality_anchor_score) - candidate.qualification_score)
            within_gap = score_gap <= float(self.charter.provider_routing_max_score_gap)
            pool_eligible = within_gap if self.charter.enable_cost_aware_fallback_routing else True
            rankings.append(
                {
                    "provider": candidate.provider,
                    "qualification_score": round(candidate.qualification_score, 4),
                    "correctness_rate": round(candidate.correctness_rate, 4),
                    "safety_rate": round(candidate.safety_rate, 4),
                    "avg_latency_ms": round(candidate.avg_latency_ms, 3),
                    "estimated_cost_per_call_usd": candidate.estimated_cost_per_call_usd,
                    "score_gap": round(score_gap, 4),
                    "pool_eligible": pool_eligible,
                }
            )
            if pool_eligible:
                pool.append(candidate)

        if not pool:
            pool = [candidates[0]]
            rankings[0]["pool_eligible"] = True
            rankings[0]["fallback_reason"] = "score_gap_override"

        selected: RoutedProviderCandidate
        strategy = "cost_aware_balance"
        if not self.charter.enable_cost_aware_fallback_routing or len(pool) == 1:
            selected = pool[0]
            for item in rankings:
                if item["provider"] == selected.provider:
                    item["routing_score"] = round(selected.qualification_score, 4)
            if not self.charter.enable_cost_aware_fallback_routing:
                strategy = "quality_only"
            else:
                strategy = "single_candidate"
            routing_score = selected.qualification_score
        else:
            resolved_costs = [
                (max_known_cost if candidate.estimated_cost_per_call_usd is None else candidate.estimated_cost_per_call_usd)
                for candidate in pool
            ]
            resolved_costs = [0.0 if value is None else float(value) for value in resolved_costs]
            max_cost = max(resolved_costs) if resolved_costs else 0.0
            min_cost = min(resolved_costs) if resolved_costs else 0.0
            weight_sum = float(self.charter.provider_routing_quality_weight + self.charter.provider_routing_cost_weight)
            quality_weight = float(self.charter.provider_routing_quality_weight) / weight_sum
            cost_weight = float(self.charter.provider_routing_cost_weight) / weight_sum
            scored_pool: list[tuple[float, RoutedProviderCandidate, float]] = []
            for candidate, resolved_cost in zip(pool, resolved_costs):
                if max_cost == min_cost:
                    cost_score = 1.0
                else:
                    cost_score = 1.0 - ((resolved_cost - min_cost) / (max_cost - min_cost))
                routing_score = (quality_weight * candidate.qualification_score) + (cost_weight * cost_score)
                scored_pool.append((routing_score, candidate, cost_score))
                for item in rankings:
                    if item["provider"] == candidate.provider:
                        item["resolved_cost_per_call_usd"] = round(resolved_cost, 6)
                        item["cost_score"] = round(cost_score, 4)
                        item["routing_score"] = round(routing_score, 4)
            scored_pool.sort(
                key=lambda item: (
                    item[0],
                    item[1].qualification_score,
                    -item[1].avg_latency_ms,
                    -(item[1].estimated_cost_per_call_usd or max_cost),
                ),
                reverse=True,
            )
            routing_score, selected, _ = scored_pool[0]

        decision = ProviderRoutingDecisionRecord.new(
            adapter_name=self.adapter_name,
            role=role,
            strategy=strategy,
            primary_provider=primary_provider,
            selected_provider=selected.provider,
            selected_routing_score=routing_score,
            selected_estimated_cost_per_call_usd=selected.estimated_cost_per_call_usd,
            quality_anchor_score=quality_anchor_score,
            provider_routing_max_score_gap=self.charter.provider_routing_max_score_gap,
            quality_weight=self.charter.provider_routing_quality_weight,
            cost_weight=self.charter.provider_routing_cost_weight,
            candidate_rankings=rankings,
        )
        self._persist(decision)
        return ProviderRoutingSelection(selected=selected, decision=decision)

    def _persist(self, decision: ProviderRoutingDecisionRecord) -> None:
        if self.ledger is not None and hasattr(self.ledger, "save_interface_provider_routing"):
            self.ledger.save_interface_provider_routing(decision.to_dict())
