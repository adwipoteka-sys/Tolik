from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable

from interfaces.adapter_schema import AdapterSafetyPolicy, InterfaceRuntimeSpec
from interfaces.canary_schema import CanaryPromotionDecision, ContinuousQualificationReport, InterfaceCanarySample
from interfaces.post_promotion_monitor import PostPromotionShadowMonitor
from interfaces.rollout_protection import RolloutProtectionAdvisor
from interfaces.provider_registry import build_cloud_transport, build_quantum_transport
from interfaces.qualification_schema import ProviderQualificationReport, ProviderRolloutDecision
from interfaces.shadow_consensus import ShadowConsensusEvaluation, ShadowConsensusScorer
from motivation.operator_charter import OperatorCharter


@dataclass(slots=True)
class _CanaryProvider:
    provider: str
    spec: InterfaceRuntimeSpec
    report: ProviderQualificationReport | None
    transport: Callable[..., dict[str, Any]]


class CanaryRolloutController:
    """Percent-based canary router with rolling re-qualification on real traffic."""

    def __init__(
        self,
        *,
        adapter_name: str,
        adapter: Any,
        candidate: _CanaryProvider,
        fallback: _CanaryProvider,
        catalog_candidates: dict[str, InterfaceRuntimeSpec],
        reports: list[ProviderQualificationReport],
        charter: OperatorCharter,
        ledger: Any | None = None,
    ) -> None:
        self.adapter_name = adapter_name
        self.adapter = adapter
        self.candidate = candidate
        self.fallback = fallback
        self.catalog_candidates = dict(catalog_candidates)
        self.reports = list(reports)
        self.charter = charter
        self.ledger = ledger
        self.sample_counter = 0
        self.canary_live_count = 0
        self.window: deque[dict[str, Any]] = deque(maxlen=charter.canary_window_size)
        self.last_requalification_report: ContinuousQualificationReport | None = None
        self.last_decision: CanaryPromotionDecision | None = None
        self.last_shadow_consensus: ShadowConsensusEvaluation | None = None
        self.completed = False
        self.baseline_latency_ms = candidate.report.avg_latency_ms if candidate.report is not None else None
        self.extra_reference_providers: list[_CanaryProvider] = self._build_extra_reference_providers()

    def summary(self) -> dict[str, Any]:
        return {
            "adapter_name": self.adapter_name,
            "rollout_stage": "canary_active" if not self.completed else "canary_resolved",
            "candidate_provider": self.candidate.provider,
            "fallback_provider": self.fallback.provider,
            "extra_reference_providers": [provider.provider for provider in self.extra_reference_providers],
            "canary_live_fraction": self.charter.canary_live_fraction,
            "sample_count": len(self.window),
            "live_canary_count": self.canary_live_count,
            "last_shadow_consensus": None if self.last_shadow_consensus is None else self.last_shadow_consensus.to_dict(),
            "last_requalification_report": None
            if self.last_requalification_report is None
            else self.last_requalification_report.to_dict(),
            "last_decision": None if self.last_decision is None else self.last_decision.to_dict(),
        }

    def select_route(self) -> dict[str, Any]:
        self.sample_counter += 1
        desired_canary_count = math.ceil(self.sample_counter * float(self.charter.canary_live_fraction))
        use_canary = self.canary_live_count < desired_canary_count
        chosen = self.candidate if use_canary else self.fallback
        if use_canary:
            self.canary_live_count += 1
        return {
            "provider": chosen.provider,
            "transport": chosen.transport,
            "policy": chosen.spec.policy,
            "rollout_stage": "canary_live" if use_canary else "stable_live",
        }

    def observe(
        self,
        *,
        routed_provider: str,
        routed_stage: str,
        operation: str,
        request_summary: dict[str, Any],
        live_result: dict[str, Any],
        live_latency_ms: float,
        shadow_runner: Callable[[str, Callable[..., dict[str, Any]], AdapterSafetyPolicy], tuple[dict[str, Any], float]],
    ) -> dict[str, Any]:
        if routed_provider == self.candidate.provider:
            candidate_result = live_result
            candidate_latency_ms = live_latency_ms
            candidate_rollout_stage = "canary_live"
            reference_candidates = [self._run_reference(self.fallback, shadow_runner)]
        else:
            candidate_result, candidate_latency_ms = shadow_runner(
                self.candidate.provider,
                self.candidate.transport,
                self.candidate.spec.policy,
            )
            candidate_rollout_stage = "stable_shadow"
            reference_candidates = [{
                "provider": self.fallback.provider,
                "result": live_result,
                "latency_ms": live_latency_ms,
            }]

        for provider in self.extra_reference_providers[: max(0, self.charter.shadow_candidate_limit - 1)]:
            if provider.provider in {self.candidate.provider, self.fallback.provider}:
                continue
            reference_candidates.append(self._run_reference(provider, shadow_runner))

        scorer = self._build_scorer(operation=operation, request_summary=request_summary)
        consensus = scorer.evaluate(
            live_provider=self.candidate.provider,
            live_result=candidate_result,
            shadow_candidates=reference_candidates,
            live_summary=self._summarize_result(candidate_result),
        )
        self.last_shadow_consensus = consensus
        if self.ledger is not None and hasattr(self.ledger, "save_interface_shadow_consensus"):
            self.ledger.save_interface_shadow_consensus(consensus.to_dict())

        reference_provider = consensus.consensus_provider if consensus.consensus_provider else self.fallback.provider
        reference_summary = consensus.consensus_summary if consensus.consensus_summary else self._summarize_result(reference_candidates[0]["result"]) if reference_candidates else {}
        reference_status = reference_candidates[0]["result"].get("status") if reference_candidates else None
        reference_latency_ms = reference_candidates[0].get("latency_ms") if reference_candidates else None

        sample = InterfaceCanarySample.new(
            adapter_name=self.adapter_name,
            candidate_provider=self.candidate.provider,
            routed_provider=routed_provider,
            reference_provider=reference_provider,
            rollout_stage=candidate_rollout_stage,
            candidate_status=str(candidate_result.get("status", "unknown")),
            reference_status=None if reference_status is None else str(reference_status),
            candidate_latency_ms=candidate_latency_ms,
            reference_latency_ms=None if reference_latency_ms is None else float(reference_latency_ms),
            correctness_pass=bool(consensus.correctness_pass),
            safety_pass=bool(candidate_result.get("status") == "ok"),
            agreement_score=consensus.live_agreement_score,
            comparison_profile=consensus.comparison_profile,
            consensus_provider=consensus.consensus_provider,
            consensus_support=consensus.consensus_support,
            consensus_strength=consensus.consensus_strength,
            reasons=list(consensus.reasons),
            request_summary=request_summary,
            candidate_summary=self._summarize_result(candidate_result),
            reference_summary=reference_summary,
        )
        if self.ledger is not None and hasattr(self.ledger, "save_interface_canary_sample"):
            self.ledger.save_interface_canary_sample(sample.to_dict())

        self.window.append(
            {
                "correctness_pass": bool(consensus.correctness_pass),
                "safety_pass": bool(candidate_result.get("status") == "ok"),
                "candidate_latency_ms": float(candidate_latency_ms),
                "agreement_score": consensus.live_agreement_score,
                "rollout_stage": candidate_rollout_stage,
            }
        )
        self.last_requalification_report = self._build_requalification_report()
        if self.ledger is not None and hasattr(self.ledger, "save_interface_requalification_report"):
            self.ledger.save_interface_requalification_report(self.last_requalification_report.to_dict())
        decision = self._maybe_finalize()
        return {
            "routed_provider": routed_provider,
            "routed_stage": routed_stage,
            "sample": sample.to_dict(),
            "shadow_consensus": consensus.to_dict(),
            "requalification_report": self.last_requalification_report.to_dict(),
            "decision": None if decision is None else decision.to_dict(),
        }

    def _run_reference(
        self,
        provider: _CanaryProvider,
        shadow_runner: Callable[[str, Callable[..., dict[str, Any]], AdapterSafetyPolicy], tuple[dict[str, Any], float]],
    ) -> dict[str, Any]:
        result, latency_ms = shadow_runner(provider.provider, provider.transport, provider.spec.policy)
        return {"provider": provider.provider, "result": result, "latency_ms": latency_ms}

    def _build_scorer(self, *, operation: str, request_summary: dict[str, Any]) -> ShadowConsensusScorer:
        return ShadowConsensusScorer(
            adapter_name=self.adapter_name,
            request_summary=request_summary,
            operation=operation,
            live_threshold_base=self.charter.canary_min_text_agreement,
            min_support=self.charter.shadow_consensus_min_support,
            pairwise_min_agreement=self.charter.shadow_consensus_pairwise_min_agreement,
        )

    def _build_requalification_report(self) -> ContinuousQualificationReport:
        sample_count = len(self.window)
        correctness_rate = (sum(1 for sample in self.window if sample["correctness_pass"]) / sample_count) if sample_count else 0.0
        safety_rate = (sum(1 for sample in self.window if sample["safety_pass"]) / sample_count) if sample_count else 0.0
        avg_latency_ms = (
            sum(float(sample["candidate_latency_ms"]) for sample in self.window) / sample_count if sample_count else 0.0
        )
        agreement_values = [float(sample["agreement_score"]) for sample in self.window if sample["agreement_score"] is not None]
        avg_agreement_score = (sum(agreement_values) / len(agreement_values)) if agreement_values else None
        live_canary_count = sum(1 for sample in self.window if sample["rollout_stage"] == "canary_live")
        shadow_only_count = sample_count - live_canary_count
        reasons: list[str] = []
        eligible = False
        if sample_count >= self.charter.canary_min_samples:
            if correctness_rate < self.charter.canary_min_correctness_rate:
                reasons.append(f"correctness_below_threshold:{correctness_rate:.3f}")
            if safety_rate < self.charter.canary_min_safety_rate:
                reasons.append(f"safety_below_threshold:{safety_rate:.3f}")
            latency_limit = self._current_latency_limit()
            if avg_latency_ms > latency_limit:
                reasons.append(f"latency_above_threshold:{avg_latency_ms:.3f}>{latency_limit:.3f}")
            eligible = not reasons
        return ContinuousQualificationReport.new(
            adapter_name=self.adapter_name,
            candidate_provider=self.candidate.provider,
            fallback_provider=self.fallback.provider,
            window_size=self.charter.canary_window_size,
            sample_count=sample_count,
            live_canary_count=live_canary_count,
            shadow_only_count=shadow_only_count,
            correctness_rate=correctness_rate,
            safety_rate=safety_rate,
            avg_candidate_latency_ms=avg_latency_ms,
            baseline_latency_ms=self.baseline_latency_ms,
            avg_agreement_score=avg_agreement_score,
            eligible=eligible,
            reasons=reasons,
        )

    def _current_latency_limit(self) -> float:
        if self.baseline_latency_ms is None or self.baseline_latency_ms <= 0.0:
            return float(self.charter.provider_rollout_max_avg_latency_ms)
        return max(
            float(self.charter.provider_rollout_max_avg_latency_ms),
            float(self.baseline_latency_ms) * float(self.charter.canary_max_latency_multiplier),
        )

    def _maybe_finalize(self) -> CanaryPromotionDecision | None:
        if self.completed or self.last_requalification_report is None:
            return None
        if self.last_requalification_report.sample_count < self.charter.canary_min_samples:
            return None
        if self.last_requalification_report.eligible and self.charter.auto_promote_canary:
            decision = CanaryPromotionDecision.new(
                adapter_name=self.adapter_name,
                candidate_provider=self.candidate.provider,
                fallback_provider=self.fallback.provider,
                action="promoted_to_live",
                sample_count=self.last_requalification_report.sample_count,
                reasons=["continuous_requalification_passed"],
            )
            self._activate_provider(self.candidate, decision)
            return decision
        if (not self.last_requalification_report.eligible) and self.charter.auto_rollback_canary:
            decision = CanaryPromotionDecision.new(
                adapter_name=self.adapter_name,
                candidate_provider=self.candidate.provider,
                fallback_provider=self.fallback.provider,
                action="rolled_back_to_fallback",
                sample_count=self.last_requalification_report.sample_count,
                reasons=list(self.last_requalification_report.reasons),
            )
            self._activate_provider(self.fallback, decision)
            return decision
        return None

    def _activate_provider(self, provider: _CanaryProvider, decision: CanaryPromotionDecision) -> None:
        new_policy = AdapterSafetyPolicy.from_dict(provider.spec.policy.to_dict())
        new_policy.allow_live_calls = True
        self.adapter.policy = new_policy
        self.adapter.set_mode("live", provider=provider.provider, live_transport=provider.transport)
        self.adapter.attach_canary_guard(None)
        promoted_decision = ProviderRolloutDecision.new(
            adapter_name=self.adapter_name,
            provider=provider.provider,
            decision="promoted",
            mode="live",
            qualification_id=provider.report.qualification_id if provider.report is not None else None,
            rollout_stage="full_live",
        )
        monitor = PostPromotionShadowMonitor(
            adapter_name=self.adapter_name,
            adapter=self.adapter,
            catalog_candidates=self.catalog_candidates,
            reports=self.reports,
            decision=promoted_decision,
            charter=self.charter,
            ledger=self.ledger,
        )
        self.adapter.attach_post_promotion_guard(monitor)
        self.last_decision = decision
        self.completed = True
        if decision.action == "rolled_back_to_fallback":
            RolloutProtectionAdvisor(adapter_name=self.adapter_name, charter=self.charter, ledger=self.ledger).record_protective_event(
                affected_provider=self.candidate.provider,
                fallback_provider=provider.provider,
                trigger_type="canary_rollback",
                reasons=list(decision.reasons),
            )
        if self.ledger is not None and hasattr(self.ledger, "save_interface_canary_decision"):
            self.ledger.save_interface_canary_decision(decision.to_dict())

    def _build_extra_reference_providers(self) -> list[_CanaryProvider]:
        builder = build_cloud_transport if self.adapter_name == "cloud_llm" else build_quantum_transport
        providers: list[_CanaryProvider] = []
        seen = {self.candidate.provider, self.fallback.provider}
        for report in sorted(self.reports, key=lambda item: item.score, reverse=True):
            if report.provider in seen:
                continue
            allowed, _ = self.charter.rollout_report_allowed(report)
            if not allowed:
                continue
            spec = self.catalog_candidates.get(report.provider)
            if spec is None:
                continue
            transport = builder(spec)
            if transport is None:
                continue
            providers.append(_CanaryProvider(provider=report.provider, spec=spec, report=report, transport=transport))
            seen.add(report.provider)
        return providers

    def _summarize_result(self, result: dict[str, Any]) -> dict[str, Any]:
        summary: dict[str, Any] = {"status": result.get("status"), "reason": result.get("reason")}
        if "text" in result:
            summary["text_preview"] = str(result.get("text", ""))[:120]
            summary["text_length"] = len(str(result.get("text", "")))
        if "factors" in result:
            summary["factors"] = list(result.get("factors", []))[:4]
        if "best_value" in result:
            summary["best_value"] = result.get("best_value")
            summary["best_index"] = result.get("best_index")
        return summary


def configure_canary_rollout(
    *,
    adapters: dict[str, Any],
    catalog: Any,
    reports: dict[str, list[ProviderQualificationReport]],
    decisions: dict[str, ProviderRolloutDecision],
    charter: OperatorCharter,
    ledger: Any | None = None,
) -> dict[str, dict[str, Any]]:
    summaries: dict[str, dict[str, Any]] = {}
    for adapter_name, adapter in adapters.items():
        adapter.attach_canary_guard(None)
        decision = decisions.get(adapter_name)
        if decision is None or decision.decision != "promoted" or decision.rollout_stage != "canary":
            summaries[adapter_name] = {"status": "inactive"}
            continue
        if decision.provider is None or decision.fallback_provider is None:
            summaries[adapter_name] = {"status": "inactive", "reason": "canary_requires_candidate_and_fallback"}
            continue
        candidates = catalog.for_adapter(adapter_name)
        builder = build_cloud_transport if adapter_name == "cloud_llm" else build_quantum_transport
        candidate_spec = candidates.get(decision.provider)
        fallback_spec = candidates.get(decision.fallback_provider)
        if candidate_spec is None or fallback_spec is None:
            summaries[adapter_name] = {"status": "inactive", "reason": "missing_canary_specs"}
            continue
        candidate_transport = builder(candidate_spec)
        fallback_transport = builder(fallback_spec)
        if candidate_transport is None or fallback_transport is None:
            summaries[adapter_name] = {"status": "inactive", "reason": "missing_canary_transport"}
            continue
        report_index = {report.provider: report for report in reports.get(adapter_name, [])}
        controller = CanaryRolloutController(
            adapter_name=adapter_name,
            adapter=adapter,
            candidate=_CanaryProvider(
                provider=decision.provider,
                spec=candidate_spec,
                report=report_index.get(decision.provider),
                transport=candidate_transport,
            ),
            fallback=_CanaryProvider(
                provider=decision.fallback_provider,
                spec=fallback_spec,
                report=report_index.get(decision.fallback_provider),
                transport=fallback_transport,
            ),
            catalog_candidates=candidates,
            reports=reports.get(adapter_name, []),
            charter=charter,
            ledger=ledger,
        )
        adapter.attach_canary_guard(controller)
        summaries[adapter_name] = controller.summary()
    return summaries
