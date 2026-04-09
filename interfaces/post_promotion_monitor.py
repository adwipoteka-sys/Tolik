from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Callable

from interfaces.adapter_schema import AdapterSafetyPolicy, InterfaceRuntimeSpec
from interfaces.post_promotion_schema import InterfaceDriftReport, InterfaceShadowRun, ProviderDemotionDecision
from interfaces.provider_qualification import ProviderCatalog
from interfaces.provider_registry import build_cloud_transport, build_quantum_transport
from interfaces.provider_routing import CostAwareFallbackRouter, RoutedProviderCandidate
from interfaces.rollout_protection import RolloutProtectionAdvisor
from interfaces.qualification_schema import ProviderQualificationReport, ProviderRolloutDecision
from interfaces.shadow_consensus import ShadowConsensusEvaluation, ShadowConsensusScorer
from motivation.operator_charter import OperatorCharter


@dataclass(slots=True)
class _ShadowProvider:
    provider: str
    spec: InterfaceRuntimeSpec
    report: ProviderQualificationReport | None
    transport: Callable[..., dict[str, Any]]


class PostPromotionShadowMonitor:
    """Mirror live traffic into shadow candidates and demote regressing providers."""

    def __init__(
        self,
        *,
        adapter_name: str,
        adapter: Any,
        catalog_candidates: dict[str, InterfaceRuntimeSpec],
        reports: list[ProviderQualificationReport],
        decision: ProviderRolloutDecision,
        charter: OperatorCharter,
        ledger: Any | None = None,
    ) -> None:
        self.adapter_name = adapter_name
        self.adapter = adapter
        self.catalog_candidates = dict(catalog_candidates)
        self.reports_by_provider = {report.provider: report for report in reports}
        self.sorted_reports = sorted(reports, key=lambda item: item.score, reverse=True)
        self.charter = charter
        self.ledger = ledger
        self.sample_counter = 0
        self.window: deque[dict[str, Any]] = deque(maxlen=charter.post_promotion_window_size)
        self.last_drift_report: InterfaceDriftReport | None = None
        self.last_demotion_decision: ProviderDemotionDecision | None = None
        self.last_shadow_consensus: ShadowConsensusEvaluation | None = None
        self.last_fallback_routing: dict[str, Any] | None = None
        self.primary_provider = decision.provider
        self.primary_report = self.reports_by_provider.get(self.primary_provider) if self.primary_provider else None
        self.primary_baseline_latency_ms = self.primary_report.avg_latency_ms if self.primary_report is not None else None
        self.shadow_providers: list[_ShadowProvider] = []
        self._rebuild_shadow_providers()

    def summary(self) -> dict[str, Any]:
        return {
            "adapter_name": self.adapter_name,
            "primary_provider": self.primary_provider,
            "shadow_providers": [shadow.provider for shadow in self.shadow_providers],
            "sample_count": len(self.window),
            "shadow_sample_rate": self.charter.shadow_sample_rate,
            "baseline_latency_ms": self.primary_baseline_latency_ms,
            "last_shadow_consensus": None if self.last_shadow_consensus is None else self.last_shadow_consensus.to_dict(),
            "last_drift_report": None if self.last_drift_report is None else self.last_drift_report.to_dict(),
            "last_demotion_decision": None if self.last_demotion_decision is None else self.last_demotion_decision.to_dict(),
            "last_fallback_routing": self.last_fallback_routing,
        }

    def observe(
        self,
        *,
        operation: str,
        request_summary: dict[str, Any],
        live_result: dict[str, Any],
        live_latency_ms: float,
        shadow_runner: Callable[[str, Callable[..., dict[str, Any]]], dict[str, Any]],
    ) -> dict[str, Any]:
        self.sample_counter += 1
        shadow_records: list[InterfaceShadowRun] = []
        shadow_candidates: list[dict[str, Any]] = []
        if self._should_shadow_sample():
            shadow_records, shadow_candidates = self._run_shadow_traffic(
                operation=operation,
                request_summary=request_summary,
                live_result=live_result,
                live_latency_ms=live_latency_ms,
                shadow_runner=shadow_runner,
            )

        consensus = self._build_shadow_consensus(
            operation=operation,
            request_summary=request_summary,
            live_result=live_result,
            shadow_candidates=shadow_candidates,
        )
        self.last_shadow_consensus = consensus
        if consensus is not None and self.ledger is not None and hasattr(self.ledger, "save_interface_shadow_consensus"):
            self.ledger.save_interface_shadow_consensus(consensus.to_dict())

        sample = self._build_sample(
            operation=operation,
            live_result=live_result,
            live_latency_ms=live_latency_ms,
            shadow_records=shadow_records,
            consensus=consensus,
        )
        self.window.append(sample)
        self.last_drift_report = self._build_drift_report()
        if self.ledger is not None and hasattr(self.ledger, "save_interface_drift_report"):
            self.ledger.save_interface_drift_report(self.last_drift_report.to_dict())
        demotion: ProviderDemotionDecision | None = None
        if self.last_drift_report.demotion_triggered and self.charter.auto_demote_on_drift:
            demotion = self._demote(self.last_drift_report.reasons)
        return {
            "shadow_records": [record.to_dict() for record in shadow_records],
            "shadow_consensus": None if consensus is None else consensus.to_dict(),
            "drift_report": self.last_drift_report.to_dict(),
            "demotion": None if demotion is None else demotion.to_dict(),
        }

    def _should_shadow_sample(self) -> bool:
        if not self.charter.enable_shadow_traffic:
            return False
        if not self.shadow_providers:
            return False
        rate = self.charter.shadow_sample_rate
        if rate <= 0.0:
            return False
        if rate >= 1.0:
            return True
        stride = max(1, round(1.0 / rate))
        return (self.sample_counter % stride) == 0

    def _run_shadow_traffic(
        self,
        *,
        operation: str,
        request_summary: dict[str, Any],
        live_result: dict[str, Any],
        live_latency_ms: float,
        shadow_runner: Callable[[str, Callable[..., dict[str, Any]]], dict[str, Any]],
    ) -> tuple[list[InterfaceShadowRun], list[dict[str, Any]]]:
        scorer = self._build_scorer(operation=operation, request_summary=request_summary)
        records: list[InterfaceShadowRun] = []
        candidates: list[dict[str, Any]] = []
        for shadow in self.shadow_providers[: self.charter.shadow_candidate_limit]:
            shadow_result, shadow_latency_ms = self._invoke_shadow(shadow, shadow_runner)
            agreement_score = scorer.score_pair(live_result, shadow_result)
            comparison = self._pair_flags(live_result, shadow_result, agreement_score, scorer.required_live_agreement)
            record = InterfaceShadowRun.new(
                adapter_name=self.adapter_name,
                operation=operation,
                primary_provider=self.primary_provider,
                shadow_provider=shadow.provider,
                live_status=str(live_result.get("status", "unknown")),
                shadow_status=str(shadow_result.get("status", "unknown")),
                live_latency_ms=live_latency_ms,
                shadow_latency_ms=shadow_latency_ms,
                correctness_pass=comparison["correctness_pass"],
                safety_pass=comparison["safety_pass"],
                agreement_score=agreement_score,
                comparison_profile=scorer.profile,
                reasons=list(comparison["reasons"]),
                request_summary=request_summary,
                live_summary=self._summarize_result(live_result),
                shadow_summary=self._summarize_result(shadow_result),
            )
            records.append(record)
            candidates.append({
                "provider": shadow.provider,
                "result": shadow_result,
                "latency_ms": shadow_latency_ms,
                "record": record,
            })
            if self.ledger is not None and hasattr(self.ledger, "save_interface_shadow_run"):
                self.ledger.save_interface_shadow_run(record.to_dict())
        return records, candidates

    def _invoke_shadow(
        self,
        shadow: _ShadowProvider,
        shadow_runner: Callable[[str, Callable[..., dict[str, Any]]], dict[str, Any]],
    ) -> tuple[dict[str, Any], float]:
        import time

        started = time.perf_counter()
        shadow_result = shadow_runner(shadow.provider, shadow.transport)
        shadow_latency_ms = (time.perf_counter() - started) * 1000.0
        return shadow_result, shadow_latency_ms

    def _build_shadow_consensus(
        self,
        *,
        operation: str,
        request_summary: dict[str, Any],
        live_result: dict[str, Any],
        shadow_candidates: list[dict[str, Any]],
    ) -> ShadowConsensusEvaluation | None:
        if not shadow_candidates:
            return None
        scorer = self._build_scorer(operation=operation, request_summary=request_summary)
        return scorer.evaluate(
            live_provider=self.primary_provider,
            live_result=live_result,
            shadow_candidates=shadow_candidates,
            live_summary=self._summarize_result(live_result),
        )

    def _build_scorer(self, *, operation: str, request_summary: dict[str, Any]) -> ShadowConsensusScorer:
        return ShadowConsensusScorer(
            adapter_name=self.adapter_name,
            request_summary=request_summary,
            operation=operation,
            live_threshold_base=self.charter.post_promotion_min_text_agreement,
            min_support=self.charter.shadow_consensus_min_support,
            pairwise_min_agreement=self.charter.shadow_consensus_pairwise_min_agreement,
        )

    def _pair_flags(
        self,
        live_result: dict[str, Any],
        shadow_result: dict[str, Any],
        agreement_score: float | None,
        required_agreement: float,
    ) -> dict[str, Any]:
        reasons: list[str] = []
        live_ok = live_result.get("status") == "ok"
        shadow_ok = shadow_result.get("status") == "ok"
        safety_pass = bool(live_ok)
        correctness_pass = bool(live_ok)
        if not live_ok:
            reasons.append(f"live_status:{live_result.get('status')}")
            return {"correctness_pass": False, "safety_pass": False, "reasons": reasons}
        if shadow_ok and agreement_score is not None:
            if agreement_score < required_agreement:
                correctness_pass = False
                reasons.append(f"shadow_disagreement:{agreement_score:.3f}")
        elif not shadow_ok:
            reasons.append(f"shadow_status:{shadow_result.get('status')}")
        return {"correctness_pass": correctness_pass, "safety_pass": safety_pass, "reasons": reasons}

    def _build_sample(
        self,
        *,
        operation: str,
        live_result: dict[str, Any],
        live_latency_ms: float,
        shadow_records: list[InterfaceShadowRun],
        consensus: ShadowConsensusEvaluation | None,
    ) -> dict[str, Any]:
        if consensus is not None:
            correctness_pass = bool(consensus.correctness_pass)
            agreement_score = consensus.live_agreement_score
            comparison_profile = consensus.comparison_profile
            consensus_support = consensus.consensus_support
        else:
            comparable = [record for record in shadow_records if record.agreement_score is not None]
            if comparable:
                correctness_pass = any(record.correctness_pass for record in comparable)
                agreement_score = sum(float(record.agreement_score or 0.0) for record in comparable) / len(comparable)
                comparison_profile = comparable[0].comparison_profile
            else:
                correctness_pass = bool(live_result.get("status") == "ok")
                agreement_score = None
                comparison_profile = None
            consensus_support = 0
        return {
            "operation": operation,
            "live_status": str(live_result.get("status", "unknown")),
            "live_latency_ms": float(live_latency_ms),
            "correctness_pass": bool(correctness_pass),
            "safety_pass": bool(live_result.get("status") == "ok"),
            "agreement_score": agreement_score,
            "comparison_profile": comparison_profile,
            "consensus_support": consensus_support,
            "shadow_comparisons": len(shadow_records),
        }

    def _build_drift_report(self) -> InterfaceDriftReport:
        sample_count = len(self.window)
        correctness_rate = (sum(1 for sample in self.window if sample["correctness_pass"]) / sample_count) if sample_count else 0.0
        safety_rate = (sum(1 for sample in self.window if sample["safety_pass"]) / sample_count) if sample_count else 0.0
        avg_live_latency_ms = (sum(float(sample["live_latency_ms"]) for sample in self.window) / sample_count) if sample_count else 0.0
        agreement_values = [float(sample["agreement_score"]) for sample in self.window if sample["agreement_score"] is not None]
        avg_agreement = (sum(agreement_values) / len(agreement_values)) if agreement_values else None
        shadow_comparisons = sum(int(sample.get("shadow_comparisons", 0)) for sample in self.window)
        reasons: list[str] = []
        demotion_triggered = False
        if sample_count >= self.charter.post_promotion_min_samples:
            if correctness_rate < self.charter.post_promotion_min_correctness_rate:
                reasons.append(f"correctness_drift:{correctness_rate:.3f}")
            if safety_rate < self.charter.post_promotion_min_safety_rate:
                reasons.append(f"safety_drift:{safety_rate:.3f}")
            latency_limit = self._current_latency_limit()
            if avg_live_latency_ms > latency_limit:
                reasons.append(f"latency_drift:{avg_live_latency_ms:.3f}>{latency_limit:.3f}")
            demotion_triggered = bool(reasons)
        return InterfaceDriftReport.new(
            adapter_name=self.adapter_name,
            provider=self.primary_provider,
            sample_count=sample_count,
            window_size=self.charter.post_promotion_window_size,
            correctness_rate=correctness_rate,
            safety_rate=safety_rate,
            avg_live_latency_ms=avg_live_latency_ms,
            baseline_latency_ms=self.primary_baseline_latency_ms,
            avg_agreement_score=avg_agreement,
            shadow_comparisons=shadow_comparisons,
            demotion_triggered=demotion_triggered,
            reasons=reasons,
        )

    def _current_latency_limit(self) -> float:
        if self.primary_baseline_latency_ms is None or self.primary_baseline_latency_ms <= 0.0:
            return float(self.charter.provider_rollout_max_avg_latency_ms)
        return max(
            float(self.charter.provider_rollout_max_avg_latency_ms),
            float(self.primary_baseline_latency_ms) * float(self.charter.post_promotion_max_latency_multiplier),
        )

    def _demote(self, reasons: list[str]) -> ProviderDemotionDecision:
        previous_provider = self.primary_provider
        fallback = self._best_fallback_provider()
        if fallback is None:
            self.adapter.set_mode("simulated", provider=None, live_transport=None)
            self.primary_provider = None
            self.primary_report = None
            self.primary_baseline_latency_ms = None
            self.shadow_providers = []
            action = "demoted_to_simulated"
            fallback_provider = None
        else:
            new_policy = AdapterSafetyPolicy.from_dict(fallback.spec.policy.to_dict())
            new_policy.allow_live_calls = True
            self.adapter.policy = new_policy
            self.adapter.set_mode("live", provider=fallback.provider, live_transport=fallback.transport)
            self.primary_provider = fallback.provider
            self.primary_report = fallback.report
            self.primary_baseline_latency_ms = fallback.report.avg_latency_ms if fallback.report is not None else None
            self._rebuild_shadow_providers()
            action = "fallback_promoted"
            fallback_provider = fallback.provider
        self.window.clear()
        decision = ProviderDemotionDecision.new(
            adapter_name=self.adapter_name,
            previous_provider=previous_provider,
            fallback_provider=fallback_provider,
            action=action,
            reasons=list(reasons),
        )
        self.last_demotion_decision = decision
        RolloutProtectionAdvisor(adapter_name=self.adapter_name, charter=self.charter, ledger=self.ledger).record_protective_event(
            affected_provider=previous_provider or "unknown_provider",
            fallback_provider=fallback_provider,
            trigger_type="drift_demotion",
            reasons=list(reasons),
        )
        if self.ledger is not None and hasattr(self.ledger, "save_interface_demotion_decision"):
            self.ledger.save_interface_demotion_decision(decision.to_dict())
        return decision

    def _best_fallback_provider(self) -> _ShadowProvider | None:
        candidates: list[RoutedProviderCandidate] = [
            RoutedProviderCandidate(provider=shadow.provider, report=shadow.report, spec=shadow.spec, transport=shadow.transport)
            for shadow in self.shadow_providers
            if shadow.provider != self.primary_provider
        ]
        if not candidates:
            self.last_fallback_routing = None
            return None
        quality_anchor = self.primary_report.score if self.primary_report is not None else None
        router = CostAwareFallbackRouter(adapter_name=self.adapter_name, charter=self.charter, ledger=self.ledger)
        selection = router.select(
            role="drift_fallback",
            primary_provider=self.primary_provider,
            candidates=candidates,
            quality_anchor_score=quality_anchor,
        )
        self.last_fallback_routing = selection.decision.to_dict()
        if selection.selected is None:
            return None
        for shadow in self.shadow_providers:
            if shadow.provider == selection.selected.provider:
                return shadow
        return None

    def _rebuild_shadow_providers(self) -> None:
        builder = build_cloud_transport if self.adapter_name == "cloud_llm" else build_quantum_transport
        shadow_providers: list[_ShadowProvider] = []
        for report in self.sorted_reports:
            if report.provider == self.primary_provider:
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
            shadow_providers.append(_ShadowProvider(provider=report.provider, spec=spec, report=report, transport=transport))
        self.shadow_providers = shadow_providers

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


def configure_post_promotion_monitors(
    *,
    adapters: dict[str, Any],
    catalog: ProviderCatalog,
    reports: dict[str, list[ProviderQualificationReport]],
    decisions: dict[str, ProviderRolloutDecision],
    charter: OperatorCharter,
    ledger: Any | None = None,
) -> dict[str, dict[str, Any]]:
    summaries: dict[str, dict[str, Any]] = {}
    for adapter_name, adapter in adapters.items():
        decision = decisions.get(adapter_name)
        if decision is None or decision.decision != "promoted" or getattr(decision, "rollout_stage", "full_live") != "full_live":
            adapter.attach_post_promotion_guard(None)
            summaries[adapter_name] = {"status": "inactive"}
            continue
        monitor = PostPromotionShadowMonitor(
            adapter_name=adapter_name,
            adapter=adapter,
            catalog_candidates=catalog.for_adapter(adapter_name),
            reports=reports.get(adapter_name, []),
            decision=decision,
            charter=charter,
            ledger=ledger,
        )
        adapter.attach_post_promotion_guard(monitor)
        summaries[adapter_name] = monitor.summary()
    return summaries
