from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from interfaces.adapter_schema import AdapterSafetyPolicy, InterfaceRuntimeSpec
from interfaces.cloud_llm import CloudLLMClient
from interfaces.provider_registry import build_cloud_transport, build_quantum_transport
from interfaces.provider_routing import CostAwareFallbackRouter, RoutedProviderCandidate
from interfaces.rollout_protection import RolloutProtectionAdvisor
from interfaces.qualification_schema import ProviderQualificationReport, ProviderRolloutDecision
from interfaces.quantum_solver import QuantumSolver
from motivation.operator_charter import OperatorCharter


@dataclass(slots=True)
class ProviderCatalog:
    cloud_llm: dict[str, InterfaceRuntimeSpec]
    quantum_solver: dict[str, InterfaceRuntimeSpec]

    def for_adapter(self, adapter_name: str) -> dict[str, InterfaceRuntimeSpec]:
        if adapter_name == "cloud_llm":
            return self.cloud_llm
        if adapter_name == "quantum_solver":
            return self.quantum_solver
        raise KeyError(adapter_name)

    def to_dict(self) -> dict[str, Any]:
        return {
            "cloud_llm": {name: spec.to_dict() for name, spec in self.cloud_llm.items()},
            "quantum_solver": {name: spec.to_dict() for name, spec in self.quantum_solver.items()},
        }


def _parse_candidates(payload: dict[str, Any] | None) -> dict[str, InterfaceRuntimeSpec]:
    if not payload:
        return {}
    candidates = payload.get("candidates") or []
    parsed: dict[str, InterfaceRuntimeSpec] = {}
    for entry in candidates:
        spec = InterfaceRuntimeSpec.from_dict(entry)
        provider = str(spec.provider or "").strip()
        if not provider:
            raise ValueError("Provider catalog entries must define a provider name")
        parsed[provider] = spec
    return parsed


def load_provider_catalog(path: str | Path | None) -> ProviderCatalog:
    if path is None:
        return ProviderCatalog(cloud_llm={}, quantum_solver={})
    catalog_path = Path(path)
    data = json.loads(catalog_path.read_text(encoding="utf-8"))
    return ProviderCatalog(
        cloud_llm=_parse_candidates(data.get("cloud_llm")),
        quantum_solver=_parse_candidates(data.get("quantum_solver")),
    )


class ProviderQualificationManager:
    """Benchmark multiple candidate providers behind the same guarded adapters."""

    def __init__(self, *, ledger: Any | None = None) -> None:
        self.ledger = ledger

    def qualify_catalog(
        self,
        catalog: ProviderCatalog,
        *,
        charter: OperatorCharter,
    ) -> dict[str, list[ProviderQualificationReport]]:
        return {
            "cloud_llm": self.qualify_cloud_llm(catalog.cloud_llm, charter=charter),
            "quantum_solver": self.qualify_quantum_solver(catalog.quantum_solver, charter=charter),
        }

    def qualify_cloud_llm(
        self,
        candidates: dict[str, InterfaceRuntimeSpec],
        *,
        charter: OperatorCharter,
    ) -> list[ProviderQualificationReport]:
        reports: list[ProviderQualificationReport] = []
        for provider, spec in candidates.items():
            transport = build_cloud_transport(spec)
            if transport is None:
                report = ProviderQualificationReport.new(
                    adapter_name="cloud_llm",
                    provider=provider,
                    total_cases=0,
                    passed_cases=0,
                    correctness_rate=0.0,
                    safety_rate=0.0,
                    avg_latency_ms=0.0,
                    score=0.0,
                    eligible=False,
                    reasons=["transport_unavailable"],
                    case_results=[],
                )
                self._persist_qualification(report)
                reports.append(report)
                continue
            policy = AdapterSafetyPolicy.from_dict(spec.policy.to_dict())
            policy.allow_live_calls = True
            if "malware" not in policy.blocked_terms:
                policy.blocked_terms.append("malware")
            adapter = CloudLLMClient(mode="live", provider=provider, policy=policy, ledger=None, live_transport=transport)

            cases = [
                (
                    "summarize_short",
                    lambda a=adapter: a.generate("Alpha. Beta. Gamma.", task="summarize"),
                    lambda result: result.get("status") == "ok"
                    and "Alpha" in str(result.get("text", ""))
                    and "Beta" in str(result.get("text", "")),
                ),
                (
                    "safe_text_generation",
                    lambda a=adapter: a.generate("Return concise safe guidance."),
                    lambda result: result.get("status") == "ok" and bool(str(result.get("text", "")).strip()),
                ),
            ]
            report = self._run_cases("cloud_llm", provider, cases, charter=charter)
            self._persist_qualification(report)
            reports.append(report)
        return sorted(reports, key=lambda item: item.score, reverse=True)

    def qualify_quantum_solver(
        self,
        candidates: dict[str, InterfaceRuntimeSpec],
        *,
        charter: OperatorCharter,
    ) -> list[ProviderQualificationReport]:
        reports: list[ProviderQualificationReport] = []
        for provider, spec in candidates.items():
            transport = build_quantum_transport(spec)
            if transport is None:
                report = ProviderQualificationReport.new(
                    adapter_name="quantum_solver",
                    provider=provider,
                    total_cases=0,
                    passed_cases=0,
                    correctness_rate=0.0,
                    safety_rate=0.0,
                    avg_latency_ms=0.0,
                    score=0.0,
                    eligible=False,
                    reasons=["transport_unavailable"],
                    case_results=[],
                )
                self._persist_qualification(report)
                reports.append(report)
                continue
            policy = AdapterSafetyPolicy.from_dict(spec.policy.to_dict())
            policy.allow_live_calls = True
            adapter = QuantumSolver(mode="live", provider=provider, policy=policy, ledger=None, live_transport=transport)
            cases = [
                (
                    "factorize_21",
                    lambda a=adapter: a.factorize(21),
                    lambda result: result.get("status") == "ok" and sorted(result.get("factors", [])) == [3, 7],
                ),
                (
                    "optimize_small_vector",
                    lambda a=adapter: a.solve_optimization({"values": [5.0, 1.0, 3.0]}),
                    lambda result: result.get("status") == "ok" and float(result.get("best_value")) == 1.0 and int(result.get("best_index")) == 1,
                ),
            ]
            report = self._run_cases("quantum_solver", provider, cases, charter=charter)
            self._persist_qualification(report)
            reports.append(report)
        return sorted(reports, key=lambda item: item.score, reverse=True)

    def _run_cases(
        self,
        adapter_name: str,
        provider: str,
        cases: list[tuple[str, Callable[[], dict[str, Any]], Callable[[dict[str, Any]], bool]]],
        *,
        charter: OperatorCharter,
    ) -> ProviderQualificationReport:
        case_results: list[dict[str, Any]] = []
        passed_cases = 0
        safety_passes = 0
        latency_values: list[float] = []
        reasons: list[str] = []
        for case_id, runner, validator in cases:
            started = time.perf_counter()
            result = runner()
            latency_ms = (time.perf_counter() - started) * 1000.0
            latency_values.append(latency_ms)
            correctness_pass = False
            safety_pass = bool(result.get("status") == "ok")
            if result.get("status") == "ok":
                try:
                    correctness_pass = bool(validator(result))
                except Exception:
                    correctness_pass = False
            else:
                reasons.append(f"{case_id}:{result.get('reason', 'unknown_rejection')}")
            if correctness_pass:
                passed_cases += 1
            if safety_pass:
                safety_passes += 1
            case_results.append(
                {
                    "case_id": case_id,
                    "status": result.get("status"),
                    "latency_ms": round(latency_ms, 3),
                    "correctness_pass": correctness_pass,
                    "safety_pass": safety_pass,
                    "reason": result.get("reason"),
                }
            )
        total_cases = len(cases)
        correctness_rate = (passed_cases / total_cases) if total_cases else 0.0
        safety_rate = (safety_passes / total_cases) if total_cases else 0.0
        avg_latency_ms = (sum(latency_values) / len(latency_values)) if latency_values else 0.0
        latency_score = 1.0 if charter.provider_rollout_max_avg_latency_ms <= 0 else max(
            0.0,
            1.0 - (avg_latency_ms / float(charter.provider_rollout_max_avg_latency_ms)),
        )
        score = (0.55 * correctness_rate) + (0.25 * safety_rate) + (0.20 * latency_score)
        if correctness_rate < charter.provider_rollout_min_correctness:
            reasons.append(f"correctness_below_threshold:{correctness_rate:.3f}")
        if safety_rate < charter.provider_rollout_min_safety:
            reasons.append(f"safety_below_threshold:{safety_rate:.3f}")
        if avg_latency_ms > charter.provider_rollout_max_avg_latency_ms:
            reasons.append(f"latency_above_threshold:{avg_latency_ms:.3f}")
        eligible = not reasons
        return ProviderQualificationReport.new(
            adapter_name=adapter_name,
            provider=provider,
            total_cases=total_cases,
            passed_cases=passed_cases,
            correctness_rate=correctness_rate,
            safety_rate=safety_rate,
            avg_latency_ms=avg_latency_ms,
            score=score,
            eligible=eligible,
            reasons=reasons,
            case_results=case_results,
        )

    def _persist_qualification(self, report: ProviderQualificationReport) -> None:
        if self.ledger is not None and hasattr(self.ledger, "save_interface_qualification"):
            self.ledger.save_interface_qualification(report.to_dict())


class CharterAwareRolloutGate:
    """Promote only charter-approved providers that passed qualification gates."""

    def __init__(self, *, ledger: Any | None = None) -> None:
        self.ledger = ledger

    def apply(
        self,
        *,
        adapters: dict[str, Any],
        catalog: ProviderCatalog,
        reports: dict[str, list[ProviderQualificationReport]],
        charter: OperatorCharter,
    ) -> dict[str, ProviderRolloutDecision]:
        decisions: dict[str, ProviderRolloutDecision] = {}
        for adapter_name, adapter in adapters.items():
            decision = self._apply_one(
                adapter_name=adapter_name,
                adapter=adapter,
                candidates=catalog.for_adapter(adapter_name),
                reports=reports.get(adapter_name, []),
                charter=charter,
            )
            decisions[adapter_name] = decision
            self._persist_rollout(decision)
        return decisions

    def _apply_one(
        self,
        *,
        adapter_name: str,
        adapter: Any,
        candidates: dict[str, InterfaceRuntimeSpec],
        reports: list[ProviderQualificationReport],
        charter: OperatorCharter,
    ) -> ProviderRolloutDecision:
        capability_ok, capability_reason = charter.goal_allowed(tags=["future_integration"], required_capabilities=[adapter_name])
        if not capability_ok:
            return ProviderRolloutDecision.new(
                adapter_name=adapter_name,
                provider=None,
                decision="deferred",
                mode=getattr(adapter, "mode", "stub"),
                reasons=[f"charter:{capability_reason}"],
            )
        if not candidates:
            return ProviderRolloutDecision.new(
                adapter_name=adapter_name,
                provider=None,
                decision="deferred",
                mode=getattr(adapter, "mode", "stub"),
                reasons=["no_candidate_providers"],
            )
        candidate_reports = sorted(reports, key=lambda item: item.score, reverse=True)
        promotable: list[RoutedProviderCandidate] = []
        rejection_reasons: list[str] = []
        protection_advisor = RolloutProtectionAdvisor(adapter_name=adapter_name, charter=charter, ledger=self.ledger)
        protection = protection_advisor.evaluate(candidate_providers=[report.provider for report in candidate_reports])
        rejection_reasons.extend(protection.reasons)
        builder = build_cloud_transport if adapter_name == "cloud_llm" else build_quantum_transport
        for report in candidate_reports:
            blocked = protection.blocked_providers.get(report.provider)
            if blocked is not None:
                rejection_reasons.append(f"provider_in_cooldown:{report.provider}:{';'.join(blocked.reasons)}")
                continue
            report_allowed, report_reasons = charter.rollout_report_allowed(report)
            if not report_allowed:
                rejection_reasons.append(f"provider_not_qualified:{report.provider}:{';'.join(report_reasons) or 'unknown'}")
                continue
            spec = candidates.get(report.provider)
            if spec is None:
                rejection_reasons.append(f"provider_spec_missing:{report.provider}")
                continue
            transport = builder(spec)
            if transport is None:
                rejection_reasons.append(f"transport_unavailable:{report.provider}")
                continue
            promotable.append(RoutedProviderCandidate(provider=report.provider, report=report, spec=spec, transport=transport))

        if not promotable:
            return ProviderRolloutDecision.new(
                adapter_name=adapter_name,
                provider=None,
                decision="deferred",
                mode=getattr(adapter, "mode", "stub"),
                reasons=rejection_reasons or ["no_qualified_provider"],
            )

        current_provider = getattr(adapter, "provider", None)
        if protection.reasons and current_provider is not None:
            for index, candidate in enumerate(promotable):
                if candidate.provider == current_provider:
                    promotable.insert(0, promotable.pop(index))
                    rejection_reasons.append(f"protective_stickiness:{current_provider}")
                    break

        primary = promotable[0]
        primary_policy = AdapterSafetyPolicy.from_dict(primary.spec.policy.to_dict())
        primary_policy.allow_live_calls = True
        adapter.policy = primary_policy
        adapter.set_mode("live", provider=primary.provider, live_transport=primary.transport)

        fallback_selection = None
        fallback_candidates = promotable[1:]
        if fallback_candidates:
            router = CostAwareFallbackRouter(adapter_name=adapter_name, charter=charter, ledger=self.ledger)
            fallback_selection = router.select(
                role="fallback",
                primary_provider=primary.provider,
                candidates=fallback_candidates,
                quality_anchor_score=primary.qualification_score,
            )

        canary_allowed = charter.enable_canary_rollout and protection.allow_canary
        if canary_allowed and fallback_selection is not None and fallback_selection.selected is not None:
            fallback = fallback_selection.selected
            return ProviderRolloutDecision.new(
                adapter_name=adapter_name,
                provider=primary.provider,
                decision="promoted",
                mode="live",
                reasons=["qualified_and_charter_approved", "canary_rollout_enabled", *rejection_reasons],
                qualification_id=primary.report.qualification_id if primary.report is not None else None,
                rollout_stage="canary",
                fallback_provider=fallback.provider,
                canary_live_fraction=charter.canary_live_fraction,
                routing_strategy=fallback_selection.decision.strategy,
                estimated_cost_per_call_usd=primary.estimated_cost_per_call_usd,
                fallback_estimated_cost_per_call_usd=fallback.estimated_cost_per_call_usd,
            )

        return ProviderRolloutDecision.new(
            adapter_name=adapter_name,
            provider=primary.provider,
            decision="promoted",
            mode="live",
            reasons=["qualified_and_charter_approved", *rejection_reasons],
            qualification_id=primary.report.qualification_id if primary.report is not None else None,
            rollout_stage="full_live",
            routing_strategy=None if fallback_selection is None else fallback_selection.decision.strategy,
            estimated_cost_per_call_usd=primary.estimated_cost_per_call_usd,
            fallback_estimated_cost_per_call_usd=None if fallback_selection is None or fallback_selection.selected is None else fallback_selection.selected.estimated_cost_per_call_usd,
        )

    def _persist_rollout(self, decision: ProviderRolloutDecision) -> None:
        if self.ledger is not None and hasattr(self.ledger, "save_interface_rollout_decision"):
            self.ledger.save_interface_rollout_decision(decision.to_dict())
