from __future__ import annotations

import json
from pathlib import Path

from autonomous_agi import _apply_provider_rollout
from interfaces.adapter_schema import AdapterSafetyPolicy, InterfaceRuntimeSpec
from interfaces.provider_routing import CostAwareFallbackRouter, RoutedProviderCandidate
from interfaces.qualification_schema import ProviderQualificationReport
from main import build_system
from motivation.operator_charter import OperatorCharter


def _write_catalog(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _report(provider: str, *, score: float, correctness: float, safety: float, latency_ms: float) -> ProviderQualificationReport:
    return ProviderQualificationReport.new(
        adapter_name="cloud_llm",
        provider=provider,
        total_cases=4,
        passed_cases=int(round(correctness * 4)),
        correctness_rate=correctness,
        safety_rate=safety,
        avg_latency_ms=latency_ms,
        score=score,
        eligible=True,
        reasons=[],
        case_results=[],
    )


def _spec(provider: str, *, cost: float | None) -> InterfaceRuntimeSpec:
    return InterfaceRuntimeSpec(
        mode="live",
        provider=provider,
        estimated_cost_per_call_usd=cost,
        policy=AdapterSafetyPolicy(allow_live_calls=True, blocked_terms=["malware"]),
    )


def _cost_aware_charter(**overrides: object) -> OperatorCharter:
    payload = {
        "allow_cloud_llm": True,
        "allow_quantum_solver": True,
        "provider_rollout_min_correctness": 1.0,
        "provider_rollout_min_safety": 1.0,
        "provider_rollout_max_avg_latency_ms": 80.0,
        "allowed_live_providers": {
            "cloud_llm": ["mock_live_fast", "mock_live_regressing", "mock_live_safe", "mock_live_safe_alt"],
            "quantum_solver": ["mock_live_precise", "mock_live_slow", "mock_live_regressing"],
        },
        "enable_canary_rollout": True,
        "canary_live_fraction": 0.5,
        "canary_window_size": 6,
        "canary_min_samples": 4,
        "canary_min_correctness_rate": 0.8,
        "canary_min_safety_rate": 1.0,
        "canary_max_latency_multiplier": 2.0,
        "canary_min_text_agreement": 0.3,
        "auto_promote_canary": True,
        "auto_rollback_canary": True,
        "enable_shadow_traffic": True,
        "shadow_sample_rate": 1.0,
        "shadow_candidate_limit": 2,
        "shadow_consensus_min_support": 2,
        "shadow_consensus_pairwise_min_agreement": 0.8,
        "post_promotion_window_size": 3,
        "post_promotion_min_samples": 3,
        "post_promotion_min_correctness_rate": 0.67,
        "post_promotion_min_safety_rate": 1.0,
        "post_promotion_max_latency_multiplier": 2.0,
        "post_promotion_min_text_agreement": 0.3,
        "auto_demote_on_drift": True,
        "enable_cost_aware_fallback_routing": True,
        "provider_routing_quality_weight": 0.8,
        "provider_routing_cost_weight": 0.2,
        "provider_routing_max_score_gap": 0.04,
    }
    payload.update(overrides)
    return OperatorCharter(**payload)


def test_cost_aware_router_prefers_cheaper_provider_within_quality_gap() -> None:
    charter = _cost_aware_charter(enable_canary_rollout=False)
    router = CostAwareFallbackRouter(adapter_name="cloud_llm", charter=charter)
    selection = router.select(
        role="fallback",
        primary_provider="mock_live_fast",
        quality_anchor_score=1.0,
        candidates=[
            RoutedProviderCandidate(
                provider="mock_live_safe",
                report=_report("mock_live_safe", score=0.975, correctness=1.0, safety=1.0, latency_ms=10.0),
                spec=_spec("mock_live_safe", cost=0.030),
            ),
            RoutedProviderCandidate(
                provider="mock_live_safe_alt",
                report=_report("mock_live_safe_alt", score=0.970, correctness=1.0, safety=1.0, latency_ms=12.0),
                spec=_spec("mock_live_safe_alt", cost=0.005),
            ),
        ],
    )

    assert selection.selected is not None
    assert selection.selected.provider == "mock_live_safe_alt"
    assert selection.decision.selected_provider == "mock_live_safe_alt"
    assert selection.decision.strategy == "cost_aware_balance"


def test_cost_aware_router_keeps_higher_quality_provider_when_cheap_option_is_too_weak() -> None:
    charter = _cost_aware_charter(enable_canary_rollout=False, provider_routing_max_score_gap=0.08)
    router = CostAwareFallbackRouter(adapter_name="cloud_llm", charter=charter)
    selection = router.select(
        role="fallback",
        primary_provider="mock_live_fast",
        quality_anchor_score=1.0,
        candidates=[
            RoutedProviderCandidate(
                provider="high_quality",
                report=_report("high_quality", score=0.96, correctness=1.0, safety=1.0, latency_ms=14.0),
                spec=_spec("high_quality", cost=0.040),
            ),
            RoutedProviderCandidate(
                provider="cheap_but_weak",
                report=_report("cheap_but_weak", score=0.80, correctness=1.0, safety=1.0, latency_ms=14.0),
                spec=_spec("cheap_but_weak", cost=0.001),
            ),
        ],
    )

    assert selection.selected is not None
    assert selection.selected.provider == "high_quality"
    weak_ranking = next(item for item in selection.decision.candidate_rankings if item["provider"] == "cheap_but_weak")
    assert weak_ranking["pool_eligible"] is False


def test_rollout_gate_chooses_cost_aware_canary_fallback_and_persists_route_audit(tmp_path: Path) -> None:
    system = build_system(tmp_path / "runtime")
    charter = _cost_aware_charter()
    catalog_path = _write_catalog(
        tmp_path / "catalog.json",
        {
            "cloud_llm": {
                "candidates": [
                    {
                        "mode": "live",
                        "provider": "mock_live_fast",
                        "estimated_cost_per_call_usd": 0.080,
                        "policy": {"allow_live_calls": True, "blocked_terms": ["malware"]},
                    },
                    {
                        "mode": "live",
                        "provider": "mock_live_safe",
                        "estimated_cost_per_call_usd": 0.030,
                        "policy": {"allow_live_calls": True, "blocked_terms": ["malware"]},
                    },
                    {
                        "mode": "live",
                        "provider": "mock_live_safe_alt",
                        "estimated_cost_per_call_usd": 0.005,
                        "policy": {"allow_live_calls": True, "blocked_terms": ["malware"]},
                    },
                ]
            },
            "quantum_solver": {
                "candidates": [
                    {"mode": "live", "provider": "mock_live_precise", "estimated_cost_per_call_usd": 0.050, "policy": {"allow_live_calls": True}},
                    {"mode": "live", "provider": "mock_live_slow", "estimated_cost_per_call_usd": 0.010, "policy": {"allow_live_calls": True}},
                ]
            },
        },
    )

    summary = _apply_provider_rollout(system, charter=charter, provider_catalog_path=catalog_path)
    cloud_decision = summary["decisions"]["cloud_llm"]

    assert cloud_decision["provider"] == "mock_live_fast"
    assert cloud_decision["fallback_provider"] == "mock_live_safe_alt"
    assert cloud_decision["routing_strategy"] == "cost_aware_balance"
    assert summary["canary_summaries"]["cloud_llm"]["fallback_provider"] == "mock_live_safe_alt"

    routing_audit = system["ledger"].load_interface_provider_routing()
    assert routing_audit
    assert any(record["selected_provider"] == "mock_live_safe_alt" for record in routing_audit)


def test_drift_demotion_uses_cost_aware_fallback_selection(tmp_path: Path) -> None:
    system = build_system(tmp_path / "runtime")
    charter = _cost_aware_charter(enable_canary_rollout=False)
    catalog_path = _write_catalog(
        tmp_path / "catalog.json",
        {
            "cloud_llm": {
                "candidates": [
                    {
                        "mode": "live",
                        "provider": "mock_live_regressing",
                        "estimated_cost_per_call_usd": 0.100,
                        "policy": {"allow_live_calls": True, "blocked_terms": ["malware"]},
                    },
                    {
                        "mode": "live",
                        "provider": "mock_live_safe",
                        "estimated_cost_per_call_usd": 0.030,
                        "policy": {"allow_live_calls": True, "blocked_terms": ["malware"]},
                    },
                    {
                        "mode": "live",
                        "provider": "mock_live_safe_alt",
                        "estimated_cost_per_call_usd": 0.005,
                        "policy": {"allow_live_calls": True, "blocked_terms": ["malware"]},
                    },
                ]
            },
            "quantum_solver": {
                "candidates": [
                    {"mode": "live", "provider": "mock_live_precise", "estimated_cost_per_call_usd": 0.050, "policy": {"allow_live_calls": True}},
                ]
            },
        },
    )
    _apply_provider_rollout(system, charter=charter, provider_catalog_path=catalog_path)

    cloud_llm = system["cloud_llm"]
    prompt = "Urgent fraud breach detected in production account."
    cloud_llm.generate(prompt, task="classify_risk")
    cloud_llm.generate(prompt, task="classify_risk")
    final = cloud_llm.generate(prompt, task="classify_risk")

    assert final["status"] == "ok"
    assert cloud_llm.provider == "mock_live_safe_alt"
    demotions = system["ledger"].load_interface_demotion_decisions()
    assert demotions
    assert demotions[0]["fallback_provider"] == "mock_live_safe_alt"

    routing_audit = system["ledger"].load_interface_provider_routing()
    assert any(record["role"] == "drift_fallback" and record["selected_provider"] == "mock_live_safe_alt" for record in routing_audit)
