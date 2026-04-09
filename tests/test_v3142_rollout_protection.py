from __future__ import annotations

import json
from pathlib import Path

from autonomous_agi import _apply_provider_rollout
from main import build_system
from motivation.operator_charter import OperatorCharter


def _write_catalog(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _regressing_quantum_catalog() -> dict:
    return {
        "cloud_llm": {
            "candidates": [
                {"mode": "live", "provider": "mock_live_safe", "policy": {"allow_live_calls": True, "blocked_terms": ["malware"]}},
                {"mode": "live", "provider": "mock_live_fast", "policy": {"allow_live_calls": True, "blocked_terms": ["malware"]}},
            ]
        },
        "quantum_solver": {
            "candidates": [
                {"mode": "live", "provider": "mock_live_regressing", "policy": {"allow_live_calls": True}},
                {"mode": "live", "provider": "mock_live_precise", "policy": {"allow_live_calls": True}},
            ]
        },
    }


def _cloud_drift_catalog() -> dict:
    return {
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
    }


def _protective_charter(**overrides: object) -> OperatorCharter:
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
        "canary_min_correctness_rate": 0.75,
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
        "rollback_cooldown_rollouts": 1,
        "anti_flap_window_rollouts": 4,
        "anti_flap_repeat_failures": 2,
        "anti_flap_freeze_rollouts": 2,
    }
    payload.update(overrides)
    return OperatorCharter(**payload)


def _force_quantum_canary_rollback(system: dict[str, object]) -> None:
    solver = system["quantum_solver"]
    solver.factorize(21)
    solver.solve_optimization({"values": [5.0, 1.0, 3.0]})
    third = solver.factorize(21)
    assert third["status"] == "rejected"
    final = solver.solve_optimization({"values": [5.0, 1.0, 3.0]})
    assert final["status"] == "ok"
    assert final["canary_rollout"]["decision"]["action"] == "rolled_back_to_fallback"


def test_canary_rollback_blocks_immediate_repromotion_with_cooldown(tmp_path: Path) -> None:
    system = build_system(tmp_path / "runtime")
    catalog_path = _write_catalog(tmp_path / "catalog.json", _regressing_quantum_catalog())
    charter = _protective_charter()

    _apply_provider_rollout(system, charter=charter, provider_catalog_path=catalog_path)
    _force_quantum_canary_rollback(system)

    protections = system["ledger"].load_interface_rollout_protections()
    assert len(protections) == 1
    assert protections[0]["trigger_type"] == "canary_rollback"
    assert protections[0]["affected_provider"] == "mock_live_regressing"
    assert protections[0]["cooldown_until_rollout_index"] == 2

    summary = _apply_provider_rollout(system, charter=charter, provider_catalog_path=catalog_path)
    decision = summary["decisions"]["quantum_solver"]

    assert decision["provider"] == "mock_live_precise"
    assert decision["rollout_stage"] == "full_live"
    assert any("provider_blocked:mock_live_regressing:until_rollout_2" in reason for reason in decision["reasons"])


def test_repeated_rollbacks_enable_adapter_level_anti_flap_freeze(tmp_path: Path) -> None:
    system = build_system(tmp_path / "runtime")
    catalog_path = _write_catalog(tmp_path / "catalog.json", _regressing_quantum_catalog())
    charter = _protective_charter()

    _apply_provider_rollout(system, charter=charter, provider_catalog_path=catalog_path)
    _force_quantum_canary_rollback(system)

    second_rollout = _apply_provider_rollout(system, charter=charter, provider_catalog_path=catalog_path)
    assert second_rollout["decisions"]["quantum_solver"]["provider"] == "mock_live_precise"
    assert second_rollout["decisions"]["quantum_solver"]["rollout_stage"] == "full_live"

    third_rollout = _apply_provider_rollout(system, charter=charter, provider_catalog_path=catalog_path)
    assert third_rollout["decisions"]["quantum_solver"]["provider"] == "mock_live_regressing"
    assert third_rollout["decisions"]["quantum_solver"]["rollout_stage"] == "canary"

    _force_quantum_canary_rollback(system)
    protections = system["ledger"].load_interface_rollout_protections()
    assert len(protections) == 2
    assert protections[-1]["anti_flap_active"] is True
    assert protections[-1]["anti_flap_until_rollout_index"] == 5

    fourth_rollout = _apply_provider_rollout(system, charter=charter, provider_catalog_path=catalog_path)
    decision = fourth_rollout["decisions"]["quantum_solver"]
    assert decision["provider"] == "mock_live_precise"
    assert decision["rollout_stage"] == "full_live"
    assert any("anti_flap_canary_freeze:until_rollout_5" in reason for reason in decision["reasons"])


def test_drift_demotion_blocks_immediate_repromotion_after_runtime_regression(tmp_path: Path) -> None:
    system = build_system(tmp_path / "runtime")
    catalog_path = _write_catalog(tmp_path / "catalog.json", _cloud_drift_catalog())
    charter = _protective_charter(enable_canary_rollout=False)

    _apply_provider_rollout(system, charter=charter, provider_catalog_path=catalog_path)

    cloud_llm = system["cloud_llm"]
    prompt = "Urgent fraud breach detected in production account."
    cloud_llm.generate(prompt, task="classify_risk")
    cloud_llm.generate(prompt, task="classify_risk")
    final = cloud_llm.generate(prompt, task="classify_risk")

    assert final["status"] == "ok"
    assert cloud_llm.provider == "mock_live_safe_alt"

    protections = system["ledger"].load_interface_rollout_protections()
    assert len(protections) == 1
    assert protections[0]["trigger_type"] == "drift_demotion"
    assert protections[0]["affected_provider"] == "mock_live_regressing"
    assert protections[0]["fallback_provider"] == "mock_live_safe_alt"

    summary = _apply_provider_rollout(system, charter=charter, provider_catalog_path=catalog_path)
    decision = summary["decisions"]["cloud_llm"]
    assert decision["provider"] == "mock_live_safe_alt"
    assert decision["rollout_stage"] == "full_live"
    assert any("provider_blocked:mock_live_regressing:until_rollout_2" in reason for reason in decision["reasons"])
