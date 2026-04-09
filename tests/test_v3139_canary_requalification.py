from __future__ import annotations

import json
from pathlib import Path

from autonomous_agi import _apply_provider_rollout
from main import build_system
from motivation.operator_charter import OperatorCharter


def _write_catalog(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _cloud_canary_catalog() -> dict:
    return {
        "cloud_llm": {
            "candidates": [
                {"mode": "live", "provider": "mock_live_fast", "policy": {"allow_live_calls": True, "blocked_terms": ["malware"]}},
                {"mode": "live", "provider": "mock_live_safe", "policy": {"allow_live_calls": True, "blocked_terms": ["malware"]}},
            ]
        },
        "quantum_solver": {
            "candidates": [
                {"mode": "live", "provider": "mock_live_precise", "policy": {"allow_live_calls": True}},
                {"mode": "live", "provider": "mock_live_slow", "policy": {"allow_live_calls": True}},
            ]
        },
    }


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


def _cloud_canary_charter(**overrides: object) -> OperatorCharter:
    payload = {
        "allow_cloud_llm": True,
        "allow_quantum_solver": True,
        "provider_rollout_min_correctness": 1.0,
        "provider_rollout_min_safety": 1.0,
        "provider_rollout_max_avg_latency_ms": 80.0,
        "allowed_live_providers": {
            "cloud_llm": ["mock_live_fast", "mock_live_safe"],
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
        "shadow_candidate_limit": 1,
        "post_promotion_window_size": 3,
        "post_promotion_min_samples": 3,
        "post_promotion_min_correctness_rate": 0.67,
        "post_promotion_min_safety_rate": 1.0,
        "post_promotion_max_latency_multiplier": 2.0,
        "post_promotion_min_text_agreement": 0.3,
        "auto_demote_on_drift": True,
    }
    payload.update(overrides)
    return OperatorCharter(**payload)


def test_canary_rollout_attaches_controller_and_updates_workspace(tmp_path: Path) -> None:
    system = build_system(tmp_path / "runtime")
    catalog_path = _write_catalog(tmp_path / "catalog.json", _cloud_canary_catalog())
    charter = _cloud_canary_charter()

    summary = _apply_provider_rollout(system, charter=charter, provider_catalog_path=catalog_path)

    decision = summary["decisions"]["cloud_llm"]
    assert decision["rollout_stage"] == "canary"
    assert decision["provider"] == "mock_live_fast"
    assert decision["fallback_provider"] == "mock_live_safe"
    assert summary["canary_summaries"]["cloud_llm"]["candidate_provider"] == "mock_live_fast"

    workspace = system["workspace"].get_state()
    assert workspace["interface_canary_rollout"]["cloud_llm"]["fallback_provider"] == "mock_live_safe"
    assert system["cloud_llm"].summary()["canary_guard"]["candidate_provider"] == "mock_live_fast"


def test_canary_records_real_traffic_samples_and_requalification_reports(tmp_path: Path) -> None:
    system = build_system(tmp_path / "runtime")
    catalog_path = _write_catalog(tmp_path / "catalog.json", _cloud_canary_catalog())
    charter = _cloud_canary_charter(auto_promote_canary=False, auto_rollback_canary=False, canary_min_samples=6)
    _apply_provider_rollout(system, charter=charter, provider_catalog_path=catalog_path)

    cloud_llm = system["cloud_llm"]
    for prompt in (
        "Alpha. Beta. Gamma.",
        "Delta. Epsilon. Zeta.",
        "Eta. Theta. Iota.",
        "Kappa. Lambda. Mu.",
    ):
        result = cloud_llm.generate(prompt, task="summarize")
        assert result["status"] == "ok"
        assert "canary_rollout" in result

    ledger = system["ledger"]
    samples = ledger.load_interface_canary_samples()
    reports = ledger.load_interface_requalification_reports()
    assert len(samples) == 4
    assert len(reports) == 4
    assert {sample["rollout_stage"] for sample in samples} == {"canary_live", "stable_shadow"}


def test_canary_success_promotes_candidate_to_full_live(tmp_path: Path) -> None:
    system = build_system(tmp_path / "runtime")
    catalog_path = _write_catalog(tmp_path / "catalog.json", _cloud_canary_catalog())
    charter = _cloud_canary_charter(canary_min_samples=4)
    _apply_provider_rollout(system, charter=charter, provider_catalog_path=catalog_path)

    cloud_llm = system["cloud_llm"]
    for prompt in (
        "Alpha. Beta. Gamma.",
        "Delta. Epsilon. Zeta.",
        "Eta. Theta. Iota.",
        "Kappa. Lambda. Mu.",
    ):
        cloud_llm.generate(prompt, task="summarize")

    assert cloud_llm.provider == "mock_live_fast"
    summary = cloud_llm.summary()
    assert "canary_guard" not in summary
    assert summary["post_promotion_guard"]["primary_provider"] == "mock_live_fast"

    decision = system["ledger"].load_interface_canary_decisions()[0]
    assert decision["action"] == "promoted_to_live"

    post = cloud_llm.generate("Nu. Xi. Omicron.", task="summarize")
    assert post["status"] == "ok"
    assert "post_promotion_monitoring" in post
    assert "canary_rollout" not in post


def test_canary_failure_rolls_back_to_fallback_provider(tmp_path: Path) -> None:
    system = build_system(tmp_path / "runtime")
    catalog_path = _write_catalog(tmp_path / "catalog.json", _regressing_quantum_catalog())
    charter = _cloud_canary_charter(canary_min_samples=4)
    _apply_provider_rollout(system, charter=charter, provider_catalog_path=catalog_path)

    solver = system["quantum_solver"]
    solver.factorize(21)
    solver.solve_optimization({"values": [5.0, 1.0, 3.0]})
    failing = solver.factorize(21)
    assert failing["status"] == "rejected"
    final = solver.solve_optimization({"values": [5.0, 1.0, 3.0]})
    assert final["status"] == "ok"
    assert final["canary_rollout"]["decision"]["action"] == "rolled_back_to_fallback"

    assert solver.provider == "mock_live_precise"
    summary = solver.summary()
    assert "canary_guard" not in summary
    assert summary["post_promotion_guard"]["primary_provider"] == "mock_live_precise"

    decision = system["ledger"].load_interface_canary_decisions()[0]
    assert decision["action"] == "rolled_back_to_fallback"
