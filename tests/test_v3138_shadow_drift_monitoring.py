from __future__ import annotations

import json
from pathlib import Path

from autonomous_agi import _apply_provider_rollout
from main import build_system
from motivation.operator_charter import OperatorCharter


def _write_catalog(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _shadow_catalog() -> dict:
    return {
        "cloud_llm": {
            "candidates": [
                {"mode": "live", "provider": "mock_live_regressing", "policy": {"allow_live_calls": True, "blocked_terms": ["malware"]}},
                {"mode": "live", "provider": "mock_live_safe", "policy": {"allow_live_calls": True, "blocked_terms": ["malware"]}},
            ]
        },
        "quantum_solver": {
            "candidates": [
                {"mode": "live", "provider": "mock_live_regressing", "policy": {"allow_live_calls": True}},
                {"mode": "live", "provider": "mock_live_precise", "policy": {"allow_live_calls": True}},
            ]
        },
    }


def _shadow_charter() -> OperatorCharter:
    return OperatorCharter(
        allow_cloud_llm=True,
        allow_quantum_solver=True,
        provider_rollout_min_correctness=1.0,
        provider_rollout_min_safety=1.0,
        provider_rollout_max_avg_latency_ms=40.0,
        allowed_live_providers={
            "cloud_llm": ["mock_live_regressing", "mock_live_safe"],
            "quantum_solver": ["mock_live_regressing", "mock_live_precise"],
        },
        enable_shadow_traffic=True,
        shadow_sample_rate=1.0,
        shadow_candidate_limit=1,
        post_promotion_window_size=3,
        post_promotion_min_samples=3,
        post_promotion_min_correctness_rate=0.67,
        post_promotion_min_safety_rate=1.0,
        post_promotion_max_latency_multiplier=2.0,
        post_promotion_min_text_agreement=0.3,
        auto_demote_on_drift=True,
    )


def test_rollout_attaches_shadow_monitor_and_persists_monitor_summary(tmp_path: Path) -> None:
    system = build_system(tmp_path / "runtime")
    catalog_path = _write_catalog(tmp_path / "catalog.json", _shadow_catalog())
    charter = _shadow_charter()

    summary = _apply_provider_rollout(system, charter=charter, provider_catalog_path=catalog_path)

    assert summary["monitor_summaries"]["cloud_llm"]["primary_provider"] == "mock_live_regressing"
    assert summary["monitor_summaries"]["cloud_llm"]["shadow_providers"] == ["mock_live_safe"]
    assert summary["monitor_summaries"]["quantum_solver"]["primary_provider"] == "mock_live_regressing"
    assert summary["monitor_summaries"]["quantum_solver"]["shadow_providers"] == ["mock_live_precise"]

    workspace = system["workspace"].get_state()
    assert workspace["post_promotion_monitoring"]["cloud_llm"]["shadow_providers"] == ["mock_live_safe"]
    assert workspace["post_promotion_monitoring"]["quantum_solver"]["shadow_providers"] == ["mock_live_precise"]


def test_shadow_traffic_records_shadow_runs_and_drift_reports(tmp_path: Path) -> None:
    system = build_system(tmp_path / "runtime")
    catalog_path = _write_catalog(tmp_path / "catalog.json", _shadow_catalog())
    charter = _shadow_charter()
    _apply_provider_rollout(system, charter=charter, provider_catalog_path=catalog_path)

    cloud_llm = system["cloud_llm"]
    first = cloud_llm.generate("Alpha. Beta. Gamma.", task="summarize")
    second = cloud_llm.generate("Delta. Epsilon. Zeta.", task="summarize")

    assert first["status"] == "ok"
    assert second["status"] == "ok"
    assert "post_promotion_monitoring" in first
    assert "drift_report" in first["post_promotion_monitoring"]

    ledger = system["ledger"]
    shadow_runs = ledger.load_interface_shadow_runs()
    drift_reports = ledger.load_interface_drift_reports()
    assert len(shadow_runs) >= 2
    assert len(drift_reports) >= 2
    assert all(run["adapter_name"] == "cloud_llm" for run in shadow_runs[:2])


def test_cloud_provider_regression_triggers_fallback_promotion(tmp_path: Path) -> None:
    system = build_system(tmp_path / "runtime")
    catalog_path = _write_catalog(tmp_path / "catalog.json", _shadow_catalog())
    charter = _shadow_charter()
    _apply_provider_rollout(system, charter=charter, provider_catalog_path=catalog_path)

    cloud_llm = system["cloud_llm"]
    assert cloud_llm.provider == "mock_live_regressing"

    cloud_llm.generate("Alpha. Beta. Gamma.", task="summarize")
    cloud_llm.generate("Delta. Epsilon. Zeta.", task="summarize")
    regressed = cloud_llm.generate("Eta. Theta. Iota.", task="summarize")

    assert regressed["status"] == "rejected"
    assert regressed["reason"].startswith("blocked_output_terms")
    assert cloud_llm.mode == "live"
    assert cloud_llm.provider == "mock_live_safe"

    demotions = system["ledger"].load_interface_demotion_decisions()
    assert len(demotions) == 1
    assert demotions[0]["adapter_name"] == "cloud_llm"
    assert demotions[0]["previous_provider"] == "mock_live_regressing"
    assert demotions[0]["fallback_provider"] == "mock_live_safe"
    assert demotions[0]["action"] == "fallback_promoted"



def test_quantum_provider_regression_triggers_fallback_promotion(tmp_path: Path) -> None:
    system = build_system(tmp_path / "runtime")
    catalog_path = _write_catalog(tmp_path / "catalog.json", _shadow_catalog())
    charter = _shadow_charter()
    _apply_provider_rollout(system, charter=charter, provider_catalog_path=catalog_path)

    quantum_solver = system["quantum_solver"]
    assert quantum_solver.provider == "mock_live_regressing"

    quantum_solver.factorize(21)
    quantum_solver.solve_optimization({"values": [5.0, 1.0, 3.0]})
    regressed = quantum_solver.factorize(21)

    assert regressed["status"] == "rejected"
    assert regressed["reason"] == "invalid_live_result"
    assert quantum_solver.mode == "live"
    assert quantum_solver.provider == "mock_live_precise"

    demotions = system["ledger"].load_interface_demotion_decisions()
    assert len(demotions) == 1
    assert demotions[0]["adapter_name"] == "quantum_solver"
    assert demotions[0]["previous_provider"] == "mock_live_regressing"
    assert demotions[0]["fallback_provider"] == "mock_live_precise"
