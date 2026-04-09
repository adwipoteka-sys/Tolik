from __future__ import annotations

import json
from pathlib import Path

from autonomous_agi import _apply_provider_rollout
from interfaces.adapter_schema import AdapterSafetyPolicy
from interfaces.cloud_llm import CloudLLMClient
from interfaces.provider_qualification import ProviderQualificationManager, load_provider_catalog
from interfaces.quantum_solver import QuantumSolver
from main import build_system
from motivation.operator_charter import OperatorCharter


def _write_catalog(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def test_cloud_llm_rejects_blocked_output_terms() -> None:
    adapter = CloudLLMClient(
        mode="live",
        provider="mock_live_risky",
        policy=AdapterSafetyPolicy(allow_live_calls=True, blocked_terms=["malware"]),
        live_transport=lambda prompt, **kwargs: {"text": "malware should be blocked", "provider": "mock_live_risky"},
    )
    result = adapter.generate("hello")
    assert result["status"] == "rejected"
    assert result["reason"].startswith("blocked_output_terms")


def test_quantum_solver_rejects_invalid_live_result() -> None:
    solver = QuantumSolver(
        mode="live",
        provider="mock_live_noisy",
        policy=AdapterSafetyPolicy(allow_live_calls=True),
        live_transport=lambda operation, payload: {"factors": [2, 10], "provider": "mock_live_noisy"}
        if operation == "factorize"
        else {"best_value": 5, "best_index": 99, "provider": "mock_live_noisy"},
    )
    factorized = solver.factorize(21)
    assert factorized["status"] == "rejected"
    assert factorized["reason"] == "invalid_live_result"

    optimized = solver.solve_optimization({"values": [5.0, 1.0, 3.0]})
    assert optimized["status"] == "rejected"
    assert optimized["reason"] == "invalid_live_result"


def test_provider_qualification_ranks_safe_providers_and_flags_risky_ones(tmp_path: Path) -> None:
    catalog_path = _write_catalog(
        tmp_path / "catalog.json",
        {
            "cloud_llm": {
                "candidates": [
                    {"mode": "live", "provider": "mock_live_fast", "policy": {"allow_live_calls": True, "blocked_terms": ["malware"]}},
                    {"mode": "live", "provider": "mock_live_risky", "policy": {"allow_live_calls": True, "blocked_terms": ["malware"]}},
                ]
            },
            "quantum_solver": {
                "candidates": [
                    {"mode": "live", "provider": "mock_live_precise", "policy": {"allow_live_calls": True}},
                    {"mode": "live", "provider": "mock_live_noisy", "policy": {"allow_live_calls": True}},
                ]
            },
        },
    )
    catalog = load_provider_catalog(catalog_path)
    charter = OperatorCharter(
        allow_cloud_llm=True,
        allow_quantum_solver=True,
        provider_rollout_min_correctness=1.0,
        provider_rollout_min_safety=1.0,
        provider_rollout_max_avg_latency_ms=40.0,
    )
    manager = ProviderQualificationManager()
    reports = manager.qualify_catalog(catalog, charter=charter)

    assert reports["cloud_llm"][0].provider == "mock_live_fast"
    risky_cloud = next(report for report in reports["cloud_llm"] if report.provider == "mock_live_risky")
    assert risky_cloud.eligible is False
    assert any("blocked_output_terms" in reason for reason in risky_cloud.reasons)

    precise_quantum = next(report for report in reports["quantum_solver"] if report.provider == "mock_live_precise")
    noisy_quantum = next(report for report in reports["quantum_solver"] if report.provider == "mock_live_noisy")
    assert precise_quantum.eligible is True
    assert noisy_quantum.eligible is False
    assert any("invalid_live_result" in reason for reason in noisy_quantum.reasons)


def test_rollout_gate_promotes_qualified_providers_and_updates_workspace(tmp_path: Path) -> None:
    runtime_dir = tmp_path / "runtime"
    system = build_system(runtime_dir)
    charter = OperatorCharter(
        allow_cloud_llm=True,
        allow_quantum_solver=True,
        provider_rollout_min_correctness=1.0,
        provider_rollout_min_safety=1.0,
        provider_rollout_max_avg_latency_ms=40.0,
        allowed_live_providers={
            "cloud_llm": ["mock_live_fast"],
            "quantum_solver": ["mock_live_precise"],
        },
    )
    catalog_path = _write_catalog(
        tmp_path / "catalog.json",
        {
            "cloud_llm": {
                "candidates": [
                    {"mode": "live", "provider": "mock_live_fast", "policy": {"allow_live_calls": True, "blocked_terms": ["malware"]}},
                    {"mode": "live", "provider": "mock_live_risky", "policy": {"allow_live_calls": True, "blocked_terms": ["malware"]}},
                ]
            },
            "quantum_solver": {
                "candidates": [
                    {"mode": "live", "provider": "mock_live_precise", "policy": {"allow_live_calls": True}},
                    {"mode": "live", "provider": "mock_live_noisy", "policy": {"allow_live_calls": True}},
                ]
            },
        },
    )

    summary = _apply_provider_rollout(system, charter=charter, provider_catalog_path=catalog_path)
    cloud_decision = summary["decisions"]["cloud_llm"]
    quantum_decision = summary["decisions"]["quantum_solver"]
    assert cloud_decision["decision"] == "promoted"
    assert cloud_decision["provider"] == "mock_live_fast"
    assert quantum_decision["decision"] == "promoted"
    assert quantum_decision["provider"] == "mock_live_precise"

    workspace = system["workspace"]
    state = workspace.get_state()
    assert state["future_interfaces"]["cloud_llm"] == "live"
    assert state["future_interfaces"]["quantum_solver"] == "live"
    assert state["interface_adapter_summary"]["cloud_llm"]["provider"] == "mock_live_fast"
    assert state["interface_adapter_summary"]["quantum_solver"]["provider"] == "mock_live_precise"
    assert state["interface_rollout_decisions"]["cloud_llm"]["decision"] == "promoted"
    assert state["provider_qualification_summary"]["cloud_llm"]

    ledger = system["ledger"]
    rollout_decisions = ledger.load_interface_rollout_decisions()
    qualifications = ledger.load_interface_qualifications()
    assert len(rollout_decisions) == 2
    assert len(qualifications) >= 4


def test_rollout_gate_respects_provider_allowlist(tmp_path: Path) -> None:
    runtime_dir = tmp_path / "runtime"
    system = build_system(runtime_dir)
    charter = OperatorCharter(
        allow_cloud_llm=True,
        allow_quantum_solver=True,
        provider_rollout_min_correctness=1.0,
        provider_rollout_min_safety=1.0,
        provider_rollout_max_avg_latency_ms=40.0,
        allowed_live_providers={
            "cloud_llm": ["mock_live_safe"],
            "quantum_solver": ["mock_live_precise"],
        },
    )
    catalog_path = _write_catalog(
        tmp_path / "catalog.json",
        {
            "cloud_llm": {
                "candidates": [
                    {"mode": "live", "provider": "mock_live_fast", "policy": {"allow_live_calls": True, "blocked_terms": ["malware"]}},
                ]
            },
            "quantum_solver": {
                "candidates": [
                    {"mode": "live", "provider": "mock_live_precise", "policy": {"allow_live_calls": True}},
                ]
            },
        },
    )

    summary = _apply_provider_rollout(system, charter=charter, provider_catalog_path=catalog_path)
    cloud_decision = summary["decisions"]["cloud_llm"]
    assert cloud_decision["decision"] == "deferred"
    assert any("provider_not_in_allowlist" in reason for reason in cloud_decision["reasons"])
    assert system["cloud_llm"].mode == "stub"
