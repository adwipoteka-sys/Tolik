from __future__ import annotations

import json
from pathlib import Path

from autonomous_agi import _apply_provider_rollout
from interfaces.shadow_consensus import ShadowConsensusScorer
from main import build_system
from motivation.operator_charter import OperatorCharter


def _write_catalog(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path



def _consensus_catalog() -> dict:
    return {
        "cloud_llm": {
            "candidates": [
                {"mode": "live", "provider": "mock_live_regressing", "policy": {"allow_live_calls": True, "blocked_terms": ["malware"]}},
                {"mode": "live", "provider": "mock_live_safe", "policy": {"allow_live_calls": True, "blocked_terms": ["malware"]}},
                {"mode": "live", "provider": "mock_live_safe_alt", "policy": {"allow_live_calls": True, "blocked_terms": ["malware"]}},
            ]
        },
        "quantum_solver": {
            "candidates": [
                {"mode": "live", "provider": "mock_live_precise", "policy": {"allow_live_calls": True}},
            ]
        },
    }



def _consensus_shadow_charter(**overrides: object) -> OperatorCharter:
    payload = {
        "allow_cloud_llm": True,
        "allow_quantum_solver": True,
        "provider_rollout_min_correctness": 1.0,
        "provider_rollout_min_safety": 1.0,
        "provider_rollout_max_avg_latency_ms": 80.0,
        "allowed_live_providers": {
            "cloud_llm": ["mock_live_regressing", "mock_live_safe", "mock_live_safe_alt"],
            "quantum_solver": ["mock_live_precise"],
        },
        "enable_shadow_traffic": True,
        "shadow_sample_rate": 1.0,
        "shadow_candidate_limit": 2,
        "shadow_consensus_min_support": 2,
        "shadow_consensus_pairwise_min_agreement": 0.8,
        "post_promotion_window_size": 3,
        "post_promotion_min_samples": 3,
        "post_promotion_min_correctness_rate": 0.8,
        "post_promotion_min_safety_rate": 1.0,
        "post_promotion_max_latency_multiplier": 2.0,
        "post_promotion_min_text_agreement": 0.3,
        "auto_demote_on_drift": True,
    }
    payload.update(overrides)
    return OperatorCharter(**payload)



def _consensus_canary_charter(**overrides: object) -> OperatorCharter:
    payload = {
        "allow_cloud_llm": True,
        "allow_quantum_solver": True,
        "provider_rollout_min_correctness": 1.0,
        "provider_rollout_min_safety": 1.0,
        "provider_rollout_max_avg_latency_ms": 80.0,
        "allowed_live_providers": {
            "cloud_llm": ["mock_live_regressing", "mock_live_safe", "mock_live_safe_alt"],
            "quantum_solver": ["mock_live_precise"],
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
        "post_promotion_min_correctness_rate": 0.8,
        "post_promotion_min_safety_rate": 1.0,
        "post_promotion_max_latency_multiplier": 2.0,
        "post_promotion_min_text_agreement": 0.3,
        "auto_demote_on_drift": True,
    }
    payload.update(overrides)
    return OperatorCharter(**payload)



def test_task_aware_consensus_rejects_high_overlap_label_mismatch() -> None:
    scorer = ShadowConsensusScorer(
        adapter_name="cloud_llm",
        request_summary={"task": "classify_risk"},
        operation="generate",
        live_threshold_base=0.3,
        min_support=2,
        pairwise_min_agreement=0.8,
    )
    evaluation = scorer.evaluate(
        live_provider="mock_live_regressing",
        live_result={"status": "ok", "text": "label:low\nconfidence:0.92"},
        shadow_candidates=[
            {"provider": "mock_live_safe", "result": {"status": "ok", "text": "label:high\nconfidence:0.91"}, "latency_ms": 10.0},
            {"provider": "mock_live_safe_alt", "result": {"status": "ok", "text": '{"label":"high","confidence":0.89}'}, "latency_ms": 12.0},
        ],
        live_summary={"status": "ok", "text_preview": "label:low"},
    )

    assert evaluation.comparison_profile == "classification"
    assert evaluation.consensus_support == 2
    assert evaluation.consensus_provider in {"mock_live_safe", "mock_live_safe_alt"}
    assert evaluation.live_agreement_score == 0.0
    assert evaluation.correctness_pass is False
    assert any(reason.startswith("consensus_disagreement") for reason in evaluation.reasons)



def test_shadow_consensus_demotes_regressing_classifier_provider(tmp_path: Path) -> None:
    system = build_system(tmp_path / "runtime")
    catalog_path = _write_catalog(tmp_path / "catalog.json", _consensus_catalog())
    charter = _consensus_shadow_charter()
    _apply_provider_rollout(system, charter=charter, provider_catalog_path=catalog_path)

    cloud_llm = system["cloud_llm"]
    assert cloud_llm.provider == "mock_live_regressing"

    prompt = "Urgent fraud breach detected in production account."
    cloud_llm.generate(prompt, task="classify_risk")
    cloud_llm.generate(prompt, task="classify_risk")
    final = cloud_llm.generate(prompt, task="classify_risk")

    assert final["status"] == "ok"
    monitoring = final["post_promotion_monitoring"]
    assert monitoring["shadow_consensus"]["comparison_profile"] == "classification"
    assert monitoring["shadow_consensus"]["consensus_support"] == 2
    assert monitoring["shadow_consensus"]["correctness_pass"] is False
    assert cloud_llm.provider == "mock_live_safe"

    ledger = system["ledger"]
    consensus_records = ledger.load_interface_shadow_consensus()
    assert any(record["comparison_profile"] == "classification" for record in consensus_records)
    demotions = ledger.load_interface_demotion_decisions()
    assert len(demotions) == 1
    assert demotions[0]["previous_provider"] == "mock_live_regressing"
    assert demotions[0]["fallback_provider"] == "mock_live_safe"



def test_canary_consensus_rolls_back_regressing_classifier_candidate(tmp_path: Path) -> None:
    system = build_system(tmp_path / "runtime")
    catalog_path = _write_catalog(tmp_path / "catalog.json", _consensus_catalog())
    charter = _consensus_canary_charter()
    _apply_provider_rollout(system, charter=charter, provider_catalog_path=catalog_path)

    cloud_llm = system["cloud_llm"]
    prompt = "Urgent fraud breach detected in production account."
    decisions = []
    for _ in range(4):
        result = cloud_llm.generate(prompt, task="classify_risk")
        if result.get("canary_rollout", {}).get("decision") is not None:
            decisions.append(result["canary_rollout"]["decision"])

    assert decisions
    assert decisions[-1]["action"] == "rolled_back_to_fallback"
    assert cloud_llm.provider == "mock_live_safe"

    samples = system["ledger"].load_interface_canary_samples()
    assert any(sample["comparison_profile"] == "classification" for sample in samples)
    assert any(sample["consensus_support"] == 2 for sample in samples)
