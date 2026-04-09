from __future__ import annotations

from automl.model_registry import ModelRegistry
from automl.response_risk_model import ResponseRiskModel, new_model_id


def _model(version: str, *, status: str = "stable") -> ResponseRiskModel:
    return ResponseRiskModel(
        model_id=new_model_id("riskmodel"),
        version=version,
        threshold=0.5,
        weights={
            "requires_verification": 1.0,
            "insufficient_evidence": 0.0,
            "high_risk_signal": 0.0,
            "critical_tag": 0.0,
        },
        status=status,
    )


def test_model_registry_candidate_canary_finalize_lifecycle() -> None:
    registry = ModelRegistry()
    stable = _model("heuristic_v1")
    candidate = _model("heuristic_v2", status="candidate")

    registry.register_stable(stable)
    registry.register_candidate(candidate)

    assert registry.get_active_model("response_risk_model").model_id == stable.model_id
    assert registry.has_candidate("response_risk_model") is True

    promoted = registry.promote_candidate_to_canary("response_risk_model")
    assert promoted.model_id == candidate.model_id
    assert registry.has_canary("response_risk_model") is True

    finalized = registry.finalize_canary("response_risk_model")
    assert finalized.model_id == candidate.model_id
    assert registry.get_active_model("response_risk_model").model_id == candidate.model_id


def test_model_registry_rolls_back_failed_canary_to_previous_stable() -> None:
    registry = ModelRegistry()
    stable = _model("heuristic_v1")
    candidate = _model("heuristic_v2", status="candidate")

    registry.register_stable(stable)
    registry.register_candidate(candidate)
    registry.promote_candidate_to_canary("response_risk_model")

    restored = registry.rollback_canary("response_risk_model")
    assert restored is not None
    assert restored.model_id == stable.model_id
    assert registry.get_active_model("response_risk_model").model_id == stable.model_id
