from __future__ import annotations

from automl.response_risk_model import RESPONSE_RISK_FAMILY, ResponseRiskTrainingExample
from benchmarks.skill_arena import SkillArenaCase
from benchmarks.transfer_suite import TransferCase
from main import build_system
from motivation.goal_schema import Goal, GoalBudget, GoalKind, GoalSource, SuccessCriterion


def _goal(
    goal_id: str,
    title: str,
    *,
    requires_verification: bool = False,
    insufficient_evidence: bool = False,
    risk_signal: float = 0.1,
) -> Goal:
    evidence = {"risk_signal": risk_signal}
    if requires_verification:
        evidence["requires_verification"] = True
    if insufficient_evidence:
        evidence["insufficient_evidence"] = True
    return Goal(
        goal_id=goal_id,
        title=title,
        description=title,
        source=GoalSource.USER,
        kind=GoalKind.USER_TASK,
        expected_gain=0.6,
        novelty=0.2,
        uncertainty_reduction=0.4,
        strategic_fit=0.7,
        risk_estimate=risk_signal,
        priority=0.7,
        risk_budget=0.1,
        resource_budget=GoalBudget(max_steps=4, max_seconds=10.0),
        success_criteria=[SuccessCriterion(metric="status", comparator="==", target="done")],
        required_capabilities=["classical_planning"],
        tags=["user"],
        evidence=evidence,
    )


def _training_examples() -> list[ResponseRiskTrainingExample]:
    return [
        ResponseRiskTrainingExample(goal=_goal("g_explicit", "Explicit verify", requires_verification=True).to_dict(), label=True),
        ResponseRiskTrainingExample(goal=_goal("g_high_risk", "High risk", risk_signal=0.9).to_dict(), label=True),
        ResponseRiskTrainingExample(goal=_goal("g_missing_evidence", "Missing evidence", insufficient_evidence=True, risk_signal=0.2).to_dict(), label=True),
        ResponseRiskTrainingExample(goal=_goal("g_routine", "Routine question", risk_signal=0.1).to_dict(), label=False),
    ]


def _anchor_cases() -> list[SkillArenaCase]:
    return [
        SkillArenaCase(
            case_id="risk_anchor_explicit_verify",
            payload={"goal": _goal("a1", "Explicit verify", requires_verification=True).to_dict()},
            expected={"predicted_verify": True, "required_steps": ["understand_request", "verify_outcome", "form_response"], "forbidden_steps": []},
            description="Explicit verification requirements must remain verified.",
        ),
        SkillArenaCase(
            case_id="risk_anchor_routine_question",
            payload={"goal": _goal("a2", "Routine question", risk_signal=0.1).to_dict()},
            expected={"predicted_verify": False, "required_steps": ["understand_request", "form_response"], "forbidden_steps": ["verify_outcome"]},
            description="Routine low-risk questions should stay lightweight.",
        ),
    ]


def _transfer_cases() -> list[TransferCase]:
    return [
        TransferCase(
            case_id="risk_transfer_high_risk_generalization",
            payload={"goal": _goal("t1", "High risk generalization", risk_signal=0.85).to_dict()},
            expected={"predicted_verify": True, "required_steps": ["understand_request", "verify_outcome", "form_response"], "forbidden_steps": []},
            description="The trained model should verify held-out high-risk tasks.",
        ),
        TransferCase(
            case_id="risk_transfer_insufficient_evidence",
            payload={"goal": _goal("t2", "Insufficient evidence", insufficient_evidence=True, risk_signal=0.2).to_dict()},
            expected={"predicted_verify": True, "required_steps": ["understand_request", "verify_outcome", "form_response"], "forbidden_steps": []},
            description="The trained model should verify held-out insufficient-evidence tasks.",
        ),
    ]


def _canary_cases() -> list[SkillArenaCase]:
    return [
        SkillArenaCase(
            case_id="risk_canary_high_risk",
            payload={"goal": _goal("c1", "Canary high risk", risk_signal=0.9).to_dict()},
            expected={"predicted_verify": True, "required_steps": ["understand_request", "verify_outcome", "form_response"], "forbidden_steps": []},
            description="Canary must verify a high-risk task.",
        ),
        SkillArenaCase(
            case_id="risk_canary_routine",
            payload={"goal": _goal("c2", "Canary routine", risk_signal=0.1).to_dict()},
            expected={"predicted_verify": False, "required_steps": ["understand_request", "form_response"], "forbidden_steps": ["verify_outcome"]},
            description="Canary must preserve the lightweight routine path.",
        ),
    ]


def _search_space() -> dict[str, list[float]]:
    return {
        "threshold": [0.5],
        "bias": [0.0],
        "weight_requires_verification": [1.0],
        "weight_insufficient_evidence": [0.0, 1.0],
        "weight_high_risk_signal": [0.0, 1.0],
        "weight_critical_tag": [0.0],
    }


def test_safe_automl_manager_trains_and_finalizes_response_risk_model(tmp_path) -> None:
    system = build_system(tmp_path / "runtime")
    registry = system["model_registry"]
    manager = system["automl_manager"]
    planning = system["planning"]

    baseline_model = registry.get_active_model(RESPONSE_RISK_FAMILY)
    assert baseline_model.should_verify(_goal("b1", "Baseline risky", risk_signal=0.9)) is False

    spec = manager.stage_response_risk_candidate(
        goal_id="goal_safe_automl",
        title="Calibrate response risk model",
        training_examples=_training_examples(),
        anchor_cases=_anchor_cases(),
        transfer_cases=_transfer_cases(),
        canary_cases=_canary_cases(),
        rationale="Train a safer verification selector from labeled response failures and risky queries.",
        search_space=_search_space(),
        threshold=0.99,
    )
    training = manager.train_candidate(spec.change_id)
    assert training.train_metrics["accuracy"] >= 1.0
    assert registry.has_candidate(RESPONSE_RISK_FAMILY) is True

    regression = manager.run_regression_gate(spec.change_id)
    assert regression.passed is True

    manager.promote_canary(spec.change_id)
    canary = manager.evaluate_canary(spec.change_id)
    assert canary.passed is True
    assert canary.rolled_back is False

    finalized = manager.finalize_model(spec.change_id)
    assert registry.get_active_model(RESPONSE_RISK_FAMILY).model_id == finalized.model_id
    planning.set_response_planning_policy("adaptive_risk_model")
    plan = planning.make_plan(_goal("heldout", "Held-out risky", risk_signal=0.9), world_state={})
    assert "verify_outcome" in [step.name for step in plan.steps]


def test_safe_automl_manager_rolls_back_failed_canary(tmp_path) -> None:
    system = build_system(tmp_path / "runtime")
    registry = system["model_registry"]
    manager = system["automl_manager"]
    planning = system["planning"]

    stable_before = registry.get_active_model(RESPONSE_RISK_FAMILY)
    failing_canary_cases = [
        SkillArenaCase(
            case_id="risk_canary_forced_failure",
            payload={"goal": _goal("f1", "Forced failure", risk_signal=0.9).to_dict()},
            expected={"predicted_verify": False, "required_steps": ["understand_request", "form_response"], "forbidden_steps": ["verify_outcome"]},
            description="This case intentionally forces the canary to fail and rollback.",
        )
    ]

    spec = manager.stage_response_risk_candidate(
        goal_id="goal_safe_automl_fail",
        title="Calibrate response risk model with forced rollback",
        training_examples=_training_examples(),
        anchor_cases=_anchor_cases(),
        transfer_cases=_transfer_cases(),
        canary_cases=failing_canary_cases,
        rationale="Train a candidate but verify rollback behavior on a failing canary.",
        search_space=_search_space(),
        threshold=0.99,
    )
    manager.train_candidate(spec.change_id)
    assert manager.run_regression_gate(spec.change_id).passed is True
    manager.promote_canary(spec.change_id)
    canary = manager.evaluate_canary(spec.change_id)

    assert canary.passed is False
    assert canary.rolled_back is True
    assert registry.get_active_model(RESPONSE_RISK_FAMILY).model_id == stable_before.model_id
    assert planning.response_risk_model.model_id == stable_before.model_id
