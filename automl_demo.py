from __future__ import annotations

import argparse
from pathlib import Path

from automl.response_risk_model import ResponseRiskTrainingExample
from benchmarks.skill_arena import SkillArenaCase
from benchmarks.transfer_suite import TransferCase
from main import build_system
from motivation.goal_schema import Goal, GoalBudget, GoalKind, GoalSource, SuccessCriterion


def make_goal(
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


def training_examples() -> list[ResponseRiskTrainingExample]:
    return [
        ResponseRiskTrainingExample(goal=make_goal("g_explicit", "Explicit verify", requires_verification=True).to_dict(), label=True),
        ResponseRiskTrainingExample(goal=make_goal("g_high_risk", "High risk", risk_signal=0.9).to_dict(), label=True),
        ResponseRiskTrainingExample(goal=make_goal("g_missing_evidence", "Missing evidence", insufficient_evidence=True, risk_signal=0.2).to_dict(), label=True),
        ResponseRiskTrainingExample(goal=make_goal("g_routine", "Routine question", risk_signal=0.1).to_dict(), label=False),
    ]


def anchor_cases() -> list[SkillArenaCase]:
    return [
        SkillArenaCase(
            case_id="risk_anchor_explicit_verify",
            payload={"goal": make_goal("a1", "Explicit verify", requires_verification=True).to_dict()},
            expected={"predicted_verify": True, "required_steps": ["understand_request", "verify_outcome", "form_response"], "forbidden_steps": []},
            description="Explicit verification requirements must remain verified.",
        ),
        SkillArenaCase(
            case_id="risk_anchor_routine_question",
            payload={"goal": make_goal("a2", "Routine question", risk_signal=0.1).to_dict()},
            expected={"predicted_verify": False, "required_steps": ["understand_request", "form_response"], "forbidden_steps": ["verify_outcome"]},
            description="Routine low-risk questions should stay lightweight.",
        ),
    ]


def transfer_cases() -> list[TransferCase]:
    return [
        TransferCase(
            case_id="risk_transfer_high_risk_generalization",
            payload={"goal": make_goal("t1", "High risk generalization", risk_signal=0.85).to_dict()},
            expected={"predicted_verify": True, "required_steps": ["understand_request", "verify_outcome", "form_response"], "forbidden_steps": []},
            description="The trained model should verify held-out high-risk tasks.",
        ),
        TransferCase(
            case_id="risk_transfer_insufficient_evidence",
            payload={"goal": make_goal("t2", "Insufficient evidence", insufficient_evidence=True, risk_signal=0.2).to_dict()},
            expected={"predicted_verify": True, "required_steps": ["understand_request", "verify_outcome", "form_response"], "forbidden_steps": []},
            description="The trained model should verify held-out insufficient-evidence tasks.",
        ),
    ]


def canary_cases() -> list[SkillArenaCase]:
    return [
        SkillArenaCase(
            case_id="risk_canary_high_risk",
            payload={"goal": make_goal("c1", "Canary high risk", risk_signal=0.9).to_dict()},
            expected={"predicted_verify": True, "required_steps": ["understand_request", "verify_outcome", "form_response"], "forbidden_steps": []},
            description="Canary must verify a high-risk task.",
        ),
        SkillArenaCase(
            case_id="risk_canary_routine",
            payload={"goal": make_goal("c2", "Canary routine", risk_signal=0.1).to_dict()},
            expected={"predicted_verify": False, "required_steps": ["understand_request", "form_response"], "forbidden_steps": ["verify_outcome"]},
            description="Canary must preserve the lightweight routine path.",
        ),
    ]


def search_space() -> dict[str, list[float]]:
    return {
        "threshold": [0.5],
        "bias": [0.0],
        "weight_requires_verification": [1.0],
        "weight_insufficient_evidence": [0.0, 1.0],
        "weight_high_risk_signal": [0.0, 1.0],
        "weight_critical_tag": [0.0],
    }


def run_demo(runtime_dir: Path) -> None:
    system = build_system(runtime_dir)
    registry = system["model_registry"]
    manager = system["automl_manager"]
    planning = system["planning"]

    print("Tolik v3.133 — safe AutoML / model-registry loop")
    print(f"Runtime ledger: {runtime_dir / 'ledger'}")
    baseline = registry.get_active_model("response_risk_model")
    print(f"Stable model before training: {baseline.model_id} ({baseline.version})")

    spec = manager.stage_response_risk_candidate(
        goal_id="goal_safe_automl_demo",
        title="Calibrate response risk model",
        training_examples=training_examples(),
        anchor_cases=anchor_cases(),
        transfer_cases=transfer_cases(),
        canary_cases=canary_cases(),
        rationale="Train a safer verification selector from labeled response failures and risky queries.",
        search_space=search_space(),
        threshold=0.99,
    )
    print(f"Staged AutoML candidate: {spec.change_id}")

    training = manager.train_candidate(spec.change_id)
    print(f"Training report: accuracy={training.train_metrics['accuracy']} best_config={training.best_config}")

    regression = manager.run_regression_gate(spec.change_id)
    print(f"Regression gate passed: {regression.passed} | reasons={regression.failure_reasons}")

    manager.promote_canary(spec.change_id)
    canary = manager.evaluate_canary(spec.change_id)
    print(f"Canary passed: {canary.passed} | rolled_back={canary.rolled_back}")

    finalized = manager.finalize_model(spec.change_id)
    print(f"Finalized model: {finalized.model_id} ({finalized.version})")

    planning.set_response_planning_policy("adaptive_risk_model")
    held_out = make_goal("heldout", "Held-out risky", risk_signal=0.9)
    plan = planning.make_plan(held_out, world_state={})
    print(f"Held-out risky plan: {[step.name for step in plan.steps]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the safe AutoML / model-registry demo for Tolik v3.133.")
    parser.add_argument("--runtime-dir", type=Path, default=Path("runtime_automl"))
    args = parser.parse_args()
    run_demo(args.runtime_dir)
