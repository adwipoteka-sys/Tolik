from __future__ import annotations

from automl.response_risk_model import ResponseRiskModel, ResponseRiskTrainingExample, train_response_risk_model
from motivation.goal_schema import Goal, GoalBudget, GoalKind, GoalSource, SuccessCriterion


def _goal(
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
        goal_id=title.lower().replace(" ", "_"),
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


def test_baseline_response_risk_model_only_triggers_on_explicit_verification() -> None:
    model = ResponseRiskModel.baseline()

    assert model.should_verify(_goal("Explicit verify", requires_verification=True)) is True
    assert model.should_verify(_goal("Risky but unflagged", risk_signal=0.9)) is False
    assert model.should_verify(_goal("Missing evidence", insufficient_evidence=True, risk_signal=0.2)) is False


def test_training_learns_high_risk_and_insufficient_evidence_patterns() -> None:
    examples = [
        ResponseRiskTrainingExample(goal=_goal("Explicit verify", requires_verification=True).to_dict(), label=True),
        ResponseRiskTrainingExample(goal=_goal("High risk", risk_signal=0.9).to_dict(), label=True),
        ResponseRiskTrainingExample(goal=_goal("Insufficient evidence", insufficient_evidence=True, risk_signal=0.2).to_dict(), label=True),
        ResponseRiskTrainingExample(goal=_goal("Routine question", risk_signal=0.1).to_dict(), label=False),
    ]

    report = train_response_risk_model(examples)
    model = report.model

    assert report.train_metrics["accuracy"] >= 1.0
    assert model.should_verify(_goal("Explicit verify", requires_verification=True)) is True
    assert model.should_verify(_goal("High risk", risk_signal=0.9)) is True
    assert model.should_verify(_goal("Insufficient evidence", insufficient_evidence=True, risk_signal=0.2)) is True
    assert model.should_verify(_goal("Routine question", risk_signal=0.1)) is False
