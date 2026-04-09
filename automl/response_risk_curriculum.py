from __future__ import annotations

from typing import Any

from automl.response_risk_model import ResponseRiskModel, ResponseRiskTrainingExample
from benchmarks.skill_arena import SkillArenaCase
from benchmarks.transfer_suite import TransferCase
from motivation.goal_schema import Goal, GoalBudget, GoalKind, GoalSource, SuccessCriterion


_RESPONSE_RISK_SIGNATURE = "response_risk_model|curriculum_upgrade"


def response_risk_signature() -> str:
    return _RESPONSE_RISK_SIGNATURE


def _goal(
    goal_id: str,
    title: str,
    *,
    requires_verification: bool = False,
    insufficient_evidence: bool = False,
    risk_signal: float = 0.1,
    critical_tag: bool = False,
) -> Goal:
    evidence: dict[str, Any] = {"risk_signal": risk_signal}
    tags = ["local_only"]
    if requires_verification:
        evidence["requires_verification"] = True
    if insufficient_evidence:
        evidence["insufficient_evidence"] = True
    if critical_tag:
        tags.append("critical_answer")
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
        tags=tags,
        evidence=evidence,
    )


def response_risk_should_verify_normatively(goal: Goal | dict[str, Any]) -> bool:
    goal_obj = goal if isinstance(goal, Goal) else Goal.from_dict(goal)
    evidence = dict(goal_obj.evidence)
    risk_signal = float(evidence.get("risk_signal", goal_obj.risk_estimate))
    return bool(
        evidence.get("requires_verification", False)
        or evidence.get("insufficient_evidence", False)
        or risk_signal >= 0.6
        or ("critical_answer" in goal_obj.tags or "high_risk" in goal_obj.tags or "critical" in goal_obj.tags)
    )


def build_training_examples() -> list[ResponseRiskTrainingExample]:
    return [
        ResponseRiskTrainingExample(goal=_goal("g_explicit", "Explicit verify", requires_verification=True).to_dict(), label=True),
        ResponseRiskTrainingExample(goal=_goal("g_high_risk", "High risk", risk_signal=0.9).to_dict(), label=True),
        ResponseRiskTrainingExample(goal=_goal("g_missing_evidence", "Missing evidence", insufficient_evidence=True, risk_signal=0.2).to_dict(), label=True),
        ResponseRiskTrainingExample(goal=_goal("g_critical_tag", "Critical answer", critical_tag=True).to_dict(), label=True),
        ResponseRiskTrainingExample(goal=_goal("g_routine", "Routine question", risk_signal=0.1).to_dict(), label=False),
    ]


def build_anchor_cases() -> list[SkillArenaCase]:
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


def build_transfer_cases() -> list[TransferCase]:
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
        TransferCase(
            case_id="risk_transfer_critical_tag",
            payload={"goal": _goal("t3", "Critical tag", critical_tag=True, risk_signal=0.1).to_dict()},
            expected={"predicted_verify": True, "required_steps": ["understand_request", "verify_outcome", "form_response"], "forbidden_steps": []},
            description="Critical tags should trigger verification even with low numeric risk.",
        ),
    ]


def build_canary_cases() -> list[SkillArenaCase]:
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


def build_search_space() -> dict[str, list[float]]:
    return {
        "threshold": [0.5],
        "bias": [0.0],
        "weight_requires_verification": [1.0],
        "weight_insufficient_evidence": [0.0, 1.0],
        "weight_high_risk_signal": [0.0, 1.0],
        "weight_critical_tag": [0.0, 0.5, 1.0],
    }


def build_audit_examples() -> list[ResponseRiskTrainingExample]:
    return [
        ResponseRiskTrainingExample(goal=_goal("audit_1", "Explicit verify", requires_verification=True).to_dict(), label=True, description="Audit explicit verify"),
        ResponseRiskTrainingExample(goal=_goal("audit_2", "High risk", risk_signal=0.9).to_dict(), label=True, description="Audit high risk"),
        ResponseRiskTrainingExample(goal=_goal("audit_3", "Insufficient evidence", insufficient_evidence=True, risk_signal=0.2).to_dict(), label=True, description="Audit insufficient evidence"),
        ResponseRiskTrainingExample(goal=_goal("audit_4", "Critical answer", critical_tag=True, risk_signal=0.1).to_dict(), label=True, description="Audit critical tag"),
        ResponseRiskTrainingExample(goal=_goal("audit_5", "Routine question", risk_signal=0.1).to_dict(), label=False, description="Audit routine low risk"),
    ]


RESPONSE_RISK_AUDIT_EXAMPLES = build_audit_examples()


def audit_response_risk_model(model: ResponseRiskModel) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    correct = 0
    false_negatives: list[str] = []
    false_positives: list[str] = []
    audit_case_ids = ["explicit", "high_risk", "insufficient_evidence", "critical_tag", "routine"]
    for case_id, example in zip(audit_case_ids, RESPONSE_RISK_AUDIT_EXAMPLES):
        goal = Goal.from_dict(example.goal)
        label = bool(example.label)
        prediction = bool(model.should_verify(goal))
        if prediction == label:
            correct += 1
        elif label:
            false_negatives.append(case_id)
        else:
            false_positives.append(case_id)
        cases.append({
            "case_id": case_id,
            "label": label,
            "prediction": prediction,
            "goal_id": goal.goal_id,
        })
    accuracy = correct / len(RESPONSE_RISK_AUDIT_EXAMPLES)
    return {
        "case_count": len(RESPONSE_RISK_AUDIT_EXAMPLES),
        "accuracy": round(accuracy, 3),
        "false_negative_count": len(false_negatives),
        "false_positive_count": len(false_positives),
        "false_negatives": false_negatives,
        "false_positives": false_positives,
        "cases": cases,
    }
