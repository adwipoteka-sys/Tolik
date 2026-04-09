from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from motivation.goal_schema import Goal, GoalSource


@dataclass(slots=True)
class PostmortemReport:
    goal_id: str
    success: bool
    expectation_gap: float
    model_error: float
    plan_error: float
    execution_error: float
    knowledge_gap: float
    regression_flag: bool
    drift_flag: bool
    root_causes: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    derived_goals: list[dict[str, Any]] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PostmortemAnalyzer:
    """Classifies the dominant failure mode of an episode."""

    def analyze(
        self,
        goal: Goal,
        trace: list[dict[str, Any]],
        expected: dict[str, Any],
        observed: dict[str, Any],
    ) -> PostmortemReport:
        step_results = [item.get("result", {}) for item in trace]
        any_failure = any(not result.get("success", True) for result in step_results)

        explicit_success = observed.get("goal_success")
        if explicit_success is None:
            success = not any_failure and not observed.get("anchor_suite_failed", False)
        else:
            success = bool(explicit_success)

        model_error = float(observed.get("prediction_error", expected.get("prediction_error", 0.0)))
        execution_error = 1.0 if any(result.get("execution_error") for result in step_results) else 0.0
        knowledge_gap = 1.0 if any(
            result.get("knowledge_gap") or result.get("insufficient_evidence")
            for result in step_results
        ) else 0.0
        regression_flag = bool(observed.get("anchor_suite_failed") or observed.get("regression_flag"))
        drift_flag = bool(observed.get("drift_flag") or goal.source == GoalSource.DRIFT_ALARM)

        plan_error = 0.0
        if (not success) and model_error < 0.4 and execution_error == 0.0 and knowledge_gap == 0.0 and not regression_flag:
            plan_error = 1.0

        expectation_gap = 0.0 if success else 1.0

        root_causes: list[str] = []
        if regression_flag:
            root_causes.append("regression_failure")
        if knowledge_gap > 0.0:
            root_causes.append("knowledge_gap")
        if execution_error > 0.0:
            root_causes.append("execution_error")
        if model_error >= 0.4:
            root_causes.append("model_error")
        if plan_error > 0.0:
            root_causes.append("plan_error")
        if not root_causes and success:
            root_causes.append("successful_execution")

        recommendations: list[str] = []
        derived_goals: list[dict[str, Any]] = []

        if "model_error" in root_causes:
            recommendations.append("Calibrate the world model on drifted or mispredicted cases.")
            derived_goals.append({"title": "Calibrate world model", "kind": "learning"})
        if "plan_error" in root_causes:
            recommendations.append("Run short planning drills and compare alternative decompositions.")
            derived_goals.append({"title": "Practice planning drills", "kind": "learning"})
        if "execution_error" in root_causes:
            recommendations.append("Rehearse the failing execution steps in a controlled simulator.")
            derived_goals.append({"title": "Rehearse failing steps", "kind": "maintenance"})
        if "knowledge_gap" in root_causes:
            recommendations.append("Study or retrieve the missing facts before retrying the task.")
            derived_goals.append({"title": "Study missing knowledge", "kind": "learning"})
        if "regression_failure" in root_causes:
            recommendations.append("Replay the anchor suite and protect the recovered skill.")
            derived_goals.append({"title": "Replay anchor suite", "kind": "regression_recovery"})
        if "successful_execution" in root_causes:
            recommendations.append("Consolidate the successful pattern for future replay.")
            derived_goals.append({"title": "Consolidate successful strategy", "kind": "learning"})

        return PostmortemReport(
            goal_id=goal.goal_id,
            success=success,
            expectation_gap=expectation_gap,
            model_error=model_error,
            plan_error=plan_error,
            execution_error=execution_error,
            knowledge_gap=knowledge_gap,
            regression_flag=regression_flag,
            drift_flag=drift_flag,
            root_causes=root_causes,
            recommendations=recommendations,
            derived_goals=derived_goals[:3],
            tags=list(goal.tags),
        )
