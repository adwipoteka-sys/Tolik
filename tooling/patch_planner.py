from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from motivation.goal_schema import Goal, GoalBudget, GoalKind, GoalSource, SuccessCriterion, new_goal_id
from metacognition.failure_miner import FailureCase


@dataclass(slots=True)
class PatchPlan:
    title: str
    description: str
    template_parameters: dict[str, Any]
    remediation_targets: list[str] = field(default_factory=list)
    priority_boost: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PatchPlanner:
    """Turns structured canary failures into bounded remediation goals."""

    def plan(self, failure: FailureCase, *, occurrence_count: int = 1) -> PatchPlan:
        capability = failure.capability
        remediation_targets: list[str] = []
        template_parameters: dict[str, Any] = {}

        if capability == "text_summarizer":
            variant = "blank_input_guard"
            template_parameters = {
                "max_sentences": 3,
                "variant": variant,
            }
            if "source_count_mismatch" in failure.violation_types:
                remediation_targets.append("count_only_normalized_inputs")
            if "sentence_limit_exceeded" in failure.violation_types:
                remediation_targets.append("respect_runtime_sentence_limit")
            if failure.input_shape.get("blank_texts_count", 0):
                remediation_targets.append("ignore_blank_inputs")
        else:
            template_parameters = dict(getattr(failure, "template_parameters", {}))
            remediation_targets.append("restore_runtime_contract")

        violation_label = ", ".join(failure.violation_types) if failure.violation_types else "runtime regression"
        title = f"Patch {capability} for {violation_label}"
        description = (
            f"Generate a patched {capability} candidate that fixes the live canary failure pattern "
            f"{failure.signature} and survives the expanded regression suite."
        )
        priority_boost = min(0.05 * max(occurrence_count - 1, 0), 0.15)
        return PatchPlan(
            title=title,
            description=description,
            template_parameters=template_parameters,
            remediation_targets=remediation_targets,
            priority_boost=priority_boost,
        )

    def build_goal(self, failure: FailureCase, *, occurrence_count: int = 1) -> Goal:
        patch = self.plan(failure, occurrence_count=occurrence_count)
        return Goal(
            goal_id=new_goal_id("toolpatch"),
            title=patch.title,
            description=patch.description,
            source=GoalSource.METACOGNITION,
            kind=GoalKind.TOOL_CREATION,
            expected_gain=min(0.82 + patch.priority_boost, 0.97),
            novelty=0.36,
            uncertainty_reduction=0.88,
            strategic_fit=0.94,
            risk_estimate=0.10,
            priority=0.86 + patch.priority_boost,
            risk_budget=0.20,
            resource_budget=GoalBudget(max_steps=8, max_seconds=25.0, max_tool_calls=1, max_api_calls=0),
            success_criteria=[SuccessCriterion(metric="tool_promoted", comparator="==", target=True)],
            required_capabilities=["classical_planning"],
            tags=["tooling", "patch", failure.capability, *failure.violation_types],
            evidence={
                "target_capability": failure.capability,
                "blocked_goal_title": f"Remediate {failure.capability} runtime regression",
                "template_parameters": patch.template_parameters,
                "canary_payload": dict(failure.payload),
                "failure_signature": failure.signature,
                "remediation_targets": list(patch.remediation_targets),
                "rollback_target": failure.rollback_target,
                "source_failure_case": failure.to_dict(),
            },
        )
