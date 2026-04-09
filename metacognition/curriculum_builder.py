from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from motivation.goal_schema import (
    Goal,
    GoalBudget,
    GoalKind,
    GoalSource,
    SuccessCriterion,
    new_goal_id,
)
from metacognition.postmortem import PostmortemReport


@dataclass(slots=True)
class CurriculumTask:
    title: str
    description: str
    source: str
    prerequisite_goal_id: str | None
    difficulty: float
    budget: GoalBudget
    success_criteria: list[SuccessCriterion]
    tags: list[str]


class CurriculumBuilder:
    """Converts postmortems into bounded, measurable training tasks."""

    def build(self, report: PostmortemReport) -> list[CurriculumTask]:
        tasks: list[CurriculumTask] = []

        def add_task(title: str, description: str, source: str, difficulty: float, tags: list[str]) -> None:
            tasks.append(
                CurriculumTask(
                    title=title,
                    description=description,
                    source=source,
                    prerequisite_goal_id=report.goal_id,
                    difficulty=difficulty,
                    budget=GoalBudget(max_steps=3, max_seconds=15.0),
                    success_criteria=[SuccessCriterion(metric="status", comparator="==", target="done")],
                    tags=tags,
                )
            )

        causes = list(report.root_causes)

        if "regression_failure" in causes:
            add_task(
                "Replay anchor regression suite",
                "Repeat anchor tasks until the previously working capability is restored.",
                "from_postmortem_regression_failure",
                0.9,
                ["regression", "anchor_suite"],
            )
        if "model_error" in causes:
            add_task(
                "Calibrate world model",
                "Train on cases where predictions diverged from observations.",
                "from_postmortem_model_error",
                0.7,
                ["model_error", "calibration"],
            )
        if "plan_error" in causes:
            add_task(
                "Run planning drills",
                "Practice short multi-step decompositions and compare alternatives.",
                "from_postmortem_plan_error",
                0.6,
                ["plan_error", "planning"],
            )
        if "knowledge_gap" in causes:
            add_task(
                "Study missing knowledge",
                "Acquire the facts or rules missing during the failed attempt.",
                "from_postmortem_knowledge_gap",
                0.5,
                ["knowledge_gap", "learning"],
            )
        if "execution_error" in causes:
            add_task(
                "Simulator rehearsal",
                "Rehearse the failing execution step in a safe simulator.",
                "from_postmortem_execution_error",
                0.55,
                ["execution_error", "rehearsal"],
            )
        if "successful_execution" in causes:
            add_task(
                "Consolidate successful strategy",
                "Store and replay the successful strategy to improve retention.",
                "from_postmortem_success",
                0.3,
                ["success", "consolidation"],
            )

        return self.prioritize(tasks)[:3]

    def prioritize(self, tasks: list[CurriculumTask]) -> list[CurriculumTask]:
        def priority_key(task: CurriculumTask) -> tuple[float, str]:
            bonus = 1.0 if "regression" in task.tags else 0.0
            return (task.difficulty + bonus, task.title)

        return sorted(tasks, key=priority_key, reverse=True)


def curriculum_task_to_goal(task: CurriculumTask) -> Goal:
    kind = GoalKind.REGRESSION_RECOVERY if "regression" in task.tags else GoalKind.LEARNING
    return Goal(
        goal_id=new_goal_id("curriculum"),
        title=task.title,
        description=task.description,
        source=GoalSource.CURRICULUM,
        kind=kind,
        expected_gain=0.74 if kind == GoalKind.REGRESSION_RECOVERY else 0.62,
        novelty=0.30,
        uncertainty_reduction=0.72,
        strategic_fit=0.80,
        risk_estimate=0.08,
        priority=0.75 if kind == GoalKind.REGRESSION_RECOVERY else 0.60,
        risk_budget=0.30,
        resource_budget=task.budget,
        success_criteria=task.success_criteria,
        parent_goal_id=task.prerequisite_goal_id,
        required_capabilities=["classical_planning"],
        tags=list(task.tags) + ["curriculum"],
        evidence={"task_source": task.source},
    )
