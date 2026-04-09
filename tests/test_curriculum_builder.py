from __future__ import annotations

from metacognition.curriculum_builder import CurriculumBuilder
from metacognition.postmortem import PostmortemReport


def make_report() -> PostmortemReport:
    return PostmortemReport(
        goal_id="g1",
        success=False,
        expectation_gap=1.0,
        model_error=0.0,
        plan_error=1.0,
        execution_error=1.0,
        knowledge_gap=1.0,
        regression_flag=True,
        drift_flag=False,
        root_causes=["regression_failure", "knowledge_gap", "execution_error", "plan_error"],
        recommendations=[],
        derived_goals=[],
        tags=["regression", "knowledge_gap"],
    )


def test_curriculum_builder_creates_bounded_tasks() -> None:
    builder = CurriculumBuilder()
    tasks = builder.build(make_report())
    assert 1 <= len(tasks) <= 3
    assert all(task.budget.max_steps > 0 for task in tasks)
    assert all(task.success_criteria for task in tasks)


def test_regression_tasks_are_prioritized_first() -> None:
    builder = CurriculumBuilder()
    tasks = builder.build(make_report())
    assert "regression" in tasks[0].tags
