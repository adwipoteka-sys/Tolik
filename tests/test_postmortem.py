from __future__ import annotations

from metacognition.postmortem import PostmortemAnalyzer
from motivation.goal_schema import (
    Goal,
    GoalBudget,
    GoalKind,
    GoalSource,
    SuccessCriterion,
)


def make_goal(kind: GoalKind = GoalKind.LEARNING, source: GoalSource = GoalSource.MEMORY_GAP) -> Goal:
    return Goal(
        goal_id="g1",
        title="Goal",
        description="Goal",
        source=source,
        kind=kind,
        expected_gain=0.7,
        novelty=0.2,
        uncertainty_reduction=0.8,
        strategic_fit=0.9,
        risk_estimate=0.1,
        priority=0.8,
        risk_budget=0.5,
        resource_budget=GoalBudget(max_steps=2, max_seconds=5.0),
        success_criteria=[SuccessCriterion(metric="status", comparator="==", target="done")],
        tags=["test"],
    )


def test_model_error_classification() -> None:
    analyzer = PostmortemAnalyzer()
    report = analyzer.analyze(
        make_goal(),
        trace=[{"step": "x", "result": {"success": False}}],
        expected={},
        observed={"goal_success": False, "prediction_error": 0.9},
    )
    assert "model_error" in report.root_causes


def test_plan_error_classification() -> None:
    analyzer = PostmortemAnalyzer()
    report = analyzer.analyze(
        make_goal(),
        trace=[{"step": "x", "result": {"success": True}}],
        expected={},
        observed={"goal_success": False, "prediction_error": 0.0},
    )
    assert "plan_error" in report.root_causes


def test_knowledge_gap_classification() -> None:
    analyzer = PostmortemAnalyzer()
    report = analyzer.analyze(
        make_goal(),
        trace=[{"step": "x", "result": {"success": False, "knowledge_gap": True}}],
        expected={},
        observed={"goal_success": False},
    )
    assert "knowledge_gap" in report.root_causes


def test_regression_flag_from_anchor_failure() -> None:
    analyzer = PostmortemAnalyzer()
    report = analyzer.analyze(
        make_goal(kind=GoalKind.REGRESSION_RECOVERY, source=GoalSource.REGRESSION_FAILURE),
        trace=[{"step": "replay_anchor_suite", "result": {"success": False}}],
        expected={},
        observed={"goal_success": False, "anchor_suite_failed": True},
    )
    assert report.regression_flag is True
    assert "regression_failure" in report.root_causes
