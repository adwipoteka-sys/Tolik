from __future__ import annotations

import pytest

from motivation.goal_schema import (
    Goal,
    GoalBudget,
    GoalKind,
    GoalSource,
    SuccessCriterion,
)


def _budget() -> GoalBudget:
    return GoalBudget(max_steps=2, max_seconds=5.0)


def _criterion() -> list[SuccessCriterion]:
    return [SuccessCriterion(metric="status", comparator="==", target="done")]


def test_goal_requires_success_criteria() -> None:
    with pytest.raises(ValueError):
        Goal(
            goal_id="g1",
            title="x",
            description="x",
            source=GoalSource.USER,
            kind=GoalKind.USER_TASK,
            expected_gain=0.5,
            novelty=0.2,
            uncertainty_reduction=0.3,
            strategic_fit=0.5,
            risk_estimate=0.1,
            priority=1.0,
            risk_budget=0.5,
            resource_budget=_budget(),
            success_criteria=[],
        )


def test_negative_budget_is_forbidden() -> None:
    with pytest.raises(ValueError):
        GoalBudget(max_steps=-1, max_seconds=1.0)


def test_invalid_comparator_raises() -> None:
    with pytest.raises(ValueError):
        SuccessCriterion(metric="m", comparator="!=", target=1)
