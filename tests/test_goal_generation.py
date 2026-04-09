from __future__ import annotations

from motivation.goal_schema import GoalSource
from motivation.internal_goal_sources import (
    detect_curiosity,
    detect_drift_alarm,
    detect_memory_gap,
    detect_regression_failure,
)


def test_drift_alarm_goal_is_generated() -> None:
    goals = detect_drift_alarm({"ood_score": 0.8}, {}, {})
    assert goals
    assert goals[0].source == GoalSource.DRIFT_ALARM


def test_memory_gap_goal_is_generated() -> None:
    goals = detect_memory_gap({"retrieval_confidence": 0.2}, {}, {})
    assert goals
    assert goals[0].source == GoalSource.MEMORY_GAP


def test_regression_failure_goal_is_generated() -> None:
    goals = detect_regression_failure({"anchor_suite_failed": True}, {}, {})
    assert goals
    assert goals[0].source == GoalSource.REGRESSION_FAILURE


def test_curiosity_goal_is_generated() -> None:
    goals = detect_curiosity({"novelty_score": 0.9, "queue_pressure": 0, "risk_estimate": 0.1}, {}, {})
    assert goals
    assert goals[0].source == GoalSource.CURIOSITY
