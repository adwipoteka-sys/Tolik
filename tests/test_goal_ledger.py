from __future__ import annotations

from pathlib import Path

from memory.goal_ledger import GoalLedger
from motivation.goal_schema import (
    Goal,
    GoalBudget,
    GoalKind,
    GoalSource,
    SuccessCriterion,
)


def make_goal() -> Goal:
    return Goal(
        goal_id="g1",
        title="Persistent goal",
        description="Persistent goal",
        source=GoalSource.USER,
        kind=GoalKind.USER_TASK,
        expected_gain=0.7,
        novelty=0.2,
        uncertainty_reduction=0.4,
        strategic_fit=0.9,
        risk_estimate=0.1,
        priority=1.0,
        risk_budget=0.5,
        resource_budget=GoalBudget(max_steps=2, max_seconds=5.0),
        success_criteria=[SuccessCriterion(metric="status", comparator="==", target="done")],
    )


def test_ledger_writes_jsonl_and_restores(tmp_path: Path) -> None:
    ledger = GoalLedger(tmp_path)
    goal = make_goal()
    ledger.append_event({"event_type": "goal_ingested", "goal_id": goal.goal_id})
    ledger.save_goal_snapshot(goal)

    history = ledger.load_history()
    active_goals = ledger.load_active_goals()

    assert history
    assert history[0]["event_type"] == "goal_ingested"
    assert len(active_goals) == 1
    assert active_goals[0].goal_id == goal.goal_id
