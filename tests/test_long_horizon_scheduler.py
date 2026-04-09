from __future__ import annotations

from memory.goal_ledger import GoalLedger
from motivation.goal_schema import Goal, GoalBudget, GoalKind, GoalSource, GoalStatus, SuccessCriterion
from planning.long_horizon_scheduler import LongHorizonScheduler



def _goal() -> Goal:
    return Goal(
        goal_id="goal_audit",
        title="Audit text_summarizer after semantic promotion",
        description="Audit text_summarizer after semantic promotion",
        source=GoalSource.SCHEDULER,
        kind=GoalKind.MAINTENANCE,
        expected_gain=0.7,
        novelty=0.2,
        uncertainty_reduction=0.8,
        strategic_fit=0.9,
        risk_estimate=0.1,
        priority=0.7,
        risk_budget=0.2,
        resource_budget=GoalBudget(max_steps=4, max_seconds=12.0),
        success_criteria=[SuccessCriterion(metric="status", comparator="==", target="done")],
        required_capabilities=["classical_planning", "text_summarizer"],
        tags=["scheduled", "skill_audit", "text_summarizer"],
        evidence={"target_capability": "text_summarizer"},
    )



def test_scheduler_releases_goal_when_due(tmp_path) -> None:
    ledger = GoalLedger(tmp_path / "ledger")
    scheduler = LongHorizonScheduler(ledger=ledger)
    scheduled = scheduler.schedule(_goal(), due_cycle=3, reason="semantic_promotion_followup")

    assert scheduled.goal.status == GoalStatus.DEFERRED
    assert scheduler.release_due(current_cycle=2) == []

    released = scheduler.release_due(current_cycle=3)
    assert len(released) == 1
    assert released[0].title == "Audit text_summarizer after semantic promotion"

    restored = LongHorizonScheduler(ledger=ledger)
    all_items = restored.list_all()
    assert len(all_items) == 1
    assert all_items[0].status == "released"
