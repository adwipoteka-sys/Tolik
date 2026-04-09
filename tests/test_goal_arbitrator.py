from __future__ import annotations

from motivation.goal_arbitrator import GoalArbitrator
from motivation.goal_schema import (
    Goal,
    GoalBudget,
    GoalKind,
    GoalSource,
    SuccessCriterion,
)


def make_goal(title: str, risk: float = 0.1, gain: float = 0.8, capabilities: list[str] | None = None) -> Goal:
    return Goal(
        goal_id=title.replace(" ", "_"),
        title=title,
        description=title,
        source=GoalSource.CURIOSITY,
        kind=GoalKind.EXPLORATION,
        expected_gain=gain,
        novelty=0.5,
        uncertainty_reduction=0.6,
        strategic_fit=0.7,
        risk_estimate=risk,
        priority=0.5,
        risk_budget=0.5,
        resource_budget=GoalBudget(max_steps=2, max_seconds=5.0),
        success_criteria=[SuccessCriterion(metric="status", comparator="==", target="done")],
        required_capabilities=capabilities or ["classical_planning"],
    )


def test_deduplicate_similar_goals() -> None:
    arb = GoalArbitrator()
    goals = [make_goal("Same Title"), make_goal("same title")]
    unique = arb.deduplicate(goals)
    assert len(unique) == 1


def test_goals_above_risk_ceiling_are_rejected() -> None:
    arb = GoalArbitrator(hard_risk_ceiling=0.55)
    goals = [make_goal("Safe", risk=0.1), make_goal("Unsafe", risk=0.8)]
    admitted = arb.admit(goals, {"available_capabilities": {"classical_planning"}})
    assert [g.title for g in admitted] == ["Safe"]


def test_goal_without_capability_is_deferred() -> None:
    arb = GoalArbitrator()
    goal = make_goal("Needs Cloud", capabilities=["cloud_llm"])
    admissible, deferred = arb.filter_by_capability([goal], {"classical_planning"})
    assert not admissible
    assert deferred[0].status.value == "deferred"


def test_rank_orders_by_score() -> None:
    arb = GoalArbitrator()
    high = make_goal("High", gain=0.9)
    low = make_goal("Low", gain=0.3)
    ranked = arb.rank([low, high], {})
    assert ranked[0].title == "High"
