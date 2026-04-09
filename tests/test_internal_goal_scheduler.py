from __future__ import annotations

from memory.improvement_ledger import CapabilityGap
from planning.internal_goal_scheduler import InternalGoalScheduler


def test_internal_goal_scheduler_selects_most_severe_gap():
    scheduler = InternalGoalScheduler()
    gaps = [
        CapabilityGap("cap_a", "low_confidence", 0.2, 0.8, 3, 0.9, 0.0),
        CapabilityGap("cap_b", "transfer_deficit", 0.7, 0.5, 2, 0.2, 0.1),
    ]
    chosen = scheduler.select_gap(gaps)
    assert chosen is not None
    assert chosen.capability_id == "cap_b"
    goal = scheduler.materialize_goal(chosen, priority=0.77)
    assert goal.goal_type == "improve_transfer"
    assert goal.priority == 0.77
