from __future__ import annotations

from memory.improvement_ledger import CapabilityGap, ImprovementGoal
from metacognition.failure_analyzer import FailureAnalyzer


def test_failure_analyzer_builds_curriculum_and_flags_tool_need():
    analyzer = FailureAnalyzer()
    gap = CapabilityGap("grounded_navigation", "transfer_deficit", 0.8, 0.4, 1, 0.2, 0.0)
    reports = [type("Report", (), {"failure_reason": "tool_missing_hint"})()]
    signature = analyzer.analyze_gap(gap, reports)
    assert signature.needs_tool_proposal is True
    goal = ImprovementGoal(
        goal_id="g1",
        capability_id="grounded_navigation",
        goal_type="improve_transfer",
        target_metric="transfer_score",
        target_value=0.8,
        priority=0.8,
        budget_steps=3,
        cooldown_cycles=3,
    )
    curriculum = analyzer.build_curriculum(goal, signature)
    assert 1 <= len(curriculum) <= 3
    assert all(item.capability_id == "grounded_navigation" for item in curriculum)
