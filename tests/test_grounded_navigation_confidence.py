from __future__ import annotations

from environments.grounded_navigation import GroundedNavigationLab


def test_grounded_navigation_result_contains_confidence_and_failure_reason():
    lab = GroundedNavigationLab()
    task = lab.get_task("nav_detour_wall")
    result = lab.solve_task(task, strategy="greedy")
    assert "confidence" in result
    assert "failure_reason" in result
    assert result["failure_reason"] == "planner_overexpansion"
