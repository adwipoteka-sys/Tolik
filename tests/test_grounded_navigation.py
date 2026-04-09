from __future__ import annotations

from environments.grounded_navigation import GroundedNavigationLab


def test_greedy_navigation_fails_detour_task():
    lab = GroundedNavigationLab()
    task = lab.get_task("nav_detour_wall")
    result = lab.solve_task(task, strategy="greedy")
    assert result["success"] is False


def test_graph_search_solves_detour_task_optimally():
    lab = GroundedNavigationLab()
    task = lab.get_task("nav_detour_wall")
    result = lab.solve_task(task, strategy="graph_search")
    assert result["success"] is True
    assert result["steps"] == result["optimal_steps"]


def test_run_batch_reports_success_rate_and_pass_flag():
    lab = GroundedNavigationLab()
    payload = {
        "tasks": [lab.get_task("nav_easy_open").to_dict(), lab.get_task("nav_detour_wall").to_dict()],
        "success_threshold": 1.0,
    }
    greedy = lab.run_batch(strategy="greedy", payload=payload)
    graph = lab.run_batch(strategy="graph_search", payload=payload)
    assert greedy["passed"] is False
    assert graph["passed"] is True
    assert greedy["success_rate"] < graph["success_rate"]
