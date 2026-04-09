from __future__ import annotations

from agency.agency_module import AgencyModule
from benchmarks.transfer_suite import TransferCase, TransferSuite
from environments.grounded_navigation import GroundedNavigationLab
from memory.goal_ledger import GoalLedger


def _heldout_cases() -> list[TransferCase]:
    lab = GroundedNavigationLab()
    return [
        TransferCase(
            case_id="bridge",
            payload={"tasks": [lab.get_task("nav_transfer_bridge").to_dict()], "success_threshold": 1.0},
            expected={"min_success_rate": 1.0, "max_mean_path_ratio": 1.0, "strategy": "graph_search", "min_task_count": 1},
        ),
        TransferCase(
            case_id="double_wall",
            payload={"tasks": [lab.get_task("nav_transfer_double_wall").to_dict()], "success_threshold": 1.0},
            expected={"min_success_rate": 1.0, "max_mean_path_ratio": 1.0, "strategy": "graph_search", "min_task_count": 1},
        ),
    ]


def test_transfer_suite_distinguishes_greedy_from_graph_search(tmp_path):
    ledger = GoalLedger(tmp_path / "ledger")
    suite = TransferSuite(ledger=ledger)
    agency = AgencyModule()
    cases = _heldout_cases()

    greedy = suite.run(
        capability="grounded_navigation",
        label="heldout_greedy",
        cases=cases,
        execute=lambda payload: agency.execute_capability("grounded_navigation", payload, rollout_stage="stable"),
        threshold=0.99,
    )
    assert greedy.passed is False
    assert greedy.mean_score < 0.99

    agency.set_navigation_strategy("graph_search")
    graph = suite.run(
        capability="grounded_navigation",
        label="heldout_graph_search",
        cases=cases,
        execute=lambda payload: agency.execute_capability("grounded_navigation", payload, rollout_stage="stable"),
        threshold=0.99,
    )
    assert graph.passed is True
    assert graph.mean_score == 1.0
    assert suite.latest_run("grounded_navigation").run_id == graph.run_id
    assert list((tmp_path / "ledger" / "transfer_runs").glob("*.json"))
