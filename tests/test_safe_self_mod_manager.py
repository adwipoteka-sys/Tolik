from __future__ import annotations

import pytest

from autonomous_agi import (
    _navigation_self_mod_anchor_cases,
    _navigation_self_mod_canary_cases,
    _navigation_transfer_cases,
)
from benchmarks.skill_arena import SkillArenaCase
from benchmarks.transfer_suite import TransferCase
from environments.grounded_navigation import GroundedNavigationLab
from main import build_system


def test_safe_self_mod_manager_promotes_graph_search_after_regression_and_canary(tmp_path) -> None:
    system = build_system(tmp_path / "runtime")
    manager = system["self_mod_manager"]
    agency = system["agency"]
    ledger = system["ledger"]

    spec = manager.stage_attribute_change(
        goal_id="goal_nav_patch",
        title="Upgrade grounded navigation strategy",
        target_component="agency",
        capability="grounded_navigation",
        parameter_name="grounded_navigation_strategy",
        candidate_value="graph_search",
        anchor_cases=_navigation_self_mod_anchor_cases(),
        transfer_cases=_navigation_transfer_cases(),
        canary_cases=_navigation_self_mod_canary_cases(),
        rationale="graph-search should replace the brittle greedy policy only after gated validation",
        threshold=0.99,
    )

    regression = manager.run_regression_gate(spec.change_id)
    assert regression.passed is True
    assert manager.get_spec(spec.change_id).status == "regression_validated"

    manager.promote_canary(spec.change_id)
    canary = manager.evaluate_canary(spec.change_id)
    assert canary.passed is True
    assert canary.rolled_back is False
    assert agency.grounded_navigation_strategy == "graph_search"

    finalized = manager.finalize_change(spec.change_id)
    assert finalized.status == "finalized"
    assert agency.grounded_navigation_strategy == "graph_search"

    assert list((ledger.root / "self_modification_specs").glob("*.json"))
    assert list((ledger.root / "self_modification_regressions").glob("*.json"))
    assert list((ledger.root / "self_modification_canaries").glob("*.json"))


def test_safe_self_mod_manager_rolls_back_bad_candidate_on_canary(tmp_path) -> None:
    system = build_system(tmp_path / "runtime")
    manager = system["self_mod_manager"]
    agency = system["agency"]

    agency.set_navigation_strategy("graph_search")
    lab = GroundedNavigationLab()
    easy_anchor = [
        SkillArenaCase(
            case_id="easy_open",
            payload={"tasks": [lab.get_task("nav_easy_open").to_dict()], "success_threshold": 1.0},
            expected={"min_success_rate": 1.0, "max_mean_path_ratio": 1.0},
            description="easy open route remains solvable",
        ),
        SkillArenaCase(
            case_id="easy_corner",
            payload={"tasks": [lab.get_task("nav_easy_corner").to_dict()], "success_threshold": 1.0},
            expected={"min_success_rate": 1.0, "max_mean_path_ratio": 1.0},
            description="easy corner route remains solvable",
        ),
    ]
    easy_transfer = [
        TransferCase(
            case_id="easy_transfer",
            payload={"tasks": [lab.get_task("nav_easy_open").to_dict()], "success_threshold": 1.0},
            expected={"min_success_rate": 1.0, "max_mean_path_ratio": 1.0, "min_task_count": 1},
            description="easy held-out route remains solvable",
        )
    ]

    spec = manager.stage_attribute_change(
        goal_id="goal_bad_nav_patch",
        title="Try brittle greedy fallback again",
        target_component="agency",
        capability="grounded_navigation",
        parameter_name="grounded_navigation_strategy",
        candidate_value="greedy",
        anchor_cases=easy_anchor,
        transfer_cases=easy_transfer,
        canary_cases=_navigation_self_mod_canary_cases(),
        rationale="test canary rollback",
        threshold=0.99,
    )

    regression = manager.run_regression_gate(spec.change_id)
    assert regression.passed is True
    manager.promote_canary(spec.change_id)

    canary = manager.evaluate_canary(spec.change_id)
    assert canary.passed is False
    assert canary.rolled_back is True
    assert agency.grounded_navigation_strategy == "graph_search"
    assert manager.get_spec(spec.change_id).status == "rolled_back"

    with pytest.raises(ValueError):
        manager.finalize_change(spec.change_id)
