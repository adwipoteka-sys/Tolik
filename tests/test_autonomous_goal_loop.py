from __future__ import annotations

from pathlib import Path

from autonomous_agi import run_capability_goal_loop_once
from main import build_system


def test_autonomous_capability_goal_loop_creates_and_executes_internal_goal(tmp_path: Path):
    system = build_system(tmp_path / "runtime")
    portfolio = system["capability_portfolio"]
    portfolio.upsert_metrics(
        capability="grounded_navigation",
        confidence=0.35,
        transfer_score=0.20,
        episodes_to_stable=7,
        regression_delta=0.2,
        maturity_stage="emerging",
    )
    result = run_capability_goal_loop_once(system, current_cycle=0)
    assert result["selected_goal"] is not None
    assert result["gap"]["capability_id"] == "grounded_navigation"
    assert len(result["curriculum"]) >= 1
    assert system["workspace"].get_state()["latest_internal_goal"] is not None
    # cooldown should suppress immediate reselection
    second = run_capability_goal_loop_once(system, current_cycle=1)
    assert second["selected_goal"] is None
