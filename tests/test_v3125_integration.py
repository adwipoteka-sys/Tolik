from __future__ import annotations

import json

from autonomous_agi import run_autonomous


def test_v3125_autonomous_run_records_transfer_and_marks_capability_ready(tmp_path, capsys):
    runtime_dir = tmp_path / "runtime_agi"
    charter_path = tmp_path / "charter.json"
    charter_path.write_text(
        """
        {
          "name": "test_charter",
          "description": "test",
          "allowed_goal_tags": ["grounded_self_training", "grounded_navigation", "maintenance", "skill_audit", "local_only"],
          "blocked_goal_tags": ["networked", "cloud_only", "unsafe"],
          "allow_grounded_navigation": true,
          "allow_tool_generation": true,
          "allow_cloud_llm": false,
          "allow_quantum_solver": false,
          "max_internal_goals_per_cycle": 1,
          "max_cycles_per_run": 6,
          "navigation_batch_size": 3,
          "navigation_max_difficulty": 2,
          "navigation_success_threshold": 0.67,
          "require_human_approval_for": ["cloud_llm", "quantum_solver", "network_install"]
        }
        """,
        encoding="utf-8",
    )

    run_autonomous(cycles=5, runtime_dir=runtime_dir, charter_path=charter_path)
    captured = capsys.readouterr().out

    assert "transfer suite" in captured
    assert "Ready capabilities: ['grounded_navigation']" in captured
    transfer_files = list((runtime_dir / "ledger" / "transfer_runs").glob("*.json"))
    portfolio_files = list((runtime_dir / "ledger" / "capability_portfolio").glob("*.json"))
    assert transfer_files
    assert portfolio_files
    state = json.loads(portfolio_files[0].read_text(encoding="utf-8"))
    assert state["capability"] == "grounded_navigation"
    assert state["maturity_stage"] == "transfer_validated"
