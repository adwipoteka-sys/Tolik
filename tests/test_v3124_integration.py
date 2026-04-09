from __future__ import annotations

from autonomous_agi import run_autonomous


def test_v3124_autonomous_run_upgrades_navigation_and_schedules_audit(tmp_path, capsys):
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

    assert "Active grounded strategy: graph_search" in captured
    assert "scheduled audit" in captured
    assert list((runtime_dir / "ledger" / "skill_arena").glob("*.json"))
    assert list((runtime_dir / "ledger" / "scheduled_goals").glob("*.json"))
