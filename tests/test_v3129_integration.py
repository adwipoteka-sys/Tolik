from __future__ import annotations

import json

from autonomous_agi import run_autonomous


def test_v3129_autonomous_run_uses_safe_self_modification_loop(tmp_path, capsys) -> None:
    runtime_dir = tmp_path / "runtime_agi"
    charter_path = tmp_path / "charter.json"
    charter_path.write_text(
        """
        {
          "name": "test_charter",
          "description": "test",
          "allowed_goal_tags": ["grounded_self_training", "grounded_navigation", "maintenance", "skill_audit", "transfer_curriculum", "navigation_route_explanation", "local_only"],
          "blocked_goal_tags": ["networked", "cloud_only", "unsafe"],
          "allow_grounded_navigation": true,
          "allow_tool_generation": true,
          "allow_cloud_llm": false,
          "allow_quantum_solver": false,
          "max_internal_goals_per_cycle": 1,
          "max_cycles_per_run": 8,
          "navigation_batch_size": 3,
          "navigation_max_difficulty": 2,
          "navigation_success_threshold": 0.67,
          "require_human_approval_for": ["cloud_llm", "quantum_solver", "network_install"]
        }
        """,
        encoding="utf-8",
    )

    run_autonomous(cycles=8, runtime_dir=runtime_dir, charter_path=charter_path)
    captured = capsys.readouterr().out

    assert "self-mod regression gate passed for grounded_navigation_strategy" in captured
    assert "self-mod canary passed for grounded_navigation_strategy" in captured
    assert "safe self-modification finalized: grounded_navigation_strategy=graph_search" in captured
    assert "completed growth path: grounded_navigation -> spatial_route_composition -> route_mission_briefing" in captured

    spec_dir = runtime_dir / "ledger" / "self_modification_specs"
    regression_dir = runtime_dir / "ledger" / "self_modification_regressions"
    canary_dir = runtime_dir / "ledger" / "self_modification_canaries"

    specs = [json.loads(path.read_text(encoding="utf-8")) for path in spec_dir.glob("*.json")]
    regressions = [json.loads(path.read_text(encoding="utf-8")) for path in regression_dir.glob("*.json")]
    canaries = [json.loads(path.read_text(encoding="utf-8")) for path in canary_dir.glob("*.json")]

    assert specs, "expected persisted self-modification specs"
    assert regressions and regressions[0]["passed"] is True
    assert canaries and canaries[0]["passed"] is True and canaries[0]["rolled_back"] is False
    assert any(spec["status"] == "finalized" for spec in specs)
