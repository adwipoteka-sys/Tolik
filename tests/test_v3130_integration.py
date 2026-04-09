from __future__ import annotations

import json

from autonomous_agi import run_autonomous


def test_v3130_autonomous_run_derives_self_mod_proposal_from_memory(tmp_path, capsys) -> None:
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

    assert "experience proposed self-mod goal: Promote graph-search grounded navigation" in captured
    assert "safe self-modification finalized: grounded_navigation_strategy=graph_search" in captured
    assert "Self-mod proposals:" in captured

    proposal_dir = runtime_dir / "ledger" / "self_modification_proposals"
    proposals = [json.loads(path.read_text(encoding="utf-8")) for path in proposal_dir.glob("*.json")]

    assert proposals, "expected persisted self-modification proposals"
    assert any(item["status"] == "finalized" for item in proposals)
    assert proposals[0]["candidate_value"] == "graph_search"
