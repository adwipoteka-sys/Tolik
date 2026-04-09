from __future__ import annotations

import json

from motivation.operator_charter import OperatorCharter, load_charter, save_charter


def test_operator_charter_blocks_forbidden_capabilities(tmp_path):
    charter = OperatorCharter(allow_cloud_llm=False, allow_quantum_solver=False)
    ok, reason = charter.goal_allowed(tags=["grounded_self_training"], required_capabilities=["classical_planning", "cloud_llm"])
    assert ok is False
    assert "cloud_llm" in str(reason)


def test_operator_charter_roundtrip(tmp_path):
    charter = OperatorCharter(name="roundtrip", navigation_batch_size=4, navigation_success_threshold=0.75)
    path = tmp_path / "charter.json"
    save_charter(charter, path)
    loaded = load_charter(path)
    assert loaded.name == "roundtrip"
    assert loaded.navigation_batch_size == 4
    assert loaded.navigation_success_threshold == 0.75
    assert json.loads(path.read_text(encoding="utf-8"))["name"] == "roundtrip"
