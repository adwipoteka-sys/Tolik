from __future__ import annotations

from memory.goal_ledger import GoalLedger
from tooling.policy_layer import PolicyLayer
from tooling.tooling_manager import ToolingManager


def test_policy_layer_rejects_try_statements() -> None:
    source = """
def run_tool(payload):
    try:
        value = int(payload.get("x", 0))
    except Exception:
        value = 0
    return {"value": value}
"""
    report = PolicyLayer().validate_source(source)
    assert report.allowed is False
    assert "try_statements_forbidden" in report.violations


def test_blank_input_guard_template_is_policy_clean(tmp_path) -> None:
    manager = ToolingManager(ledger=GoalLedger(tmp_path / "ledger"))
    spec = manager.generator.make_spec(
        "text_summarizer",
        parameters={"max_sentences": 3, "variant": "blank_input_guard"},
    )
    tool = manager.generator.generate(spec)
    assert tool.validation.allowed is True
    assert "try:" not in tool.source_code
    assert "try_statements_forbidden" not in tool.validation.violations
