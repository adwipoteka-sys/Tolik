from __future__ import annotations

from memory.goal_ledger import GoalLedger
from tooling.tooling_manager import ToolingManager


def test_controlled_generator_can_register_and_execute_summarizer(tmp_path) -> None:
    manager = ToolingManager(ledger=GoalLedger(tmp_path / "ledger"))
    spec = manager.generator.make_spec("text_summarizer")
    tool = manager.generator.generate(spec)
    assert tool.validation.allowed is True
    runtime = manager.sandbox.load_callable(tool.source_code)
    manager.registry.register(tool, runtime)

    result = manager.registry.execute_by_capability(
        "text_summarizer",
        {"texts": ["Alpha system became stable.", "Beta notes confirm shorter plans."], "max_sentences": 2},
    )
    assert result["source_count"] == 2
    assert "stable" in result["summary"].lower() or "shorter" in result["summary"].lower()
