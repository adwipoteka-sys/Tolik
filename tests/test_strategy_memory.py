from __future__ import annotations

from memory.goal_ledger import GoalLedger
from memory.strategy_memory import StrategyMemory


def test_strategy_memory_persists_and_rehydrates(tmp_path) -> None:
    ledger = GoalLedger(tmp_path / "ledger")
    memory = StrategyMemory(ledger=ledger)
    pattern = memory.register_patch_strategy(
        failure_signature="text_summarizer|blank_inputs|limit=2",
        capability="text_summarizer",
        template_parameters={"max_sentences": 3, "variant": "blank_input_guard"},
        remediation_targets=["ignore_blank_inputs"],
        source_goal_id="goal_patch",
        source_tool_name="generated_text_summarizer_v3",
    )

    restored = StrategyMemory(ledger=ledger)
    by_signature = restored.get_by_signature("text_summarizer|blank_inputs|limit=2")
    assert by_signature is not None
    assert by_signature.strategy_id == pattern.strategy_id
    assert by_signature.template_parameters["variant"] == "blank_input_guard"


def test_strategy_memory_tracks_reuse_outcomes(tmp_path) -> None:
    ledger = GoalLedger(tmp_path / "ledger")
    memory = StrategyMemory(ledger=ledger)
    pattern = memory.register_patch_strategy(
        failure_signature="text_summarizer|blank_inputs|limit=2",
        capability="text_summarizer",
        template_parameters={"max_sentences": 3, "variant": "blank_input_guard"},
    )

    updated = memory.record_outcome(pattern.strategy_id, passed=True)
    assert updated is not None
    assert updated.uses == 1
    assert updated.wins == 1
    assert updated.success_rate() == 1.0
