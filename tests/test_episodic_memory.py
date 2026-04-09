from __future__ import annotations

from memory.episodic_memory import EpisodicMemory
from memory.goal_ledger import GoalLedger
from motivation.goal_schema import Goal, GoalBudget, GoalKind, GoalSource, SuccessCriterion


def _goal() -> Goal:
    return Goal(
        goal_id="goal_episode",
        title="Patch summarizer",
        description="Patch summarizer",
        source=GoalSource.METACOGNITION,
        kind=GoalKind.TOOL_CREATION,
        expected_gain=0.8,
        novelty=0.3,
        uncertainty_reduction=0.8,
        strategic_fit=0.9,
        risk_estimate=0.1,
        priority=0.8,
        risk_budget=0.2,
        resource_budget=GoalBudget(max_steps=8, max_seconds=25.0, max_tool_calls=1, max_api_calls=0),
        success_criteria=[SuccessCriterion(metric="tool_promoted", comparator="==", target=True)],
        required_capabilities=["classical_planning"],
        tags=["tooling", "patch", "text_summarizer"],
        evidence={"target_capability": "text_summarizer"},
    )



def test_episode_memory_records_and_rehydrates(tmp_path) -> None:
    ledger = GoalLedger(tmp_path / "ledger")
    memory = EpisodicMemory(ledger=ledger)
    goal = _goal()

    record = memory.record_goal_episode(
        goal,
        cycle=4,
        trace=[{"step": "finalize_rollout", "result": {"success": True}}],
        outcome={"success": True},
        pattern_key="text_summarizer|blank_input_guard",
        lesson="Ignore blank inputs.",
    )

    assert record.capability == "text_summarizer"
    assert memory.support_count("text_summarizer|blank_input_guard") == 1

    restored = EpisodicMemory(ledger=ledger)
    assert len(restored.list_episodes()) == 1
    assert restored.list_episodes()[0].episode_id == record.episode_id
