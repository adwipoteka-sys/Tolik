from __future__ import annotations

from memory.episodic_memory import EpisodicMemory
from memory.goal_ledger import GoalLedger
from memory.memory_module import MemoryModule
from memory.semantic_promoter import SemanticPromoter
from motivation.goal_schema import Goal, GoalBudget, GoalKind, GoalSource, SuccessCriterion
from reasoning.reasoning_module import ReasoningModule


PATTERN = "text_summarizer|blank_input_guard"
LESSON = "Ignore blank inputs and enforce runtime sentence limits for text_summarizer canaries."



def _goal(goal_id: str) -> Goal:
    return Goal(
        goal_id=goal_id,
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



def test_semantic_promoter_promotes_repeated_success(tmp_path) -> None:
    ledger = GoalLedger(tmp_path / "ledger")
    episodic = EpisodicMemory(ledger=ledger)
    memory = MemoryModule(goal_ledger=ledger)
    promoter = SemanticPromoter(reasoning=ReasoningModule(), ledger=ledger, min_support=2)

    episodic.record_goal_episode(_goal("goal_patch_a"), cycle=4, trace=[], outcome={"success": True}, pattern_key=PATTERN, lesson=LESSON)
    episodic.record_goal_episode(_goal("goal_patch_b"), cycle=5, trace=[], outcome={"success": True}, pattern_key=PATTERN, lesson=LESSON)

    promotions = promoter.consolidate(memory=memory, episodic_memory=episodic, pattern_key=PATTERN)

    assert len(promotions) == 1
    promotion = promotions[0]
    assert promotion.summary == LESSON
    assert promotion.support_count == 2
    assert promotion.fact_key in memory.semantic_facts()

    restored = SemanticPromoter(reasoning=ReasoningModule(), ledger=ledger, min_support=2)
    assert restored.get(PATTERN) is not None
