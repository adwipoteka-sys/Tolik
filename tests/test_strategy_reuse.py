from __future__ import annotations

from memory.goal_ledger import GoalLedger
from memory.strategy_memory import StrategyMemory
from motivation.goal_schema import Goal, GoalBudget, GoalKind, GoalSource, SuccessCriterion
from tooling.tooling_manager import ToolingManager


def _goal() -> Goal:
    return Goal(
        goal_id="goal_strategy_reuse",
        title="Reuse learned strategy for text_summarizer edge canary",
        description="Reuse learned strategy",
        source=GoalSource.METACOGNITION,
        kind=GoalKind.TOOL_CREATION,
        expected_gain=0.7,
        novelty=0.2,
        uncertainty_reduction=0.8,
        strategic_fit=0.9,
        risk_estimate=0.1,
        priority=0.8,
        risk_budget=0.2,
        resource_budget=GoalBudget(max_steps=8, max_seconds=25.0, max_tool_calls=1, max_api_calls=0),
        success_criteria=[SuccessCriterion(metric="tool_promoted", comparator="==", target=True)],
        required_capabilities=["classical_planning"],
        tags=["tooling", "strategy_reuse"],
        evidence={
            "target_capability": "text_summarizer",
            "blocked_goal_title": "Validate reusable summarizer strategy",
            "strategy_lookup": {"reuse_learned_strategy": True},
            "canary_payload": {
                "texts": [
                    "Stable behavior followed policy tuning.",
                    "   ",
                    "Noise handling improved after switching strategy.",
                    "Operator review became faster.",
                ],
                "max_sentences": 2,
            },
        },
    )


def test_design_tool_spec_reuses_learned_strategy(tmp_path) -> None:
    ledger = GoalLedger(tmp_path / "ledger")
    strategies = StrategyMemory(ledger=ledger)
    pattern = strategies.register_patch_strategy(
        failure_signature="text_summarizer|sentence_limit_exceeded+source_count_mismatch|blank_inputs|limit=2",
        capability="text_summarizer",
        template_parameters={"max_sentences": 3, "variant": "blank_input_guard"},
        remediation_targets=["count_only_normalized_inputs", "respect_runtime_sentence_limit", "ignore_blank_inputs"],
    )
    manager = ToolingManager(ledger=ledger, strategy_memory=strategies)
    goal = _goal()

    spec = manager.design_tool_spec(goal)
    hint = manager.get_strategy_hint(goal.goal_id)

    assert spec.parameters["variant"] == "blank_input_guard"
    assert hint is not None
    assert hint["strategy_id"] == pattern.strategy_id


def test_strategy_reuse_candidate_passes_edge_canary(tmp_path) -> None:
    ledger = GoalLedger(tmp_path / "ledger")
    strategies = StrategyMemory(ledger=ledger)
    pattern = strategies.register_patch_strategy(
        failure_signature="text_summarizer|sentence_limit_exceeded+source_count_mismatch|blank_inputs|limit=2",
        capability="text_summarizer",
        template_parameters={"max_sentences": 3, "variant": "blank_input_guard"},
    )
    manager = ToolingManager(ledger=ledger, strategy_memory=strategies)
    manager.seed_stable_tool("text_summarizer")
    goal = _goal()

    manager.design_tool_spec(goal)
    manager.generate_tool_code(goal)
    manager.register_tool(goal)
    report = manager.benchmark_tool(goal)
    assert report.passed is True
    manager.promote_canary(goal)
    outcome = manager.evaluate_canary(goal)

    assert outcome["passed"] is True
    reused = strategies.get_by_id(pattern.strategy_id)
    assert reused is not None
    assert reused.uses == 1
    assert reused.wins == 1
