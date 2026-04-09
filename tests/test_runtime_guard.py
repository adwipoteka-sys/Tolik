from __future__ import annotations

from motivation.goal_schema import Goal, GoalBudget, GoalKind, GoalSource, SuccessCriterion, new_goal_id
from tooling.runtime_guard import RuntimeGuard
from tooling.tool_spec import GeneratedTool, ToolSpec, ToolValidationReport
from tooling.tool_templates import render_text_summarizer
from tooling.sandbox import RestrictedSandbox
from tooling.tooling_manager import ToolingManager
from memory.goal_ledger import GoalLedger
from tooling.tool_registry import ToolRegistry


def _upgrade_goal() -> Goal:
    return Goal(
        goal_id=new_goal_id("tool"),
        title="Roll out canary upgrade for text_summarizer",
        description="Upgrade summarizer safely.",
        source=GoalSource.METACOGNITION,
        kind=GoalKind.TOOL_CREATION,
        expected_gain=0.74,
        novelty=0.41,
        uncertainty_reduction=0.70,
        strategic_fit=0.86,
        risk_estimate=0.10,
        priority=0.82,
        risk_budget=0.20,
        resource_budget=GoalBudget(max_steps=8, max_seconds=25.0, max_tool_calls=1, max_api_calls=0),
        success_criteria=[SuccessCriterion(metric="tool_promoted", comparator="==", target=True)],
        required_capabilities=["classical_planning"],
        tags=["tooling", "text_summarizer", "upgrade"],
        evidence={
            "target_capability": "text_summarizer",
            "template_parameters": {"variant": "counts_raw_inputs"},
            "canary_payload": {
                "texts": ["Stable output should survive rollback.", "   ", "Blank inputs must be handled."],
                "max_sentences": 2,
            },
        },
    )


def test_runtime_guard_detects_blank_input_source_count_regression() -> None:
    spec = ToolSpec(
        name="generated_text_summarizer_v2",
        capability="text_summarizer",
        description="regressive demo",
        template_name="text_summarizer",
        parameters={"max_sentences": 2, "variant": "counts_raw_inputs"},
    )
    runtime = RestrictedSandbox().load_callable(render_text_summarizer(spec))
    payload = {
        "texts": [
            "Stable output should survive rollback.",
            "   ",
            "Blank inputs must be handled.",
        ],
        "max_sentences": 2,
    }
    output = runtime(payload)
    assessment = RuntimeGuard().assess(
        capability="text_summarizer",
        tool_name=spec.name,
        payload=payload,
        output=output,
    )
    assert assessment.passed is False
    assert "source_count_mismatch" in assessment.violations


def test_tooling_manager_canary_failure_rolls_back_to_stable(tmp_path) -> None:
    ledger = GoalLedger(tmp_path / "ledger")
    manager = ToolingManager(ledger=ledger)
    stable = manager.seed_stable_tool("text_summarizer", name="generated_text_summarizer_v1")

    goal = _upgrade_goal()
    manager.design_tool_spec(goal)
    manager.generate_tool_code(goal)
    manager.validate_tool_code(goal)
    manager.register_tool(goal)
    benchmark = manager.benchmark_tool(goal)
    assert benchmark.passed is True
    manager.promote_canary(goal)

    outcome = manager.evaluate_canary(goal)
    assert outcome["passed"] is False
    assert outcome["rolled_back"] is True
    assert manager.registry.get_active_tool("text_summarizer").name == stable.name
    assert manager.registry.has_canary("text_summarizer") is False
