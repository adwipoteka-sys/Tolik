from __future__ import annotations

from metacognition.failure_miner import FailureMiner
from motivation.goal_schema import Goal, GoalBudget, GoalKind, GoalSource, SuccessCriterion, new_goal_id
from tooling.runtime_guard import RuntimeAssessment
from tooling.tool_spec import GeneratedTool, ToolSpec, ToolValidationReport


def _goal() -> Goal:
    return Goal(
        goal_id=new_goal_id("tool"),
        title="Roll out canary upgrade for text_summarizer",
        description="Upgrade summarizer safely.",
        source=GoalSource.METACOGNITION,
        kind=GoalKind.TOOL_CREATION,
        expected_gain=0.7,
        novelty=0.4,
        uncertainty_reduction=0.8,
        strategic_fit=0.9,
        risk_estimate=0.1,
        priority=0.8,
        risk_budget=0.2,
        resource_budget=GoalBudget(max_steps=8, max_seconds=25.0),
        success_criteria=[SuccessCriterion(metric="tool_promoted", comparator="==", target=True)],
        required_capabilities=["classical_planning"],
        tags=["tooling", "text_summarizer"],
        evidence={"target_capability": "text_summarizer"},
    )


def test_failure_miner_structures_canary_failure() -> None:
    tool = GeneratedTool(
        spec=ToolSpec(
            name="generated_text_summarizer_v2",
            capability="text_summarizer",
            description="regressive",
            template_name="text_summarizer",
            parameters={"variant": "counts_raw_inputs"},
        ),
        source_code="def run_tool(payload):\n    return {}\n",
        validation=ToolValidationReport(allowed=True),
        version="v3.120",
    )
    payload = {
        "texts": ["alpha stable.", "   ", "beta concise."],
        "max_sentences": 2,
    }
    assessment = RuntimeAssessment(
        capability="text_summarizer",
        tool_name=tool.name,
        passed=False,
        score=0.5,
        violations=["source_count_mismatch", "sentence_limit_exceeded"],
        details={"source_count": 3, "sentences_used": 3},
    )
    failure = FailureMiner().mine_canary_failure(
        goal=_goal(),
        tool=tool,
        payload=payload,
        assessment=assessment,
        output={"summary": "alpha stable. beta concise. gamma extra.", "source_count": 3, "sentences_used": 3},
        rollback_target="generated_text_summarizer_v1",
    )

    assert failure.capability == "text_summarizer"
    assert failure.expected["source_count"] == 2
    assert failure.actual["source_count"] == 3
    assert failure.input_shape["blank_texts_count"] == 1
    assert failure.rollback_target == "generated_text_summarizer_v1"
    assert "source_count_mismatch" in failure.violation_types
    assert failure.signature.startswith("text_summarizer|")
