from __future__ import annotations

from metacognition.failure_miner import FailureCase
from tooling.patch_planner import PatchPlanner


def _failure_case() -> FailureCase:
    return FailureCase(
        case_id="failure_tool_1",
        goal_id="tool_goal_1",
        capability="text_summarizer",
        tool_name="generated_text_summarizer_v2",
        tool_version="v3.120",
        rollout_stage="canary",
        payload={
            "texts": ["alpha", "   ", "beta"],
            "max_sentences": 2,
        },
        input_shape={"texts_count": 3, "nonempty_texts_count": 2, "blank_texts_count": 1, "requested_max_sentences": 2},
        violation_types=["source_count_mismatch", "sentence_limit_exceeded"],
        expected={"source_count": 2, "max_sentences": 2, "required_summary_nonempty": True},
        actual={"source_count": 3, "sentences_used": 3},
        rollback_target="generated_text_summarizer_v1",
        signature="text_summarizer|source_count_mismatch+sentence_limit_exceeded|blank_inputs|limit=2",
        notes=["ignore_blank_inputs"],
    )


def test_patch_planner_builds_tool_creation_goal() -> None:
    goal = PatchPlanner().build_goal(_failure_case(), occurrence_count=2)
    assert goal.kind.name == "TOOL_CREATION"
    assert goal.evidence["template_parameters"]["variant"] == "blank_input_guard"
    assert "ignore_blank_inputs" in goal.evidence["remediation_targets"]
    assert goal.evidence["failure_signature"].startswith("text_summarizer|")
