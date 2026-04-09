from __future__ import annotations

from metacognition.failure_miner import FailureCase
from metacognition.postmortem_clusterer import PostmortemClusterer
from motivation.curriculum_registry import CurriculumRegistry


def _failure_case() -> FailureCase:
    return FailureCase(
        case_id="failure_tool_1",
        goal_id="tool_goal_1",
        capability="text_summarizer",
        tool_name="generated_text_summarizer_v2",
        tool_version="v3.120",
        rollout_stage="canary",
        payload={"texts": ["alpha", "   ", "beta"], "max_sentences": 2},
        input_shape={"texts_count": 3, "nonempty_texts_count": 2, "blank_texts_count": 1, "requested_max_sentences": 2},
        violation_types=["source_count_mismatch"],
        expected={"source_count": 2, "max_sentences": 2, "required_summary_nonempty": True},
        actual={"source_count": 3, "sentences_used": 2},
        rollback_target="generated_text_summarizer_v1",
        signature="text_summarizer|source_count_mismatch|blank_inputs|limit=2",
    )


def test_curriculum_registry_tracks_open_and_closed_patterns() -> None:
    failure = _failure_case()
    clusterer = PostmortemClusterer()
    cluster = clusterer.add_failure(failure)

    registry = CurriculumRegistry()
    pattern = registry.register_failure(failure, cluster)
    assert pattern.status == "open"
    registry.attach_remediation_goal(failure.signature, "Patch text_summarizer")
    assert registry.get(failure.signature).remediation_goals == ["Patch text_summarizer"]
    registry.mark_closed(failure.signature, "generated_text_summarizer_v3")
    assert registry.get(failure.signature).status == "closed"
