from __future__ import annotations

from memory.goal_ledger import GoalLedger
from metacognition.failure_miner import FailureCase
from metacognition.postmortem_clusterer import FailureCluster
from motivation.autonomous_goal_manager import AutonomousGoalManager
from motivation.curriculum_registry import CurriculumRegistry
from motivation.goal_arbitrator import GoalArbitrator
from motivation.goal_schema import Goal, GoalBudget, GoalKind, GoalSource, GoalStatus, SuccessCriterion
from tooling.tooling_manager import ToolingManager


def _make_goal(goal_id: str = "g_restore") -> Goal:
    return Goal(
        goal_id=goal_id,
        title="Recoverable goal",
        description="Recoverable goal",
        source=GoalSource.USER,
        kind=GoalKind.USER_TASK,
        expected_gain=0.7,
        novelty=0.2,
        uncertainty_reduction=0.4,
        strategic_fit=0.9,
        risk_estimate=0.1,
        priority=1.0,
        risk_budget=0.5,
        resource_budget=GoalBudget(max_steps=2, max_seconds=5.0),
        success_criteria=[SuccessCriterion(metric="status", comparator="==", target="done")],
    )


def test_goal_manager_recovers_active_goal_from_snapshots(tmp_path) -> None:
    ledger = GoalLedger(tmp_path / "ledger")
    original = _make_goal()
    original.status = GoalStatus.ACTIVE
    ledger.save_goal_snapshot(original)

    restored = ledger.load_active_goals()
    assert len(restored) == 1
    assert restored[0].status == GoalStatus.ACTIVE

    manager = AutonomousGoalManager(ledger=ledger, arbitrator=GoalArbitrator())
    assert len(manager.active) == 1
    assert manager.active[0].goal_id == original.goal_id


def test_curriculum_registry_persists_patterns(tmp_path) -> None:
    ledger = GoalLedger(tmp_path / "ledger")
    registry = CurriculumRegistry(ledger=ledger)
    failure = FailureCase(
        case_id="failure_demo",
        goal_id="goal_demo",
        capability="text_summarizer",
        tool_name="generated_text_summarizer_v2",
        tool_version="v3.121",
        rollout_stage="canary",
        payload={"texts": ["A", " ", "B"], "max_sentences": 2},
        input_shape={"requested_max_sentences": 2, "blank_texts_count": 1},
        violation_types=["source_count_mismatch"],
        expected={"source_count": 2, "max_sentences": 2},
        actual={"source_count": 3},
        rollback_target="generated_text_summarizer_v1",
        signature="text_summarizer|source_count_mismatch|blank_inputs|limit=2",
        notes=["ignore_blank_inputs"],
    )
    cluster = FailureCluster(signature=failure.signature, capability=failure.capability, occurrence_count=1)
    registry.register_failure(failure, cluster)
    registry.attach_remediation_goal(failure.signature, "Patch text_summarizer for source_count_mismatch")
    registry.mark_closed(failure.signature, "generated_text_summarizer_v3")

    restored = CurriculumRegistry(ledger=ledger)
    assert restored.closed_patterns()[0].signature == failure.signature
    assert restored.closed_patterns()[0].closed_by_tool == "generated_text_summarizer_v3"


def test_tooling_manager_builds_proactive_patch_goals_from_persisted_failure(tmp_path) -> None:
    ledger = GoalLedger(tmp_path / "ledger")
    failure = FailureCase(
        case_id="failure_demo",
        goal_id="goal_demo",
        capability="text_summarizer",
        tool_name="generated_text_summarizer_v2",
        tool_version="v3.121",
        rollout_stage="canary",
        payload={"texts": ["A", " ", "B"], "max_sentences": 2},
        input_shape={"requested_max_sentences": 2, "blank_texts_count": 1},
        violation_types=["source_count_mismatch", "sentence_limit_exceeded"],
        expected={"source_count": 2, "max_sentences": 2},
        actual={"source_count": 3, "sentences_used": 3},
        rollback_target="generated_text_summarizer_v1",
        signature="text_summarizer|sentence_limit_exceeded+source_count_mismatch|blank_inputs|limit=2",
        notes=["ignore_blank_inputs", "respect_runtime_sentence_limit"],
    )
    ledger.save_failure_case(failure)
    registry = CurriculumRegistry(ledger=ledger)
    cluster = FailureCluster(
        signature=failure.signature,
        capability=failure.capability,
        occurrence_count=2,
        latest_goal_id=failure.goal_id,
        latest_tool_name=failure.tool_name,
    )
    registry.register_failure(failure, cluster)

    tooling = ToolingManager(ledger=ledger)
    goals = tooling.build_proactive_patch_goals(existing_goal_titles=set())
    assert len(goals) == 1
    assert goals[0].evidence["failure_signature"] == failure.signature
    assert "patch" in goals[0].tags
