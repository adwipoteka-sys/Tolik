from __future__ import annotations

from experiments.curriculum_experiment_scheduler import CurriculumExperimentScheduler
from experiments.experiment_schema import ExperimentKind, ExperimentStatus
from memory.goal_ledger import GoalLedger
from motivation.operator_charter import OperatorCharter


def test_experiment_scheduler_selects_highest_admissible_candidate(tmp_path) -> None:
    ledger = GoalLedger(tmp_path / "ledger")
    scheduler = CurriculumExperimentScheduler(ledger=ledger)
    charter = OperatorCharter()

    policy_candidate = scheduler.make_proposal(
        experiment_kind=ExperimentKind.POLICY_CHANGE,
        source_signature="response_planning|verify_before_answer_patch",
        title="Policy upgrade",
        description="Promote a safer policy.",
        expected_utility=0.82,
        estimated_risk=0.08,
        estimated_cost=0.22,
        confidence=0.84,
        curriculum_signals=["response_planning|verify_before_answer_patch", "root_cause:plan_error"],
        tags=["response_planning_patch", "planning", "maintenance", "local_only"],
        required_capabilities=["classical_planning", "response_planning"],
    )
    model_candidate = scheduler.make_proposal(
        experiment_kind=ExperimentKind.MODEL_UPGRADE,
        source_signature="response_risk_model|curriculum_upgrade",
        title="Model upgrade",
        description="Upgrade the response risk model.",
        expected_utility=0.74,
        estimated_risk=0.09,
        estimated_cost=0.42,
        confidence=0.78,
        curriculum_signals=["response_risk_model|curriculum_upgrade", "false_negative:high_risk"],
        tags=["automl_model_upgrade", "planning", "maintenance", "local_only", "response_risk_model"],
        required_capabilities=["classical_planning"],
    )

    selected, assessment, ranked = scheduler.select(
        [model_candidate, policy_candidate],
        context={
            "goal_queue_size": 0,
            "scheduled_goal_count": 0,
            "available_capabilities": ["classical_planning", "response_planning"],
            "latest_pattern_key": "response_planning|verify_before_answer_patch",
        },
        current_cycle=3,
        charter=charter,
    )

    assert selected is not None
    assert assessment is not None
    assert selected.experiment_kind == ExperimentKind.POLICY_CHANGE
    assert ranked[0].proposal_id == selected.proposal_id
    assert ranked[0].admissible is True


def test_experiment_scheduler_cooldown_blocks_duplicate_signature(tmp_path) -> None:
    ledger = GoalLedger(tmp_path / "ledger")
    scheduler = CurriculumExperimentScheduler(ledger=ledger, cooldown_cycles=2)
    charter = OperatorCharter()

    candidate = scheduler.make_proposal(
        experiment_kind=ExperimentKind.MODEL_UPGRADE,
        source_signature="response_risk_model|curriculum_upgrade",
        title="Model upgrade",
        description="Upgrade the response risk model.",
        expected_utility=0.78,
        estimated_risk=0.09,
        estimated_cost=0.30,
        confidence=0.80,
        curriculum_signals=["response_risk_model|curriculum_upgrade"],
        tags=["automl_model_upgrade", "planning", "maintenance", "local_only", "response_risk_model"],
        required_capabilities=["classical_planning"],
    )

    selected, _, _ = scheduler.select([candidate], context={"available_capabilities": ["classical_planning"]}, current_cycle=1, charter=charter)
    assert selected is not None
    scheduler.record_materialized(selected.proposal_id, current_cycle=1, goal_id="goal_1")
    scheduler.record_outcome(selected.proposal_id, success=True, current_cycle=1, note="completed")

    duplicate = scheduler.make_proposal(
        experiment_kind=ExperimentKind.MODEL_UPGRADE,
        source_signature="response_risk_model|curriculum_upgrade",
        title="Model upgrade",
        description="Upgrade the response risk model again.",
        expected_utility=0.80,
        estimated_risk=0.09,
        estimated_cost=0.30,
        confidence=0.81,
        curriculum_signals=["response_risk_model|curriculum_upgrade"],
        tags=["automl_model_upgrade", "planning", "maintenance", "local_only", "response_risk_model"],
        required_capabilities=["classical_planning"],
    )

    selected_again, _, ranked = scheduler.select([duplicate], context={"available_capabilities": ["classical_planning"]}, current_cycle=2, charter=charter)
    assert selected_again is None
    assert ranked[0].admissible is False
    proposal = scheduler.latest_for_signature(ExperimentKind.MODEL_UPGRADE, "response_risk_model|curriculum_upgrade")
    assert proposal is not None
    assert proposal.status in {ExperimentStatus.BLOCKED, ExperimentStatus.COMPLETED, ExperimentStatus.FAILED}
    assert proposal.cooldown_until_cycle == 3
