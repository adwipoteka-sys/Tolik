from __future__ import annotations

from experiments.curriculum_experiment_scheduler import CurriculumExperimentScheduler
from experiments.experiment_board import ExperimentBoard
from experiments.experiment_board_schema import ExperimentCampaignStatus
from experiments.experiment_schema import ExperimentKind
from memory.goal_ledger import GoalLedger
from motivation.operator_charter import OperatorCharter


def test_experiment_board_releases_queued_campaign_next_cycle(tmp_path) -> None:
    ledger = GoalLedger(tmp_path / "ledger")
    scheduler = CurriculumExperimentScheduler(ledger=ledger)
    board = ExperimentBoard(ledger=ledger)
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
        curriculum_signals=["response_risk_model|curriculum_upgrade"],
        tags=["automl_model_upgrade", "planning", "maintenance", "local_only", "response_risk_model"],
        required_capabilities=["classical_planning"],
    )

    selected, _, ranked = scheduler.select(
        [policy_candidate, model_candidate],
        context={
            "goal_queue_size": 0,
            "scheduled_goal_count": 0,
            "available_capabilities": ["classical_planning", "response_planning"],
            "latest_pattern_key": "response_planning|verify_before_answer_patch",
        },
        current_cycle=1,
        charter=charter,
    )
    assert selected is not None

    board.refresh_candidate_pool(
        [policy_candidate, model_candidate],
        ranked,
        current_cycle=1,
        selected_proposal_id=selected.proposal_id,
        context={"future_interfaces": {}},
        charter=charter,
    )

    queued = board.campaign_for_signature(ExperimentKind.MODEL_UPGRADE, model_candidate.source_signature)
    assert queued is not None
    assert queued.status == ExperimentCampaignStatus.QUEUED
    assert queued.defer_reason == "not_selected_this_cycle"
    assert queued.defer_until_cycle == 2

    released = board.release_due_candidates(current_cycle=2, context={"future_interfaces": {}}, charter=charter)
    assert [candidate.source_signature for candidate in released] == [model_candidate.source_signature]


def test_experiment_board_defers_until_live_interface_then_releases(tmp_path) -> None:
    ledger = GoalLedger(tmp_path / "ledger")
    board = ExperimentBoard(ledger=ledger)
    scheduler = CurriculumExperimentScheduler(ledger=ledger)
    charter = OperatorCharter(allow_quantum_solver=True)

    quantum_candidate = scheduler.make_proposal(
        experiment_kind=ExperimentKind.POLICY_CHANGE,
        source_signature="future_interface|quantum_solver_probe",
        title="Probe deferred quantum solver campaign when live access is available",
        description="Wait for live quantum access.",
        expected_utility=0.56,
        estimated_risk=0.05,
        estimated_cost=0.18,
        confidence=0.72,
        curriculum_signals=["future_interface|quantum_solver_probe", "mode:stub"],
        tags=["maintenance", "future_integration", "deferred_execution", "local_only"],
        required_capabilities=["classical_planning", "quantum_solver"],
        evidence={
            "requires_live_interface": True,
            "required_interface": "quantum_solver",
            "retry_after_cycles": 2,
            "campaign_budget": {"max_total_cost": 0.54, "max_total_risk": 0.16, "max_attempts": 2},
        },
    )

    campaign, ticket = board.stage_selected_execution(
        quantum_candidate,
        current_cycle=1,
        context={"future_interfaces": {"quantum_solver": "stub"}},
        charter=charter,
    )
    assert ticket["materialize_now"] is False
    assert campaign.status == ExperimentCampaignStatus.DEFERRED
    assert campaign.defer_until_cycle == 3
    assert campaign.defer_reason.startswith("awaiting_live_interface:quantum_solver")

    assert board.release_due_candidates(current_cycle=2, context={"future_interfaces": {"quantum_solver": "stub"}}, charter=charter) == []
    released = board.release_due_candidates(current_cycle=3, context={"future_interfaces": {"quantum_solver": "live"}}, charter=charter)
    assert [candidate.source_signature for candidate in released] == [quantum_candidate.source_signature]


def test_experiment_board_stops_campaign_when_budget_is_exhausted(tmp_path) -> None:
    ledger = GoalLedger(tmp_path / "ledger")
    board = ExperimentBoard(ledger=ledger)
    scheduler = CurriculumExperimentScheduler(ledger=ledger)
    charter = OperatorCharter()

    candidate = scheduler.make_proposal(
        experiment_kind=ExperimentKind.MODEL_UPGRADE,
        source_signature="response_risk_model|budgeted_upgrade",
        title="Budgeted model upgrade",
        description="Spend campaign budget only twice.",
        expected_utility=0.70,
        estimated_risk=0.10,
        estimated_cost=0.30,
        confidence=0.80,
        curriculum_signals=["response_risk_model|budgeted_upgrade"],
        tags=["automl_model_upgrade", "planning", "maintenance", "local_only", "response_risk_model"],
        required_capabilities=["classical_planning"],
        evidence={"campaign_budget": {"max_total_cost": 0.50, "max_total_risk": 0.20, "max_attempts": 2}},
    )

    campaign, ticket = board.stage_selected_execution(candidate, current_cycle=1, context={"future_interfaces": {}}, charter=charter)
    assert ticket["materialize_now"] is True
    board.record_materialized(candidate.proposal_id, goal_id="goal_upgrade")
    board.record_outcome(candidate.proposal_id, success=False, current_cycle=1, note="retry", cooldown_until_cycle=2)

    campaign, ticket = board.stage_selected_execution(candidate, current_cycle=3, context={"future_interfaces": {}}, charter=charter)
    assert ticket["materialize_now"] is False
    assert campaign.status == ExperimentCampaignStatus.BUDGET_EXHAUSTED
    assert campaign.defer_reason == "cost_budget_exhausted"
