from __future__ import annotations

from autonomous_agi import _collect_experiment_candidates
from experiments.experiment_board import ExperimentBoard
from main import build_system
from motivation.operator_charter import OperatorCharter


def test_board_persists_future_campaigns_and_defers_them_under_default_charter(tmp_path) -> None:
    system = build_system(tmp_path / "runtime")
    scheduler = system["experiment_scheduler"]
    board: ExperimentBoard = system["experiment_board"]
    workspace = system["workspace"]

    candidates = _collect_experiment_candidates(system)
    assert any("quantum_solver" in candidate.required_capabilities for candidate in candidates)
    assert any("cloud_llm" in candidate.required_capabilities for candidate in candidates)

    selected, _, ranked = scheduler.select(
        candidates,
        context=workspace.get_state(),
        current_cycle=1,
        charter=OperatorCharter(),
    )

    board.refresh_candidate_pool(
        candidates,
        ranked,
        current_cycle=1,
        selected_proposal_id=selected.proposal_id if selected is not None else None,
        context=workspace.get_state(),
        charter=OperatorCharter(),
    )

    campaigns = board.list_campaigns()
    assert len(campaigns) >= 3
    deferred = [campaign for campaign in campaigns if campaign.defer_reason]
    assert any("charter:" in (campaign.defer_reason or "") for campaign in deferred)
    assert board.summary()["deferred_campaign_count"] >= 2
    assert board.list_cycle_budgets()[0].deferred_campaign_ids
