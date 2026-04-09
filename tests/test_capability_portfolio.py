from __future__ import annotations

from memory.capability_portfolio import CapabilityPortfolio
from memory.goal_ledger import GoalLedger


def test_capability_portfolio_tracks_transfer_validated_state_and_persists(tmp_path):
    ledger = GoalLedger(tmp_path / "ledger")
    portfolio = CapabilityPortfolio(ledger=ledger)

    portfolio.register_skill_validation(
        capability="grounded_navigation",
        strategy="graph_search",
        run_id="skill_1",
        mean_score=1.0,
        passed=True,
        pattern_key="grounded_navigation|graph_search_patch",
    )
    portfolio.register_semantic_promotions(
        capability="grounded_navigation",
        fact_keys=["semantic:grounded_navigation__graph_search_patch"],
        pattern_key="grounded_navigation|graph_search_patch",
    )
    portfolio.register_transfer_validation(
        capability="grounded_navigation",
        strategy="graph_search",
        run_id="transfer_1",
        mean_score=1.0,
        passed=True,
    )
    portfolio.register_scheduled_maintenance(capability="grounded_navigation", schedule_id="sched_1")

    state = portfolio.get("grounded_navigation")
    assert state is not None
    assert state.maturity_stage == "transfer_validated"
    assert state.latest_transfer_run_id == "transfer_1"
    assert portfolio.ready_for_unattended_use("grounded_navigation") is True
    assert portfolio.summary()["grounded_navigation"]["promotion_count"] == 1

    restored = CapabilityPortfolio(ledger=ledger)
    restored_state = restored.get("grounded_navigation")
    assert restored_state is not None
    assert restored_state.maturity_stage == "transfer_validated"
    assert restored_state.maintenance_schedule_ids == ["sched_1"]
    assert list((tmp_path / "ledger" / "capability_portfolio").glob("*.json"))
