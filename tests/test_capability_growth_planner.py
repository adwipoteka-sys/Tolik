from __future__ import annotations

from memory.capability_graph import CapabilityGraph
from memory.capability_portfolio import CapabilityPortfolio
from memory.goal_ledger import GoalLedger
from planning.capability_growth_planner import CapabilityGrowthPlanner


def test_capability_growth_planner_prefers_multi_edge_path_with_higher_total_value(tmp_path):
    ledger = GoalLedger(tmp_path / "ledger")
    portfolio = CapabilityPortfolio(ledger=ledger)
    graph = CapabilityGraph(ledger=ledger)
    planner = CapabilityGrowthPlanner(ledger=ledger)

    portfolio.register_transfer_validation(
        capability="grounded_navigation",
        strategy="graph_search",
        run_id="transfer_grounded",
        mean_score=1.0,
        passed=True,
    )
    graph.sync_from_portfolio(portfolio)

    plans = planner.propose(
        graph=graph,
        portfolio=portfolio,
        executable_targets={"spatial_route_composition", "route_mission_briefing", "navigation_route_explanation"},
        horizon=2,
        limit=3,
    )

    assert plans
    best = plans[0]
    assert best.root_capability == "grounded_navigation"
    assert best.path_targets == ["spatial_route_composition", "route_mission_briefing"]
    assert best.total_score > plans[-1].total_score
