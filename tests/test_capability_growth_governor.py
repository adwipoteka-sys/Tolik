from __future__ import annotations

from memory.capability_graph import CapabilityGraph, CapabilityTransferEdge
from memory.capability_portfolio import CapabilityPortfolio
from memory.goal_ledger import GoalLedger
from planning.capability_growth_governor import CapabilityGrowthGovernor
from planning.capability_growth_planner import CapabilityGrowthPlanner


def test_growth_governor_prefers_multi_edge_path_when_utility_outweighs_cost_and_risk(tmp_path):
    ledger = GoalLedger(tmp_path / "ledger")
    portfolio = CapabilityPortfolio(ledger=ledger)
    graph = CapabilityGraph(ledger=ledger)
    planner = CapabilityGrowthPlanner(ledger=ledger)
    governor = CapabilityGrowthGovernor(ledger=ledger)

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

    selected, assessment, ranked = governor.select(plans, graph=graph, portfolio=portfolio, context={})

    assert selected is not None
    assert assessment is not None
    assert selected.path_targets == ["spatial_route_composition", "route_mission_briefing"]
    assert ranked[0].path_targets == ["spatial_route_composition", "route_mission_briefing"]
    assert ranked[0].admissible is True
    assert ranked[0].composite_score > ranked[-1].composite_score


def test_growth_governor_blocks_high_risk_branch_even_if_nominal_strategic_value_is_high(tmp_path):
    ledger = GoalLedger(tmp_path / "ledger")
    portfolio = CapabilityPortfolio(ledger=ledger)
    graph = CapabilityGraph(ledger=ledger)
    planner = CapabilityGrowthPlanner(ledger=ledger)
    governor = CapabilityGrowthGovernor(ledger=ledger)

    graph.add_transfer_edge(
        CapabilityTransferEdge(
            source_capability="grounded_navigation",
            target_capability="risky_capability",
            support_capabilities=("local_llm", "quantum_solver"),
            description="High-upside but brittle quantum-heavy leap.",
            strength=0.15,
            strategic_value=1.0,
        )
    )
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
        executable_targets={"risky_capability"},
        horizon=1,
        limit=3,
    )

    selected, assessment, ranked = governor.select(plans, graph=graph, portfolio=portfolio, context={})

    assert plans
    assert selected is None
    assert assessment is None
    assert ranked[0].path_targets == ["risky_capability"]
    assert ranked[0].admissible is False
    assert ranked[0].estimated_risk > governor.hard_risk_ceiling


def test_growth_governor_goal_overrides_are_embedded_into_materialized_goal(tmp_path):
    ledger = GoalLedger(tmp_path / "ledger")
    portfolio = CapabilityPortfolio(ledger=ledger)
    graph = CapabilityGraph(ledger=ledger)
    planner = CapabilityGrowthPlanner(ledger=ledger)
    governor = CapabilityGrowthGovernor(ledger=ledger)

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
    selected, assessment, _ = governor.select(plans, graph=graph, portfolio=portfolio, context={})

    assert selected is not None and assessment is not None
    goal = planner.materialize_next_goal(
        selected,
        tool_payload={"tasks": []},
        overrides=governor.goal_overrides_for(assessment, selected),
    )

    assert goal.evidence["growth_assessment_id"] == assessment.assessment_id
    assert goal.risk_estimate == assessment.estimated_risk
    assert goal.priority >= 0.67
    assert goal.resource_budget.max_steps >= 4
