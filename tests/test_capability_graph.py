from __future__ import annotations

from memory.capability_graph import CapabilityGraph
from memory.capability_portfolio import CapabilityPortfolio
from memory.goal_ledger import GoalLedger
from metacognition.transfer_curriculum import TransferCurriculumBuilder


def test_capability_graph_unlocks_cross_skill_transfer_from_transfer_validated_source(tmp_path):
    ledger = GoalLedger(tmp_path / "ledger")
    portfolio = CapabilityPortfolio(ledger=ledger)
    graph = CapabilityGraph(ledger=ledger)

    portfolio.register_transfer_validation(
        capability="grounded_navigation",
        strategy="graph_search",
        run_id="transfer_1",
        mean_score=1.0,
        passed=True,
    )
    graph.sync_from_portfolio(portfolio)

    builder = TransferCurriculumBuilder()
    tasks = builder.build(graph, existing_titles=set())
    assert len(tasks) == 1
    task = tasks[0]
    assert task.source_capability == "grounded_navigation"
    assert task.target_capability == "navigation_route_explanation"
    assert "navigation_route_explanation" in task.tags

    graph.mark_transfer_goal_suggested(source_capability="grounded_navigation", target_capability="navigation_route_explanation")
    assert builder.build(graph, existing_titles=set()) == []
    assert list((tmp_path / "ledger" / "capability_graph").glob("*.json"))
