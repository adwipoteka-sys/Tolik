from __future__ import annotations

from pathlib import Path

from agency.agency_module import AgencyModule
from core.global_workspace import GlobalWorkspace
from memory.goal_ledger import GoalLedger
from memory.memory_module import MemoryModule
from motivation.autonomous_goal_manager import AutonomousGoalManager
from motivation.goal_arbitrator import GoalArbitrator
from tooling.tooling_manager import ToolingManager


def test_tool_goal_unblocks_deferred_goal(tmp_path: Path) -> None:
    ledger = GoalLedger(tmp_path / "ledger")
    manager = AutonomousGoalManager(ledger=ledger, arbitrator=GoalArbitrator())
    tooling = ToolingManager(ledger=ledger)
    agency = AgencyModule(tool_registry=tooling.registry)
    memory = MemoryModule(goal_ledger=ledger)
    workspace = GlobalWorkspace({"available_capabilities": sorted(agency.list_capabilities())})

    blocked = manager.ingest_external_goal(
        "Summarize notes",
        required_capabilities=["classical_planning", "text_summarizer"],
        evidence={"tool_payload": {"texts": ["A.", "B."]}},
    )
    manager.admit_candidates([], context=workspace.get_state())
    # explicit capability pass to mark the already ingested goal as deferred would not happen above,
    # so run admission on the concrete goal.
    manager.pending.clear()
    manager.admit_candidates([blocked], context=workspace.get_state())
    assert blocked.status.value == "deferred"

    tool_goals = manager.create_tooling_goals_for_deferred(tooling.supported_capabilities())
    admitted_tool_goals = manager.admit_candidates(tool_goals, context=workspace.get_state())
    assert admitted_tool_goals
    tool_goal = admitted_tool_goals[0]

    tooling.design_tool_spec(tool_goal)
    tooling.generate_tool_code(tool_goal)
    tooling.validate_tool_code(tool_goal)
    registered = tooling.register_tool(tool_goal)
    agency.add_capability(registered.capability)
    memory.store_tool(registered)
    workspace.update({"available_capabilities": sorted(agency.list_capabilities())})

    unblocked = manager.reactivate_deferred_goals_for_capability("text_summarizer")
    assert any(goal.goal_id == blocked.goal_id for goal in unblocked)
    assert blocked.status.value == "pending"
