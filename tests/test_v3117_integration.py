from __future__ import annotations

from pathlib import Path

from main import build_system, seed_demo_goals
from metacognition.curriculum_builder import curriculum_task_to_goal


def test_v3117_safe_tool_integration_cycle(tmp_path: Path) -> None:
    system = build_system(tmp_path)
    seed_demo_goals(system)

    manager = system["manager"]
    memory = system["memory"]
    metacognition = system["metacognition"]
    planning = system["planning"]
    tooling = system["tooling"]
    agency = system["agency"]
    workspace = system["workspace"]

    admitted_internal = manager.admit_candidates(
        manager.generate_candidates(workspace.get_state(), memory.get_recent_events(), metacognition.analyze()),
        context=workspace.get_state(),
    )
    tool_goals = manager.create_tooling_goals_for_deferred(tooling.supported_capabilities())
    admitted_tool_goals = manager.admit_candidates(tool_goals, context=workspace.get_state())
    assert admitted_tool_goals or admitted_internal

    first_goal = manager.select_next_goal(context=workspace.get_state())
    assert first_goal is not None
    first_plan = planning.make_plan(first_goal, workspace.get_state())
    assert first_plan.steps

    if first_goal.kind.name == "TOOL_CREATION":
        tooling.design_tool_spec(first_goal)
        tooling.generate_tool_code(first_goal)
        validation = tooling.validate_tool_code(first_goal)
        assert validation["allowed"] is True
        tool = tooling.register_tool(first_goal)
        agency.add_capability(tool.capability)
        manager.reactivate_deferred_goals_for_capability(tool.capability)
        manager.complete_goal(first_goal.goal_id, {"success": True})

    workspace.update({"available_capabilities": sorted(agency.list_capabilities())})
    next_goal = manager.select_next_goal(context=workspace.get_state())
    assert next_goal is not None
    next_plan = planning.make_plan(next_goal, workspace.get_state())

    trace = []
    for step in next_plan.steps:
        result = agency.execute(step.name, goal=next_goal, workspace_state=workspace.get_state())
        manager.update_progress(next_goal.goal_id, {"step": step.name, "result": result})
        trace.append({"step": step.name, "result": result})

    report, tasks = metacognition.run_postmortem(
        next_goal,
        trace,
        {"goal_id": next_goal.goal_id, "plan_len": len(next_plan.steps)},
        {"goal_success": all(item["result"].get("success", True) for item in trace)},
    )
    memory.store_postmortem(report)
    manager.complete_goal(next_goal.goal_id, {"success": report.success})
    curriculum_goals = [curriculum_task_to_goal(task) for task in tasks]
    manager.admit_candidates(curriculum_goals, context=workspace.get_state())

    assert list((tmp_path / "ledger" / "postmortems").glob("*.json"))
