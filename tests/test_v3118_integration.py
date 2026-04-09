from __future__ import annotations

from main import build_system, seed_demo_goals


def test_v3119_canary_rollout_finalizes_before_use(tmp_path) -> None:
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
    assert first_goal.kind.name == "TOOL_CREATION"
    first_plan = planning.make_plan(first_goal, workspace.get_state())
    step_names = [step.name for step in first_plan.steps]
    assert step_names[-3:] == ["promote_canary", "evaluate_canary", "finalize_rollout"]

    tooling.design_tool_spec(first_goal)
    tooling.generate_tool_code(first_goal)
    validation = tooling.validate_tool_code(first_goal)
    assert validation["allowed"] is True
    tooling.register_tool(first_goal)
    report = tooling.benchmark_tool(first_goal)
    assert report.passed is True
    tooling.promote_canary(first_goal)
    canary = tooling.evaluate_canary(first_goal)
    assert canary["passed"] is True
    tool = tooling.finalize_rollout(first_goal)
    memory.store_tool(tool)
    manager.reactivate_deferred_goals_for_capability(tool.capability)
    manager.complete_goal(first_goal.goal_id, {"success": True})

    workspace.update({"available_capabilities": sorted(agency.list_capabilities())})
    next_goal = manager.select_next_goal(context=workspace.get_state())
    assert next_goal is not None
    assert "text_summarizer" in agency.list_capabilities()

    user_goal = next((goal for goal in manager.all_goals() if goal.title == "Summarize the current research notes"), None)
    assert user_goal is not None
    result = agency.execute("run_capability:text_summarizer", goal=user_goal, workspace_state=workspace.get_state())
    assert result["success"] is True
    assert result["output"]["summary"]
