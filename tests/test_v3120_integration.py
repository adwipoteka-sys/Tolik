from __future__ import annotations

from main import build_system, seed_demo_goals, seed_upgrade_goal


def test_v3120_closed_loop_patch_rollout(tmp_path) -> None:
    system = build_system(tmp_path)
    seed_demo_goals(system)

    manager = system["manager"]
    memory = system["memory"]
    metacognition = system["metacognition"]
    planning = system["planning"]
    tooling = system["tooling"]
    agency = system["agency"]
    workspace = system["workspace"]

    # Admit initial tooling goal for missing summarizer.
    manager.admit_candidates(
        manager.generate_candidates(workspace.get_state(), memory.get_recent_events(), metacognition.analyze()),
        context=workspace.get_state(),
    )
    tool_goals = manager.create_tooling_goals_for_deferred(tooling.supported_capabilities())
    manager.admit_candidates(tool_goals, context=workspace.get_state())

    tool_goal = manager.select_next_goal(context=workspace.get_state())
    assert tool_goal is not None and tool_goal.kind.name == "TOOL_CREATION"

    tooling.design_tool_spec(tool_goal)
    tooling.generate_tool_code(tool_goal)
    tooling.validate_tool_code(tool_goal)
    tooling.register_tool(tool_goal)
    assert tooling.benchmark_tool(tool_goal).passed is True
    tooling.promote_canary(tool_goal)
    assert tooling.evaluate_canary(tool_goal)["passed"] is True
    stable_v1 = tooling.finalize_rollout(tool_goal)
    memory.store_tool(stable_v1)
    manager.reactivate_deferred_goals_for_capability(stable_v1.capability)
    manager.complete_goal(tool_goal.goal_id, {"success": True})

    # Now create the regressive upgrade candidate.
    seed_upgrade_goal(system)
    upgrade_goal = next(goal for goal in manager.pending if goal.title == "Roll out canary upgrade for text_summarizer")
    manager.activate_goal(upgrade_goal.goal_id)

    tooling.design_tool_spec(upgrade_goal)
    tooling.generate_tool_code(upgrade_goal)
    tooling.validate_tool_code(upgrade_goal)
    tooling.register_tool(upgrade_goal)
    first_report = tooling.benchmark_tool(upgrade_goal)
    assert first_report.passed is True  # still passes old suite before expansion
    tooling.promote_canary(upgrade_goal)
    failed_outcome = tooling.evaluate_canary(upgrade_goal)
    assert failed_outcome["passed"] is False
    assert failed_outcome["rolled_back"] is True
    assert failed_outcome["patch_goal"] is not None
    assert failed_outcome["regression_case"]["case_id"].startswith("regression_text_summarizer_")

    patch_goal = tooling.get_patch_goal(upgrade_goal.goal_id)
    assert patch_goal is not None

    # Admit and execute the patch goal; the expanded benchmark suite must now be enforced.
    admitted_patch = manager.admit_candidates([patch_goal], context=workspace.get_state())
    assert admitted_patch
    manager.activate_goal(patch_goal.goal_id)

    tooling.design_tool_spec(patch_goal)
    tooling.generate_tool_code(patch_goal)
    tooling.validate_tool_code(patch_goal)
    tooling.register_tool(patch_goal)
    patched_report = tooling.benchmark_tool(patch_goal)
    assert patched_report.passed is True
    assert len(patched_report.cases) >= 3  # base cases + expanded regression case
    tooling.promote_canary(patch_goal)
    patched_outcome = tooling.evaluate_canary(patch_goal)
    assert patched_outcome["passed"] is True
    stable_v3 = tooling.finalize_rollout(patch_goal)
    memory.store_tool(stable_v3)
    manager.complete_goal(patch_goal.goal_id, {"success": True})

    assert stable_v3.name.endswith("_v3")
    assert tooling.registry.get_active_tool("text_summarizer").name == stable_v3.name
    # The new stable tool should respect runtime limits and blank-input handling.
    payload = {
        "texts": ["Alpha stable.", "   ", "Beta concise.", "Gamma extra."],
        "max_sentences": 2,
    }
    result = agency.execute_capability("text_summarizer", payload)
    assert result["success"] is True
    assert result["output"]["source_count"] == 3
    assert result["output"]["sentences_used"] <= 2
