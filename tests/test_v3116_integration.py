from __future__ import annotations

from pathlib import Path

from agency.agency_module import AgencyModule
from core.global_workspace import GlobalWorkspace
from memory.goal_ledger import GoalLedger
from memory.memory_module import MemoryModule
from metacognition.curriculum_builder import curriculum_task_to_goal
from metacognition.metacognition_module import MetacognitionModule
from motivation.autonomous_goal_manager import AutonomousGoalManager
from motivation.goal_arbitrator import GoalArbitrator
from planning.planning_module import PlanningModule


def test_v3116_integration_cycle(tmp_path: Path) -> None:
    ledger = GoalLedger(tmp_path / "ledger")
    manager = AutonomousGoalManager(ledger=ledger, arbitrator=GoalArbitrator())
    memory = MemoryModule(goal_ledger=ledger)
    metacognition = MetacognitionModule()
    planning = PlanningModule()
    agency = AgencyModule()
    workspace = GlobalWorkspace(
        {
            "novelty_score": 0.1,
            "retrieval_confidence": 0.2,
            "available_capabilities": ["classical_planning", "local_llm", "classical_simulation"],
        }
    )

    candidates = manager.generate_candidates(workspace.get_state(), memory.get_recent_events(), metacognition.analyze())
    admitted = manager.admit_candidates(candidates, context=workspace.get_state())
    assert admitted

    goal = manager.select_next_goal(context=workspace.get_state())
    assert goal is not None

    plan = planning.make_plan(goal, workspace.get_state())
    trace = []
    for step in plan.steps:
        result = agency.execute(step.name, goal=goal, workspace_state=workspace.get_state())
        manager.update_progress(goal.goal_id, {"step": step.name, "result": result})
        trace.append({"step": step.name, "result": result})

    report, tasks = metacognition.run_postmortem(
        goal,
        trace,
        {"goal_id": goal.goal_id, "plan_len": len(plan.steps)},
        {"goal_success": all(item["result"].get("success", True) for item in trace)},
    )
    memory.store_postmortem(report)
    manager.complete_goal(goal.goal_id, {"success": report.success})

    curriculum_goals = [curriculum_task_to_goal(task) for task in tasks]
    manager.admit_candidates(curriculum_goals, context=workspace.get_state())

    history = ledger.load_history(limit=100)
    assert any(event["event_type"] == "goal_admitted" for event in history)
    assert any(event["event_type"] == "goal_completed" for event in history)
    assert list((tmp_path / "ledger" / "postmortems").glob("*.json"))
    assert tasks
