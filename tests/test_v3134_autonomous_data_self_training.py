from __future__ import annotations

from autonomous_agi import _materialize_experiment_goal, _run_automl_step
from experiments.experiment_sources import build_response_risk_model_upgrade_candidate
from main import build_system
from motivation.goal_schema import Goal, GoalBudget, GoalKind, GoalSource, SuccessCriterion
from motivation.operator_charter import OperatorCharter


def _risky_goal() -> Goal:
    return Goal(
        goal_id="goal_v3134_risky",
        title="Risky held-out request",
        description="Risky held-out request",
        source=GoalSource.USER,
        kind=GoalKind.USER_TASK,
        expected_gain=0.7,
        novelty=0.2,
        uncertainty_reduction=0.4,
        strategic_fit=0.8,
        risk_estimate=0.9,
        priority=0.8,
        risk_budget=0.2,
        resource_budget=GoalBudget(max_steps=4, max_seconds=10.0),
        success_criteria=[SuccessCriterion(metric="status", comparator="==", target="done")],
        required_capabilities=["classical_planning"],
        tags=["user"],
        evidence={"risk_signal": 0.9},
    )


def test_autonomous_data_pipeline_drives_safe_self_training_model_upgrade(tmp_path) -> None:
    system = build_system(tmp_path / "runtime")
    scheduler = system["experiment_scheduler"]
    registry = system["model_registry"]
    automl_manager = system["automl_manager"]
    planning = system["planning"]
    data_pipeline = system["response_risk_data_pipeline"]

    snapshot, report = data_pipeline.refresh_snapshot(current_cycle=1)
    assert snapshot.status == "approved"
    assert report["train_example_count"] >= 5

    candidate = build_response_risk_model_upgrade_candidate(
        registry,
        automl_manager=automl_manager,
        scheduler=scheduler,
        data_pipeline=data_pipeline,
    )
    assert candidate is not None
    assert candidate.evidence["dataset_snapshot_id"] == snapshot.snapshot_id
    assert len(candidate.evidence["training_examples"]) >= 5

    selected, assessment, ranked = scheduler.select(
        [candidate],
        context={
            "goal_queue_size": 0,
            "scheduled_goal_count": 0,
            "available_capabilities": ["classical_planning"],
        },
        current_cycle=1,
        charter=OperatorCharter(),
    )
    assert selected is not None
    assert assessment is not None
    assert ranked[0].admissible is True

    goal = _materialize_experiment_goal(system, selected)
    scheduler.record_materialized(selected.proposal_id, current_cycle=1, goal_id=goal.goal_id)
    for step in planning.make_plan(goal, world_state={}).steps:
        if step.name == "record_learning":
            continue
        result = _run_automl_step(step.name, goal, system)
        assert result["success"] is True

    scheduler.record_outcome(selected.proposal_id, success=True, current_cycle=1, note="v3134_finalized")
    risky_plan = planning.make_plan(_risky_goal(), world_state={})
    assert "verify_outcome" in [step.name for step in risky_plan.steps]
