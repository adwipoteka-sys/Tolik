from __future__ import annotations

import argparse
from pathlib import Path

from autonomous_agi import _materialize_experiment_goal, _run_automl_step
from experiments.experiment_schema import ExperimentKind
from experiments.experiment_sources import build_response_risk_model_upgrade_candidate
from main import build_system
from motivation.operator_charter import load_charter


def main() -> None:
    parser = argparse.ArgumentParser(description="Demonstrate curriculum-governed experiment scheduling in Tolik v3.135.")
    parser.add_argument("--runtime-dir", type=Path, default=Path("runtime_v3135_scheduler_demo"), help="Runtime directory for the ledger.")
    parser.add_argument("--charter", type=Path, default=Path("configs/operator_charter.example.json"), help="Operator charter JSON.")
    args = parser.parse_args()

    system = build_system(args.runtime_dir)
    scheduler = system["experiment_scheduler"]
    registry = system["model_registry"]
    automl_manager = system["automl_manager"]
    data_pipeline = system["response_risk_data_pipeline"]
    workspace = system["workspace"]
    charter = load_charter(args.charter)

    policy_candidate = scheduler.make_proposal(
        experiment_kind=ExperimentKind.POLICY_CHANGE,
        source_signature="response_planning|verify_before_answer_patch",
        title="Synthetic response-planning patch proposal",
        description="Demonstrate that the scheduler can compare a policy-change experiment with a model-upgrade experiment.",
        expected_utility=0.74,
        estimated_risk=0.08,
        estimated_cost=0.22,
        confidence=0.78,
        curriculum_signals=["response_planning|verify_before_answer_patch", "root_cause:plan_error"],
        tags=["response_planning_patch", "planning", "maintenance", "local_only"],
        required_capabilities=["classical_planning"],
        evidence={"synthetic_demo": True},
    )
    model_candidate = build_response_risk_model_upgrade_candidate(
        registry,
        automl_manager=automl_manager,
        scheduler=scheduler,
        data_pipeline=data_pipeline,
    )
    candidates = [policy_candidate]
    if model_candidate is not None:
        candidates.append(model_candidate)

    selected, assessment, ranked = scheduler.select(
        candidates,
        context=workspace.get_state(),
        current_cycle=1,
        charter=charter,
    )

    print("Tolik v3.135 — experiment scheduler demo")
    print(f"Runtime ledger: {args.runtime_dir / 'ledger'}")
    for item in ranked:
        print(
            f"- {item.experiment_kind.value}:{item.source_signature} | admissible={item.admissible} "
            f"utility={item.expected_utility:.3f} curriculum={item.curriculum_alignment:.3f} "
            f"risk={item.estimated_risk:.3f} cost={item.estimated_cost:.3f} composite={item.composite_score:.3f}"
        )

    if selected is None or assessment is None:
        print("No admissible experiment was selected.")
        return

    goal = _materialize_experiment_goal(system, selected)
    scheduler.record_materialized(selected.proposal_id, current_cycle=1, goal_id=goal.goal_id)
    print(f"\nSelected experiment: {selected.title}")
    print(f"Materialized goal: {goal.title}")

    if selected.experiment_kind == ExperimentKind.MODEL_UPGRADE:
        print("Executing safe model-upgrade steps...")
        for step in system["planning"].make_plan(goal, world_state=workspace.get_state()).steps:
            if step.name == "record_learning":
                print(f"  - {step.name}: {{'success': True}}")
                continue
            result = _run_automl_step(step.name, goal, system)
            print(f"  - {step.name}: {result}")
        scheduler.record_outcome(selected.proposal_id, success=True, current_cycle=1, note="demo_finalized")
        print(f"Active response policy: {system['planning'].response_planning_policy}")
    else:
        scheduler.record_outcome(selected.proposal_id, success=True, current_cycle=1, note="demo_materialized")
        print("Policy-change candidate selected; demo stops before execution because it is synthetic.")


if __name__ == "__main__":
    main()
