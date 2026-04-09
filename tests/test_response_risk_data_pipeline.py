from __future__ import annotations

from automl.response_risk_data_pipeline import ResponseRiskDataAcquisitionPipeline
from automl.training_data_registry import CurriculumDataRegistry
from memory.episodic_memory import EpisodicMemory
from memory.goal_ledger import GoalLedger
from motivation.goal_schema import Goal, GoalBudget, GoalKind, GoalSource, SuccessCriterion
from main import build_system


def _user_goal(goal_id: str, *, risk_signal: float, insufficient_evidence: bool = False) -> Goal:
    evidence = {"risk_signal": risk_signal}
    if insufficient_evidence:
        evidence["insufficient_evidence"] = True
    return Goal(
        goal_id=goal_id,
        title="User query",
        description="User query",
        source=GoalSource.USER,
        kind=GoalKind.USER_TASK,
        expected_gain=0.6,
        novelty=0.2,
        uncertainty_reduction=0.4,
        strategic_fit=0.7,
        risk_estimate=risk_signal,
        priority=0.7,
        risk_budget=0.1,
        resource_budget=GoalBudget(max_steps=4, max_seconds=10.0),
        success_criteria=[SuccessCriterion(metric="status", comparator="==", target="done")],
        required_capabilities=["classical_planning"],
        tags=["user"],
        evidence=evidence,
    )


def test_response_risk_data_pipeline_collects_episode_supervision(tmp_path) -> None:
    ledger = GoalLedger(tmp_path / "ledger")
    system = build_system(tmp_path / "runtime")
    registry = CurriculumDataRegistry(ledger=ledger)
    episodic = EpisodicMemory(ledger=ledger)

    episodic.record_goal_episode(
        _user_goal("goal_episode_high_risk", risk_signal=0.9),
        cycle=3,
        trace=[{"step": "understand_request", "result": {"success": True}}, {"step": "form_response", "result": {"success": True}}],
        outcome={"success": True},
        tags=["user"],
    )

    pipeline = ResponseRiskDataAcquisitionPipeline(
        ledger=ledger,
        registry=system["model_registry"],
        episodic_memory=episodic,
        data_registry=registry,
    )
    snapshot, report = pipeline.refresh_snapshot(current_cycle=3)

    assert snapshot.status == "approved"
    assert report["train_example_count"] >= 6
    assert any(example.source_type == "episode_supervision" for example in registry.list_examples(model_family="response_risk_model", split="train"))
    bundle = pipeline.latest_training_bundle()
    assert bundle is not None
    assert len(bundle.training_examples) >= 6
    assert len(bundle.canary_cases) >= 2
