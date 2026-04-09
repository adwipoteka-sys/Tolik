from __future__ import annotations

from automl.training_data_registry import CurriculumDataRegistry
from automl.training_data_schema import CurriculumDatasetExample
from memory.goal_ledger import GoalLedger


def _example(example_id: str = "ex_1") -> CurriculumDatasetExample:
    return CurriculumDatasetExample(
        example_id=example_id,
        model_family="response_risk_model",
        split="train",
        source_type="unit_test",
        source_signature="unit:1",
        description="Synthetic example",
        payload={"goal_id": "goal_1", "risk_signal": 0.9},
        target={"label": True},
        tags=["unit_test"],
        difficulty=0.4,
        quality_score=0.9,
    )


def test_curriculum_data_registry_dedupes_and_persists(tmp_path) -> None:
    ledger = GoalLedger(tmp_path / "ledger")
    registry = CurriculumDataRegistry(ledger=ledger)

    first = registry.register_example(_example("ex_a"))
    second = registry.register_example(_example("ex_b"))

    assert first.example_id == second.example_id
    assert len(registry.list_examples(model_family="response_risk_model", split="train")) == 1

    snapshot = registry.create_snapshot(
        model_family="response_risk_model",
        title="Unit snapshot",
        description="Synthetic snapshot",
        example_ids=[first.example_id],
    )
    assert snapshot.stats["example_count"] == 1
    assert snapshot.stats["train_example_count"] == 1

    restored = CurriculumDataRegistry(ledger=ledger)
    assert len(restored.list_examples(model_family="response_risk_model")) == 1
    assert restored.latest_snapshot(model_family="response_risk_model") is not None
