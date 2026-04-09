from __future__ import annotations

from collections import Counter
from typing import Any

from automl.training_data_schema import CurriculumDatasetExample, CurriculumDatasetSnapshot, new_dataset_snapshot_id
from memory.goal_ledger import GoalLedger


class CurriculumDataRegistry:
    """Persistent registry of curriculum examples and validated training snapshots."""

    def __init__(self, ledger: GoalLedger | None = None) -> None:
        self.ledger = ledger
        self._examples: dict[str, CurriculumDatasetExample] = {}
        self._snapshots: dict[str, CurriculumDatasetSnapshot] = {}
        self._fingerprint_to_example_id: dict[str, str] = {}
        if self.ledger is not None:
            self._rehydrate()

    def _rehydrate(self) -> None:
        for payload in self.ledger.load_dataset_examples():
            example = CurriculumDatasetExample.from_dict(payload)
            self._examples[example.example_id] = example
            self._fingerprint_to_example_id[example.fingerprint()] = example.example_id
        for payload in self.ledger.load_dataset_snapshots():
            snapshot = CurriculumDatasetSnapshot.from_dict(payload)
            self._snapshots[snapshot.snapshot_id] = snapshot

    def _persist_example(self, example: CurriculumDatasetExample) -> CurriculumDatasetExample:
        self._examples[example.example_id] = example
        self._fingerprint_to_example_id[example.fingerprint()] = example.example_id
        if self.ledger is not None:
            self.ledger.save_dataset_example(example.to_dict())
        return example

    def _persist_snapshot(self, snapshot: CurriculumDatasetSnapshot) -> CurriculumDatasetSnapshot:
        self._snapshots[snapshot.snapshot_id] = snapshot
        if self.ledger is not None:
            self.ledger.save_dataset_snapshot(snapshot.to_dict())
        return snapshot

    def register_example(self, example: CurriculumDatasetExample) -> CurriculumDatasetExample:
        existing_id = self._fingerprint_to_example_id.get(example.fingerprint())
        if existing_id is not None:
            existing = self._examples[existing_id]
            existing.quality_score = max(existing.quality_score, example.quality_score)
            existing.lineage = {**existing.lineage, **example.lineage}
            return self._persist_example(existing)
        return self._persist_example(example)

    def get_example(self, example_id: str) -> CurriculumDatasetExample:
        return self._examples[example_id]

    def list_examples(self, *, model_family: str | None = None, split: str | None = None) -> list[CurriculumDatasetExample]:
        examples = list(self._examples.values())
        if model_family is not None:
            examples = [example for example in examples if example.model_family == model_family]
        if split is not None:
            examples = [example for example in examples if example.split == split]
        return sorted(examples, key=lambda item: (item.created_at, item.example_id))

    def _compute_snapshot_stats(self, example_ids: list[str]) -> dict[str, Any]:
        examples = [self._examples[example_id] for example_id in example_ids]
        split_counts = Counter(example.split for example in examples)
        source_counts = Counter(example.source_type for example in examples)
        train_examples = [example for example in examples if example.split == "train"]
        positive_count = sum(1 for example in train_examples if bool(example.target.get("label", False)))
        negative_count = sum(1 for example in train_examples if not bool(example.target.get("label", False)))
        mean_quality = round(sum(example.quality_score for example in examples) / max(1, len(examples)), 3)
        return {
            "example_count": len(examples),
            "split_counts": dict(split_counts),
            "source_counts": dict(source_counts),
            "train_example_count": len(train_examples),
            "positive_train_count": positive_count,
            "negative_train_count": negative_count,
            "mean_quality": mean_quality,
        }

    def create_snapshot(
        self,
        *,
        model_family: str,
        title: str,
        description: str,
        example_ids: list[str],
        status: str = "approved",
        extra_stats: dict[str, Any] | None = None,
    ) -> CurriculumDatasetSnapshot:
        stats = self._compute_snapshot_stats(example_ids)
        if extra_stats:
            stats.update(dict(extra_stats))
        snapshot = CurriculumDatasetSnapshot(
            snapshot_id=new_dataset_snapshot_id(),
            model_family=model_family,
            title=title,
            description=description,
            example_ids=list(example_ids),
            status=status,
            stats=stats,
        )
        return self._persist_snapshot(snapshot)

    def list_snapshots(self, *, model_family: str | None = None, status: str | None = None) -> list[CurriculumDatasetSnapshot]:
        snapshots = list(self._snapshots.values())
        if model_family is not None:
            snapshots = [snapshot for snapshot in snapshots if snapshot.model_family == model_family]
        if status is not None:
            snapshots = [snapshot for snapshot in snapshots if snapshot.status == status]
        return sorted(snapshots, key=lambda item: (item.created_at, item.snapshot_id))

    def latest_snapshot(self, *, model_family: str, status: str | None = None) -> CurriculumDatasetSnapshot | None:
        snapshots = self.list_snapshots(model_family=model_family, status=status)
        return snapshots[-1] if snapshots else None

    def get_snapshot_examples(self, snapshot_id: str, *, split: str | None = None) -> list[CurriculumDatasetExample]:
        snapshot = self._snapshots[snapshot_id]
        examples = [self._examples[example_id] for example_id in snapshot.example_ids]
        if split is not None:
            examples = [example for example in examples if example.split == split]
        return sorted(examples, key=lambda item: (item.created_at, item.example_id))
