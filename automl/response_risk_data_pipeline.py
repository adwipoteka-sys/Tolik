from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from automl.model_registry import ModelRegistry
from automl.response_risk_curriculum import (
    RESPONSE_RISK_AUDIT_EXAMPLES,
    audit_response_risk_model,
    build_anchor_cases,
    build_canary_cases,
    build_search_space,
    build_training_examples,
    build_transfer_cases,
    response_risk_should_verify_normatively,
)
from automl.response_risk_model import RESPONSE_RISK_FAMILY, ResponseRiskModel, ResponseRiskTrainingExample
from automl.training_data_registry import CurriculumDataRegistry
from automl.training_data_schema import CurriculumDatasetExample, CurriculumDatasetSnapshot, new_dataset_example_id
from benchmarks.skill_arena import SkillArenaCase
from benchmarks.transfer_suite import TransferCase
from memory.episodic_memory import EpisodicMemory, EpisodeRecord
from memory.goal_ledger import GoalLedger
from motivation.goal_schema import Goal


@dataclass(slots=True)
class ResponseRiskSelfTrainingBundle:
    snapshot: CurriculumDatasetSnapshot
    training_examples: list[ResponseRiskTrainingExample]
    anchor_cases: list[SkillArenaCase]
    transfer_cases: list[TransferCase]
    canary_cases: list[SkillArenaCase]
    search_space: dict[str, list[float]]


class ResponseRiskDataAcquisitionPipeline:
    """Collects, validates, snapshots, and serves autonomous response-risk self-training data."""

    def __init__(
        self,
        *,
        ledger: GoalLedger,
        registry: ModelRegistry,
        episodic_memory: EpisodicMemory,
        data_registry: CurriculumDataRegistry,
        min_quality: float = 0.65,
    ) -> None:
        self.ledger = ledger
        self.registry = registry
        self.episodic_memory = episodic_memory
        self.data_registry = data_registry
        self.min_quality = min_quality

    def _seed_examples(self) -> list[CurriculumDatasetExample]:
        examples: list[CurriculumDatasetExample] = []
        for index, example in enumerate(build_training_examples(), start=1):
            examples.append(
                CurriculumDatasetExample(
                    example_id=new_dataset_example_id("rrtrain"),
                    model_family=RESPONSE_RISK_FAMILY,
                    split="train",
                    source_type="seed_curriculum",
                    source_signature=f"seed_train:{index}",
                    description=example.description or f"Seed response-risk example {index}",
                    payload=dict(example.goal),
                    target={"label": bool(example.label)},
                    tags=["seed", "response_risk_model"],
                    difficulty=0.35,
                    quality_score=0.98,
                    lineage={"origin": "bootstrap_seed"},
                )
            )
        for case in build_anchor_cases():
            examples.append(self._case_to_dataset_example(case, split="anchor", source_type="anchor_suite"))
        for case in build_transfer_cases():
            examples.append(self._transfer_case_to_dataset_example(case, split="transfer", source_type="transfer_suite"))
        for case in build_canary_cases():
            examples.append(self._case_to_dataset_example(case, split="canary", source_type="canary_suite"))
        return examples

    def _case_to_dataset_example(self, case: SkillArenaCase, *, split: str, source_type: str) -> CurriculumDatasetExample:
        return CurriculumDatasetExample(
            example_id=new_dataset_example_id("rrcase"),
            model_family=RESPONSE_RISK_FAMILY,
            split=split,
            source_type=source_type,
            source_signature=case.case_id,
            description=case.description,
            payload=case.payload,
            target=case.expected,
            tags=[split, "response_risk_model"],
            difficulty=0.55,
            quality_score=0.95,
            lineage={"origin": source_type},
        )

    def _transfer_case_to_dataset_example(self, case: TransferCase, *, split: str, source_type: str) -> CurriculumDatasetExample:
        return CurriculumDatasetExample(
            example_id=new_dataset_example_id("rrtransfer"),
            model_family=RESPONSE_RISK_FAMILY,
            split=split,
            source_type=source_type,
            source_signature=case.case_id,
            description=case.description,
            payload=case.payload,
            target=case.expected,
            tags=[split, "response_risk_model"],
            difficulty=0.65,
            quality_score=0.95,
            lineage={"origin": source_type},
        )

    def _episode_to_training_example(self, episode: EpisodeRecord) -> CurriculumDatasetExample | None:
        if not episode.goal_payload:
            return None
        if episode.kind != "user_task":
            return None
        try:
            goal = Goal.from_dict(episode.goal_payload)
        except Exception:
            return None
        label = response_risk_should_verify_normatively(goal)
        quality = 0.86 if episode.success else 0.74
        return CurriculumDatasetExample(
            example_id=new_dataset_example_id("rrep"),
            model_family=RESPONSE_RISK_FAMILY,
            split="train",
            source_type="episode_supervision",
            source_signature=f"episode:{episode.episode_id}",
            description=f"Episode-derived supervision: {episode.title}",
            payload=goal.to_dict(),
            target={"label": bool(label)},
            tags=["episode_supervision", *episode.tags],
            difficulty=min(1.0, 0.30 + (0.40 if label else 0.10)),
            quality_score=quality,
            lineage={
                "episode_id": episode.episode_id,
                "cycle": episode.cycle,
                "success": episode.success,
            },
        )

    def _episode_canary_examples(self) -> list[CurriculumDatasetExample]:
        canaries: list[CurriculumDatasetExample] = []
        for episode in reversed(self.episodic_memory.list_episodes()):
            if len(canaries) >= 2:
                break
            derived = self._episode_to_training_example(episode)
            if derived is None:
                continue
            requires_verify = bool(derived.target.get("label", False))
            required_steps = ["understand_request", "form_response"]
            forbidden_steps: list[str] = [] if requires_verify else ["verify_outcome"]
            if requires_verify:
                required_steps.insert(1, "verify_outcome")
            canaries.append(
                CurriculumDatasetExample(
                    example_id=new_dataset_example_id("rrcanary"),
                    model_family=RESPONSE_RISK_FAMILY,
                    split="canary",
                    source_type="episode_canary",
                    source_signature=f"episode_canary:{episode.episode_id}",
                    description=f"Recent real-task canary derived from episode '{episode.title}'.",
                    payload={"goal": dict(derived.payload)},
                    target={
                        "predicted_verify": requires_verify,
                        "required_steps": required_steps,
                        "forbidden_steps": forbidden_steps,
                    },
                    tags=["episode_canary", *episode.tags],
                    difficulty=min(1.0, derived.difficulty + 0.1),
                    quality_score=max(0.7, derived.quality_score - 0.05),
                    lineage={"episode_id": episode.episode_id},
                )
            )
        return canaries

    def _audit_counterexamples(self, model: ResponseRiskModel) -> list[CurriculumDatasetExample]:
        examples: list[CurriculumDatasetExample] = []
        for index, example in enumerate(RESPONSE_RISK_AUDIT_EXAMPLES, start=1):
            prediction = bool(model.should_verify(example.goal))
            if prediction == bool(example.label):
                continue
            examples.append(
                CurriculumDatasetExample(
                    example_id=new_dataset_example_id("rraud"),
                    model_family=RESPONSE_RISK_FAMILY,
                    split="train",
                    source_type="audit_counterexample",
                    source_signature=f"audit:{model.model_id}:{index}",
                    description=example.description or f"Audit counterexample {index}",
                    payload=dict(example.goal),
                    target={"label": bool(example.label)},
                    tags=["audit_counterexample", "response_risk_model"],
                    difficulty=0.72,
                    quality_score=0.93,
                    lineage={"model_id": model.model_id, "prediction": prediction},
                )
            )
        return examples

    def _validation_status(self, example_ids: list[str]) -> tuple[str, dict[str, Any]]:
        stats = self.data_registry._compute_snapshot_stats(example_ids)
        train_count = int(stats.get("train_example_count", 0))
        positive_count = int(stats.get("positive_train_count", 0))
        negative_count = int(stats.get("negative_train_count", 0))
        mean_quality = float(stats.get("mean_quality", 0.0))
        split_counts = dict(stats.get("split_counts", {}))
        reasons: list[str] = []
        if train_count < 5:
            reasons.append("insufficient_train_examples")
        if positive_count == 0 or negative_count == 0:
            reasons.append("missing_label_balance")
        for required_split in ("anchor", "transfer", "canary"):
            if int(split_counts.get(required_split, 0)) == 0:
                reasons.append(f"missing_{required_split}")
        if mean_quality < self.min_quality:
            reasons.append("mean_quality_below_threshold")
        status = "approved" if not reasons else "rejected"
        return status, {"validation_reasons": reasons}

    def refresh_snapshot(self, *, current_cycle: int | None = None) -> tuple[CurriculumDatasetSnapshot, dict[str, Any]]:
        active_model = self.registry.get_active_model(RESPONSE_RISK_FAMILY)
        collected: list[CurriculumDatasetExample] = []
        collected.extend(self._seed_examples())
        collected.extend(example for example in (self._episode_to_training_example(item) for item in self.episodic_memory.list_episodes()) if example is not None)
        collected.extend(self._audit_counterexamples(active_model))
        collected.extend(self._episode_canary_examples())

        kept_ids: list[str] = []
        dropped_low_quality = 0
        for example in collected:
            if example.quality_score < self.min_quality:
                dropped_low_quality += 1
                continue
            kept_ids.append(self.data_registry.register_example(example).example_id)

        audit = audit_response_risk_model(active_model)
        status, validation = self._validation_status(kept_ids)
        snapshot = self.data_registry.create_snapshot(
            model_family=RESPONSE_RISK_FAMILY,
            title=f"Autonomous response-risk curriculum snapshot{'' if current_cycle is None else f' cycle {current_cycle}'}",
            description="Validated autonomous curriculum dataset for guarded response-risk self-training.",
            example_ids=kept_ids,
            status=status,
            extra_stats={
                "cycle": current_cycle,
                "audit_accuracy": audit["accuracy"],
                "audit_false_negative_count": audit["false_negative_count"],
                "audit_false_positive_count": audit["false_positive_count"],
                "dropped_low_quality": dropped_low_quality,
                **validation,
            },
        )
        report = {
            "snapshot_id": snapshot.snapshot_id,
            "status": snapshot.status,
            "train_example_count": snapshot.stats.get("train_example_count", 0),
            "audit_accuracy": audit["accuracy"],
            "audit_false_negative_count": audit["false_negative_count"],
            "dropped_low_quality": dropped_low_quality,
        }
        self.ledger.append_event(
            {
                "event_type": "self_training_data_refreshed",
                "model_family": RESPONSE_RISK_FAMILY,
                "snapshot_id": snapshot.snapshot_id,
                "status": snapshot.status,
                "cycle": current_cycle,
                **report,
            }
        )
        return snapshot, report

    def latest_training_bundle(self) -> ResponseRiskSelfTrainingBundle | None:
        snapshot = self.data_registry.latest_snapshot(model_family=RESPONSE_RISK_FAMILY, status="approved")
        if snapshot is None:
            return None
        training_examples = [example.to_training_example() for example in self.data_registry.get_snapshot_examples(snapshot.snapshot_id, split="train")]
        anchor_cases = [example.to_skill_case() for example in self.data_registry.get_snapshot_examples(snapshot.snapshot_id, split="anchor")]
        transfer_cases = [example.to_transfer_case() for example in self.data_registry.get_snapshot_examples(snapshot.snapshot_id, split="transfer")]
        canary_cases = [example.to_skill_case() for example in self.data_registry.get_snapshot_examples(snapshot.snapshot_id, split="canary")]
        return ResponseRiskSelfTrainingBundle(
            snapshot=snapshot,
            training_examples=training_examples,
            anchor_cases=anchor_cases,
            transfer_cases=transfer_cases,
            canary_cases=canary_cases,
            search_space=build_search_space(),
        )
