from __future__ import annotations

import copy
from typing import Any

from automl.model_registry import ModelRegistry
from automl.model_schema import (
    AutoMLCanaryReport,
    AutoMLRegressionReport,
    AutoMLSpec,
    AutoMLTrainingReport,
    new_automl_change_id,
)
from automl.response_risk_model import (
    RESPONSE_RISK_FAMILY,
    ResponseRiskModel,
    ResponseRiskTrainingExample,
    ResponseRiskTrainingReport,
    train_response_risk_model,
)
from benchmarks.skill_arena import SkillArena, SkillArenaCase, SkillArenaRun
from benchmarks.transfer_suite import TransferCase, TransferRun, TransferSuite
from core.event_types import GoalEventType
from memory.goal_ledger import GoalLedger
from motivation.goal_schema import Goal
from planning.planning_module import PlanningModule


class SafeAutoMLManager:
    """Runs training, regression gate, canary, and finalize for internal models."""

    def __init__(
        self,
        *,
        ledger: GoalLedger,
        registry: ModelRegistry,
        components: dict[str, Any],
        skill_arena: SkillArena,
        transfer_suite: TransferSuite,
    ) -> None:
        self.ledger = ledger
        self.registry = registry
        self.components = dict(components)
        self.skill_arena = skill_arena
        self.transfer_suite = transfer_suite
        self._specs: dict[str, AutoMLSpec] = {}
        self._training_reports: dict[str, AutoMLTrainingReport] = {}
        self._regression_reports: dict[str, AutoMLRegressionReport] = {}
        self._canary_reports: dict[str, AutoMLCanaryReport] = {}
        self._rehydrate()

    def _rehydrate(self) -> None:
        for payload in self.ledger.load_automl_specs():
            spec = AutoMLSpec.from_dict(payload)
            self._specs[spec.change_id] = spec
        for payload in self.ledger.load_automl_training_reports():
            report = AutoMLTrainingReport.from_dict(payload)
            self._training_reports[report.change_id] = report
        for payload in self.ledger.load_automl_regression_reports():
            report = AutoMLRegressionReport.from_dict(payload)
            self._regression_reports[report.change_id] = report
        for payload in self.ledger.load_automl_canary_reports():
            report = AutoMLCanaryReport.from_dict(payload)
            self._canary_reports[report.change_id] = report

    def _resolve_component(self, name: str) -> Any:
        if name not in self.components:
            raise KeyError(f"Unknown AutoML component: {name}")
        return self.components[name]

    def _install_model(self, model: ResponseRiskModel, *, target_component: str, target_attribute: str) -> None:
        component = self._resolve_component(target_component)
        if not hasattr(component, target_attribute):
            raise AttributeError(f"Component '{target_component}' has no attribute '{target_attribute}'")
        setattr(component, target_attribute, model)

    def stage_response_risk_candidate(
        self,
        *,
        goal_id: str,
        title: str,
        training_examples: list[ResponseRiskTrainingExample],
        anchor_cases: list[SkillArenaCase],
        transfer_cases: list[TransferCase],
        canary_cases: list[SkillArenaCase],
        rationale: str,
        search_space: dict[str, list[float]],
        threshold: float = 0.99,
        allowed_regression_delta: float = 0.0,
    ) -> AutoMLSpec:
        baseline_model = self.registry.get_active_model(RESPONSE_RISK_FAMILY)
        spec = AutoMLSpec(
            change_id=new_automl_change_id(),
            goal_id=goal_id,
            title=title,
            model_family=RESPONSE_RISK_FAMILY,
            target_component="planning",
            target_attribute="response_risk_model",
            baseline_model=baseline_model.to_dict(),
            training_examples=list(training_examples),
            search_space={key: list(values) for key, values in search_space.items()},
            anchor_cases=list(anchor_cases),
            transfer_cases=list(transfer_cases),
            canary_cases=list(canary_cases),
            rationale=rationale,
            threshold=threshold,
            allowed_regression_delta=allowed_regression_delta,
        )
        self._specs[spec.change_id] = spec
        self.ledger.save_automl_spec(spec.to_dict())
        self.ledger.append_event(
            {
                "event_type": GoalEventType.MODEL_CANDIDATE_STAGED.value,
                "goal_id": goal_id,
                "change_id": spec.change_id,
                "model_family": spec.model_family,
                "baseline_model_id": baseline_model.model_id,
            }
        )
        return spec

    def train_candidate(self, change_id: str) -> AutoMLTrainingReport:
        spec = self._specs[change_id]
        if spec.model_family != RESPONSE_RISK_FAMILY:
            raise ValueError(f"Unsupported AutoML family: {spec.model_family}")
        training: ResponseRiskTrainingReport = train_response_risk_model(
            list(spec.training_examples),
            search_space=spec.search_space,
            version_prefix="response_risk_model",
        )
        candidate_model = training.model
        self.registry.register_candidate(candidate_model)
        report = AutoMLTrainingReport(
            change_id=change_id,
            model_family=spec.model_family,
            baseline_model_id=str(spec.baseline_model.get("model_id")),
            candidate_model=candidate_model.to_dict(),
            train_metrics=dict(training.train_metrics),
            best_config=dict(training.best_config),
            leaderboard=list(training.leaderboard),
        )
        self._training_reports[change_id] = report
        spec.status = "trained"
        self.ledger.save_automl_spec(spec.to_dict())
        self.ledger.save_automl_training_report(report.to_dict())
        self.ledger.append_event(
            {
                "event_type": GoalEventType.MODEL_TRAINED.value,
                "goal_id": spec.goal_id,
                "change_id": change_id,
                "model_family": spec.model_family,
                "candidate_model_id": candidate_model.model_id,
                "accuracy": report.train_metrics.get("accuracy"),
            }
        )
        return report

    def _execute_response_risk_model(self, model: ResponseRiskModel, payload: dict[str, Any], *, rollout_stage: str) -> dict[str, Any]:
        planning = copy.deepcopy(self._resolve_component("planning"))
        if not isinstance(planning, PlanningModule):
            raise TypeError("planning component must be a PlanningModule")
        planning.set_response_risk_model(copy.deepcopy(model))
        planning.set_response_planning_policy("adaptive_risk_model")
        raw_goal = payload.get("goal")
        goal = raw_goal if isinstance(raw_goal, Goal) else Goal.from_dict(raw_goal)
        world_state = dict(payload.get("world_state", {}))
        plan = planning.make_plan(goal, world_state)
        steps = [step.name for step in plan.steps]
        return {
            "predicted_verify": "verify_outcome" in steps,
            "steps": steps,
            "policy": planning.response_planning_policy,
            "model_id": model.model_id,
            "rollout_stage": rollout_stage,
        }

    def _run_anchor(self, model: ResponseRiskModel, spec: AutoMLSpec, *, label: str, rollout_stage: str) -> SkillArenaRun:
        return self.skill_arena.run(
            capability=spec.model_family,
            label=label,
            cases=list(spec.anchor_cases),
            execute=lambda payload: self._execute_response_risk_model(model, payload, rollout_stage=rollout_stage),
            threshold=spec.threshold,
        )

    def _run_transfer(self, model: ResponseRiskModel, spec: AutoMLSpec, *, label: str) -> TransferRun:
        return self.transfer_suite.run(
            capability=spec.model_family,
            label=label,
            cases=list(spec.transfer_cases),
            execute=lambda payload: self._execute_response_risk_model(model, payload, rollout_stage="shadow_transfer"),
            threshold=spec.threshold,
        )

    def run_regression_gate(self, change_id: str) -> AutoMLRegressionReport:
        spec = self._specs[change_id]
        baseline_model = ResponseRiskModel.from_dict(spec.baseline_model)
        training_report = self._training_reports.get(change_id)
        if training_report is None:
            raise ValueError("train the candidate before running regression gate")
        candidate_model = ResponseRiskModel.from_dict(training_report.candidate_model)

        baseline_anchor = self._run_anchor(baseline_model, spec, label=f"{change_id}_baseline_anchor", rollout_stage="shadow_baseline")
        candidate_anchor = self._run_anchor(candidate_model, spec, label=f"{change_id}_candidate_anchor", rollout_stage="shadow_candidate")
        candidate_transfer = self._run_transfer(candidate_model, spec, label=f"{change_id}_candidate_transfer")

        regression_case_ids: list[str] = []
        failure_reasons: list[str] = []
        for base_case, cand_case in zip(baseline_anchor.cases, candidate_anchor.cases):
            if cand_case.score + spec.allowed_regression_delta < base_case.score:
                regression_case_ids.append(base_case.case_id)
        if regression_case_ids:
            failure_reasons.append(f"anchor_regression:{','.join(regression_case_ids)}")
        if not candidate_anchor.passed:
            failure_reasons.append("candidate_anchor_failed")
        if not candidate_transfer.passed:
            failure_reasons.append("candidate_transfer_failed")
        if candidate_anchor.mean_score + spec.allowed_regression_delta < baseline_anchor.mean_score:
            failure_reasons.append("candidate_mean_below_baseline")

        report = AutoMLRegressionReport(
            change_id=change_id,
            model_family=spec.model_family,
            baseline_anchor_run=baseline_anchor.to_dict(),
            candidate_anchor_run=candidate_anchor.to_dict(),
            candidate_transfer_run=candidate_transfer.to_dict(),
            passed=not failure_reasons,
            regression_case_ids=regression_case_ids,
            failure_reasons=failure_reasons,
        )
        self._regression_reports[change_id] = report
        spec.status = "regression_validated" if report.passed else "regression_rejected"
        self.ledger.save_automl_spec(spec.to_dict())
        self.ledger.save_automl_regression_report(report.to_dict())
        self.ledger.append_event(
            {
                "event_type": GoalEventType.MODEL_REGRESSION_EVALUATED.value,
                "goal_id": spec.goal_id,
                "change_id": change_id,
                "model_family": spec.model_family,
                "passed": report.passed,
                "failure_reasons": list(report.failure_reasons),
            }
        )
        return report

    def promote_canary(self, change_id: str) -> AutoMLSpec:
        spec = self._specs[change_id]
        regression = self._regression_reports.get(change_id)
        if regression is None or not regression.passed:
            raise ValueError("regression gate must pass before canary promotion")
        self.registry.promote_candidate_to_canary(spec.model_family)
        spec.status = "canary_ready"
        self.ledger.save_automl_spec(spec.to_dict())
        self.ledger.append_event(
            {
                "event_type": GoalEventType.MODEL_CANARY_PROMOTED.value,
                "goal_id": spec.goal_id,
                "change_id": change_id,
                "model_family": spec.model_family,
                "candidate_model_id": self.registry.get_canary_model(spec.model_family).model_id,
            }
        )
        return spec

    def evaluate_canary(self, change_id: str) -> AutoMLCanaryReport:
        spec = self._specs[change_id]
        if spec.status not in {"canary_ready", "canary_active"}:
            raise ValueError("AutoML candidate is not in canary-ready state")
        active_before = self.registry.get_active_model(spec.model_family)
        canary_model = self.registry.get_canary_model(spec.model_family)
        self._install_model(canary_model, target_component=spec.target_component, target_attribute=spec.target_attribute)
        canary_run = self.skill_arena.run(
            capability=spec.model_family,
            label=f"{change_id}_canary",
            cases=list(spec.canary_cases),
            execute=lambda payload: self._execute_response_risk_model(canary_model, payload, rollout_stage="canary"),
            threshold=spec.threshold,
        )
        rolled_back = not canary_run.passed
        if rolled_back:
            restored = self.registry.rollback_canary(spec.model_family)
            if restored is None:
                raise RuntimeError("expected stable model during canary rollback")
            self._install_model(restored, target_component=spec.target_component, target_attribute=spec.target_attribute)
            spec.status = "rolled_back"
            active_after_model_id = restored.model_id
        else:
            spec.status = "canary_active"
            active_after_model_id = canary_model.model_id
        report = AutoMLCanaryReport(
            change_id=change_id,
            model_family=spec.model_family,
            canary_run=canary_run.to_dict(),
            passed=canary_run.passed,
            rolled_back=rolled_back,
            active_before_model_id=active_before.model_id,
            active_after_model_id=active_after_model_id,
        )
        self._canary_reports[change_id] = report
        self.ledger.save_automl_spec(spec.to_dict())
        self.ledger.save_automl_canary_report(report.to_dict())
        self.ledger.append_event(
            {
                "event_type": GoalEventType.MODEL_ROLLED_BACK.value if rolled_back else GoalEventType.MODEL_CANARY_EVALUATED.value,
                "goal_id": spec.goal_id,
                "change_id": change_id,
                "model_family": spec.model_family,
                "passed": report.passed,
                "rolled_back": rolled_back,
                "active_after_model_id": active_after_model_id,
            }
        )
        return report

    def finalize_model(self, change_id: str) -> ResponseRiskModel:
        spec = self._specs[change_id]
        report = self._canary_reports.get(change_id)
        if report is None or not report.passed or report.rolled_back:
            raise ValueError("canary must pass before model finalization")
        model = self.registry.finalize_canary(spec.model_family)
        self._install_model(model, target_component=spec.target_component, target_attribute=spec.target_attribute)
        spec.status = "finalized"
        self.ledger.save_automl_spec(spec.to_dict())
        self.ledger.append_event(
            {
                "event_type": GoalEventType.MODEL_FINALIZED.value,
                "goal_id": spec.goal_id,
                "change_id": change_id,
                "model_family": spec.model_family,
                "stable_model_id": model.model_id,
            }
        )
        return model

    def rollback_model(self, change_id: str, *, reason: str) -> ResponseRiskModel | None:
        spec = self._specs[change_id]
        restored = self.registry.rollback_family(spec.model_family)
        if restored is not None:
            self._install_model(restored, target_component=spec.target_component, target_attribute=spec.target_attribute)
        spec.status = "rolled_back"
        self.ledger.save_automl_spec(spec.to_dict())
        self.ledger.append_event(
            {
                "event_type": GoalEventType.MODEL_ROLLED_BACK.value,
                "goal_id": spec.goal_id,
                "change_id": change_id,
                "model_family": spec.model_family,
                "reason": reason,
                "restored_model_id": restored.model_id if restored is not None else None,
            }
        )
        return restored

    def get_spec(self, change_id: str) -> AutoMLSpec | None:
        return self._specs.get(change_id)

    def get_training_report(self, change_id: str) -> AutoMLTrainingReport | None:
        return self._training_reports.get(change_id)

    def get_regression_report(self, change_id: str) -> AutoMLRegressionReport | None:
        return self._regression_reports.get(change_id)

    def get_canary_report(self, change_id: str) -> AutoMLCanaryReport | None:
        return self._canary_reports.get(change_id)

    def list_specs(self) -> list[AutoMLSpec]:
        return sorted(self._specs.values(), key=lambda item: item.created_at)
