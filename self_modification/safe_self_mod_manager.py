from __future__ import annotations

import copy
from typing import Any, Callable

from agency.agency_module import AgencyModule
from benchmarks.skill_arena import SkillArena, SkillArenaCase, SkillArenaRun
from benchmarks.transfer_suite import TransferCase, TransferRun, TransferSuite
from core.event_types import GoalEventType
from memory.goal_ledger import GoalLedger
from self_modification.change_schema import (
    RegressionGateReport,
    SelfModificationCanaryReport,
    SelfModificationSpec,
    new_change_id,
)


SelfModExecutor = Callable[[dict[str, Any], dict[str, Any], str], dict[str, Any]]


class SafeSelfModificationManager:
    """Runs regression-gated, canary-guarded internal self-modifications with rollback."""

    def __init__(
        self,
        *,
        ledger: GoalLedger,
        components: dict[str, Any],
        skill_arena: SkillArena,
        transfer_suite: TransferSuite,
        executors: dict[str, SelfModExecutor] | None = None,
    ) -> None:
        self.ledger = ledger
        self.components = dict(components)
        self.skill_arena = skill_arena
        self.transfer_suite = transfer_suite
        self.executors = dict(executors or {})
        self._specs: dict[str, SelfModificationSpec] = {}
        self._regression_reports: dict[str, RegressionGateReport] = {}
        self._canary_reports: dict[str, SelfModificationCanaryReport] = {}
        self._rehydrate()

    def _rehydrate(self) -> None:
        for payload in self.ledger.load_self_modification_specs():
            spec = SelfModificationSpec.from_dict(payload)
            self._specs[spec.change_id] = spec
        for payload in self.ledger.load_self_modification_regressions():
            report = RegressionGateReport.from_dict(payload)
            self._regression_reports[report.change_id] = report
        for payload in self.ledger.load_self_modification_canaries():
            report = SelfModificationCanaryReport.from_dict(payload)
            self._canary_reports[report.change_id] = report

    def _resolve_component(self, components: dict[str, Any], name: str) -> Any:
        if name not in components:
            raise KeyError(f"Unknown self-modification component: {name}")
        return components[name]

    def _apply_candidate(self, components: dict[str, Any], spec: SelfModificationSpec) -> None:
        component = self._resolve_component(components, spec.target_component)
        if not hasattr(component, spec.parameter_name):
            raise AttributeError(f"Component '{spec.target_component}' has no field '{spec.parameter_name}'")
        setattr(component, spec.parameter_name, spec.candidate_value)

    def stage_attribute_change(
        self,
        *,
        goal_id: str,
        title: str,
        target_component: str,
        capability: str,
        parameter_name: str,
        candidate_value: Any,
        anchor_cases: list[SkillArenaCase],
        transfer_cases: list[TransferCase],
        canary_cases: list[SkillArenaCase],
        rationale: str,
        threshold: float = 0.99,
        allowed_regression_delta: float = 0.0,
    ) -> SelfModificationSpec:
        component = self._resolve_component(self.components, target_component)
        baseline_value = getattr(component, parameter_name)
        spec = SelfModificationSpec(
            change_id=new_change_id(),
            goal_id=goal_id,
            title=title,
            target_component=target_component,
            capability=capability,
            parameter_name=parameter_name,
            baseline_value=baseline_value,
            candidate_value=candidate_value,
            rationale=rationale,
            anchor_cases=list(anchor_cases),
            transfer_cases=list(transfer_cases),
            canary_cases=list(canary_cases),
            threshold=threshold,
            allowed_regression_delta=allowed_regression_delta,
        )
        self._specs[spec.change_id] = spec
        self.ledger.save_self_modification_spec(spec.to_dict())
        self.ledger.append_event(
            {
                "event_type": GoalEventType.SELF_MOD_SPEC_STAGED.value,
                "goal_id": goal_id,
                "change_id": spec.change_id,
                "capability": capability,
                "parameter_name": parameter_name,
                "baseline_value": baseline_value,
                "candidate_value": candidate_value,
            }
        )
        return spec

    def _execute_with_components(self, components: dict[str, Any], capability: str, payload: dict[str, Any], *, rollout_stage: str) -> dict[str, Any]:
        executor = self.executors.get(capability)
        if executor is not None:
            return executor(components, payload, rollout_stage)
        agency = components.get("agency")
        if isinstance(agency, AgencyModule):
            return agency.execute_capability(capability, payload, rollout_stage=rollout_stage)
        raise TypeError(f"Unsupported self-mod execution for capability={capability!r}")

    def _run_anchor(self, components: dict[str, Any], spec: SelfModificationSpec, *, label: str, rollout_stage: str) -> SkillArenaRun:
        return self.skill_arena.run(
            capability=spec.capability,
            label=label,
            cases=list(spec.anchor_cases),
            execute=lambda payload: self._execute_with_components(components, spec.capability, payload, rollout_stage=rollout_stage),
            threshold=spec.threshold,
        )

    def _run_transfer(self, components: dict[str, Any], spec: SelfModificationSpec, *, label: str) -> TransferRun:
        return self.transfer_suite.run(
            capability=spec.capability,
            label=label,
            cases=list(spec.transfer_cases),
            execute=lambda payload: self._execute_with_components(components, spec.capability, payload, rollout_stage="shadow_transfer"),
            threshold=spec.threshold,
        )

    def run_regression_gate(self, change_id: str) -> RegressionGateReport:
        spec = self._specs[change_id]
        baseline_components = copy.deepcopy(self.components)
        candidate_components = copy.deepcopy(self.components)
        self._apply_candidate(candidate_components, spec)

        baseline_anchor = self._run_anchor(baseline_components, spec, label=f"{change_id}_baseline_anchor", rollout_stage="shadow_baseline")
        candidate_anchor = self._run_anchor(candidate_components, spec, label=f"{change_id}_candidate_anchor", rollout_stage="shadow_candidate")
        candidate_transfer = self._run_transfer(candidate_components, spec, label=f"{change_id}_candidate_transfer")

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

        report = RegressionGateReport(
            change_id=change_id,
            capability=spec.capability,
            baseline_anchor_run=baseline_anchor.to_dict(),
            candidate_anchor_run=candidate_anchor.to_dict(),
            candidate_transfer_run=candidate_transfer.to_dict(),
            passed=not failure_reasons,
            regression_case_ids=regression_case_ids,
            failure_reasons=failure_reasons,
        )
        self._regression_reports[change_id] = report
        spec.status = "regression_validated" if report.passed else "regression_rejected"
        self.ledger.save_self_modification_spec(spec.to_dict())
        self.ledger.save_self_modification_regression(report.to_dict())
        self.ledger.append_event(
            {
                "event_type": GoalEventType.SELF_MOD_REGRESSION_EVALUATED.value,
                "goal_id": spec.goal_id,
                "change_id": change_id,
                "capability": spec.capability,
                "passed": report.passed,
                "failure_reasons": list(report.failure_reasons),
            }
        )
        return report

    def promote_canary(self, change_id: str) -> SelfModificationSpec:
        spec = self._specs[change_id]
        report = self._regression_reports.get(change_id)
        if report is None or not report.passed:
            raise ValueError("regression gate must pass before canary promotion")
        spec.status = "canary_ready"
        self.ledger.save_self_modification_spec(spec.to_dict())
        self.ledger.append_event(
            {
                "event_type": GoalEventType.SELF_MOD_CANARY_PROMOTED.value,
                "goal_id": spec.goal_id,
                "change_id": change_id,
                "capability": spec.capability,
                "candidate_value": spec.candidate_value,
            }
        )
        return spec

    def evaluate_canary(self, change_id: str) -> SelfModificationCanaryReport:
        spec = self._specs[change_id]
        if spec.status not in {"canary_ready", "canary_active"}:
            raise ValueError("self-modification candidate is not in canary-ready state")
        component = self._resolve_component(self.components, spec.target_component)
        active_before = getattr(component, spec.parameter_name)
        self._apply_candidate(self.components, spec)
        canary_run = self.skill_arena.run(
            capability=spec.capability,
            label=f"{change_id}_canary",
            cases=list(spec.canary_cases),
            execute=lambda payload: self._execute_with_components(self.components, spec.capability, payload, rollout_stage="canary"),
            threshold=spec.threshold,
        )
        rolled_back = not canary_run.passed
        if rolled_back:
            setattr(component, spec.parameter_name, active_before)
            spec.status = "rolled_back"
        else:
            spec.status = "canary_active"
        report = SelfModificationCanaryReport(
            change_id=change_id,
            capability=spec.capability,
            canary_run=canary_run.to_dict(),
            passed=canary_run.passed,
            rolled_back=rolled_back,
            active_before=active_before,
            restored_value=active_before if rolled_back else spec.candidate_value,
            active_value_after=getattr(component, spec.parameter_name),
        )
        self._canary_reports[change_id] = report
        self.ledger.save_self_modification_spec(spec.to_dict())
        self.ledger.save_self_modification_canary(report.to_dict())
        event_type = GoalEventType.SELF_MOD_ROLLED_BACK.value if rolled_back else GoalEventType.SELF_MOD_CANARY_EVALUATED.value
        self.ledger.append_event(
            {
                "event_type": event_type,
                "goal_id": spec.goal_id,
                "change_id": change_id,
                "capability": spec.capability,
                "passed": report.passed,
                "rolled_back": rolled_back,
                "active_value_after": report.active_value_after,
            }
        )
        return report

    def finalize_change(self, change_id: str) -> SelfModificationSpec:
        spec = self._specs[change_id]
        report = self._canary_reports.get(change_id)
        if report is None or not report.passed or report.rolled_back:
            raise ValueError("canary must pass before self-modification finalization")
        component = self._resolve_component(self.components, spec.target_component)
        if getattr(component, spec.parameter_name) != spec.candidate_value:
            setattr(component, spec.parameter_name, spec.candidate_value)
        spec.status = "finalized"
        self.ledger.save_self_modification_spec(spec.to_dict())
        self.ledger.append_event(
            {
                "event_type": GoalEventType.SELF_MOD_FINALIZED.value,
                "goal_id": spec.goal_id,
                "change_id": change_id,
                "capability": spec.capability,
                "parameter_name": spec.parameter_name,
                "stable_value": spec.candidate_value,
            }
        )
        return spec

    def rollback_change(self, change_id: str, reason: str) -> SelfModificationSpec:
        spec = self._specs[change_id]
        component = self._resolve_component(self.components, spec.target_component)
        setattr(component, spec.parameter_name, spec.baseline_value)
        spec.status = "rolled_back"
        self.ledger.save_self_modification_spec(spec.to_dict())
        self.ledger.append_event(
            {
                "event_type": GoalEventType.SELF_MOD_ROLLED_BACK.value,
                "goal_id": spec.goal_id,
                "change_id": change_id,
                "capability": spec.capability,
                "reason": reason,
                "restored_value": spec.baseline_value,
            }
        )
        return spec

    def get_spec(self, change_id: str) -> SelfModificationSpec | None:
        return self._specs.get(change_id)

    def get_regression_report(self, change_id: str) -> RegressionGateReport | None:
        return self._regression_reports.get(change_id)

    def get_canary_report(self, change_id: str) -> SelfModificationCanaryReport | None:
        return self._canary_reports.get(change_id)

    def list_specs(self) -> list[SelfModificationSpec]:
        return sorted(self._specs.values(), key=lambda item: item.created_at)
