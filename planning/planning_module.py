from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from automl.response_risk_model import ResponseRiskModel
from motivation.goal_schema import Goal, GoalKind, GoalSource


@dataclass(slots=True)
class PlanStep:
    name: str
    description: str


@dataclass(slots=True)
class Plan:
    goal_id: str
    steps: list[PlanStep] = field(default_factory=list)


class PlanningModule:
    """Turns structured goals into executable step sequences."""

    def __init__(
        self,
        *,
        response_planning_policy: str = "single_pass",
        response_risk_model: ResponseRiskModel | None = None,
    ) -> None:
        if response_planning_policy not in {"single_pass", "verify_before_answer", "adaptive_risk_model"}:
            raise ValueError(f"Unsupported response planning policy: {response_planning_policy}")
        self.response_planning_policy = response_planning_policy
        self.response_risk_model = response_risk_model

    def set_response_planning_policy(self, policy: str) -> None:
        if policy not in {"single_pass", "verify_before_answer", "adaptive_risk_model"}:
            raise ValueError(f"Unsupported response planning policy: {policy}")
        self.response_planning_policy = policy

    def set_response_risk_model(self, model: ResponseRiskModel | None) -> None:
        self.response_risk_model = model

    def _should_verify(self, goal: Goal, world_state: dict[str, Any]) -> bool:
        if self.response_planning_policy == "verify_before_answer":
            return bool(goal.evidence.get("requires_verification", False))
        if self.response_planning_policy == "adaptive_risk_model":
            if self.response_risk_model is None:
                return False
            return self.response_risk_model.should_verify(goal, world_state)
        return False

    def make_plan(self, goal: Goal, world_state: dict[str, Any] | None = None) -> Plan:
        if not isinstance(goal, Goal):
            raise TypeError("PlanningModule.make_plan expects a Goal instance.")

        world_state = world_state or {}
        steps: list[PlanStep]

        if goal.kind == GoalKind.TOOL_CREATION:
            steps = [
                PlanStep("design_tool_spec", "Build a safe, structured tool specification."),
                PlanStep("generate_tool_code", "Generate code only from approved templates."),
                PlanStep("validate_tool_code", "Run policy validation over the generated source."),
                PlanStep("register_tool", "Load the tool into a candidate runtime slot."),
                PlanStep("benchmark_tool", "Run offline deterministic benchmarks before rollout."),
                PlanStep("promote_canary", "Expose the candidate to a guarded canary slot."),
                PlanStep("evaluate_canary", "Run live runtime checks before trust is increased."),
                PlanStep("finalize_rollout", "Finalize stable promotion only if canary checks pass."),
            ]
        elif goal.kind == GoalKind.USER_TASK and "text_summarizer" in goal.required_capabilities:
            steps = [
                PlanStep("understand_request", "Interpret the user goal."),
                PlanStep("run_capability:text_summarizer", "Use the stable summarizer tool."),
            ]
            if self._should_verify(goal, world_state):
                steps.append(PlanStep("verify_outcome", "Verify the synthesized answer before responding."))
            steps.append(PlanStep("form_response", "Produce a concise answer or action."))
        elif goal.kind == GoalKind.USER_TASK:
            steps = [
                PlanStep("understand_request", "Interpret the user goal."),
            ]
            if self._should_verify(goal, world_state):
                steps.append(PlanStep("verify_outcome", "Verify the answer before responding."))
            steps.append(PlanStep("form_response", "Produce a concise answer or action."))
        elif goal.kind == GoalKind.MAINTENANCE and (goal.evidence.get("self_modification") or "grounded_navigation_patch" in goal.tags):
            steps = [
                PlanStep("stage_self_mod_candidate", "Create a shadow self-modification candidate for grounded navigation."),
                PlanStep("run_self_mod_regression_gate", "Require anchor and transfer validation before live rollout."),
                PlanStep("promote_self_mod_canary", "Expose the internal change to a guarded canary slot."),
                PlanStep("evaluate_self_mod_canary", "Run limited live tasks and rollback on regression."),
                PlanStep("finalize_self_mod", "Commit the internal modification only after a clean canary."),
                PlanStep("record_learning", "Persist the grounded-navigation self-modification outcome."),
            ]
        elif goal.kind == GoalKind.MAINTENANCE and "automl_model_upgrade" in goal.tags:
            steps = [
                PlanStep("stage_model_candidate", "Create a guarded candidate model upgrade from curriculum evidence."),
                PlanStep("train_model_candidate", "Train the candidate in an offline shadow lane."),
                PlanStep("run_model_regression_gate", "Require anchor and transfer validation before live exposure."),
                PlanStep("promote_model_canary", "Expose the candidate model to a guarded canary slot."),
                PlanStep("evaluate_model_canary", "Run limited live checks and rollback on regression."),
                PlanStep("finalize_model", "Finalize the stable model only after a clean canary."),
                PlanStep("record_learning", "Persist the model-upgrade outcome and audit trail."),
            ]
        elif goal.kind == GoalKind.MAINTENANCE and goal.evidence.get("interface_probe"):
            interface_name = str(goal.evidence.get("required_interface", "external_interface"))
            action = str(goal.evidence.get("experiment_action", "probe_external_interface"))
            steps = [
                PlanStep(action, f"Probe deferred {interface_name} execution through the safe adapter."),
                PlanStep("record_learning", "Persist the interface probe outcome and campaign notes."),
            ]
        elif goal.kind == GoalKind.MAINTENANCE and "grounded_navigation_audit" in goal.tags:
            steps = [
                PlanStep("run_capability:grounded_navigation", "Audit the grounded navigation skill on deterministic tasks."),
                PlanStep("record_learning", "Persist grounded navigation audit findings."),
            ]
        elif goal.kind == GoalKind.MAINTENANCE and "skill_audit" in goal.tags:
            steps = [
                PlanStep("review_episode_cluster", "Inspect the recent episode cluster for drift signals."),
                PlanStep("refresh_skill_snapshot", "Refresh the stable skill snapshot after audit."),
                PlanStep("record_learning", "Persist audit findings into memory."),
            ]
        elif goal.kind == GoalKind.LEARNING and "grounded_self_training" in goal.tags:
            steps = [
                PlanStep("run_capability:grounded_navigation", "Practice grounded navigation in a deterministic micro-world."),
                PlanStep("record_learning", "Persist the grounded practice outcome."),
            ]
        elif goal.kind == GoalKind.LEARNING and ("transfer_curriculum" in goal.tags or goal.evidence.get("curriculum_type") in {"capability_transfer", "capability_growth"}):
            steps = [
                PlanStep(f"run_capability:{goal.evidence.get('target_capability', 'unknown')}", "Practice the transferred capability on deterministic tasks."),
                PlanStep("refresh_skill_snapshot", "Refresh the stable skill snapshot after transfer practice."),
                PlanStep("record_learning", "Persist cross-skill transfer results."),
            ]
        elif goal.kind == GoalKind.LEARNING and "semantic_consolidation" in goal.tags:
            steps = [
                PlanStep("review_episode_cluster", "Inspect repeated successful episodes."),
                PlanStep("record_learning", "Persist consolidated semantic knowledge."),
            ]
        elif goal.kind == GoalKind.EXPLORATION:
            steps = [
                PlanStep("scan_novelty", "Inspect the novel observation."),
                PlanStep("store_findings", "Persist useful findings into memory."),
            ]
        elif goal.kind == GoalKind.MAINTENANCE:
            steps = [
                PlanStep("diagnose_drift", "Confirm or reject distribution shift."),
                PlanStep("refresh_baseline", "Refresh drift baselines or calibration."),
            ]
        elif goal.kind == GoalKind.REGRESSION_RECOVERY:
            steps = [
                PlanStep("replay_anchor_suite", "Replay anchor tasks to recover the lost skill."),
                PlanStep("compare_baseline", "Compare current results with baseline."),
            ]
        elif goal.source == GoalSource.MEMORY_GAP or "knowledge_gap" in goal.tags:
            steps = [
                PlanStep("retrieve_missing_knowledge", "Fill the missing knowledge slot."),
                PlanStep("record_learning", "Store the learned fact or rule."),
            ]
        else:
            steps = [
                PlanStep("study_gap", "Study the identified weakness."),
                PlanStep("record_learning", "Persist the outcome."),
            ]

        return Plan(goal_id=goal.goal_id, steps=steps)

    def evaluate_plan(self, plan: Plan) -> bool:
        names = [step.name for step in plan.steps]
        return len(names) == len(set(names))

    def replan(self, goal: Goal, feedback: dict[str, Any] | None = None) -> Plan:
        base_plan = self.make_plan(goal)
        if feedback and feedback.get("needs_verification"):
            base_plan.steps.append(PlanStep("verify_outcome", "Verify the revised outcome."))
        return base_plan
