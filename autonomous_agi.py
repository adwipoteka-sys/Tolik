from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from agency.agency_module import AgencyModule
from automl.response_risk_curriculum import build_anchor_cases, build_canary_cases, build_search_space, build_training_examples, build_transfer_cases
from automl.response_risk_model import ResponseRiskTrainingExample
from benchmarks.skill_arena import SkillArena, SkillArenaCase
from benchmarks.report_types import NavigationConfidenceReport, SkillArenaReport, TransferEvaluationReport
from benchmarks.transfer_suite import TransferCase, TransferSuite
from core.global_workspace import GlobalWorkspace
from experiments.curriculum_experiment_scheduler import CurriculumExperimentScheduler
from experiments.experiment_board import ExperimentBoard
from experiments.experiment_schema import ExperimentKind, ExperimentProposal
from experiments.experiment_sources import build_future_interface_candidates, build_response_risk_model_upgrade_candidate, build_self_mod_experiment_candidates
from interfaces.provider_qualification import load_provider_catalog
from interfaces.canary_rollout import configure_canary_rollout
from interfaces.post_promotion_monitor import configure_post_promotion_monitors
from automl.response_risk_data_pipeline import ResponseRiskDataAcquisitionPipeline
from environments.grounded_navigation import GroundedNavigationLab
from environments.route_briefing import RouteBriefingLab
from environments.spatial_route import SpatialRouteLab
from language.language_module import LanguageModule
from main import build_system
from memory.capability_graph import CapabilityGraph
from memory.capability_portfolio import CapabilityPortfolio
from memory.improvement_ledger import ImprovementGoal, ImprovementLedger
from memory.episodic_memory import EpisodicMemory
from memory.memory_module import MemoryModule
from metacognition.metacognition_module import MetacognitionModule
from metacognition.failure_analyzer import FailureAnalyzer
from motivation.autonomous_goal_manager import AutonomousGoalManager
from motivation.goal_schema import Goal, GoalBudget, GoalKind, GoalSource, SuccessCriterion, new_goal_id
from motivation.operator_charter import OperatorCharter, load_charter
from planning.capability_growth_planner import CapabilityGrowthPlanner, CapabilityGrowthPlan
from planning.capability_growth_governor import CapabilityGrowthGovernor, GrowthPlanAssessment
from planning.long_horizon_scheduler import LongHorizonScheduler
from planning.internal_goal_scheduler import InternalGoalScheduler
from planning.planning_module import PlanningModule
from self_modification.safe_self_mod_manager import SafeSelfModificationManager
from policy.tool_proposal_guard import ToolProposal, ToolProposalGuard
from integrations.quantum_solver import DeferredQuantumTask
from integrations.cloud_llm_client import DeferredLLMTask


GROUNDED_NAVIGATION_PATTERN = "grounded_navigation|graph_search_patch"
SPATIAL_ROUTE_PATTERN = "spatial_route_composition|waypoint_graph_reuse"
ROUTE_BRIEFING_PATTERN = "route_mission_briefing|multi_leg_route_briefing"
ROUTE_EXPLANATION_PATTERN = "navigation_route_explanation|grounded_trace_explanation"

GROUNDED_NAVIGATION_LESSON = "When navigation requires a detour, upgrade from greedy moves to graph search."
SPATIAL_ROUTE_LESSON = "Compose transfer-validated navigation primitives across waypoint chains."
ROUTE_BRIEFING_LESSON = "Summarize validated waypoint routes into concise mission briefings."
ROUTE_EXPLANATION_LESSON = "Explain grounded detours from transfer-validated graph-search traces."

EXECUTABLE_GROWTH_TARGETS = {"spatial_route_composition", "route_mission_briefing", "navigation_route_explanation"}


def _apply_provider_rollout(system: dict[str, object], *, charter: OperatorCharter, provider_catalog_path: Path | None) -> dict[str, Any]:
    workspace: GlobalWorkspace = system["workspace"]  # type: ignore[assignment]
    quantum_solver = system["quantum_solver"]
    cloud_llm = system["cloud_llm"]
    if provider_catalog_path is None:
        workspace.update({"provider_qualification_summary": {}, "interface_rollout_decisions": {}, "interface_canary_rollout": {}, "post_promotion_monitoring": {}})
        return {"reports": {}, "decisions": {}, "canary_summaries": {}, "monitor_summaries": {}}

    catalog = load_provider_catalog(provider_catalog_path)
    qualification_manager = system["provider_qualification_manager"]
    rollout_gate = system["rollout_gate"]
    reports = qualification_manager.qualify_catalog(catalog, charter=charter)
    decisions = rollout_gate.apply(
        adapters={"cloud_llm": cloud_llm, "quantum_solver": quantum_solver},
        catalog=catalog,
        reports=reports,
        charter=charter,
    )
    canary_summaries = configure_canary_rollout(
        adapters={"cloud_llm": cloud_llm, "quantum_solver": quantum_solver},
        catalog=catalog,
        reports=reports,
        decisions=decisions,
        charter=charter,
        ledger=system.get("ledger"),
    )
    monitor_summaries = configure_post_promotion_monitors(
        adapters={"cloud_llm": cloud_llm, "quantum_solver": quantum_solver},
        catalog=catalog,
        reports=reports,
        decisions=decisions,
        charter=charter,
        ledger=system.get("ledger"),
    )
    workspace.update(
        {
            "provider_catalog": catalog.to_dict(),
            "provider_qualification_summary": {
                adapter_name: [report.to_dict() for report in adapter_reports]
                for adapter_name, adapter_reports in reports.items()
            },
            "interface_rollout_decisions": {
                adapter_name: decision.to_dict() for adapter_name, decision in decisions.items()
            },
            "interface_canary_rollout": canary_summaries,
            "post_promotion_monitoring": monitor_summaries,
            "future_interfaces": {
                "quantum_solver": quantum_solver.mode,
                "cloud_llm": cloud_llm.mode,
            },
            "interface_adapter_summary": {
                "quantum_solver": quantum_solver.summary(),
                "cloud_llm": cloud_llm.summary(),
            },
        }
    )
    return {
        "reports": {adapter_name: [report.to_dict() for report in adapter_reports] for adapter_name, adapter_reports in reports.items()},
        "decisions": {adapter_name: decision.to_dict() for adapter_name, decision in decisions.items()},
        "canary_summaries": canary_summaries,
        "monitor_summaries": monitor_summaries,
    }


def _navigation_failure_practice_payload() -> dict[str, Any]:
    lab = GroundedNavigationLab()
    return {
        "tasks": [
            lab.get_task("nav_detour_wall").to_dict(),
            lab.get_task("nav_detour_channel").to_dict(),
        ],
        "success_threshold": 1.0,
    }



def _navigation_audit_payload() -> dict[str, Any]:
    return _navigation_failure_practice_payload()



def _route_practice_payload() -> dict[str, Any]:
    lab = SpatialRouteLab()
    return {
        "tasks": [
            lab.get_task("route_train_open_chain").to_dict(),
            lab.get_task("route_train_detour_chain").to_dict(),
        ],
        "success_threshold": 1.0,
    }



def _route_briefing_practice_payload() -> dict[str, Any]:
    lab = RouteBriefingLab()
    return {
        "tasks": [
            lab.get_task("route_train_open_chain").to_dict(),
            lab.get_task("route_train_detour_chain").to_dict(),
        ]
    }



def _route_explanation_practice_payload() -> dict[str, Any]:
    lab = GroundedNavigationLab()
    return {
        "tasks": [
            lab.get_task("nav_detour_wall").to_dict(),
            lab.get_task("nav_detour_channel").to_dict(),
        ],
        "success_threshold": 1.0,
        "detour_explanation_threshold": 1.0,
    }



def _growth_payload(target_capability: str) -> dict[str, Any]:
    if target_capability == "spatial_route_composition":
        return _route_practice_payload()
    if target_capability == "route_mission_briefing":
        return _route_briefing_practice_payload()
    if target_capability == "navigation_route_explanation":
        return _route_explanation_practice_payload()
    raise KeyError(f"No payload builder for capability: {target_capability}")



def _make_grounded_training_goal() -> Goal:
    return Goal(
        goal_id=new_goal_id("curiosity"),
        title="Practice grounded navigation in detour tasks",
        description="Run grounded self-training on detour-heavy navigation tasks.",
        source=GoalSource.CURIOSITY,
        kind=GoalKind.LEARNING,
        expected_gain=0.84,
        novelty=0.44,
        uncertainty_reduction=0.76,
        strategic_fit=0.92,
        risk_estimate=0.08,
        priority=0.86,
        risk_budget=0.18,
        resource_budget=GoalBudget(max_steps=3, max_seconds=20.0, max_tool_calls=0, max_api_calls=0),
        success_criteria=[SuccessCriterion(metric="success_rate", comparator=">=", target=1.0)],
        required_capabilities=["classical_planning", "grounded_navigation"],
        tags=["grounded_self_training", "grounded_navigation", "local_only"],
        evidence={"tool_payload": _navigation_failure_practice_payload()},
    )



def _admit_self_modification_goals_from_memory(
    system: dict[str, object],
    *,
    charter: OperatorCharter,
) -> list[Goal]:
    manager: AutonomousGoalManager = system["manager"]  # type: ignore[assignment]
    workspace: GlobalWorkspace = system["workspace"]  # type: ignore[assignment]
    self_mod_proposer = system["self_mod_proposer"]

    proposals = self_mod_proposer.propose_from_memory()
    if not proposals:
        workspace.update(
            {
                "self_mod_proposal_count": len(self_mod_proposer.list_proposals()),
                "latest_self_mod_proposal": None,
            }
        )
        return []

    workspace.update(
        {
            "self_mod_proposal_count": len(self_mod_proposer.list_proposals()),
            "latest_self_mod_proposal": proposals[-1].to_dict(),
        }
    )
    goals = [self_mod_proposer.materialize_goal(proposal.proposal_id) for proposal in proposals]
    return _admit_with_charter(manager, goals, charter=charter, context=workspace.get_state())



def _make_navigation_audit_goal(source: GoalSource = GoalSource.METACOGNITION) -> Goal:
    return Goal(
        goal_id=new_goal_id("audit" if source != GoalSource.SCHEDULER else "sched"),
        title="Audit grounded navigation on detour tasks",
        description="Run a deterministic audit over detour-heavy grounded navigation tasks.",
        source=source,
        kind=GoalKind.MAINTENANCE,
        expected_gain=0.80,
        novelty=0.20,
        uncertainty_reduction=0.82,
        strategic_fit=0.94,
        risk_estimate=0.05,
        priority=0.83,
        risk_budget=0.12,
        resource_budget=GoalBudget(max_steps=3, max_seconds=15.0, max_tool_calls=0, max_api_calls=0),
        success_criteria=[SuccessCriterion(metric="success_rate", comparator=">=", target=1.0)],
        required_capabilities=["classical_planning", "grounded_navigation"],
        tags=["grounded_navigation_audit", "grounded_navigation", "skill_audit", "local_only"],
        evidence={"tool_payload": _navigation_audit_payload()},
    )



def _admit_with_charter(
    manager: AutonomousGoalManager,
    goals: list[Goal],
    *,
    charter: OperatorCharter,
    context: dict[str, Any],
) -> list[Goal]:
    allowed: list[Goal] = []
    for goal in goals:
        ok, reason = charter.goal_allowed(tags=goal.tags, required_capabilities=goal.required_capabilities)
        if ok:
            allowed.append(goal)
        else:
            print(f"[charter] blocked goal '{goal.title}': {reason}")
    if not allowed:
        return []
    return manager.admit_candidates(allowed[: charter.max_internal_goals_per_cycle], context=context)



def _navigation_skill_arena_cases() -> list[SkillArenaCase]:
    lab = GroundedNavigationLab()
    return [
        SkillArenaCase(
            case_id="nav_detour_curriculum",
            payload=_navigation_audit_payload(),
            expected={"min_success_rate": 1.0, "max_mean_path_ratio": 1.0, "strategy": "graph_search"},
            description="Detour tasks must be solved optimally after the patch.",
        ),
        SkillArenaCase(
            case_id="nav_easy_regression_guard",
            payload={
                "tasks": [lab.get_task("nav_easy_open").to_dict(), lab.get_task("nav_easy_corner").to_dict()],
                "success_threshold": 1.0,
            },
            expected={"min_success_rate": 1.0, "max_mean_path_ratio": 1.0, "strategy": "graph_search"},
            description="Easy grounded tasks must remain solved after the upgrade.",
        ),
    ]



def _navigation_transfer_cases() -> list[TransferCase]:
    lab = GroundedNavigationLab()
    return [
        TransferCase(
            case_id="nav_transfer_bridge",
            payload={"tasks": [lab.get_task("nav_transfer_bridge").to_dict()], "success_threshold": 1.0},
            expected={"min_success_rate": 1.0, "max_mean_path_ratio": 1.0, "strategy": "graph_search", "min_task_count": 1},
            description="Held-out bridge task must generalize.",
        ),
        TransferCase(
            case_id="nav_transfer_double_wall",
            payload={"tasks": [lab.get_task("nav_transfer_double_wall").to_dict()], "success_threshold": 1.0},
            expected={"min_success_rate": 1.0, "max_mean_path_ratio": 1.0, "strategy": "graph_search", "min_task_count": 1},
            description="Held-out multi-detour task must generalize.",
        ),
    ]



def _navigation_self_mod_anchor_cases() -> list[SkillArenaCase]:
    lab = GroundedNavigationLab()
    return [
        SkillArenaCase(
            case_id="nav_anchor_easy_open",
            payload={"tasks": [lab.get_task("nav_easy_open").to_dict()], "success_threshold": 1.0},
            expected={"min_success_rate": 1.0, "max_mean_path_ratio": 1.0},
            description="Easy open task must remain solved after internal modification.",
        ),
        SkillArenaCase(
            case_id="nav_anchor_easy_corner",
            payload={"tasks": [lab.get_task("nav_easy_corner").to_dict()], "success_threshold": 1.0},
            expected={"min_success_rate": 1.0, "max_mean_path_ratio": 1.0},
            description="Easy corner task must remain solved after internal modification.",
        ),
    ]


def _navigation_self_mod_canary_cases() -> list[SkillArenaCase]:
    return _navigation_skill_arena_cases()


def _stage_self_modification_from_goal(system: dict[str, object], goal: Goal):
    self_mod_manager: SafeSelfModificationManager = system["self_mod_manager"]  # type: ignore[assignment]
    self_mod_proposer = system["self_mod_proposer"]
    workspace: GlobalWorkspace = system["workspace"]  # type: ignore[assignment]
    self_mod_proposer = system["self_mod_proposer"]

    proposal_id = goal.evidence.get("self_mod_proposal_id")
    if not isinstance(proposal_id, str):
        raise ValueError("missing self_mod_proposal_id in goal evidence")
    proposal = self_mod_proposer.get(proposal_id)
    if proposal is None:
        raise ValueError(f"unknown self-modification proposal: {proposal_id}")

    change_id = goal.evidence.get("self_mod_change_id")
    if isinstance(change_id, str):
        existing = self_mod_manager.get_spec(change_id)
        if existing is not None:
            return existing, proposal

    spec = self_mod_manager.stage_attribute_change(
        goal_id=goal.goal_id,
        title=proposal.title,
        target_component=proposal.target_component,
        capability=proposal.capability,
        parameter_name=proposal.parameter_name,
        candidate_value=proposal.candidate_value,
        anchor_cases=list(proposal.anchor_cases),
        transfer_cases=list(proposal.transfer_cases),
        canary_cases=list(proposal.canary_cases),
        rationale=proposal.rationale,
        threshold=proposal.threshold,
        allowed_regression_delta=proposal.allowed_regression_delta,
    )
    goal.evidence["self_mod_change_id"] = spec.change_id
    workspace.update(
        {
            "self_mod_proposal_count": len(self_mod_proposer.list_proposals()),
            "latest_self_mod_proposal": proposal.to_dict(),
            "self_mod_change_count": len(self_mod_manager.list_specs()),
            "latest_self_mod_status": spec.status,
        }
    )
    return spec, proposal



def _run_self_modification_step(step_name: str, goal: Goal, system: dict[str, object]) -> dict[str, Any]:
    self_mod_manager: SafeSelfModificationManager = system["self_mod_manager"]  # type: ignore[assignment]
    self_mod_proposer = system["self_mod_proposer"]
    agency: AgencyModule = system["agency"]  # type: ignore[assignment]
    planning: PlanningModule = system["planning"]  # type: ignore[assignment]
    memory: MemoryModule = system["memory"]  # type: ignore[assignment]
    workspace: GlobalWorkspace = system["workspace"]  # type: ignore[assignment]

    spec, proposal = _stage_self_modification_from_goal(system, goal)
    if step_name == "stage_self_mod_candidate":
        return {
            "step": step_name,
            "success": True,
            "proposal_id": proposal.proposal_id,
            "change_id": spec.change_id,
            "baseline_value": spec.baseline_value,
            "candidate_value": spec.candidate_value,
            "message": f"staged safe self-modification candidate for {spec.parameter_name}",
        }
    if step_name == "run_self_mod_regression_gate":
        report = self_mod_manager.run_regression_gate(spec.change_id)
        if not report.passed:
            self_mod_proposer.mark_rolled_back(proposal.proposal_id, reason="regression_rejected")
        workspace.update({
            "latest_self_mod_status": self_mod_manager.get_spec(spec.change_id).status,
            "latest_self_mod_proposal": self_mod_proposer.get(proposal.proposal_id).to_dict() if self_mod_proposer.get(proposal.proposal_id) else None,
        })
        return {
            "step": step_name,
            "success": report.passed,
            "proposal_id": proposal.proposal_id,
            "change_id": spec.change_id,
            "regression_case_ids": list(report.regression_case_ids),
            "failure_reasons": list(report.failure_reasons),
            "message": (
                f"self-mod regression gate passed for {spec.parameter_name}"
                if report.passed
                else f"self-mod regression gate rejected {spec.parameter_name}"
            ),
        }
    if step_name == "promote_self_mod_canary":
        staged = self_mod_manager.promote_canary(spec.change_id)
        workspace.update({"latest_self_mod_status": staged.status, "latest_self_mod_proposal": proposal.to_dict()})
        return {
            "step": step_name,
            "success": True,
            "proposal_id": proposal.proposal_id,
            "change_id": spec.change_id,
            "message": f"self-mod canary promoted for {spec.parameter_name}",
        }
    if step_name == "evaluate_self_mod_canary":
        report = self_mod_manager.evaluate_canary(spec.change_id)
        if report.rolled_back:
            self_mod_proposer.mark_rolled_back(proposal.proposal_id, reason="canary_failed")
        workspace.update({
            "latest_self_mod_status": self_mod_manager.get_spec(spec.change_id).status,
            "latest_self_mod_proposal": self_mod_proposer.get(proposal.proposal_id).to_dict() if self_mod_proposer.get(proposal.proposal_id) else None,
            "navigation_strategy": agency.grounded_navigation_strategy,
            "response_planning_policy": planning.response_planning_policy,
            "retrieval_policy": memory.retrieval_policy,
        })
        return {
            "step": step_name,
            "success": report.passed and not report.rolled_back,
            "proposal_id": proposal.proposal_id,
            "change_id": spec.change_id,
            "rolled_back": report.rolled_back,
            "active_value_after": report.active_value_after,
            "message": (
                f"self-mod canary passed for {spec.parameter_name}"
                if report.passed and not report.rolled_back
                else f"self-mod canary rolled back {spec.parameter_name}"
            ),
        }
    if step_name == "finalize_self_mod":
        finalized = self_mod_manager.finalize_change(spec.change_id)
        self_mod_proposer.mark_finalized(proposal.proposal_id)
        workspace.update({
            "latest_self_mod_status": finalized.status,
            "latest_self_mod_proposal": self_mod_proposer.get(proposal.proposal_id).to_dict() if self_mod_proposer.get(proposal.proposal_id) else None,
            "navigation_strategy": agency.grounded_navigation_strategy,
            "response_planning_policy": planning.response_planning_policy,
            "retrieval_policy": memory.retrieval_policy,
        })
        return {
            "step": step_name,
            "success": True,
            "proposal_id": proposal.proposal_id,
            "change_id": spec.change_id,
            "stable_value": finalized.candidate_value,
            "message": f"safe self-modification finalized: {finalized.parameter_name}={finalized.candidate_value}",
        }
    raise ValueError(f"Unsupported self-modification step: {step_name}")


def _materialize_model_upgrade_goal(proposal: ExperimentProposal) -> Goal:
    return Goal(
        goal_id=new_goal_id("automlgoal"),
        title=proposal.title,
        description=proposal.description,
        source=GoalSource.SCHEDULER,
        kind=GoalKind.MAINTENANCE,
        expected_gain=max(0.62, min(0.98, proposal.expected_utility)),
        novelty=max(0.26, min(0.74, 0.18 + (0.06 * len(proposal.curriculum_signals)))),
        uncertainty_reduction=max(0.58, min(0.95, 0.40 + (0.45 * proposal.confidence))),
        strategic_fit=max(0.76, min(0.98, 0.52 + (0.30 * proposal.expected_utility))),
        risk_estimate=proposal.estimated_risk,
        priority=max(0.72, min(0.98, 0.62 + (0.42 * proposal.expected_utility))),
        risk_budget=max(0.15, min(0.24, proposal.estimated_risk + 0.05)),
        resource_budget=GoalBudget(max_steps=7, max_seconds=30.0, max_tool_calls=0, max_api_calls=0),
        success_criteria=[SuccessCriterion(metric="status", comparator="==", target="finalized")],
        required_capabilities=list(proposal.required_capabilities),
        tags=list(proposal.tags),
        evidence={
            **proposal.evidence,
            "automl_upgrade": True,
            "experiment_proposal_id": proposal.proposal_id,
            "curriculum_signals": list(proposal.curriculum_signals),
        },
    )


def _materialize_interface_probe_goal(proposal: ExperimentProposal) -> Goal:
    return Goal(
        goal_id=new_goal_id("iface"),
        title=proposal.title,
        description=proposal.description,
        source=GoalSource.SCHEDULER,
        kind=GoalKind.MAINTENANCE,
        expected_gain=max(0.52, min(0.88, proposal.expected_utility)),
        novelty=max(0.24, min(0.70, 0.18 + (0.05 * len(proposal.curriculum_signals)))),
        uncertainty_reduction=max(0.44, min(0.86, 0.34 + (0.36 * proposal.confidence))),
        strategic_fit=max(0.60, min(0.92, 0.46 + (0.28 * proposal.expected_utility))),
        risk_estimate=proposal.estimated_risk,
        priority=max(0.58, min(0.90, 0.48 + (0.34 * proposal.expected_utility))),
        risk_budget=max(0.10, min(0.18, proposal.estimated_risk + 0.04)),
        resource_budget=GoalBudget(max_steps=2, max_seconds=12.0, max_tool_calls=0, max_api_calls=1),
        success_criteria=[SuccessCriterion(metric="status", comparator="==", target="ok")],
        required_capabilities=list(proposal.required_capabilities),
        tags=list(proposal.tags),
        evidence={
            **proposal.evidence,
            "interface_probe": True,
            "experiment_proposal_id": proposal.proposal_id,
            "curriculum_signals": list(proposal.curriculum_signals),
        },
    )


def _materialize_experiment_goal(system: dict[str, object], proposal: ExperimentProposal) -> Goal:
    if proposal.experiment_kind == ExperimentKind.POLICY_CHANGE:
        if "self_mod_proposal_id" in proposal.evidence:
            self_mod_proposer = system["self_mod_proposer"]
            goal = self_mod_proposer.materialize_goal(str(proposal.evidence["self_mod_proposal_id"]))
        else:
            goal = _materialize_interface_probe_goal(proposal)
        goal.evidence["experiment_proposal_id"] = proposal.proposal_id
        goal.evidence["experiment_kind"] = proposal.experiment_kind.value
        return goal
    if proposal.experiment_kind == ExperimentKind.MODEL_UPGRADE:
        goal = _materialize_model_upgrade_goal(proposal)
        goal.evidence["experiment_kind"] = proposal.experiment_kind.value
        return goal
    raise ValueError(f"Unsupported experiment kind: {proposal.experiment_kind}")


def _stage_automl_from_goal(system: dict[str, object], goal: Goal):
    automl_manager = system["automl_manager"]
    workspace: GlobalWorkspace = system["workspace"]  # type: ignore[assignment]

    change_id = goal.evidence.get("automl_change_id")
    if isinstance(change_id, str):
        existing = automl_manager.get_spec(change_id)
        if existing is not None:
            return existing

    training_examples = [ResponseRiskTrainingExample.from_dict(item) for item in goal.evidence.get("training_examples", [])]
    anchor_cases = [SkillArenaCase.from_dict(item) for item in goal.evidence.get("anchor_cases", [])]
    transfer_cases = [TransferCase.from_dict(item) for item in goal.evidence.get("transfer_cases", [])]
    canary_cases = [SkillArenaCase.from_dict(item) for item in goal.evidence.get("canary_cases", [])]
    search_space = {key: list(values) for key, values in goal.evidence.get("search_space", {}).items()}
    if not training_examples:
        training_examples = list(build_training_examples())
    if not anchor_cases:
        anchor_cases = list(build_anchor_cases())
    if not transfer_cases:
        transfer_cases = list(build_transfer_cases())
    if not canary_cases:
        canary_cases = list(build_canary_cases())
    if not search_space:
        search_space = build_search_space()

    spec = automl_manager.stage_response_risk_candidate(
        goal_id=goal.goal_id,
        title=goal.title,
        training_examples=training_examples,
        anchor_cases=anchor_cases,
        transfer_cases=transfer_cases,
        canary_cases=canary_cases,
        rationale=goal.description,
        search_space=search_space,
        threshold=0.99,
    )
    goal.evidence["automl_change_id"] = spec.change_id
    workspace.update({
        "latest_model_status": spec.status,
    })
    return spec


def _run_interface_probe_step(step_name: str, goal: Goal, system: dict[str, object]) -> dict[str, Any]:
    quantum_solver = system["quantum_solver"]
    cloud_llm = system["cloud_llm"]
    workspace: GlobalWorkspace = system["workspace"]  # type: ignore[assignment]

    if step_name == "probe_quantum_interface":
        result = quantum_solver.factorize(21)
        workspace.update({"future_interfaces": {**workspace.get("future_interfaces", {}), "quantum_solver": quantum_solver.mode}})
        return {
            "step": step_name,
            "success": result.get("status") == "ok",
            "status": result.get("status"),
            "mode": quantum_solver.mode,
            "result": result,
        }
    if step_name == "probe_cloud_llm_interface":
        result = cloud_llm.generate("Alpha. Beta. Gamma.", task="summarize")
        workspace.update({"future_interfaces": {**workspace.get("future_interfaces", {}), "cloud_llm": cloud_llm.mode}})
        return {
            "step": step_name,
            "success": result.get("status") == "ok",
            "status": result.get("status"),
            "mode": cloud_llm.mode,
            "result": result,
        }
    raise ValueError(f"Unsupported interface probe step: {step_name}")


def _run_automl_step(step_name: str, goal: Goal, system: dict[str, object]) -> dict[str, Any]:
    automl_manager = system["automl_manager"]
    registry = system["model_registry"]
    planning: PlanningModule = system["planning"]  # type: ignore[assignment]
    workspace: GlobalWorkspace = system["workspace"]  # type: ignore[assignment]

    spec = _stage_automl_from_goal(system, goal)
    if step_name == "stage_model_candidate":
        return {
            "step": step_name,
            "success": True,
            "change_id": spec.change_id,
            "baseline_model_id": spec.baseline_model.get("model_id"),
            "message": "staged safe model-upgrade candidate",
        }
    if step_name == "train_model_candidate":
        report = automl_manager.train_candidate(spec.change_id)
        workspace.update({
            "latest_model_status": automl_manager.get_spec(spec.change_id).status,
            "stable_models": registry.stable_model_names(),
        })
        return {
            "step": step_name,
            "success": True,
            "change_id": spec.change_id,
            "candidate_model_id": report.candidate_model.get("model_id"),
            "train_metrics": dict(report.train_metrics),
            "message": "trained candidate response-risk model",
        }
    if step_name == "run_model_regression_gate":
        report = automl_manager.run_regression_gate(spec.change_id)
        workspace.update({
            "latest_model_status": automl_manager.get_spec(spec.change_id).status,
        })
        return {
            "step": step_name,
            "success": report.passed,
            "change_id": spec.change_id,
            "failure_reasons": list(report.failure_reasons),
            "message": "model regression gate passed" if report.passed else "model regression gate rejected candidate",
        }
    if step_name == "promote_model_canary":
        staged = automl_manager.promote_canary(spec.change_id)
        workspace.update({
            "latest_model_status": staged.status,
        })
        return {
            "step": step_name,
            "success": True,
            "change_id": spec.change_id,
            "message": "model canary promoted",
        }
    if step_name == "evaluate_model_canary":
        report = automl_manager.evaluate_canary(spec.change_id)
        workspace.update({
            "latest_model_status": automl_manager.get_spec(spec.change_id).status,
            "stable_models": registry.stable_model_names(),
        })
        return {
            "step": step_name,
            "success": report.passed and not report.rolled_back,
            "change_id": spec.change_id,
            "rolled_back": report.rolled_back,
            "active_after_model_id": report.active_after_model_id,
            "message": "model canary passed" if report.passed and not report.rolled_back else "model canary rolled back",
        }
    if step_name == "finalize_model":
        model = automl_manager.finalize_model(spec.change_id)
        planning.set_response_risk_model(model)
        planning.set_response_planning_policy("adaptive_risk_model")
        workspace.update({
            "latest_model_status": automl_manager.get_spec(spec.change_id).status,
            "stable_models": registry.stable_model_names(),
            "response_planning_policy": planning.response_planning_policy,
        })
        return {
            "step": step_name,
            "success": True,
            "change_id": spec.change_id,
            "stable_model_id": model.model_id,
            "policy": planning.response_planning_policy,
            "message": "safe model-upgrade finalized",
        }
    raise ValueError(f"Unsupported AutoML step: {step_name}")


def _refresh_autonomous_training_data(system: dict[str, object], *, cycle: int) -> dict[str, Any]:
    data_pipeline: ResponseRiskDataAcquisitionPipeline = system["response_risk_data_pipeline"]  # type: ignore[assignment]
    data_registry = system["curriculum_data_registry"]
    workspace: GlobalWorkspace = system["workspace"]  # type: ignore[assignment]

    snapshot, report = data_pipeline.refresh_snapshot(current_cycle=cycle)
    workspace.update(
        {
            "training_dataset_size": report.get("train_example_count", 0),
            "latest_dataset_snapshot": snapshot.to_dict(),
            "dataset_snapshot_count": len(data_registry.list_snapshots(model_family="response_risk_model")),
        }
    )
    print(
        f"[data_pipeline] snapshot={snapshot.snapshot_id} status={snapshot.status} "
        f"train={report.get('train_example_count', 0)} audit_acc={report.get('audit_accuracy', 0.0):.3f}"
    )
    return report


def _collect_experiment_candidates(system: dict[str, object]) -> list[ExperimentProposal]:
    experiment_scheduler: CurriculumExperimentScheduler = system["experiment_scheduler"]  # type: ignore[assignment]
    experiment_board: ExperimentBoard = system["experiment_board"]  # type: ignore[assignment]
    self_mod_proposer = system["self_mod_proposer"]
    automl_manager = system["automl_manager"]
    registry = system["model_registry"]
    data_pipeline: ResponseRiskDataAcquisitionPipeline = system["response_risk_data_pipeline"]  # type: ignore[assignment]
    quantum_solver = system["quantum_solver"]
    cloud_llm = system["cloud_llm"]

    candidates: list[ExperimentProposal] = []
    candidates.extend(build_self_mod_experiment_candidates(self_mod_proposer, scheduler=experiment_scheduler))
    model_upgrade = build_response_risk_model_upgrade_candidate(
        registry,
        automl_manager=automl_manager,
        scheduler=experiment_scheduler,
        data_pipeline=data_pipeline,
    )
    if model_upgrade is not None:
        candidates.append(model_upgrade)
    candidates.extend(
        build_future_interface_candidates(
            scheduler=experiment_scheduler,
            quantum_solver=quantum_solver,
            cloud_llm=cloud_llm,
        )
    )
    deduped: dict[tuple[str, str], ExperimentProposal] = {}
    for candidate in candidates:
        deduped[(candidate.experiment_kind.value, candidate.source_signature)] = candidate
    return list(deduped.values())


def _route_skill_arena_cases() -> list[SkillArenaCase]:
    return [
        SkillArenaCase(
            case_id="route_curriculum",
            payload=_route_practice_payload(),
            expected={"min_success_rate": 1.0, "max_mean_path_ratio": 1.0, "strategy": "graph_search"},
            description="Waypoint composition should reuse grounded graph-search skills.",
        ),
    ]



def _route_transfer_cases() -> list[TransferCase]:
    lab = SpatialRouteLab()
    return [
        TransferCase(
            case_id="route_transfer_switchback",
            payload={"tasks": [lab.get_task("route_transfer_switchback").to_dict()], "success_threshold": 1.0},
            expected={"min_success_rate": 1.0, "max_mean_path_ratio": 1.0, "strategy": "graph_search", "min_task_count": 1},
            description="Held-out switchback route must generalize.",
        ),
        TransferCase(
            case_id="route_transfer_bridge_loop",
            payload={"tasks": [lab.get_task("route_transfer_bridge_loop").to_dict()], "success_threshold": 1.0},
            expected={"min_success_rate": 1.0, "max_mean_path_ratio": 1.0, "strategy": "graph_search", "min_task_count": 1},
            description="Held-out bridge-loop route must generalize.",
        ),
    ]



def _route_briefing_skill_arena_cases() -> list[SkillArenaCase]:
    lab = RouteBriefingLab()
    return [
        SkillArenaCase(
            case_id="briefing_curriculum",
            payload=_route_briefing_practice_payload(),
            expected={
                "strategy": "graph_search",
                "task_count": 2,
                "reports": [
                    lab.brief_task(lab.get_task("route_train_open_chain"), strategy="graph_search"),
                    lab.brief_task(lab.get_task("route_train_detour_chain"), strategy="graph_search"),
                ],
            },
            description="Mission briefings should summarize waypoint curricula exactly.",
        ),
    ]



def _route_briefing_transfer_cases() -> list[TransferCase]:
    lab = RouteBriefingLab()
    return [
        TransferCase(
            case_id="briefing_transfer",
            payload={
                "tasks": [
                    lab.get_task("route_transfer_switchback").to_dict(),
                    lab.get_task("route_transfer_bridge_loop").to_dict(),
                ]
            },
            expected={
                "strategy": "graph_search",
                "task_count": 2,
                "reports": [
                    lab.brief_task(lab.get_task("route_transfer_switchback"), strategy="graph_search"),
                    lab.brief_task(lab.get_task("route_transfer_bridge_loop"), strategy="graph_search"),
                ],
            },
            description="Held-out mission briefings should generalize from route composition.",
        ),
    ]



def _route_explanation_skill_arena_cases() -> list[SkillArenaCase]:
    return [
        SkillArenaCase(
            case_id="route_explainer_curriculum",
            payload=_route_explanation_practice_payload(),
            expected={
                "min_success_rate": 1.0,
                "max_mean_path_ratio": 1.0,
                "min_detour_explanation_rate": 1.0,
                "strategy": "graph_search",
            },
            description="Detour explanations must be exact on curriculum tasks.",
        ),
    ]



def _route_explanation_transfer_cases() -> list[TransferCase]:
    lab = GroundedNavigationLab()
    return [
        TransferCase(
            case_id="route_explainer_transfer_bridge",
            payload={
                "tasks": [lab.get_task("nav_transfer_bridge").to_dict()],
                "success_threshold": 1.0,
                "detour_explanation_threshold": 1.0,
            },
            expected={
                "min_success_rate": 1.0,
                "max_mean_path_ratio": 1.0,
                "min_detour_explanation_rate": 1.0,
                "strategy": "graph_search",
                "min_task_count": 1,
            },
            description="Held-out bridge detour should remain explainable.",
        ),
        TransferCase(
            case_id="route_explainer_transfer_double_wall",
            payload={
                "tasks": [lab.get_task("nav_transfer_double_wall").to_dict()],
                "success_threshold": 1.0,
                "detour_explanation_threshold": 1.0,
            },
            expected={
                "min_success_rate": 1.0,
                "max_mean_path_ratio": 1.0,
                "min_detour_explanation_rate": 1.0,
                "strategy": "graph_search",
                "min_task_count": 1,
            },
            description="Held-out multi-detour task should remain explainable.",
        ),
    ]



def _infer_pattern(goal: Goal) -> tuple[str | None, str | None]:
    target = str(goal.evidence.get("target_capability", ""))
    if "grounded_self_training" in goal.tags:
        return GROUNDED_NAVIGATION_PATTERN, GROUNDED_NAVIGATION_LESSON
    if "grounded_navigation_patch" in goal.tags:
        return GROUNDED_NAVIGATION_PATTERN, GROUNDED_NAVIGATION_LESSON
    if target == "spatial_route_composition":
        return SPATIAL_ROUTE_PATTERN, SPATIAL_ROUTE_LESSON
    if target == "route_mission_briefing":
        return ROUTE_BRIEFING_PATTERN, ROUTE_BRIEFING_LESSON
    if target == "navigation_route_explanation":
        return ROUTE_EXPLANATION_PATTERN, ROUTE_EXPLANATION_LESSON
    if "grounded_navigation_audit" in goal.tags:
        return GROUNDED_NAVIGATION_PATTERN, GROUNDED_NAVIGATION_LESSON
    return None, None



def _run_navigation_validation(system: dict[str, object]) -> dict[str, Any]:
    agency: AgencyModule = system["agency"]  # type: ignore[assignment]
    capability_portfolio: CapabilityPortfolio = system["capability_portfolio"]  # type: ignore[assignment]
    capability_graph: CapabilityGraph = system["capability_graph"]  # type: ignore[assignment]
    skill_arena: SkillArena = system["skill_arena"]  # type: ignore[assignment]
    transfer_suite: TransferSuite = system["transfer_suite"]  # type: ignore[assignment]
    workspace: GlobalWorkspace = system["workspace"]  # type: ignore[assignment]

    skill_run = skill_arena.run(
        capability="grounded_navigation",
        label="post_patch_navigation",
        cases=_navigation_skill_arena_cases(),
        execute=lambda payload: agency.execute_capability("grounded_navigation", payload, rollout_stage="stable"),
        threshold=0.99,
    )
    capability_portfolio.register_skill_validation(
        capability="grounded_navigation",
        strategy=agency.grounded_navigation_strategy,
        run_id=skill_run.run_id,
        mean_score=skill_run.mean_score,
        passed=skill_run.passed,
        pattern_key=GROUNDED_NAVIGATION_PATTERN,
    )
    transfer_run = transfer_suite.run(
        capability="grounded_navigation",
        label="heldout_navigation_transfer",
        cases=_navigation_transfer_cases(),
        execute=lambda payload: agency.execute_capability("grounded_navigation", payload, rollout_stage="stable"),
        threshold=0.99,
    )
    capability_state = capability_portfolio.register_transfer_validation(
        capability="grounded_navigation",
        strategy=agency.grounded_navigation_strategy,
        run_id=transfer_run.run_id,
        mean_score=transfer_run.mean_score,
        passed=transfer_run.passed,
    )
    capability_graph.sync_from_portfolio(capability_portfolio)
    workspace.update({
        "latest_skill_score": skill_run.mean_score,
        "latest_transfer_score": transfer_run.mean_score,
        "capability_portfolio": capability_portfolio.summary(),
        "capability_graph": capability_graph.summary(),
        "ready_capabilities": [cap for cap in sorted(capability_portfolio.summary()) if capability_portfolio.ready_for_unattended_use(cap)],
    })
    return {"skill_run": skill_run.to_dict(), "transfer_run": transfer_run.to_dict(), "capability_state": capability_state.to_dict()}



def _run_growth_validation(system: dict[str, object], target_capability: str) -> dict[str, Any]:
    agency: AgencyModule = system["agency"]  # type: ignore[assignment]
    capability_portfolio: CapabilityPortfolio = system["capability_portfolio"]  # type: ignore[assignment]
    capability_graph: CapabilityGraph = system["capability_graph"]  # type: ignore[assignment]
    skill_arena: SkillArena = system["skill_arena"]  # type: ignore[assignment]
    transfer_suite: TransferSuite = system["transfer_suite"]  # type: ignore[assignment]
    workspace: GlobalWorkspace = system["workspace"]  # type: ignore[assignment]

    if target_capability == "spatial_route_composition":
        skill_cases = _route_skill_arena_cases()
        transfer_cases = _route_transfer_cases()
    elif target_capability == "route_mission_briefing":
        skill_cases = _route_briefing_skill_arena_cases()
        transfer_cases = _route_briefing_transfer_cases()
    elif target_capability == "navigation_route_explanation":
        skill_cases = _route_explanation_skill_arena_cases()
        transfer_cases = _route_explanation_transfer_cases()
    else:
        raise KeyError(f"Unknown growth target: {target_capability}")

    skill_run = skill_arena.run(
        capability=target_capability,
        label=f"{target_capability}_skill_arena",
        cases=skill_cases,
        execute=lambda payload: agency.execute_capability(target_capability, payload, rollout_stage="stable"),
        threshold=0.99,
    )
    capability_portfolio.register_skill_validation(
        capability=target_capability,
        strategy=agency.grounded_navigation_strategy,
        run_id=skill_run.run_id,
        mean_score=skill_run.mean_score,
        passed=skill_run.passed,
        pattern_key={
            "spatial_route_composition": SPATIAL_ROUTE_PATTERN,
            "route_mission_briefing": ROUTE_BRIEFING_PATTERN,
            "navigation_route_explanation": ROUTE_EXPLANATION_PATTERN,
        }[target_capability],
    )
    transfer_run = transfer_suite.run(
        capability=target_capability,
        label=f"{target_capability}_transfer_suite",
        cases=transfer_cases,
        execute=lambda payload: agency.execute_capability(target_capability, payload, rollout_stage="stable"),
        threshold=0.99,
    )
    capability_state = capability_portfolio.register_transfer_validation(
        capability=target_capability,
        strategy=agency.grounded_navigation_strategy,
        run_id=transfer_run.run_id,
        mean_score=transfer_run.mean_score,
        passed=transfer_run.passed,
    )
    if transfer_run.passed:
        agency.add_capability(target_capability)
    capability_graph.sync_from_portfolio(capability_portfolio)
    workspace.update({
        "capability_portfolio": capability_portfolio.summary(),
        "capability_graph": capability_graph.summary(),
        "ready_capabilities": [cap for cap in sorted(capability_portfolio.summary()) if capability_portfolio.ready_for_unattended_use(cap)],
    })
    return {"skill_run": skill_run.to_dict(), "transfer_run": transfer_run.to_dict(), "capability_state": capability_state.to_dict()}



def _run_and_store_postmortem(goal: Goal, trace: list[dict[str, Any]], outcome: dict[str, Any], system: dict[str, object]) -> dict[str, Any]:
    memory: MemoryModule = system["memory"]  # type: ignore[assignment]
    metacognition: MetacognitionModule = system["metacognition"]  # type: ignore[assignment]
    workspace: GlobalWorkspace = system["workspace"]  # type: ignore[assignment]

    observed = {
        "goal_success": bool(outcome.get("success", False)),
        "anchor_suite_failed": any(item.get("result", {}).get("regression_failure") for item in trace),
        "prediction_error": 0.10 if (not outcome.get("success", False) and "grounded_self_training" in goal.tags) else 0.0,
        "regression_flag": any(item.get("result", {}).get("regression_failure") for item in trace),
        "drift_flag": False,
    }
    report, tasks = metacognition.run_postmortem(goal, trace, expected={}, observed=observed)
    memory.store_postmortem(report)
    workspace.update({
        "latest_postmortem": report.to_dict(),
        "latest_curriculum_task_count": len(tasks),
    })
    return report.to_dict()



def _propose_growth_goal(system: dict[str, object], *, charter: OperatorCharter) -> Goal | None:
    manager: AutonomousGoalManager = system["manager"]  # type: ignore[assignment]
    capability_graph: CapabilityGraph = system["capability_graph"]  # type: ignore[assignment]
    capability_portfolio: CapabilityPortfolio = system["capability_portfolio"]  # type: ignore[assignment]
    growth_planner: CapabilityGrowthPlanner = system["growth_planner"]  # type: ignore[assignment]
    growth_governor: CapabilityGrowthGovernor = system["growth_governor"]  # type: ignore[assignment]
    workspace: GlobalWorkspace = system["workspace"]  # type: ignore[assignment]

    capability_graph.sync_from_portfolio(capability_portfolio)
    plans = growth_planner.propose(
        graph=capability_graph,
        portfolio=capability_portfolio,
        executable_targets=EXECUTABLE_GROWTH_TARGETS,
        horizon=2,
        limit=3,
    )
    if not plans:
        workspace.update({"next_growth_path": None, "growth_candidates": [], "growth_governor_decision": None})
        return None

    selected, assessment, ranked = growth_governor.select(
        plans,
        graph=capability_graph,
        portfolio=capability_portfolio,
        context=workspace.get_state(),
    )
    workspace.update(
        {
            "next_growth_path": selected.to_dict() if selected is not None else None,
            "growth_candidates": [assessment_item.to_dict() for assessment_item in ranked],
            "growth_governor_decision": assessment.to_dict() if assessment is not None else None,
        }
    )

    for assessment_item in ranked:
        path = " -> ".join([assessment_item.root_capability, *assessment_item.path_targets])
        disposition = "admit" if assessment_item.admissible else "defer"
        print(
            f"[governor] {disposition} {path} | utility={assessment_item.expected_utility:.3f} "
            f"risk={assessment_item.estimated_risk:.3f} cost={assessment_item.estimated_cost:.3f} "
            f"confidence={assessment_item.confidence:.3f} composite={assessment_item.composite_score:.3f}"
        )

    if selected is None or assessment is None:
        print("[governor] no admissible growth paths under current utility/risk budget")
        return None

    goal = growth_planner.materialize_next_goal(
        selected,
        tool_payload=_growth_payload(selected.next_step().target_capability),
        overrides=growth_governor.goal_overrides_for(assessment, selected),
    )
    admitted = _admit_with_charter(manager, [goal], charter=charter, context=workspace.get_state())
    if admitted:
        growth_planner.mark_status(selected.plan_id, "admitted", goal_id=admitted[0].goal_id, assessment_id=assessment.assessment_id)
        return admitted[0]

    print(f"[governor] charter blocked growth path: {' -> '.join([assessment.root_capability, *assessment.path_targets])}")
    return None



def run_autonomous(cycles: int, runtime_dir: Path, charter_path: Path | None, interfaces_config_path: Path | None = None, provider_catalog_path: Path | None = None) -> None:
    charter = load_charter(charter_path)
    system = build_system(runtime_dir, interfaces_config_path=interfaces_config_path)

    manager: AutonomousGoalManager = system["manager"]  # type: ignore[assignment]
    memory: MemoryModule = system["memory"]  # type: ignore[assignment]
    episodic_memory: EpisodicMemory = system["episodic_memory"]  # type: ignore[assignment]
    planning: PlanningModule = system["planning"]  # type: ignore[assignment]
    agency: AgencyModule = system["agency"]  # type: ignore[assignment]
    language: LanguageModule = system["language"]  # type: ignore[assignment]
    metacognition: MetacognitionModule = system["metacognition"]  # type: ignore[assignment]
    workspace: GlobalWorkspace = system["workspace"]  # type: ignore[assignment]
    capability_portfolio: CapabilityPortfolio = system["capability_portfolio"]  # type: ignore[assignment]
    capability_graph: CapabilityGraph = system["capability_graph"]  # type: ignore[assignment]
    growth_planner: CapabilityGrowthPlanner = system["growth_planner"]  # type: ignore[assignment]
    growth_governor: CapabilityGrowthGovernor = system["growth_governor"]  # type: ignore[assignment]
    scheduler: LongHorizonScheduler = system["scheduler"]  # type: ignore[assignment]
    experiment_scheduler: CurriculumExperimentScheduler = system["experiment_scheduler"]  # type: ignore[assignment]
    experiment_board: ExperimentBoard = system["experiment_board"]  # type: ignore[assignment]
    self_mod_manager: SafeSelfModificationManager = system["self_mod_manager"]  # type: ignore[assignment]
    self_mod_proposer = system["self_mod_proposer"]
    quantum_solver = system["quantum_solver"]
    cloud_llm = system["cloud_llm"]

    manager.restore_from_ledger()
    workspace.update(
        {
            "operator_charter": charter.to_dict(),
            "goal_queue_size": len(manager.pending),
            "capability_portfolio": capability_portfolio.summary(),
            "capability_graph": capability_graph.summary(),
            "ready_capabilities": [cap for cap in sorted(capability_portfolio.summary()) if capability_portfolio.ready_for_unattended_use(cap)],
            "navigation_strategy": agency.grounded_navigation_strategy,
            "response_planning_policy": planning.response_planning_policy,
            "retrieval_policy": memory.retrieval_policy,
            "next_growth_path": None,
            "growth_governor_decision": None,
            "self_mod_proposal_count": len(self_mod_proposer.list_proposals()),
            "latest_self_mod_proposal": self_mod_proposer.list_proposals()[-1].to_dict() if self_mod_proposer.list_proposals() else None,
            "self_mod_change_count": len(self_mod_manager.list_specs()),
            "latest_self_mod_status": None,
            "experiment_proposal_count": len(experiment_scheduler.list_proposals()),
            "experiment_assessment_count": len(experiment_scheduler.list_assessments()),
            "experiment_campaign_count": len(experiment_board.list_campaigns()),
            "deferred_experiment_count": experiment_board.summary()["deferred_campaign_count"],
            "latest_experiment_decision": None,
            "interface_adapter_summary": {
                "quantum_solver": quantum_solver.summary(),
                "cloud_llm": cloud_llm.summary(),
            },
        }
    )

    rollout_summary = _apply_provider_rollout(system, charter=charter, provider_catalog_path=provider_catalog_path)

    print("Tolik v3.138 — shadow traffic + post-promotion drift guard")
    print(f"Runtime ledger: {runtime_dir / 'ledger'}")
    print(f"Charter: {charter_path or 'DEFAULT'}")
    print(f"Provider catalog: {provider_catalog_path or 'NONE'}")
    decisions = rollout_summary.get("decisions", {})
    if decisions:
        for adapter_name, decision in decisions.items():
            print(
                f"Rollout {adapter_name}: {decision.get('decision')} | provider={decision.get('provider')} "
                f"| mode={decision.get('mode')} | reasons={decision.get('reasons', [])}"
            )
    print()

    patch_seeded = bool(self_mod_proposer.list_proposals()) or agency.grounded_navigation_strategy == "graph_search"
    navigation_validated = capability_portfolio.ready_for_unattended_use("grounded_navigation")

    for cycle in range(1, min(cycles, charter.max_cycles_per_run) + 1):
        workspace.update(
            {
                "current_cycle": cycle,
                "available_capabilities": sorted(agency.list_capabilities()),
                "goal_queue_size": len(manager.pending),
                "episode_count": len(episodic_memory.list_episodes()),
                "scheduled_goal_count": len(scheduler.pending()),
                "capability_portfolio": capability_portfolio.summary(),
                "capability_graph": capability_graph.summary(),
                "ready_capabilities": [cap for cap in sorted(capability_portfolio.summary()) if capability_portfolio.ready_for_unattended_use(cap)],
                "navigation_strategy": agency.grounded_navigation_strategy,
                "response_planning_policy": planning.response_planning_policy,
                "retrieval_policy": memory.retrieval_policy,
                "self_mod_proposal_count": len(self_mod_proposer.list_proposals()),
                "latest_self_mod_proposal": self_mod_proposer.list_proposals()[-1].to_dict() if self_mod_proposer.list_proposals() else None,
                "experiment_proposal_count": len(experiment_scheduler.list_proposals()),
                "experiment_assessment_count": len(experiment_scheduler.list_assessments()),
                "experiment_campaign_count": len(experiment_board.list_campaigns()),
                "deferred_experiment_count": experiment_board.summary()["deferred_campaign_count"],
                "interface_adapter_summary": {
                    "quantum_solver": quantum_solver.summary(),
                    "cloud_llm": cloud_llm.summary(),
                },
            }
        )

        existing_titles = {goal.title for goal in manager.pending + manager.active + manager.all_goals()}
        released = scheduler.release_due(current_cycle=cycle, existing_goal_titles=existing_titles)
        admitted = _admit_with_charter(manager, released, charter=charter, context=workspace.get_state()) if released else []

        if not manager.pending and not manager.active:
            if not navigation_validated:
                if not patch_seeded:
                    admitted.extend(_admit_with_charter(manager, [_make_grounded_training_goal()], charter=charter, context=workspace.get_state()))
                else:
                    admitted.extend(_admit_self_modification_goals_from_memory(system, charter=charter))
            else:
                _refresh_autonomous_training_data(system, cycle=cycle)
                direct_candidates = _collect_experiment_candidates(system)
                deferred_candidates = experiment_board.release_due_candidates(
                    current_cycle=cycle,
                    context=workspace.get_state(),
                    charter=charter,
                )
                experiment_candidate_map: dict[tuple[str, str], ExperimentProposal] = {}
                for candidate in [*deferred_candidates, *direct_candidates]:
                    experiment_candidate_map[(candidate.experiment_kind.value, candidate.source_signature)] = candidate
                experiment_candidates = list(experiment_candidate_map.values())
                selected_experiment, assessment, ranked_experiments = experiment_scheduler.select(
                    experiment_candidates,
                    context=workspace.get_state(),
                    current_cycle=cycle,
                    charter=charter,
                )
                experiment_board.refresh_candidate_pool(
                    experiment_candidates,
                    ranked_experiments,
                    current_cycle=cycle,
                    selected_proposal_id=selected_experiment.proposal_id if selected_experiment is not None else None,
                    context=workspace.get_state(),
                    charter=charter,
                )
                workspace.update({
                    "experiment_proposal_count": len(experiment_scheduler.list_proposals()),
                    "experiment_assessment_count": len(experiment_scheduler.list_assessments()),
                    "experiment_campaign_count": len(experiment_board.list_campaigns()),
                    "deferred_experiment_count": experiment_board.summary()["deferred_campaign_count"],
                    "latest_experiment_decision": assessment.to_dict() if assessment is not None else None,
                })
                for experiment_assessment in ranked_experiments:
                    kind = experiment_assessment.experiment_kind.value
                    disposition = "admit" if experiment_assessment.admissible else "defer"
                    print(
                        f"[experiment_governor] {disposition} {kind}:{experiment_assessment.source_signature} "
                        f"| utility={experiment_assessment.expected_utility:.3f} curriculum={experiment_assessment.curriculum_alignment:.3f} "
                        f"risk={experiment_assessment.estimated_risk:.3f} cost={experiment_assessment.estimated_cost:.3f} "
                        f"composite={experiment_assessment.composite_score:.3f}"
                    )
                if selected_experiment is not None and assessment is not None:
                    campaign, ticket = experiment_board.stage_selected_execution(
                        selected_experiment,
                        current_cycle=cycle,
                        context=workspace.get_state(),
                        charter=charter,
                    )
                    workspace.update({
                        "experiment_campaign_count": len(experiment_board.list_campaigns()),
                        "deferred_experiment_count": experiment_board.summary()["deferred_campaign_count"],
                    })
                    if ticket.get("materialize_now"):
                        experiment_goal = _materialize_experiment_goal(system, selected_experiment)
                        admitted_goal = _admit_with_charter(manager, [experiment_goal], charter=charter, context=workspace.get_state())
                        if admitted_goal:
                            experiment_scheduler.record_materialized(selected_experiment.proposal_id, current_cycle=cycle, goal_id=admitted_goal[0].goal_id)
                            experiment_board.record_materialized(selected_experiment.proposal_id, goal_id=admitted_goal[0].goal_id)
                            admitted.append(admitted_goal[0])
                    else:
                        print(
                            f"[experiment_board] deferred {campaign.source_signature} "
                            f"until_cycle={campaign.defer_until_cycle} reason={campaign.defer_reason}"
                        )
                        growth_goal = _propose_growth_goal(system, charter=charter)
                        if growth_goal is not None:
                            admitted.append(growth_goal)
                else:
                    growth_goal = _propose_growth_goal(system, charter=charter)
                    if growth_goal is not None:
                        admitted.append(growth_goal)

        print(f"[cycle {cycle}] admitted: {[goal.title for goal in admitted]}")

        goal = manager.select_next_goal(context=workspace.get_state())
        if goal is None:
            print(f"[cycle {cycle}] no executable goals left.\n")
            continue

        growth_plan_id = goal.evidence.get("growth_plan_id")
        if isinstance(growth_plan_id, str) and growth_planner.get(growth_plan_id) is not None:
            growth_planner.mark_status(growth_plan_id, "running", goal_id=goal.goal_id)

        print(f"[cycle {cycle}] active goal: {goal.title} ({goal.source.value})")
        plan = planning.make_plan(goal, world_state=workspace.get_state())
        print(f"[cycle {cycle}] plan: {[step.name for step in plan.steps]}")

        trace: list[dict[str, Any]] = []
        latest_response: str | None = None
        for step in plan.steps:
            if step.name in {"stage_self_mod_candidate", "run_self_mod_regression_gate", "promote_self_mod_canary", "evaluate_self_mod_canary", "finalize_self_mod"}:
                result = _run_self_modification_step(step.name, goal, system)
            elif step.name in {"stage_model_candidate", "train_model_candidate", "run_model_regression_gate", "promote_model_canary", "evaluate_model_canary", "finalize_model"}:
                result = _run_automl_step(step.name, goal, system)
            elif step.name in {"probe_quantum_interface", "probe_cloud_llm_interface"}:
                result = _run_interface_probe_step(step.name, goal, system)
            else:
                result = agency.execute(step.name, goal=goal, workspace_state=workspace.get_state())
            if step.name == "form_response":
                latest_response = language.generate_response({"goal": goal.title, "state": workspace.get_state()})
                result["response"] = latest_response
            manager.update_progress(goal.goal_id, {"step": step.name, "result": result})
            metacognition.log_event("agency", step.name, result)
            memory.add_experience({"goal_id": goal.goal_id, "step": step.name, **result})
            trace.append({"step": step.name, "result": result})
            print(f"  - {step.name}: {result}")

        success = all(item["result"].get("success", True) for item in trace)
        outcome = {"success": success}
        manager.complete_goal(goal.goal_id, outcome)
        experiment_proposal_id = goal.evidence.get("experiment_proposal_id")
        if isinstance(experiment_proposal_id, str):
            persisted = experiment_scheduler.record_outcome(experiment_proposal_id, success=success, current_cycle=cycle, note=goal.title)
            experiment_board.record_outcome(
                experiment_proposal_id,
                success=success,
                current_cycle=cycle,
                note=goal.title,
                cooldown_until_cycle=persisted.cooldown_until_cycle,
            )
            workspace.update({
                "experiment_proposal_count": len(experiment_scheduler.list_proposals()),
                "experiment_assessment_count": len(experiment_scheduler.list_assessments()),
                "experiment_campaign_count": len(experiment_board.list_campaigns()),
                "deferred_experiment_count": experiment_board.summary()["deferred_campaign_count"],
            })
        pattern_key, lesson = _infer_pattern(goal)
        episode = episodic_memory.record_goal_episode(
            goal,
            cycle=cycle,
            trace=trace,
            outcome=outcome,
            workspace_excerpt={
                "navigation_strategy": agency.grounded_navigation_strategy,
                "response_planning_policy": planning.response_planning_policy,
                "retrieval_policy": memory.retrieval_policy,
            },
            pattern_key=pattern_key,
            lesson=lesson,
            tags=list(goal.tags),
        )
        workspace.update({"episode_count": len(episodic_memory.list_episodes()), "latest_pattern_key": episode.pattern_key})
        postmortem = _run_and_store_postmortem(goal, trace, outcome, system)
        insights = metacognition.analyze()
        print(f"[cycle {cycle}] meta: {insights}")
        if latest_response:
            print(f"[cycle {cycle}] language response: {latest_response}")
        print()

        if "grounded_self_training" in goal.tags and not success and not patch_seeded:
            self_mod_admitted = _admit_self_modification_goals_from_memory(system, charter=charter)
            if self_mod_admitted:
                print(f"[cycle {cycle}] experience proposed self-mod goal: {self_mod_admitted[0].title}")
                patch_seeded = True

        if "grounded_navigation_patch" in goal.tags and success:
            scheduled_audit = _admit_with_charter(manager, [_make_navigation_audit_goal()], charter=charter, context=workspace.get_state())
            if scheduled_audit:
                print(f"[cycle {cycle}] scheduled audit: {scheduled_audit[0].title}")

        if "grounded_navigation_audit" in goal.tags and success and not navigation_validated:
            validation = _run_navigation_validation(system)
            print(f"[cycle {cycle}] skill arena: {validation['skill_run']}")
            print(f"[cycle {cycle}] transfer suite: {validation['transfer_run']}")
            print(f"[cycle {cycle}] capability state: {validation['capability_state']}")
            print(f"[cycle {cycle}] capability graph unlocked next skills from grounded_navigation")
            print("Ready capabilities: ['grounded_navigation']")
            navigation_validated = True
            growth_goal = _propose_growth_goal(system, charter=charter)
            if growth_goal is not None:
                selected = workspace.get("next_growth_path")
                path_str = " -> ".join([selected.get("root_capability"), *selected.get("path_targets", [])]) if isinstance(selected, dict) else growth_goal.title
                decision = workspace.get("growth_governor_decision")
                if isinstance(decision, dict):
                    utility = float(decision.get("expected_utility", 0.0))
                    risk = float(decision.get("estimated_risk", 0.0))
                    cost = float(decision.get("estimated_cost", 0.0))
                    composite = float(decision.get("composite_score", 0.0))
                    print(
                        f"[cycle {cycle}] growth governor selected path: {path_str} "
                        f"| utility={utility:.3f} risk={risk:.3f} cost={cost:.3f} composite={composite:.3f}"
                    )
                print(f"[cycle {cycle}] growth planner selected path: {path_str}")

        if goal.evidence.get("curriculum_type") == "capability_growth":
            target_capability = str(goal.evidence.get("target_capability", ""))
            plan_id = str(goal.evidence.get("growth_plan_id", ""))
            validation = _run_growth_validation(system, target_capability)
            print(f"[cycle {cycle}] {target_capability} skill arena: {validation['skill_run']}")
            print(f"[cycle {cycle}] {target_capability} transfer suite: {validation['transfer_run']}")
            if target_capability == "navigation_route_explanation":
                print(f"[cycle {cycle}] route explanation transfer suite: {validation['transfer_run']}")
            print(f"[cycle {cycle}] {target_capability} capability state: {validation['capability_state']}")
            passed = bool(validation["transfer_run"]["passed"])
            score = float(validation["transfer_run"]["mean_score"])
            growth_plan = growth_planner.mark_step_completed(plan_id, capability=target_capability, passed=passed, run_id=str(validation["transfer_run"]["run_id"]), score=score)
            capability_graph.sync_from_portfolio(capability_portfolio)
            workspace.update({"next_growth_path": growth_plan.to_dict(), "capability_graph": capability_graph.summary()})
            if passed and growth_plan.status == "running":
                follow_on_assessment = growth_governor.assess_candidates([growth_plan], graph=capability_graph, portfolio=capability_portfolio, context=workspace.get_state())[0]
                workspace.update({"growth_governor_decision": follow_on_assessment.to_dict()})
                follow_on_goal = growth_planner.materialize_next_goal(
                    growth_plan,
                    tool_payload=_growth_payload(growth_plan.next_step().target_capability),
                    overrides=growth_governor.goal_overrides_for(follow_on_assessment, growth_plan),
                )
                scheduled = scheduler.schedule(
                    follow_on_goal,
                    due_cycle=cycle + 1,
                    reason=f"growth_follow_on:{growth_plan.plan_id}",
                )
                print(f"[cycle {cycle}] scheduled follow-on growth goal: {scheduled.goal.title}")
            elif growth_plan.status == "completed":
                print(f"[cycle {cycle}] completed growth path: {' -> '.join([growth_plan.root_capability, *growth_plan.path_targets])}")

    print("Autonomous AGI-path run finished.")
    print(f"Active grounded strategy: {agency.grounded_navigation_strategy}")
    print(f"Episodes stored: {len(episodic_memory.list_episodes())}")
    print(f"Capability portfolio: {capability_portfolio.summary()}")
    print(f"Capability graph: {capability_graph.summary()}")
    print(f"Ready capabilities: {[cap for cap in sorted(capability_portfolio.summary()) if capability_portfolio.ready_for_unattended_use(cap)]}")
    print(f"Growth plans: {[plan.to_dict() for plan in growth_planner.list_plans()]}")
    print(f"Growth assessments: {[assessment.to_dict() for assessment in growth_governor.list_assessments()]}")
    print(f"Scheduled goals: {[item.to_dict() for item in scheduler.list_all()]}")
    print(f"Experiment proposals: {[proposal.to_dict() for proposal in experiment_scheduler.list_proposals()]}")
    print(f"Experiment assessments: {[assessment.to_dict() for assessment in experiment_scheduler.list_assessments()]}")
    print(f"Experiment campaigns: {[campaign.to_dict() for campaign in experiment_board.list_campaigns()]}")
    print(f"Experiment cycle budgets: {[budget.to_dict() for budget in experiment_board.list_cycle_budgets()]}")
    print(f"Self-mod proposals: {[proposal.to_dict() for proposal in self_mod_proposer.list_proposals()]}")
    print(f"Self modifications: {[spec.to_dict() for spec in self_mod_manager.list_specs()]}")




def _capability_goal_payload(capability_id: str, scenario_id: str) -> dict[str, Any]:
    nav_lab = GroundedNavigationLab()
    if capability_id == "grounded_navigation":
        if scenario_id == "anchor_easy":
            return {"tasks": [nav_lab.get_task("nav_easy_open").to_dict()], "success_threshold": 1.0}
        if scenario_id == "anchor_detour":
            return {"tasks": [nav_lab.get_task("nav_detour_wall").to_dict()], "success_threshold": 1.0}
        if scenario_id == "heldout_transfer":
            return {"tasks": [nav_lab.get_task("nav_transfer_bridge").to_dict()], "success_threshold": 1.0}
    return {}


def _has_open_user_goal(manager: AutonomousGoalManager) -> bool:
    for goal in manager.all_goals():
        if goal.source == GoalSource.USER and goal.status.value in {"pending", "active"}:
            return True
    return False


def run_capability_goal_loop_once(system: dict[str, object], *, current_cycle: int = 0, background_mode: bool = False) -> dict[str, Any]:
    workspace: GlobalWorkspace = system["workspace"]  # type: ignore[assignment]
    portfolio: CapabilityPortfolio = system["capability_portfolio"]  # type: ignore[assignment]
    manager: AutonomousGoalManager = system["manager"]  # type: ignore[assignment]
    scheduler: InternalGoalScheduler = system["internal_goal_scheduler"]  # type: ignore[assignment]
    failure_analyzer: FailureAnalyzer = system["failure_analyzer"]  # type: ignore[assignment]
    improvement_ledger: ImprovementLedger = system["improvement_ledger"]  # type: ignore[assignment]
    tool_guard: ToolProposalGuard = system["tool_proposal_guard"]  # type: ignore[assignment]
    deferred_quantum = system["deferred_quantum"]
    deferred_cloud_llm = system["deferred_cloud_llm"]
    agency: AgencyModule = system["agency"]  # type: ignore[assignment]

    previous_snapshot = portfolio.snapshot()
    gaps = portfolio.identify_gaps(current_step=current_cycle)
    workspace.update({"capability_gaps": [gap.to_dict() for gap in gaps]})
    if not gaps:
        workspace.update({"latest_internal_goal": None})
        return {"selected_goal": None, "gaps": []}
    if _has_open_user_goal(manager) and not background_mode:
        return {"selected_goal": None, "reason": "user_goal_preempts", "gaps": [gap.to_dict() for gap in gaps]}
    if not agency.can_run_internal_goal_now(workspace.get_state()) and not background_mode:
        return {"selected_goal": None, "reason": "agency_busy", "gaps": [gap.to_dict() for gap in gaps]}

    selected_gap = scheduler.select_gap(gaps)
    if selected_gap is None:
        return {"selected_goal": None, "gaps": [gap.to_dict() for gap in gaps]}

    state = portfolio.get(selected_gap.capability_id)
    episodes_to_stable = state.episodes_to_stable if state else None
    priority = scheduler.score_gap(selected_gap, episodes_to_stable=episodes_to_stable, missing_tool=(selected_gap.gap_type == "insufficient_evidence"))
    internal_goal = scheduler.materialize_goal(selected_gap, priority=priority)
    improvement_ledger.save_goal(internal_goal)
    portfolio.apply_cooldown(selected_gap.capability_id, current_cycle=current_cycle)

    runtime_goal = Goal(
        goal_id=internal_goal.goal_id,
        title=f"Improve {selected_gap.capability_id} via {internal_goal.goal_type}",
        description=f"Autonomous improvement goal for {selected_gap.capability_id} ({selected_gap.gap_type}).",
        source=GoalSource.METACOGNITION,
        kind=GoalKind.LEARNING if internal_goal.goal_type != "maintenance_eval" else GoalKind.MAINTENANCE,
        expected_gain=min(max(priority, 0.05), 1.0),
        novelty=0.20,
        uncertainty_reduction=selected_gap.severity,
        strategic_fit=0.82,
        risk_estimate=0.08,
        priority=min(max(priority, 0.05), 1.0),
        risk_budget=0.30,
        resource_budget=GoalBudget(max_steps=3, max_seconds=15.0, max_tool_calls=0, max_api_calls=0),
        success_criteria=[SuccessCriterion(metric=internal_goal.target_metric, comparator=">=", target=internal_goal.target_value)],
        required_capabilities=["classical_planning"],
        tags=["capability_gap", selected_gap.gap_type, selected_gap.capability_id],
        evidence={"capability_id": selected_gap.capability_id, "gap_type": selected_gap.gap_type},
    )

    admitted = manager.admit_candidates([runtime_goal], workspace.get_state())
    active_goal = runtime_goal
    if admitted:
        maybe_active = manager.select_next_goal(workspace.get_state())
        if maybe_active is not None:
            active_goal = maybe_active
    signature = failure_analyzer.analyze_gap(selected_gap, workspace.get_state().get("latest_reports", []))
    curriculum = failure_analyzer.build_curriculum(internal_goal, signature)
    for item in curriculum:
        item.payload = _capability_goal_payload(selected_gap.capability_id, item.scenario_id)
    improvement_ledger.save_curriculum_items(curriculum)

    remaining_steps = internal_goal.budget_steps
    trace: list[dict[str, Any]] = []
    confidence_scores: list[float] = []
    transfer_scores: list[float] = []
    for item in curriculum:
        if not agency.respect_self_improvement_budget(remaining_steps):
            break
        result = agency.execute_curriculum_item(item, workspace.get_state())
        trace.append({"item": item.to_dict(), "result": result})
        output = result.get("output", {}) if isinstance(result, dict) else {}
        if item.benchmark_name == "transfer_suite":
            transfer_scores.append(float(output.get("confidence", output.get("success_rate", 0.0)) or 0.0))
        else:
            confidence_scores.append(float(output.get("confidence", output.get("success_rate", 0.0)) or 0.0))
        remaining_steps -= 1

    outcome = {
        "capability_id": selected_gap.capability_id,
        "confidence": round(sum(confidence_scores) / len(confidence_scores), 3) if confidence_scores else None,
        "transfer_score": round(sum(transfer_scores) / len(transfer_scores), 3) if transfer_scores else None,
        "episodes_to_stable": 1 if confidence_scores else None,
        "reset_cooldown": False,
    }
    portfolio.mark_improvement_result(internal_goal.goal_id, outcome)

    if signature.needs_tool_proposal:
        proposal = ToolProposal(
            capability_id=selected_gap.capability_id,
            problem_signature=signature.reason,
            interface_spec={"capability": selected_gap.capability_id, "action": "tool_stub"},
            test_spec=[{"scenario": item.scenario_id} for item in curriculum],
        )
        review = tool_guard.review(proposal)
        improvement_ledger.save_tool_proposal({**proposal.to_dict(), "review": {"allowed": review.allowed, "status": review.status, "reasons": review.reasons}})

    if signature.needs_external_backend:
        if "quantum" in signature.reason:
            task = DeferredQuantumTask(capability_id=selected_gap.capability_id, payload={"reason": signature.reason}).to_dict()
        else:
            task = DeferredLLMTask(capability_id=selected_gap.capability_id, prompt=f"Deferred external help for {selected_gap.capability_id}", context={"reason": signature.reason}).to_dict()
        improvement_ledger.save_deferred_task(task)

    workspace.update({
        "latest_internal_goal": runtime_goal.to_dict(),
        "latest_curriculum": [item.to_dict() for item in curriculum],
        "latest_reports": [result["result"]["output"] for result in trace if isinstance(result.get("result"), dict) and isinstance(result["result"].get("output"), dict)],
        "capability_portfolio": portfolio.summary(),
    })

    if any(not step["result"].get("success", False) for step in trace):
        manager.complete_goal(active_goal.goal_id, {"success": False, "goal_success": False})
    else:
        manager.complete_goal(active_goal.goal_id, {"success": True, "goal_success": True})

    return {
        "selected_goal": runtime_goal.to_dict(),
        "gap": selected_gap.to_dict(),
        "curriculum": [item.to_dict() for item in curriculum],
        "trace": trace,
        "portfolio": portfolio.summary(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Tolik v3.138 AGI-path with provider qualification, shadow traffic, drift monitoring, safe external adapters, and experiment-board continuity.")
    parser.add_argument("--cycles", type=int, default=8, help="Maximum autonomous cycles.")
    parser.add_argument("--runtime-dir", type=Path, default=Path("runtime_agi"), help="Ledger directory for autonomous runs.")
    parser.add_argument("--charter", type=Path, default=Path("configs/operator_charter.example.json"), help="Path to operator charter JSON.")
    parser.add_argument("--interfaces-config", type=Path, default=None, help="Optional JSON config for safe external adapter modes/providers.")
    parser.add_argument("--provider-catalog", type=Path, default=None, help="Optional JSON config describing candidate providers for qualification and rollout.")
    args = parser.parse_args()
    run_autonomous(cycles=args.cycles, runtime_dir=args.runtime_dir, charter_path=args.charter, interfaces_config_path=args.interfaces_config, provider_catalog_path=args.provider_catalog)


if __name__ == "__main__":
    main()
