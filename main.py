from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from agency.agency_module import AgencyModule
from automl.model_registry import ModelRegistry
from automl.response_risk_model import ResponseRiskModel
from automl.safe_automl_manager import SafeAutoMLManager
from automl.training_data_registry import CurriculumDataRegistry
from automl.response_risk_data_pipeline import ResponseRiskDataAcquisitionPipeline
from benchmarks.skill_arena import SkillArena, SkillArenaCase
from benchmarks.transfer_suite import TransferSuite
from core.global_workspace import GlobalWorkspace
from experiments.curriculum_experiment_scheduler import CurriculumExperimentScheduler
from experiments.experiment_board import ExperimentBoard
from interfaces.interface_loader import load_interface_runtime
from interfaces.provider_qualification import CharterAwareRolloutGate, ProviderQualificationManager
from language.language_module import LanguageModule
from memory.episodic_memory import EpisodicMemory
from memory.capability_graph import CapabilityGraph
from memory.capability_portfolio import CapabilityPortfolio
from memory.goal_ledger import GoalLedger
from memory.improvement_ledger import ImprovementLedger
from memory.memory_module import MemoryModule
from memory.semantic_promoter import SemanticPromoter
from memory.strategy_memory import StrategyMemory
from metacognition.curriculum_builder import curriculum_task_to_goal
from metacognition.transfer_curriculum import TransferCurriculumBuilder
from metacognition.metacognition_module import MetacognitionModule
from metacognition.failure_analyzer import FailureAnalyzer
from motivation.autonomous_goal_manager import AutonomousGoalManager
from motivation.goal_arbitrator import GoalArbitrator
from motivation.goal_schema import Goal, GoalBudget, GoalKind, GoalSource, SuccessCriterion, new_goal_id
from planning.capability_growth_planner import CapabilityGrowthPlanner
from planning.capability_growth_governor import CapabilityGrowthGovernor
from planning.long_horizon_scheduler import LongHorizonScheduler
from planning.internal_goal_scheduler import InternalGoalScheduler
from planning.planning_module import PlanningModule
from reasoning.reasoning_module import ReasoningModule
from self_modification.evaluation_harness import default_self_mod_executors
from self_modification.experience_self_mod_proposer import ExperienceSelfModificationProposer
from self_modification.safe_self_mod_manager import SafeSelfModificationManager
from tooling.tooling_manager import ToolingManager
from policy.tool_proposal_guard import ToolProposalGuard
from integrations.quantum_solver import QuantumSolver
from integrations.cloud_llm_client import CloudLLMClient


RESEARCH_NOTES = [
    "The prototype reached stable behavior after policy tuning and produced shorter plans than baseline.",
    "Noise handling improved once the controller stopped retrying the same failing path and switched strategy.",
    "The next iteration should compress reports into a concise summary for quick operator review.",
]

CANARY_EDGE_CASE_NOTES = [
    "The prototype reached stable behavior after policy tuning and produced shorter plans than baseline.",
    "   ",
    "Noise handling improved once the controller stopped retrying the same failing path and switched strategy.",
    "The next iteration should compress reports into a concise summary for quick operator review.",
]

FOLLOWUP_NOTES = [
    "Fallback remained stable after the canary rollback.",
    "Operators kept receiving concise summaries from the trusted stable tool.",
    "The next candidate should broaden benchmark coverage for blank inputs.",
]

PATCH_NOTES = [
    "Failure mining converted the live rollback into a structured regression case.",
    "The benchmark suite now includes the blank-input canary scenario.",
    "The patched summarizer respects runtime sentence limits and ignores blank inputs.",
]


BLANK_INPUT_GUARD_PATTERN = "text_summarizer|blank_input_guard"
BLANK_INPUT_GUARD_LESSON = "Ignore blank inputs and enforce runtime sentence limits for text_summarizer canaries."


def build_system(runtime_dir: Path, interfaces_config_path: Path | None = None) -> dict[str, object]:
    ledger = GoalLedger(runtime_dir / "ledger")
    manager = AutonomousGoalManager(ledger=ledger, arbitrator=GoalArbitrator())
    strategy_memory = StrategyMemory(ledger=ledger)
    improvement_ledger = ImprovementLedger(goal_ledger=ledger)
    memory = MemoryModule(goal_ledger=ledger, strategy_memory=strategy_memory)
    episodic_memory = EpisodicMemory(ledger=ledger)
    reasoning = ReasoningModule()
    semantic_promoter = SemanticPromoter(reasoning=reasoning, ledger=ledger, min_support=2)
    capability_portfolio = CapabilityPortfolio(ledger=ledger)
    capability_graph = CapabilityGraph(ledger=ledger)
    growth_planner = CapabilityGrowthPlanner(ledger=ledger)
    growth_governor = CapabilityGrowthGovernor(ledger=ledger)
    model_registry = ModelRegistry(ledger=ledger)
    if not model_registry.has_family("response_risk_model"):
        baseline_model = ResponseRiskModel.baseline()
        model_registry.register_stable(baseline_model)
        ledger.append_event({
            "event_type": "model_stable_registered",
            "model_family": baseline_model.family,
            "model_id": baseline_model.model_id,
            "version": baseline_model.version,
        })
    planning = PlanningModule(response_risk_model=model_registry.get_active_model("response_risk_model"))
    tooling = ToolingManager(ledger=ledger, strategy_memory=strategy_memory)
    agency = AgencyModule(tool_registry=tooling.registry)
    language = LanguageModule()
    metacognition = MetacognitionModule()
    failure_analyzer = FailureAnalyzer()
    workspace = GlobalWorkspace()
    skill_arena = SkillArena(ledger=ledger)
    transfer_suite = TransferSuite(ledger=ledger)
    self_mod_components = {"agency": agency, "planning": planning, "memory": memory}
    self_mod_manager = SafeSelfModificationManager(ledger=ledger, components=self_mod_components, skill_arena=skill_arena, transfer_suite=transfer_suite, executors=default_self_mod_executors())
    self_mod_proposer = ExperienceSelfModificationProposer(ledger=ledger, episodic_memory=episodic_memory, components=self_mod_components)
    automl_manager = SafeAutoMLManager(ledger=ledger, registry=model_registry, components={"planning": planning}, skill_arena=skill_arena, transfer_suite=transfer_suite)
    curriculum_data_registry = CurriculumDataRegistry(ledger=ledger)
    response_risk_data_pipeline = ResponseRiskDataAcquisitionPipeline(ledger=ledger, registry=model_registry, episodic_memory=episodic_memory, data_registry=curriculum_data_registry)
    snapshot, data_report = response_risk_data_pipeline.refresh_snapshot(current_cycle=0)
    scheduler = LongHorizonScheduler(ledger=ledger)
    internal_goal_scheduler = InternalGoalScheduler()
    experiment_scheduler = CurriculumExperimentScheduler(ledger=ledger)
    experiment_board = ExperimentBoard(ledger=ledger)
    transfer_curriculum = TransferCurriculumBuilder()
    quantum_solver, cloud_llm, interface_summaries = load_interface_runtime(interfaces_config_path, ledger=ledger)
    provider_qualification_manager = ProviderQualificationManager(ledger=ledger)
    rollout_gate = CharterAwareRolloutGate(ledger=ledger)
    tool_proposal_guard = ToolProposalGuard()
    deferred_quantum = QuantumSolver(available=False)
    deferred_cloud_llm = CloudLLMClient(available=False)

    workspace.update(
        {
            "novelty_score": 0.20,
            "risk_estimate": 0.08,
            "queue_pressure": 0,
            "retrieval_confidence": 0.95,
            "ood_score": 0.15,
            "regression_score": 0.98,
            "anchor_suite_failed": False,
            "knowledge_source_available": True,
            "available_capabilities": sorted(agency.list_capabilities()),
            "goal_queue_size": 0,
            "notes_batch": list(RESEARCH_NOTES),
            "canary_notes_batch": list(CANARY_EDGE_CASE_NOTES),
            "followup_notes_batch": list(FOLLOWUP_NOTES),
            "patch_notes_batch": list(PATCH_NOTES),
            "future_interfaces": {
                "quantum_solver": quantum_solver.mode,
                "cloud_llm": cloud_llm.mode,
            },
            "interface_adapter_summary": interface_summaries,
            "provider_qualification_summary": {},
            "interface_rollout_decisions": {},
            "interface_canary_rollout": {},
            "post_promotion_monitoring": {},
            "interface_drift_summary": {},
            "interface_demotion_decisions": [],
            "episode_count": len(episodic_memory.list_episodes()),
            "latest_skill_score": None,
            "latest_transfer_score": None,
            "latest_pattern_key": None,
            "capability_portfolio": capability_portfolio.summary(),
            "scheduled_goal_count": len(scheduler.pending()),
            "capability_graph": capability_graph.summary(),
            "next_growth_path": None,
            "growth_governor_decision": None,
            "self_mod_proposal_count": 0,
            "latest_self_mod_proposal": None,
            "self_mod_change_count": 0,
            "latest_self_mod_status": None,
            "response_planning_policy": planning.response_planning_policy,
            "retrieval_policy": memory.retrieval_policy,
            "model_registry_families": ["response_risk_model"],
            "stable_models": model_registry.stable_model_names(),
            "latest_model_status": None,
            "experiment_proposal_count": 0,
            "experiment_assessment_count": 0,
            "experiment_campaign_count": len(experiment_board.list_campaigns()),
            "deferred_experiment_count": experiment_board.summary()["deferred_campaign_count"],
            "latest_experiment_decision": None,
            "training_dataset_size": data_report.get("train_example_count", 0),
            "latest_dataset_snapshot": snapshot.to_dict(),
            "dataset_snapshot_count": len(curriculum_data_registry.list_snapshots(model_family="response_risk_model")),
            "active_user_goal": False,
            "capability_goal_loop_enabled": True,
            "latest_internal_goal": None,
        }
    )

    return {
        "ledger": ledger,
        "manager": manager,
        "memory": memory,
        "episodic_memory": episodic_memory,
        "semantic_promoter": semantic_promoter,
        "capability_portfolio": capability_portfolio,
        "capability_graph": capability_graph,
        "strategy_memory": strategy_memory,
        "improvement_ledger": improvement_ledger,
        "reasoning": reasoning,
        "planning": planning,
        "model_registry": model_registry,
        "automl_manager": automl_manager,
        "curriculum_data_registry": curriculum_data_registry,
        "response_risk_data_pipeline": response_risk_data_pipeline,
        "tooling": tooling,
        "agency": agency,
        "language": language,
        "metacognition": metacognition,
        "failure_analyzer": failure_analyzer,
        "workspace": workspace,
        "skill_arena": skill_arena,
        "transfer_suite": transfer_suite,
        "self_mod_manager": self_mod_manager,
        "self_mod_proposer": self_mod_proposer,
        "scheduler": scheduler,
        "internal_goal_scheduler": internal_goal_scheduler,
        "experiment_scheduler": experiment_scheduler,
        "experiment_board": experiment_board,
        "transfer_curriculum": transfer_curriculum,
        "growth_planner": growth_planner,
        "growth_governor": growth_governor,
        "quantum_solver": quantum_solver,
        "cloud_llm": cloud_llm,
        "interface_adapter_summary": interface_summaries,
        "provider_qualification_manager": provider_qualification_manager,
        "rollout_gate": rollout_gate,
        "tool_proposal_guard": tool_proposal_guard,
        "deferred_quantum": deferred_quantum,
        "deferred_cloud_llm": deferred_cloud_llm,
    }


def seed_demo_goals(system: dict[str, object]) -> None:
    manager: AutonomousGoalManager = system["manager"]  # type: ignore[assignment]
    memory: MemoryModule = system["memory"]  # type: ignore[assignment]
    workspace: GlobalWorkspace = system["workspace"]  # type: ignore[assignment]

    if manager.all_goals():
        return

    memory.store_fact("France", {"fact": "Paris"})
    user_goal = manager.ingest_external_goal(
        "Summarize the current research notes",
        required_capabilities=["classical_planning", "text_summarizer"],
        tags=["user", "summarize", "notes"],
        evidence={
            "raw_text": "Summarize the current research notes",
            "response_mode": "summary",
            "tool_payload": {
                "texts": list(workspace.get("notes_batch", [])),
                "max_sentences": 2,
            },
        },
    )
    manager.pending = [goal for goal in manager.pending if goal.goal_id != user_goal.goal_id]
    manager.admit_candidates([user_goal], context=workspace.get_state())


def _make_tool_goal(
    *,
    capability: str,
    title: str,
    description: str,
    source: GoalSource,
    expected_gain: float,
    novelty: float,
    uncertainty_reduction: float,
    strategic_fit: float,
    risk_estimate: float,
    priority: float,
    tags: list[str],
    evidence: dict[str, Any],
) -> Goal:
    return Goal(
        goal_id=new_goal_id("tool"),
        title=title,
        description=description,
        source=source,
        kind=GoalKind.TOOL_CREATION,
        expected_gain=expected_gain,
        novelty=novelty,
        uncertainty_reduction=uncertainty_reduction,
        strategic_fit=strategic_fit,
        risk_estimate=risk_estimate,
        priority=priority,
        risk_budget=0.20,
        resource_budget=GoalBudget(max_steps=8, max_seconds=25.0, max_tool_calls=1, max_api_calls=0),
        success_criteria=[SuccessCriterion(metric="tool_promoted", comparator="==", target=True)],
        required_capabilities=["classical_planning"],
        tags=tags,
        evidence={"target_capability": capability, **evidence},
    )



def _make_scheduled_audit_goal(*, capability: str, due_cycle: int) -> Goal:
    return Goal(
        goal_id=new_goal_id("sched"),
        title=f"Audit {capability} after semantic promotion",
        description=f"Scheduled long-horizon audit for {capability} after semantic consolidation.",
        source=GoalSource.SCHEDULER,
        kind=GoalKind.MAINTENANCE,
        expected_gain=0.71,
        novelty=0.18,
        uncertainty_reduction=0.82,
        strategic_fit=0.91,
        risk_estimate=0.05,
        priority=0.73,
        risk_budget=0.12,
        resource_budget=GoalBudget(max_steps=4, max_seconds=12.0, max_tool_calls=0, max_api_calls=0),
        success_criteria=[SuccessCriterion(metric="status", comparator="==", target="done")],
        required_capabilities=["classical_planning", capability],
        tags=["scheduled", "skill_audit", capability],
        evidence={
            "target_capability": capability,
            "scheduled_due_cycle": due_cycle,
            "pattern_key": BLANK_INPUT_GUARD_PATTERN,
        },
    )


def seed_upgrade_goal(system: dict[str, object]) -> None:
    manager: AutonomousGoalManager = system["manager"]  # type: ignore[assignment]
    workspace: GlobalWorkspace = system["workspace"]  # type: ignore[assignment]
    goal = _make_tool_goal(
        capability="text_summarizer",
        title="Roll out canary upgrade for text_summarizer",
        description="Generate a new summarizer version, benchmark it offline, then subject it to a guarded canary.",
        source=GoalSource.METACOGNITION,
        expected_gain=0.74,
        novelty=0.41,
        uncertainty_reduction=0.70,
        strategic_fit=0.86,
        risk_estimate=0.10,
        priority=0.82,
        tags=["tooling", "text_summarizer", "upgrade"],
        evidence={
            "blocked_goal_title": "Improve text_summarizer robustness",
            "template_parameters": {"variant": "counts_raw_inputs"},
            "canary_payload": {
                "texts": list(workspace.get("canary_notes_batch", [])),
                "max_sentences": 2,
            },
        },
    )
    manager.admit_candidates([goal], context=workspace.get_state())


def seed_patch_followup_goal(system: dict[str, object]) -> None:
    manager: AutonomousGoalManager = system["manager"]  # type: ignore[assignment]
    workspace: GlobalWorkspace = system["workspace"]  # type: ignore[assignment]
    manager.ingest_external_goal(
        "Summarize the post-patch notes",
        required_capabilities=["classical_planning", "text_summarizer"],
        tags=["user", "followup", "summarize", "patch"],
        evidence={
            "raw_text": "Summarize the post-patch notes",
            "response_mode": "summary",
            "tool_payload": {
                "texts": list(workspace.get("patch_notes_batch", [])),
                "max_sentences": 2,
            },
        },
    )



def seed_strategy_reuse_goal(system: dict[str, object]) -> None:
    manager: AutonomousGoalManager = system["manager"]  # type: ignore[assignment]
    workspace: GlobalWorkspace = system["workspace"]  # type: ignore[assignment]
    goal = _make_tool_goal(
        capability="text_summarizer",
        title="Reuse learned strategy for text_summarizer edge canary",
        description="Generate a new summarizer candidate by reusing the learned remediation strategy and validate it against the prior edge case.",
        source=GoalSource.METACOGNITION,
        expected_gain=0.78,
        novelty=0.28,
        uncertainty_reduction=0.81,
        strategic_fit=0.92,
        risk_estimate=0.07,
        priority=0.84,
        tags=["tooling", "text_summarizer", "strategy_reuse"],
        evidence={
            "blocked_goal_title": "Validate reusable summarizer strategy",
            "strategy_lookup": {"reuse_learned_strategy": True},
            "canary_payload": {
                "texts": list(workspace.get("canary_notes_batch", [])),
                "max_sentences": 2,
            },
        },
    )
    manager.admit_candidates([goal], context=workspace.get_state())



def seed_proactive_remediation_goals(system: dict[str, object]) -> None:
    manager: AutonomousGoalManager = system["manager"]  # type: ignore[assignment]
    workspace: GlobalWorkspace = system["workspace"]  # type: ignore[assignment]
    tooling: ToolingManager = system["tooling"]  # type: ignore[assignment]
    existing_titles = {goal.title for goal in manager.pending + manager.active + manager.all_goals()}
    patch_goals = tooling.build_proactive_patch_goals(existing_goal_titles=existing_titles)
    if patch_goals:
        manager.admit_candidates(patch_goals, context=workspace.get_state())



def _run_tooling_step(
    step_name: str,
    goal: Any,
    tooling: ToolingManager,
    agency: AgencyModule,
    memory: MemoryModule,
    manager: AutonomousGoalManager,
    workspace: GlobalWorkspace,
) -> dict[str, Any]:
    if step_name == "design_tool_spec":
        spec = tooling.design_tool_spec(goal)
        result = {
            "step": step_name,
            "success": True,
            "tool_name": spec.name,
            "capability": spec.capability,
            "parameters": dict(spec.parameters),
        }
        strategy_hint = tooling.get_strategy_hint(goal.goal_id)
        if strategy_hint is not None:
            result["strategy_hint"] = strategy_hint
        return result
    if step_name == "generate_tool_code":
        generated = tooling.generate_tool_code(goal)
        return {
            "step": step_name,
            "success": generated.validation.allowed,
            "tool_name": generated.name,
            "capability": generated.capability,
            "violations": list(generated.validation.violations),
        }
    if step_name == "validate_tool_code":
        report = tooling.validate_tool_code(goal)
        return {"step": step_name, "success": bool(report["allowed"]), **report}
    if step_name == "register_tool":
        tool = tooling.register_tool(goal)
        return {
            "step": step_name,
            "success": True,
            "tool_registered": True,
            "tool_name": tool.name,
            "capability": tool.capability,
            "stage": "candidate",
        }
    if step_name == "benchmark_tool":
        report = tooling.benchmark_tool(goal)
        workspace.update({"latest_tool_benchmark": report.to_dict()})
        return {
            "step": step_name,
            "success": report.passed,
            "tool_name": report.tool_name,
            "capability": report.capability,
            "mean_score": report.mean_score,
            "threshold": report.threshold,
            "case_scores": [case.score for case in report.cases],
            "case_ids": [case.case_id for case in report.cases],
            "errors": list(report.errors),
        }
    if step_name == "promote_canary":
        try:
            tool = tooling.promote_canary(goal)
        except ValueError as exc:
            return {"step": step_name, "success": False, "canary_promoted": False, "message": str(exc), "execution_error": True}
        return {
            "step": step_name,
            "success": True,
            "canary_promoted": True,
            "tool_name": tool.name,
            "capability": tool.capability,
            "stage": "canary",
        }
    if step_name == "evaluate_canary":
        try:
            outcome = tooling.evaluate_canary(goal)
        except ValueError as exc:
            return {"step": step_name, "success": False, "message": str(exc), "execution_error": True}
        workspace.update({"latest_canary_assessment": outcome["assessment"], "latest_runtime_health": outcome["health"]})
        result: dict[str, Any] = {
            "step": step_name,
            "success": bool(outcome["passed"]),
            "tool_name": tooling._generated[goal.goal_id].name,
            "capability": tooling._generated[goal.goal_id].capability,
            "live_score": outcome["assessment"]["score"],
            "violations": list(outcome["assessment"]["violations"]),
            "rolled_back": bool(outcome["rolled_back"]),
            "restored_tool_name": outcome["restored_tool_name"],
        }
        if outcome.get("failure_case"):
            result["failure_case"] = outcome["failure_case"]
            result["cluster"] = outcome.get("cluster")
            result["regression_case"] = outcome.get("regression_case")
            result["patch_goal"] = outcome.get("patch_goal")
        return result
    if step_name == "finalize_rollout":
        try:
            tool = tooling.finalize_rollout(goal)
        except ValueError as exc:
            return {"step": step_name, "success": False, "tool_promoted": False, "message": str(exc), "execution_error": True}
        memory.store_tool(tool)
        workspace.update({"available_capabilities": sorted(agency.list_capabilities())})
        unblocked = manager.reactivate_deferred_goals_for_capability(tool.capability)
        return {
            "step": step_name,
            "success": True,
            "tool_promoted": True,
            "tool_name": tool.name,
            "capability": tool.capability,
            "stage": "stable",
            "unblocked_goals": [item.title for item in unblocked],
        }
    raise ValueError(f"Unsupported tooling step: {step_name}")



def _infer_episode_pattern(goal: Goal) -> tuple[str | None, str | None]:
    capability = str(goal.evidence.get("target_capability", "")).strip() or None
    if capability == "text_summarizer" and goal.kind == GoalKind.TOOL_CREATION and ({"patch", "strategy_reuse"} & set(goal.tags)):
        return BLANK_INPUT_GUARD_PATTERN, BLANK_INPUT_GUARD_LESSON
    if goal.kind == GoalKind.MAINTENANCE and "skill_audit" in goal.tags and capability == "text_summarizer":
        return BLANK_INPUT_GUARD_PATTERN, "Scheduled audits preserve the stable blank-input remediation for text_summarizer."
    if capability == "text_summarizer" and goal.kind == GoalKind.USER_TASK:
        return "text_summarizer|stable_summary_delivery", "Stable text_summarizer delivery keeps user summaries concise and operator-ready."
    return None, None



def _build_skill_arena_cases(workspace: GlobalWorkspace) -> list[SkillArenaCase]:
    return [
        SkillArenaCase(
            case_id="arena_summary_stability",
            payload={"texts": list(workspace.get("notes_batch", [])), "max_sentences": 2},
            expected={"source_count": 3, "max_sentences": 2, "required_summary_nonempty": True},
            description="Baseline notes should remain concise.",
        ),
        SkillArenaCase(
            case_id="arena_blank_input_guard",
            payload={"texts": list(workspace.get("canary_notes_batch", [])), "max_sentences": 2},
            expected={"source_count": 3, "max_sentences": 2, "required_summary_nonempty": True},
            description="Blank-input canary must stay fixed after rollout.",
        ),
        SkillArenaCase(
            case_id="arena_patch_notes",
            payload={"texts": list(workspace.get("patch_notes_batch", [])), "max_sentences": 2},
            expected={"source_count": 3, "max_sentences": 2, "required_summary_nonempty": True},
            description="Post-patch operator notes must remain concise.",
        ),
    ]



def _run_skill_consolidation(system: dict[str, object], *, current_cycle: int) -> dict[str, Any]:
    agency: AgencyModule = system["agency"]  # type: ignore[assignment]
    memory: MemoryModule = system["memory"]  # type: ignore[assignment]
    episodic_memory: EpisodicMemory = system["episodic_memory"]  # type: ignore[assignment]
    semantic_promoter: SemanticPromoter = system["semantic_promoter"]  # type: ignore[assignment]
    skill_arena: SkillArena = system["skill_arena"]  # type: ignore[assignment]
    scheduler: LongHorizonScheduler = system["scheduler"]  # type: ignore[assignment]
    workspace: GlobalWorkspace = system["workspace"]  # type: ignore[assignment]

    run = skill_arena.run(
        capability="text_summarizer",
        label="post_strategy_reuse",
        cases=_build_skill_arena_cases(workspace),
        execute=lambda payload: agency.execute_capability("text_summarizer", payload, rollout_stage="stable"),
    )
    workspace.update({"latest_skill_score": run.mean_score})
    promotions = semantic_promoter.consolidate(memory=memory, episodic_memory=episodic_memory, pattern_key=BLANK_INPUT_GUARD_PATTERN)
    for promotion in promotions:
        memory.store_semantic_promotion(promotion.fact_key, promotion.to_dict())
    scheduled = None
    if promotions:
        scheduled = scheduler.schedule(
            _make_scheduled_audit_goal(capability="text_summarizer", due_cycle=current_cycle + 2),
            due_cycle=current_cycle + 2,
            reason="semantic_promotion_followup",
        )
        workspace.update({"scheduled_goal_count": len(scheduler.pending())})
    return {
        "skill_run": run.to_dict(),
        "promotions": [item.to_dict() for item in promotions],
        "scheduled": scheduled.to_dict() if scheduled else None,
    }



def run_demo(cycles: int, runtime_dir: Path, interfaces_config_path: Path | None = None) -> None:
    system = build_system(runtime_dir, interfaces_config_path=interfaces_config_path)
    manager: AutonomousGoalManager = system["manager"]  # type: ignore[assignment]
    memory: MemoryModule = system["memory"]  # type: ignore[assignment]
    episodic_memory: EpisodicMemory = system["episodic_memory"]  # type: ignore[assignment]
    semantic_promoter: SemanticPromoter = system["semantic_promoter"]  # type: ignore[assignment]
    planning: PlanningModule = system["planning"]  # type: ignore[assignment]
    tooling: ToolingManager = system["tooling"]  # type: ignore[assignment]
    agency: AgencyModule = system["agency"]  # type: ignore[assignment]
    language: LanguageModule = system["language"]  # type: ignore[assignment]
    metacognition: MetacognitionModule = system["metacognition"]  # type: ignore[assignment]
    workspace: GlobalWorkspace = system["workspace"]  # type: ignore[assignment]
    skill_arena: SkillArena = system["skill_arena"]  # type: ignore[assignment]
    scheduler: LongHorizonScheduler = system["scheduler"]  # type: ignore[assignment]
    capability_graph: CapabilityGraph = system["capability_graph"]  # type: ignore[assignment]

    manager.restore_from_ledger()
    seed_demo_goals(system)
    seed_proactive_remediation_goals(system)

    upgrade_seeded = False
    reuse_seeded = False
    followup_seeded = False
    admitted_patch_signatures: set[str] = set()
    semantic_consolidated = False

    print("Tolik v3.138 — safe external adapters + shadow traffic drift guard demo")
    print(f"Runtime ledger: {runtime_dir / 'ledger'}\n")

    for cycle in range(1, cycles + 1):
        workspace.update(
            {
                "current_cycle": cycle,
                "available_capabilities": sorted(agency.list_capabilities()),
                "goal_queue_size": len(manager.pending),
                "episode_count": len(episodic_memory.list_episodes()),
                "scheduled_goal_count": len(scheduler.pending()),
            "capability_graph": capability_graph.summary(),
            }
        )

        existing_titles = {goal.title for goal in manager.pending + manager.active + manager.all_goals()}
        released = scheduler.release_due(current_cycle=cycle, existing_goal_titles=existing_titles)
        admitted_scheduled = manager.admit_candidates(released, context=workspace.get_state()) if released else []

        meta_summary = metacognition.analyze()
        memory_summary = memory.get_recent_events()

        candidates = manager.generate_candidates(workspace_state=workspace.get_state(), memory_summary=memory_summary, meta_summary=meta_summary)
        admitted = manager.admit_candidates(candidates, context=workspace.get_state())

        tooling_candidates = manager.create_tooling_goals_for_deferred(tooling.supported_capabilities())
        admitted_tools = manager.admit_candidates(tooling_candidates, context=workspace.get_state()) if tooling_candidates else []

        admitted_titles = [goal.title for goal in admitted_scheduled + admitted + admitted_tools]
        print(f"[cycle {cycle}] admitted: {admitted_titles}")

        goal = manager.select_next_goal(context=workspace.get_state())
        if goal is None:
            print(f"[cycle {cycle}] no executable goals left.\n")
            continue

        print(f"[cycle {cycle}] active goal: {goal.title} ({goal.source.value})")
        plan = planning.make_plan(goal, world_state=workspace.get_state())
        print(f"[cycle {cycle}] plan: {[step.name for step in plan.steps]}")

        trace: list[dict[str, Any]] = []
        latest_tool_output: dict[str, Any] | None = None

        for step in plan.steps:
            if step.name in {
                "design_tool_spec",
                "generate_tool_code",
                "validate_tool_code",
                "register_tool",
                "benchmark_tool",
                "promote_canary",
                "evaluate_canary",
                "finalize_rollout",
            }:
                result = _run_tooling_step(step_name=step.name, goal=goal, tooling=tooling, agency=agency, memory=memory, manager=manager, workspace=workspace)
            else:
                result = agency.execute(step.name, goal=goal, workspace_state=workspace.get_state())
                if step.name.startswith("run_capability:") and result.get("success"):
                    latest_tool_output = dict(result.get("output", {}))
                    workspace.update({"latest_tool_output": latest_tool_output})
                if step.name == "form_response" and goal.kind == GoalKind.USER_TASK:
                    if latest_tool_output and latest_tool_output.get("summary"):
                        result["response"] = language.generate_response({"answer": latest_tool_output["summary"]})
                    else:
                        result["response"] = language.generate_response({"answer": "ok"})
                if step.name == "retrieve_missing_knowledge" and result.get("success", False):
                    memory.store_fact("resolved_memory_gap", {"fact": "knowledge acquired"})
                    workspace.update({"retrieval_confidence": 0.95})

            manager.update_progress(goal.goal_id, {"step": step.name, "result": result})
            metacognition.log_event("agency", step.name, result)
            memory.add_experience({"goal_id": goal.goal_id, "step": step.name, **result})
            trace.append({"step": step.name, "result": result})
            print(f"  - {step.name}: {result}")

        observed = {
            "goal_success": all(item["result"].get("success", True) for item in trace),
            "drift_flag": goal.source.value == "drift_alarm",
            "regression_flag": goal.source.value == "regression_failure",
            "anchor_suite_failed": workspace.get("anchor_suite_failed", False),
        }
        expected = {"plan_len": len(plan.steps), "goal_id": goal.goal_id}
        report, tasks = metacognition.run_postmortem(goal, trace, expected, observed)
        memory.store_postmortem(report)
        outcome = {"success": report.success, "postmortem": report.to_dict()}
        manager.complete_goal(goal.goal_id, outcome)

        pattern_key, lesson = _infer_episode_pattern(goal)
        episode = episodic_memory.record_goal_episode(
            goal,
            cycle=cycle,
            trace=trace,
            outcome=outcome,
            workspace_excerpt={
                "latest_tool_output": dict(latest_tool_output or {}),
                "latest_skill_score": workspace.get("latest_skill_score"),
            },
            pattern_key=pattern_key,
            lesson=lesson,
            tags=list(goal.tags),
        )
        workspace.update({"episode_count": len(episodic_memory.list_episodes()), "latest_pattern_key": episode.pattern_key})

        curriculum_goals = [curriculum_task_to_goal(task) for task in tasks]
        manager.admit_candidates(curriculum_goals, context=workspace.get_state())

        print(f"[cycle {cycle}] postmortem causes: {report.root_causes}")
        print(f"[cycle {cycle}] curriculum: {[task.title for task in tasks]}")
        if goal.kind == GoalKind.USER_TASK:
            response = latest_tool_output.get("summary") if latest_tool_output else "ok"
            print(f"[cycle {cycle}] language response: {language.generate_response({'answer': response})}")
        print()

        if goal.title == "Summarize the current research notes" and report.success and not upgrade_seeded:
            seed_upgrade_goal(system)
            upgrade_seeded = True

        canary_outcome = tooling.get_canary_outcome(goal.goal_id)
        if canary_outcome and canary_outcome.get("patch_goal"):
            patch_goal = tooling.get_patch_goal(goal.goal_id)
            if patch_goal is not None:
                signature = str(patch_goal.evidence.get("failure_signature", ""))
                if signature and signature not in admitted_patch_signatures:
                    manager.admit_candidates([patch_goal], context=workspace.get_state())
                    admitted_patch_signatures.add(signature)

        if goal.kind == GoalKind.TOOL_CREATION and "patch" in goal.tags and report.success and not reuse_seeded:
            seed_strategy_reuse_goal(system)
            reuse_seeded = True

        if goal.kind == GoalKind.TOOL_CREATION and "strategy_reuse" in goal.tags and report.success and not followup_seeded:
            consolidation = _run_skill_consolidation(system, current_cycle=cycle)
            print(f"[cycle {cycle}] skill arena: {consolidation['skill_run']}")
            print(f"[cycle {cycle}] semantic promotions: {consolidation['promotions']}")
            if consolidation["scheduled"] is not None:
                print(f"[cycle {cycle}] scheduled audit: {consolidation['scheduled']}")
            semantic_consolidated = semantic_consolidated or bool(consolidation["promotions"])
            seed_patch_followup_goal(system)
            followup_seeded = True

    print("Demo finished.")
    print(f"Registered tools: {tooling.registry.list_tools()}")
    print(f"Stable tools: {tooling.registry.stable_tool_names()}")
    print(f"Canary tools: {tooling.registry.canary_tool_names()}")
    print(f"Expanded regression cases: {[case.case_id for case in tooling.benchmark_expander.get_cases('text_summarizer')]}")
    print(f"Failure pattern registry (open): {[item.to_dict() for item in tooling.curriculum_registry.open_patterns()]}")
    print(f"Failure pattern registry (closed): {[item.to_dict() for item in tooling.curriculum_registry.closed_patterns()]}")
    print(f"Strategy patterns: {tooling.list_strategy_patterns()}")
    print(f"Episode memory: {[record.to_dict() for record in episodic_memory.list_episodes()]}")
    print(f"Semantic promotions: {[item.to_dict() for item in semantic_promoter.list_promotions()]}")
    print(f"Skill arena runs: {[run.to_dict() for run in skill_arena.list_runs()]}")
    print(f"Scheduled goals: {[item.to_dict() for item in scheduler.list_all()]}")
    print(f"Semantic consolidation reached: {semantic_consolidated}")
    print(f"Saved goal snapshots: {len(list((runtime_dir / 'ledger' / 'snapshots').glob('*.json')))}")
    print(f"Saved postmortems: {len(list((runtime_dir / 'ledger' / 'postmortems').glob('*.json')))}")



def main() -> None:
    parser = argparse.ArgumentParser(description="Run Tolik v3.138 demo with safe external adapters, provider rollout, shadow traffic, and drift guard support.")
    parser.add_argument("--cycles", type=int, default=7, help="Maximum number of demo cycles.")
    parser.add_argument("--runtime-dir", type=Path, default=Path("runtime"), help="Directory used for ledger snapshots and postmortems.")
    parser.add_argument("--interfaces-config", type=Path, default=None, help="Optional JSON config for safe external adapter modes/providers.")
    args = parser.parse_args()
    run_demo(cycles=args.cycles, runtime_dir=args.runtime_dir, interfaces_config_path=args.interfaces_config)


if __name__ == "__main__":
    main()
