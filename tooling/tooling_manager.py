from __future__ import annotations

import re
from typing import Any

from benchmarks.benchmark_expander import BenchmarkExpander, RegressionCase
from core.event_types import GoalEventType
from memory.goal_ledger import GoalLedger
from memory.strategy_memory import StrategyMemory
from metacognition.failure_miner import FailureCase, FailureMiner
from metacognition.postmortem_clusterer import FailureCluster, PostmortemClusterer
from motivation.curriculum_registry import CurriculumRegistry
from motivation.goal_schema import Goal
from tooling.controlled_code_generator import ControlledCodeGenerator
from tooling.patch_planner import PatchPlanner
from tooling.policy_layer import PolicyLayer
from tooling.runtime_guard import RuntimeAssessment, RuntimeGuard, RuntimeMonitor
from tooling.sandbox import RestrictedSandbox
from tooling.tool_evaluator import ToolEvaluationReport, ToolEvaluator
from tooling.tool_registry import ToolRegistry
from tooling.tool_spec import GeneratedTool, ToolSpec


class ToolingManager:
    """Orchestrates safe tool generation, canary failure mining, benchmark expansion, and rollback."""

    def __init__(
        self,
        ledger: GoalLedger,
        policy_layer: PolicyLayer | None = None,
        generator: ControlledCodeGenerator | None = None,
        sandbox: RestrictedSandbox | None = None,
        registry: ToolRegistry | None = None,
        evaluator: ToolEvaluator | None = None,
        runtime_guard: RuntimeGuard | None = None,
        runtime_monitor: RuntimeMonitor | None = None,
        failure_miner: FailureMiner | None = None,
        benchmark_expander: BenchmarkExpander | None = None,
        postmortem_clusterer: PostmortemClusterer | None = None,
        curriculum_registry: CurriculumRegistry | None = None,
        patch_planner: PatchPlanner | None = None,
        strategy_memory: StrategyMemory | None = None,
    ) -> None:
        self.ledger = ledger
        self.policy_layer = policy_layer or PolicyLayer()
        self.generator = generator or ControlledCodeGenerator(policy_layer=self.policy_layer)
        self.sandbox = sandbox or RestrictedSandbox()
        self.registry = registry or ToolRegistry()
        self.benchmark_expander = benchmark_expander or BenchmarkExpander(ledger=self.ledger)
        self.evaluator = evaluator or ToolEvaluator(benchmark_expander=self.benchmark_expander)
        self.runtime_guard = runtime_guard or RuntimeGuard()
        self.runtime_monitor = runtime_monitor or RuntimeMonitor()
        self.failure_miner = failure_miner or FailureMiner()
        self.postmortem_clusterer = postmortem_clusterer or PostmortemClusterer()
        self.curriculum_registry = curriculum_registry or CurriculumRegistry(ledger=self.ledger)
        self.patch_planner = patch_planner or PatchPlanner()
        self.strategy_memory = strategy_memory or StrategyMemory(ledger=self.ledger)
        self._specs: dict[str, ToolSpec] = {}
        self._generated: dict[str, GeneratedTool] = {}
        self._evaluations: dict[str, ToolEvaluationReport] = {}
        self._tool_sequence_by_capability: dict[str, int] = {}
        self._canary_outcomes: dict[str, dict[str, Any]] = {}
        self._failure_cases: dict[str, FailureCase] = {}
        self._regression_cases: dict[str, RegressionCase] = {}
        self._clusters: dict[str, FailureCluster] = {}
        self._patch_goals: dict[str, Goal] = {}
        self._strategy_hint_by_goal: dict[str, str] = {}
        self._latest_failure_by_signature: dict[str, FailureCase] = {}
        self._rehydrate_failure_memory()


    def _rehydrate_failure_memory(self) -> None:
        for failure in self.ledger.load_failure_cases():
            self._failure_cases[failure.goal_id] = failure
            self._latest_failure_by_signature[failure.signature] = failure
            cluster = self.postmortem_clusterer.add_failure(failure)
            self._clusters[failure.goal_id] = cluster
        for pattern in self.curriculum_registry.closed_patterns():
            self.postmortem_clusterer.mark_closed(pattern.signature)

    def build_proactive_patch_goals(self, existing_goal_titles: set[str] | None = None) -> list[Goal]:
        existing = {" ".join(title.lower().split()) for title in (existing_goal_titles or set())}
        goals: list[Goal] = []
        for pattern in self.curriculum_registry.open_patterns():
            failure = self._latest_failure_by_signature.get(pattern.signature)
            if failure is None:
                continue
            goal = self.patch_planner.build_goal(failure, occurrence_count=max(pattern.occurrence_count, 1))
            normalized_title = " ".join(goal.title.lower().split())
            if normalized_title in existing:
                continue
            existing.add(normalized_title)
            self.curriculum_registry.attach_remediation_goal(pattern.signature, goal.title)
            goals.append(goal)
        return goals

    def supported_capabilities(self) -> set[str]:
        return self.generator.supported_capabilities()

    def _next_sequence(self, capability: str) -> int:
        sequence = self._tool_sequence_by_capability.get(capability, 0) + 1
        self._tool_sequence_by_capability[capability] = sequence
        return sequence

    def _remember_sequence_from_name(self, capability: str, name: str) -> None:
        match = re.search(r"_v(\d+)$", name)
        if not match:
            return
        sequence = int(match.group(1))
        self._tool_sequence_by_capability[capability] = max(self._tool_sequence_by_capability.get(capability, 0), sequence)

    def seed_stable_tool(
        self,
        capability: str,
        *,
        name: str | None = None,
        description: str | None = None,
        parameters: dict[str, Any] | None = None,
        version: str = "v3.121-seed",
    ) -> GeneratedTool:
        sequence = self._tool_sequence_by_capability.get(capability, 0)
        if sequence <= 0:
            sequence = 1
            self._tool_sequence_by_capability[capability] = sequence
        tool_name = name or f"generated_{capability}_v{sequence}"
        spec = self.generator.make_spec(
            capability,
            name=tool_name,
            description=description or f"Stable seed tool for capability {capability}.",
            parameters=parameters,
        )
        tool = self.generator.generate(spec)
        tool.version = version
        runtime_callable = self.sandbox.load_callable(tool.source_code)
        self.registry.register_stable(tool, runtime_callable)
        self.ledger.append_event(
            {
                "event_type": GoalEventType.TOOL_PROMOTED.value,
                "goal_id": f"seed:{capability}",
                "tool_name": tool.name,
                "capability": tool.capability,
                "mode": "seed_stable",
            }
        )
        self._remember_sequence_from_name(capability, tool.name)
        return tool

    def design_tool_spec(self, goal: Goal) -> ToolSpec:
        capability = str(goal.evidence.get("target_capability", "")).strip()
        if not capability:
            raise ValueError("tool creation goal is missing target_capability")
        sequence = self._next_sequence(capability)
        explicit_parameters = dict(goal.evidence.get("template_parameters", {}))
        preferred_signature = str(goal.evidence.get("failure_signature", "")).strip() or None
        strategy_lookup = dict(goal.evidence.get("strategy_lookup", {}))
        if preferred_signature is None:
            preferred_signature = str(strategy_lookup.get("signature", "")).strip() or None
        allow_capability_fallback = bool(strategy_lookup.get("reuse_learned_strategy", False))
        template_parameters = dict(explicit_parameters)
        chosen_strategy = None
        learned_parameters, learned_pattern = self.strategy_memory.select_parameters(
            capability,
            preferred_signature=preferred_signature,
            allow_capability_fallback=allow_capability_fallback,
        )
        if learned_pattern is not None and learned_parameters is not None:
            chosen_strategy = learned_pattern
            merged_parameters = dict(learned_parameters)
            merged_parameters.update(template_parameters)
            template_parameters = merged_parameters
            self._strategy_hint_by_goal[goal.goal_id] = learned_pattern.strategy_id
        spec = self.generator.make_spec(
            capability,
            name=f"generated_{capability}_v{sequence}",
            description=f"Generated for blocked goal {goal.evidence.get('blocked_goal_title', goal.title)}",
            parameters=template_parameters,
        )
        self._specs[goal.goal_id] = spec
        event = {
            "event_type": GoalEventType.TOOL_DESIGNED.value,
            "goal_id": goal.goal_id,
            "tool_name": spec.name,
            "capability": spec.capability,
            "parameters": dict(spec.parameters),
        }
        if chosen_strategy is not None:
            event.update({
                "strategy_id": chosen_strategy.strategy_id,
                "strategy_signature": chosen_strategy.signature,
            })
            self.ledger.append_event({
                "event_type": GoalEventType.STRATEGY_PATTERN_REUSED.value,
                "goal_id": goal.goal_id,
                "strategy_id": chosen_strategy.strategy_id,
                "signature": chosen_strategy.signature,
                "capability": capability,
            })
        self.ledger.append_event(event)
        return spec

    def generate_tool_code(self, goal: Goal) -> GeneratedTool:
        spec = self._specs[goal.goal_id]
        tool = self.generator.generate(spec)
        self._generated[goal.goal_id] = tool
        self.ledger.append_event(
            {
                "event_type": GoalEventType.TOOL_GENERATED.value,
                "goal_id": goal.goal_id,
                "tool_name": tool.name,
                "capability": tool.capability,
                "allowed": tool.validation.allowed,
            }
        )
        return tool

    def validate_tool_code(self, goal: Goal) -> dict[str, Any]:
        tool = self._generated[goal.goal_id]
        report = tool.validation
        self.ledger.append_event(
            {
                "event_type": GoalEventType.TOOL_VALIDATED.value,
                "goal_id": goal.goal_id,
                "tool_name": tool.name,
                "allowed": report.allowed,
                "violations": list(report.violations),
            }
        )
        return report.to_dict()

    def register_tool(self, goal: Goal) -> GeneratedTool:
        tool = self._generated[goal.goal_id]
        if not tool.validation.allowed:
            raise ValueError("cannot register tool that failed policy validation")
        runtime_callable = self.sandbox.load_callable(tool.source_code)
        self.registry.register_candidate(tool, runtime_callable)
        self.ledger.append_event(
            {
                "event_type": GoalEventType.TOOL_REGISTERED.value,
                "goal_id": goal.goal_id,
                "tool_name": tool.name,
                "capability": tool.capability,
                "stage": "candidate",
            }
        )
        return tool

    def benchmark_tool(self, goal: Goal) -> ToolEvaluationReport:
        tool = self._generated[goal.goal_id]
        report = self.evaluator.evaluate(
            tool_name=tool.name,
            capability=tool.capability,
            runtime_callable=lambda payload: self.registry.execute_candidate(tool.capability, payload),
        )
        self._evaluations[goal.goal_id] = report
        self.ledger.append_event(
            {
                "event_type": GoalEventType.TOOL_BENCHMARKED.value,
                "goal_id": goal.goal_id,
                "tool_name": tool.name,
                "capability": tool.capability,
                "mean_score": report.mean_score,
                "passed": report.passed,
                "case_count": len(report.cases),
            }
        )
        return report

    def promote_tool(self, goal: Goal) -> GeneratedTool:
        tool = self._generated[goal.goal_id]
        report = self._evaluations.get(goal.goal_id)
        if report is None:
            raise ValueError("tool must be benchmarked before promotion")
        if not report.passed:
            self.registry.discard_candidate(tool.capability)
            raise ValueError("tool failed benchmark gate")
        promoted = self.registry.promote_candidate(tool.capability)
        self.ledger.append_event(
            {
                "event_type": GoalEventType.TOOL_PROMOTED.value,
                "goal_id": goal.goal_id,
                "tool_name": promoted.name,
                "capability": promoted.capability,
                "mode": "direct",
                "mean_score": report.mean_score,
            }
        )
        return promoted

    def promote_canary(self, goal: Goal) -> GeneratedTool:
        tool = self._generated[goal.goal_id]
        report = self._evaluations.get(goal.goal_id)
        if report is None:
            raise ValueError("tool must be benchmarked before canary promotion")
        if not report.passed:
            self.registry.discard_candidate(tool.capability)
            raise ValueError("tool failed benchmark gate")
        promoted = self.registry.promote_candidate_to_canary(tool.capability)
        self.ledger.append_event(
            {
                "event_type": GoalEventType.TOOL_CANARY_PROMOTED.value,
                "goal_id": goal.goal_id,
                "tool_name": promoted.name,
                "capability": promoted.capability,
                "mean_score": report.mean_score,
                "stable_before_canary": self.registry.get_active_tool(tool.capability).name if self.registry.has_capability(tool.capability) else None,
            }
        )
        return promoted

    def _mine_failure(
        self,
        *,
        goal: Goal,
        tool: GeneratedTool,
        payload: dict[str, Any],
        assessment: RuntimeAssessment,
        output: dict[str, Any] | None,
        restored_tool_name: str | None,
    ) -> tuple[FailureCase, FailureCluster, RegressionCase, Goal | None]:
        failure = self.failure_miner.mine_canary_failure(
            goal=goal,
            tool=tool,
            payload=payload,
            assessment=assessment,
            output=output,
            rollback_target=restored_tool_name,
        )
        self._failure_cases[goal.goal_id] = failure
        self.ledger.save_failure_case(failure)
        self.ledger.append_event(
            {
                "event_type": GoalEventType.CANARY_FAILURE_MINED.value,
                "goal_id": goal.goal_id,
                "tool_name": tool.name,
                "capability": tool.capability,
                "signature": failure.signature,
                "violations": list(failure.violation_types),
            }
        )

        cluster = self.postmortem_clusterer.add_failure(failure)
        self._clusters[goal.goal_id] = cluster
        self.curriculum_registry.register_failure(failure, cluster)

        regression_case = self.benchmark_expander.expand_from_failure(failure)
        self._regression_cases[goal.goal_id] = regression_case
        self.ledger.append_event(
            {
                "event_type": GoalEventType.BENCHMARK_EXPANDED.value,
                "goal_id": goal.goal_id,
                "capability": tool.capability,
                "case_id": regression_case.case_id,
                "signature": failure.signature,
            }
        )

        patch_goal: Goal | None = None
        self._latest_failure_by_signature[failure.signature] = failure
        if self.curriculum_registry.is_open(failure.signature):
            patch_goal = self.patch_planner.build_goal(failure, occurrence_count=cluster.occurrence_count)
            self._patch_goals[goal.goal_id] = patch_goal
            self.curriculum_registry.attach_remediation_goal(failure.signature, patch_goal.title)
            self.ledger.append_event(
                {
                    "event_type": GoalEventType.PATCH_GOAL_PLANNED.value,
                    "goal_id": goal.goal_id,
                    "capability": tool.capability,
                    "title": patch_goal.title,
                    "signature": failure.signature,
                }
            )
        return failure, cluster, regression_case, patch_goal

    def evaluate_canary(self, goal: Goal) -> dict[str, Any]:
        tool = self._generated[goal.goal_id]
        if not self.registry.has_canary(tool.capability):
            raise ValueError("canary tool is not active")

        payload = dict(goal.evidence.get("canary_payload", {}))
        if not payload:
            payload = dict(goal.evidence.get("tool_payload", {}))
        if not payload:
            if tool.capability == "text_summarizer":
                payload = {
                    "texts": [
                        "Canary execution should validate a live edge case.",
                        "The rollout gate must reject brittle tools.",
                    ],
                    "max_sentences": 2,
                }
            elif tool.capability == "keyword_extractor":
                payload = {"text": "alpha beta beta gamma"}
            else:
                payload = {"values": [1, 2, 3]}

        output: dict[str, Any] | None = None
        error: Exception | None = None
        try:
            output = self.registry.execute_canary(tool.capability, payload)
        except Exception as exc:  # pragma: no cover
            error = exc

        assessment = self.runtime_guard.assess(
            capability=tool.capability,
            tool_name=tool.name,
            payload=payload,
            output=output,
            error=error,
        )
        health = self.runtime_monitor.record(assessment)
        rolled_back = False
        restored_tool_name: str | None = None

        if not assessment.passed and self.runtime_monitor.should_rollback(tool.capability):
            restored = self.registry.rollback_canary(tool.capability)
            restored_tool_name = restored.name if restored is not None else None
            health = self.runtime_monitor.mark_rollback(tool.capability, restored_tool_name)
            rolled_back = True
            self.ledger.append_event(
                {
                    "event_type": GoalEventType.TOOL_ROLLED_BACK.value,
                    "goal_id": goal.goal_id,
                    "capability": tool.capability,
                    "tool_name": restored_tool_name,
                    "reason": assessment.reason or "canary_runtime_failure",
                }
            )

        outcome: dict[str, Any] = {
            "passed": assessment.passed and not rolled_back,
            "rolled_back": rolled_back,
            "assessment": assessment.to_dict(),
            "health": health.to_dict(),
            "output": output,
            "restored_tool_name": restored_tool_name,
        }

        if not outcome["passed"]:
            failure, cluster, regression_case, patch_goal = self._mine_failure(
                goal=goal,
                tool=tool,
                payload=payload,
                assessment=assessment,
                output=output,
                restored_tool_name=restored_tool_name,
            )
            outcome.update(
                {
                    "failure_case": failure.to_dict(),
                    "cluster": cluster.to_dict(),
                    "regression_case": regression_case.to_dict(),
                    "patch_goal": patch_goal.to_dict() if patch_goal is not None else None,
                }
            )

        strategy_id = self._strategy_hint_by_goal.get(goal.goal_id)
        if strategy_id is not None:
            self.strategy_memory.record_outcome(strategy_id, passed=bool(outcome["passed"]))

        self._canary_outcomes[goal.goal_id] = outcome
        self.ledger.append_event(
            {
                "event_type": GoalEventType.TOOL_CANARY_EVALUATED.value,
                "goal_id": goal.goal_id,
                "tool_name": tool.name,
                "capability": tool.capability,
                "passed": outcome["passed"],
                "rolled_back": rolled_back,
                "score": assessment.score,
                "violations": list(assessment.violations),
            }
        )
        return outcome

    def finalize_rollout(self, goal: Goal) -> GeneratedTool:
        tool = self._generated[goal.goal_id]
        outcome = self._canary_outcomes.get(goal.goal_id)
        if outcome is None:
            raise ValueError("canary must be evaluated before final rollout")
        if not outcome.get("passed") or outcome.get("rolled_back"):
            raise ValueError("canary gate failed; rollout stays on stable version")
        if not self.registry.has_canary(tool.capability):
            raise ValueError("canary slot is empty")

        promoted = self.registry.finalize_canary(tool.capability)
        failure_signature = goal.evidence.get("failure_signature")
        if isinstance(failure_signature, str) and failure_signature:
            self.curriculum_registry.mark_closed(failure_signature, promoted.name)
            self.postmortem_clusterer.mark_closed(failure_signature)
            pattern = self.strategy_memory.register_patch_strategy(
                failure_signature=failure_signature,
                capability=promoted.capability,
                template_parameters=dict(self._specs[goal.goal_id].parameters),
                remediation_targets=list(goal.evidence.get("remediation_targets", [])),
                source_goal_id=goal.goal_id,
                source_tool_name=promoted.name,
                evidence={
                    "rollback_target": goal.evidence.get("rollback_target"),
                    "source_failure_goal_id": dict(goal.evidence.get("source_failure_case", {})).get("goal_id"),
                },
            )
            self.ledger.append_event(
                {
                    "event_type": GoalEventType.STRATEGY_PATTERN_STORED.value,
                    "goal_id": goal.goal_id,
                    "strategy_id": pattern.strategy_id,
                    "signature": pattern.signature,
                    "capability": pattern.capability,
                    "tool_name": promoted.name,
                }
            )
        self.ledger.append_event(
            {
                "event_type": GoalEventType.TOOL_PROMOTED.value,
                "goal_id": goal.goal_id,
                "tool_name": promoted.name,
                "capability": promoted.capability,
                "mode": "finalize_canary",
                "live_score": outcome["assessment"]["score"],
            }
        )
        return promoted

    def get_strategy_hint(self, goal_id: str) -> dict[str, Any] | None:
        strategy_id = self._strategy_hint_by_goal.get(goal_id)
        if not strategy_id:
            return None
        pattern = self.strategy_memory.get_by_id(strategy_id)
        if pattern is None:
            return None
        return {
            "strategy_id": pattern.strategy_id,
            "signature": pattern.signature,
            "template_parameters": dict(pattern.template_parameters),
            "success_rate": round(pattern.success_rate(), 4),
            "uses": pattern.uses,
            "wins": pattern.wins,
        }

    def list_strategy_patterns(self) -> list[dict[str, Any]]:
        return [pattern.to_dict() for pattern in self.strategy_memory.list_patterns()]

    def get_canary_outcome(self, goal_id: str) -> dict[str, Any] | None:
        outcome = self._canary_outcomes.get(goal_id)
        return dict(outcome) if outcome is not None else None

    def get_patch_goal(self, goal_id: str) -> Goal | None:
        return self._patch_goals.get(goal_id)

    def get_failure_case(self, goal_id: str) -> FailureCase | None:
        return self._failure_cases.get(goal_id)

    def rollback_capability(self, capability: str, reason: str) -> GeneratedTool | None:
        restored = self.registry.rollback_capability(capability)
        self.ledger.append_event(
            {
                "event_type": GoalEventType.TOOL_ROLLED_BACK.value,
                "goal_id": f"rollback:{capability}",
                "capability": capability,
                "tool_name": restored.name if restored else None,
                "reason": reason,
            }
        )
        return restored
