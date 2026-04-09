from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from environments.grounded_navigation import GroundedNavigationLab
from environments.route_briefing import RouteBriefingLab
from environments.spatial_route import SpatialRouteLab
from tooling.tool_registry import ToolRegistry

DEFAULT_CAPABILITIES = {
    "classical_planning",
    "classical_simulation",
    "local_llm",
    "grounded_navigation",
}


@dataclass
class AgencyModule:
    """Executes abstract plan steps in deterministic demo and autonomous-training environments."""

    capabilities: set[str] = field(default_factory=lambda: set(DEFAULT_CAPABILITIES))
    failure_map: dict[str, dict[str, Any]] = field(default_factory=dict)
    tool_registry: ToolRegistry = field(default_factory=ToolRegistry)
    grounded_lab: GroundedNavigationLab = field(default_factory=GroundedNavigationLab)
    spatial_route_lab: SpatialRouteLab = field(default_factory=SpatialRouteLab)
    route_briefing_lab: RouteBriefingLab = field(default_factory=RouteBriefingLab)
    grounded_navigation_strategy: str = "greedy"

    def list_capabilities(self) -> set[str]:
        return set(self.capabilities) | self.tool_registry.capabilities()

    def add_capability(self, capability: str) -> None:
        self.capabilities.add(capability)

    def set_navigation_strategy(self, strategy: str) -> None:
        if strategy not in {"greedy", "graph_search"}:
            raise ValueError(f"Unsupported grounded navigation strategy: {strategy}")
        self.grounded_navigation_strategy = strategy

    def execute_capability(
        self,
        capability: str,
        payload: dict[str, Any],
        *,
        rollout_stage: str = "stable",
    ) -> dict[str, Any]:
        if capability == "grounded_navigation":
            output = self.grounded_lab.run_batch(strategy=self.grounded_navigation_strategy, payload=payload)
            return {
                "step": f"run_capability:{capability}",
                "success": bool(output.get("passed", False)),
                "capability": capability,
                "tool_name": f"builtin_grounded_navigation_{self.grounded_navigation_strategy}",
                "rollout_stage": rollout_stage,
                "output": output,
            }

        if capability == "navigation_route_explanation":
            output = self.grounded_lab.explain_batch(strategy=self.grounded_navigation_strategy, payload=payload)
            return {
                "step": f"run_capability:{capability}",
                "success": bool(output.get("passed", False)),
                "capability": capability,
                "tool_name": f"builtin_navigation_route_explanation_{self.grounded_navigation_strategy}",
                "rollout_stage": rollout_stage,
                "output": output,
            }

        if capability == "spatial_route_composition":
            output = self.spatial_route_lab.run_batch(strategy=self.grounded_navigation_strategy, payload=payload)
            return {
                "step": f"run_capability:{capability}",
                "success": bool(output.get("passed", False)),
                "capability": capability,
                "tool_name": f"builtin_spatial_route_composition_{self.grounded_navigation_strategy}",
                "rollout_stage": rollout_stage,
                "output": output,
            }

        if capability == "route_mission_briefing":
            output = self.route_briefing_lab.run_batch(strategy=self.grounded_navigation_strategy, payload=payload)
            return {
                "step": f"run_capability:{capability}",
                "success": bool(output.get("passed", False)),
                "capability": capability,
                "tool_name": f"builtin_route_mission_briefing_{self.grounded_navigation_strategy}",
                "rollout_stage": rollout_stage,
                "output": output,
            }

        if rollout_stage == "canary":
            if not self.tool_registry.has_canary(capability):
                return {
                    "step": f"run_capability:{capability}",
                    "success": False,
                    "missing_capability": capability,
                    "rollout_stage": rollout_stage,
                }
            tool = self.tool_registry.get_canary_tool(capability)
            output = self.tool_registry.execute_canary(capability, payload)
        else:
            if not self.tool_registry.has_capability(capability):
                return {
                    "step": f"run_capability:{capability}",
                    "success": False,
                    "missing_capability": capability,
                    "rollout_stage": rollout_stage,
                }
            tool = self.tool_registry.get_active_tool(capability)
            output = self.tool_registry.execute_by_capability(capability, payload)

        return {
            "step": f"run_capability:{capability}",
            "success": True,
            "capability": capability,
            "tool_name": tool.name,
            "rollout_stage": rollout_stage,
            "output": output,
        }

    def can_run_internal_goal_now(self, workspace_state: dict[str, Any] | None = None) -> bool:
        workspace_state = workspace_state or {}
        return not bool(workspace_state.get("active_user_goal") or workspace_state.get("user_goal_active"))

    def respect_self_improvement_budget(self, remaining_steps: int) -> bool:
        return remaining_steps > 0

    def execute_curriculum_item(self, item: Any, workspace_state: dict[str, Any] | None = None) -> dict[str, Any]:
        workspace_state = workspace_state or {}
        capability = getattr(item, "capability_id", None) or workspace_state.get("current_capability")
        payload = getattr(item, "payload", {}) or {}
        benchmark_name = getattr(item, "benchmark_name", "skill_arena")
        if capability:
            result = self.execute_capability(str(capability), payload, rollout_stage="stable")
            result["benchmark_name"] = benchmark_name
            result["scenario_id"] = getattr(item, "scenario_id", None)
            result["action_hint"] = getattr(item, "action_hint", None)
            return result
        return {"step": "execute_curriculum_item", "success": False, "execution_error": True, "message": "No capability bound to curriculum item."}

    def execute(
        self,
        step_name: str,
        goal: Any | None = None,
        workspace_state: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        workspace_state = workspace_state or {}
        if step_name.startswith("run_capability:"):
            capability = step_name.split(":", 1)[1]
            payload = {}
            if goal is not None:
                payload = dict(getattr(goal, "evidence", {}).get("tool_payload", {}))
            return self.execute_capability(capability, payload, rollout_stage="stable")

        if step_name == "bootstrap_transfer_capability":
            target_capability = None
            if goal is not None:
                target_capability = getattr(goal, "evidence", {}).get("target_capability")
            if not target_capability:
                return {
                    "step": step_name,
                    "success": False,
                    "execution_error": True,
                    "message": "Missing target_capability in goal evidence.",
                }
            self.add_capability(str(target_capability))
            return {
                "step": step_name,
                "success": True,
                "target_capability": target_capability,
                "capability_bootstrapped": True,
                "available_capabilities": sorted(self.list_capabilities()),
            }

        result: dict[str, Any] = {
            "step": step_name,
            "success": True,
        }

        defaults: dict[str, dict[str, Any]] = {
            "understand_request": {"observation": "request_understood"},
            "form_response": {"response": f"Handled goal: {getattr(goal, 'title', 'unknown')}"},
            "scan_novelty": {"novelty_sample": workspace_state.get("novelty_score", 0.0)},
            "store_findings": {"stored": True},
            "diagnose_drift": {"drift_confirmed": workspace_state.get("ood_score", 0.0) > 0.6},
            "refresh_baseline": {"baseline_refreshed": True},
            "retrieve_missing_knowledge": {"fact_acquired": True},
            "record_learning": {"memory_written": True},
            "study_gap": {"study_session": "completed"},
            "replay_anchor_suite": {"anchor_suite_passed": not workspace_state.get("anchor_suite_failed", False)},
            "compare_baseline": {"delta": workspace_state.get("regression_score", 1.0)},
            "verify_outcome": {"verified": True},
            "design_tool_spec": {"tool_spec_designed": True},
            "generate_tool_code": {"tool_code_generated": True},
            "validate_tool_code": {"tool_code_validated": True},
            "register_tool": {"tool_registered": True, "stage": "candidate"},
            "benchmark_tool": {"benchmarked": True},
            "promote_tool": {"tool_promoted": True},
            "review_episode_cluster": {
                "cluster_reviewed": True,
                "episode_count": workspace_state.get("episode_count", 0),
                "latest_pattern": workspace_state.get("latest_pattern_key"),
            },
            "refresh_skill_snapshot": {
                "skill_snapshot_refreshed": True,
                "latest_skill_score": workspace_state.get("latest_skill_score"),
            },
        }

        result.update(defaults.get(step_name, {}))

        if step_name == "upgrade_navigation_strategy":
            previous = self.grounded_navigation_strategy
            self.set_navigation_strategy("graph_search")
            result.update({
                "strategy_before": previous,
                "strategy_after": self.grounded_navigation_strategy,
                "strategy_upgraded": previous != self.grounded_navigation_strategy,
            })

        if step_name == "review_navigation_patch":
            result.update({
                "active_strategy": self.grounded_navigation_strategy,
                "graph_search_enabled": self.grounded_navigation_strategy == "graph_search",
            })

        if step_name == "retrieve_missing_knowledge" and workspace_state.get("knowledge_source_available") is False:
            result.update(
                {
                    "success": False,
                    "knowledge_gap": True,
                    "message": "Knowledge source unavailable.",
                }
            )

        if step_name == "replay_anchor_suite" and workspace_state.get("anchor_suite_failed", False):
            result.update(
                {
                    "success": False,
                    "regression_failure": True,
                    "message": "Anchor suite still failing.",
                }
            )

        if step_name in self.failure_map:
            result.update(self.failure_map[step_name])

        if (
            result.get("success") is False
            and "execution_error" not in result
            and "knowledge_gap" not in result
            and "regression_failure" not in result
            and "missing_capability" not in result
        ):
            result["execution_error"] = True

        return result
