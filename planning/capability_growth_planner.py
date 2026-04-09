from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from memory.capability_graph import CapabilityGraph, CapabilityTransferEdge
from memory.capability_portfolio import CapabilityPortfolio
from memory.goal_ledger import GoalLedger
from motivation.goal_schema import Goal, GoalBudget, GoalKind, GoalSource, SuccessCriterion, new_goal_id


@dataclass(slots=True)
class CapabilityGrowthStep:
    source_capability: str
    target_capability: str
    relation_type: str
    strength: float
    strategic_value: float
    score: float
    depth: int
    rationale: str
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CapabilityGrowthStep":
        return cls(**dict(data))


@dataclass(slots=True)
class CapabilityGrowthPlan:
    plan_id: str
    root_capability: str
    steps: list[CapabilityGrowthStep]
    total_score: float
    horizon: int
    status: str = "proposed"
    active_step_index: int = 0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    evidence: dict[str, Any] = field(default_factory=dict)

    @property
    def path_targets(self) -> list[str]:
        return [step.target_capability for step in self.steps]

    @property
    def terminal_capability(self) -> str:
        return self.steps[-1].target_capability if self.steps else self.root_capability

    def next_step(self) -> CapabilityGrowthStep | None:
        if self.active_step_index >= len(self.steps):
            return None
        return self.steps[self.active_step_index]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["steps"] = [step.to_dict() for step in self.steps]
        payload["path_targets"] = self.path_targets
        payload["terminal_capability"] = self.terminal_capability
        payload["created_at"] = self.created_at.isoformat()
        payload["updated_at"] = self.updated_at.isoformat()
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CapabilityGrowthPlan":
        raw = dict(data)
        raw.pop("path_targets", None)
        raw.pop("terminal_capability", None)
        raw["steps"] = [CapabilityGrowthStep.from_dict(item) for item in raw["steps"]]
        raw["created_at"] = datetime.fromisoformat(raw["created_at"])
        raw["updated_at"] = datetime.fromisoformat(raw["updated_at"])
        return cls(**raw)


class CapabilityGrowthPlanner:
    """Plans multi-edge growth paths and tracks progress through them."""

    def __init__(self, ledger: GoalLedger | None = None) -> None:
        self.ledger = ledger
        self._plans: dict[str, CapabilityGrowthPlan] = {}
        if self.ledger is not None:
            self._rehydrate()

    def _rehydrate(self) -> None:
        for payload in self.ledger.load_capability_growth_plans():
            plan = CapabilityGrowthPlan.from_dict(payload)
            self._plans[plan.plan_id] = plan

    def _persist(self, plan: CapabilityGrowthPlan) -> CapabilityGrowthPlan:
        plan.updated_at = datetime.now(timezone.utc)
        self._plans[plan.plan_id] = plan
        if self.ledger is not None:
            self.ledger.save_capability_growth_plan(plan.to_dict())
        return plan

    def list_plans(self) -> list[CapabilityGrowthPlan]:
        return sorted(self._plans.values(), key=lambda item: (item.created_at, item.plan_id))

    def get(self, plan_id: str) -> CapabilityGrowthPlan | None:
        return self._plans.get(plan_id)

    def active_targets(self) -> set[str]:
        targets: set[str] = set()
        for plan in self._plans.values():
            if plan.status in {"admitted", "running"}:
                targets.update(plan.path_targets)
        return targets

    def propose(
        self,
        *,
        graph: CapabilityGraph,
        portfolio: CapabilityPortfolio,
        executable_targets: set[str],
        horizon: int = 2,
        limit: int = 3,
        discount: float = 0.85,
    ) -> list[CapabilityGrowthPlan]:
        ready = {state.capability for state in portfolio.list_states() if portfolio.ready_for_unattended_use(state.capability)}
        validated = {state.capability for state in portfolio.list_states() if state.maturity_stage == "transfer_validated"}
        blocked_targets = validated | self.active_targets()

        candidates: list[CapabilityGrowthPlan] = []
        for root in sorted(ready):
            for edge in graph.list_transfer_edges(source_capability=root):
                if edge.target_capability not in executable_targets or edge.target_capability in blocked_targets:
                    continue
                candidates.extend(
                    self._enumerate_paths(
                        graph=graph,
                        edge=edge,
                        executable_targets=executable_targets,
                        blocked_targets=blocked_targets,
                        visited={root},
                        horizon=horizon,
                        discount=discount,
                        depth=0,
                    )
                )

        ranked = sorted(candidates, key=lambda item: (-item.total_score, item.terminal_capability, item.plan_id))
        return [self._persist(plan) for plan in ranked[:limit]]

    def _enumerate_paths(
        self,
        *,
        graph: CapabilityGraph,
        edge: CapabilityTransferEdge,
        executable_targets: set[str],
        blocked_targets: set[str],
        visited: set[str],
        horizon: int,
        discount: float,
        depth: int,
    ) -> list[CapabilityGrowthPlan]:
        if edge.target_capability in visited:
            return []
        current_step = self._edge_to_step(edge=edge, depth=depth)
        plans = [self._build_plan(root_capability=edge.source_capability, steps=[current_step], discount=discount)]
        if depth + 1 >= horizon:
            return plans
        next_visited = set(visited)
        next_visited.add(edge.target_capability)
        for child in graph.list_transfer_edges(source_capability=edge.target_capability):
            if child.target_capability not in executable_targets or child.target_capability in blocked_targets or child.target_capability in next_visited:
                continue
            for child_plan in self._enumerate_paths(
                graph=graph,
                edge=child,
                executable_targets=executable_targets,
                blocked_targets=blocked_targets,
                visited=next_visited,
                horizon=horizon,
                discount=discount,
                depth=depth + 1,
            ):
                merged_steps = [CapabilityGrowthStep.from_dict(current_step.to_dict())]
                merged_steps.extend(CapabilityGrowthStep.from_dict(step.to_dict()) for step in child_plan.steps)
                plans.append(self._build_plan(root_capability=edge.source_capability, steps=merged_steps, discount=discount))
        return plans

    def _edge_to_step(self, *, edge: CapabilityTransferEdge, depth: int) -> CapabilityGrowthStep:
        score = round((edge.strength * 0.7) + (edge.strategic_value * 0.3), 3)
        rationale = (
            f"{edge.source_capability} can unlock {edge.target_capability} at depth {depth + 1}; "
            f"transfer strength {edge.strength:.2f}, strategic value {edge.strategic_value:.2f}."
        )
        return CapabilityGrowthStep(
            source_capability=edge.source_capability,
            target_capability=edge.target_capability,
            relation_type=edge.relation,
            strength=edge.strength,
            strategic_value=edge.strategic_value,
            score=score,
            depth=depth,
            rationale=rationale,
            evidence={
                "support_capabilities": list(edge.support_capabilities),
                "source_stage_required": edge.source_stage_required,
                "description": edge.description,
            },
        )

    def _build_plan(self, *, root_capability: str, steps: list[CapabilityGrowthStep], discount: float) -> CapabilityGrowthPlan:
        total = 0.0
        for step in steps:
            total += step.score * (discount ** step.depth)
        total += 0.05 * len(steps)
        total = round(total, 3)
        return CapabilityGrowthPlan(
            plan_id=f"growth_{uuid4().hex[:12]}",
            root_capability=root_capability,
            steps=steps,
            total_score=total,
            horizon=len(steps),
            evidence={"discount": discount, "path_targets": [step.target_capability for step in steps]},
        )

    def mark_status(self, plan_id: str, status: str, **extra_evidence: Any) -> CapabilityGrowthPlan:
        plan = self._plans[plan_id]
        plan.status = status
        if extra_evidence:
            plan.evidence.update(extra_evidence)
        return self._persist(plan)

    def mark_step_completed(self, plan_id: str, *, capability: str, passed: bool, run_id: str | None = None, score: float | None = None) -> CapabilityGrowthPlan:
        plan = self._plans[plan_id]
        step = plan.next_step()
        if step is None:
            raise ValueError(f"Growth plan {plan_id} has no remaining steps.")
        if step.target_capability != capability:
            raise ValueError(f"Expected {step.target_capability}, got {capability}.")
        plan.evidence.setdefault("step_outcomes", []).append({"capability": capability, "passed": passed, "run_id": run_id, "score": score})
        if passed:
            plan.active_step_index += 1
            plan.status = "completed" if plan.active_step_index >= len(plan.steps) else "running"
        else:
            plan.status = "failed"
        return self._persist(plan)

    def materialize_next_goal(
        self,
        plan: CapabilityGrowthPlan,
        *,
        tool_payload: dict[str, Any],
        overrides: dict[str, Any] | None = None,
        success_threshold: float = 0.99,
        source: GoalSource = GoalSource.CURRICULUM,
    ) -> Goal:
        step = plan.next_step()
        if step is None:
            raise ValueError(f"Growth plan {plan.plan_id} has no next step.")

        overrides = dict(overrides or {})
        evidence = {
            "curriculum_type": "capability_growth",
            "growth_plan_id": plan.plan_id,
            "path_root": plan.root_capability,
            "path_targets": list(plan.path_targets),
            "current_step_index": plan.active_step_index,
            "source_capability": step.source_capability,
            "target_capability": step.target_capability,
            "transfer_score": step.score,
            "tool_payload": tool_payload,
        }
        evidence.update(overrides.pop("evidence", {}))

        tags = ["grounded_navigation", "local_only"]
        extra_tags = overrides.pop("tags", [])
        tags = list(dict.fromkeys([*tags, *extra_tags]))

        resource_budget = overrides.pop("resource_budget", GoalBudget(max_steps=4, max_seconds=18.0, max_tool_calls=0, max_api_calls=0))
        if isinstance(resource_budget, dict):
            resource_budget = GoalBudget(**resource_budget)

        return Goal(
            goal_id=new_goal_id("growth"),
            title=f"Bootstrap {step.target_capability} along growth path from {plan.root_capability}",
            description=(
                f"Advance the multi-step capability growth path by building {step.target_capability} from "
                f"validated prerequisite {step.source_capability}."
            ),
            source=source,
            kind=GoalKind.LEARNING,
            expected_gain=overrides.pop("expected_gain", 0.88),
            novelty=overrides.pop("novelty", 0.49),
            uncertainty_reduction=overrides.pop("uncertainty_reduction", 0.85),
            strategic_fit=overrides.pop("strategic_fit", 0.95),
            risk_estimate=overrides.pop("risk_estimate", 0.07),
            priority=overrides.pop("priority", max(0.72, min(0.97, plan.total_score / max(1, len(plan.steps))))),
            risk_budget=overrides.pop("risk_budget", 0.16),
            resource_budget=resource_budget,
            success_criteria=[SuccessCriterion(metric="success_rate", comparator=">=", target=success_threshold)],
            required_capabilities=["classical_planning", step.source_capability, *step.evidence.get("support_capabilities", [])],
            tags=tags,
            evidence=evidence,
        )
