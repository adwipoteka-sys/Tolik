from __future__ import annotations

from dataclasses import dataclass

from memory.capability_graph import CapabilityGraph, CapabilityTransferEdge
from motivation.goal_schema import Goal, GoalBudget, GoalKind, GoalSource, SuccessCriterion, new_goal_id
from environments.grounded_navigation import GroundedNavigationLab


@dataclass(slots=True)
class TransferCurriculumTask:
    source_capability: str
    target_capability: str
    title: str
    description: str
    budget: GoalBudget
    success_criteria: list[SuccessCriterion]
    tags: list[str]
    required_capabilities: list[str]
    evidence: dict[str, object]


class TransferCurriculumBuilder:
    """Builds next-skill curriculum goals from the capability graph."""

    def __init__(self) -> None:
        self.lab = GroundedNavigationLab()

    def build(self, graph: CapabilityGraph, *, existing_titles: set[str] | None = None) -> list[TransferCurriculumTask]:
        tasks: list[TransferCurriculumTask] = []
        for edge in graph.suggest_transfer_edges(existing_titles=existing_titles):
            task = self._from_edge(edge)
            if task is not None:
                tasks.append(task)
        return tasks

    def _from_edge(self, edge: CapabilityTransferEdge) -> TransferCurriculumTask | None:
        if edge.target_capability == "navigation_route_explanation":
            return self._route_explanation_task(edge)
        return None

    def _route_explanation_task(self, edge: CapabilityTransferEdge) -> TransferCurriculumTask:
        practice_tasks = [
            self.lab.get_task("nav_detour_wall").to_dict(),
            self.lab.get_task("nav_detour_channel").to_dict(),
        ]
        return TransferCurriculumTask(
            source_capability=edge.source_capability,
            target_capability=edge.target_capability,
            title="Bootstrap navigation route explanation from grounded navigation",
            description="Reuse transfer-validated grounded navigation traces to build a new route-explanation skill.",
            budget=GoalBudget(max_steps=4, max_seconds=18.0, max_tool_calls=0, max_api_calls=0),
            success_criteria=[SuccessCriterion(metric="success_rate", comparator=">=", target=1.0)],
            tags=["transfer_curriculum", "grounded_navigation", "navigation_route_explanation", "local_only"],
            required_capabilities=[edge.source_capability, *edge.support_capabilities],
            evidence={
                "source_capability": edge.source_capability,
                "target_capability": edge.target_capability,
                "tool_payload": {
                    "tasks": practice_tasks,
                    "success_threshold": 1.0,
                    "detour_explanation_threshold": 1.0,
                },
                "edge_relation": edge.relation,
                "edge_description": edge.description,
            },
        )


def transfer_task_to_goal(task: TransferCurriculumTask) -> Goal:
    return Goal(
        goal_id=new_goal_id("transfer"),
        title=task.title,
        description=task.description,
        source=GoalSource.CURRICULUM,
        kind=GoalKind.LEARNING,
        expected_gain=0.79,
        novelty=0.46,
        uncertainty_reduction=0.77,
        strategic_fit=0.90,
        risk_estimate=0.07,
        priority=0.82,
        risk_budget=0.18,
        resource_budget=task.budget,
        success_criteria=task.success_criteria,
        required_capabilities=list(task.required_capabilities),
        tags=list(task.tags),
        evidence=dict(task.evidence),
    )
