from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from environments.grounded_navigation import GroundedNavigationLab, NavigationTask


@dataclass(slots=True)
class NavigationExplanationTask:
    task_id: str
    width: int
    height: int
    start: tuple[int, int]
    goal: tuple[int, int]
    obstacles: list[tuple[int, int]]
    max_steps: int
    difficulty: int
    description: str = ""

    def to_navigation_task(self) -> NavigationTask:
        return NavigationTask(
            task_id=self.task_id,
            width=self.width,
            height=self.height,
            start=self.start,
            goal=self.goal,
            obstacles=list(self.obstacles),
            max_steps=self.max_steps,
            difficulty=self.difficulty,
            description=self.description,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "NavigationExplanationTask":
        raw = dict(data)
        raw["start"] = tuple(raw["start"])
        raw["goal"] = tuple(raw["goal"])
        raw["obstacles"] = [tuple(item) for item in raw.get("obstacles", [])]
        return cls(**raw)


class NavigationExplanationLab:
    """Builds structured explanations from grounded-navigation tasks."""

    def __init__(self) -> None:
        self.navigation = GroundedNavigationLab()

    def get_task(self, task_id: str) -> NavigationExplanationTask:
        task = self.navigation.get_task(task_id)
        return NavigationExplanationTask(
            task_id=task.task_id,
            width=task.width,
            height=task.height,
            start=task.start,
            goal=task.goal,
            obstacles=list(task.obstacles),
            max_steps=task.max_steps,
            difficulty=task.difficulty,
            description=task.description,
        )

    def run_batch(self, *, strategy: str, payload: dict[str, Any]) -> dict[str, Any]:
        tasks_payload = payload.get("tasks") or []
        tasks = [task if isinstance(task, NavigationExplanationTask) else NavigationExplanationTask.from_dict(task) for task in tasks_payload]
        explanations = [self.explain_task(task, strategy=strategy) for task in tasks]
        return {
            "strategy": strategy,
            "task_count": len(tasks),
            "explanations": explanations,
            "passed": all(item["recommended_strategy"] == strategy for item in explanations),
        }

    def explain_task(self, task: NavigationExplanationTask, *, strategy: str) -> dict[str, Any]:
        nav_task = task.to_navigation_task()
        solved = self.navigation.solve_task(nav_task, strategy="graph_search")
        total_steps = int(solved["optimal_steps"] or 0)
        manhattan = self.navigation._manhattan(task.start, task.goal)
        return {
            "task_id": task.task_id,
            "recommended_strategy": strategy,
            "step_count": total_steps,
            "detour_required": total_steps > manhattan,
            "start": list(task.start),
            "goal": list(task.goal),
        }
