from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from environments.grounded_navigation import GroundedNavigationLab, NavigationTask
from environments.spatial_route import RouteTask, SpatialRouteLab


@dataclass(slots=True)
class RouteBriefingTask:
    task_id: str
    width: int
    height: int
    start: tuple[int, int]
    checkpoints: list[tuple[int, int]]
    goal: tuple[int, int]
    obstacles: list[tuple[int, int]]
    max_steps: int
    difficulty: int
    description: str = ""

    def to_route_task(self) -> RouteTask:
        return RouteTask(
            task_id=self.task_id,
            width=self.width,
            height=self.height,
            start=self.start,
            checkpoints=list(self.checkpoints),
            goal=self.goal,
            obstacles=list(self.obstacles),
            max_steps=self.max_steps,
            difficulty=self.difficulty,
            description=self.description,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RouteBriefingTask":
        raw = dict(data)
        raw["start"] = tuple(raw["start"])
        raw["goal"] = tuple(raw["goal"])
        raw["checkpoints"] = [tuple(item) for item in raw.get("checkpoints", [])]
        raw["obstacles"] = [tuple(item) for item in raw.get("obstacles", [])]
        return cls(**raw)


class RouteBriefingLab:
    """Builds concise mission briefings from waypoint-composition tasks."""

    def __init__(self) -> None:
        self.route_lab = SpatialRouteLab()
        self.navigation = GroundedNavigationLab()

    def get_task(self, task_id: str) -> RouteBriefingTask:
        task = self.route_lab.get_task(task_id)
        return RouteBriefingTask(
            task_id=task.task_id,
            width=task.width,
            height=task.height,
            start=task.start,
            checkpoints=list(task.checkpoints),
            goal=task.goal,
            obstacles=list(task.obstacles),
            max_steps=task.max_steps,
            difficulty=task.difficulty,
            description=task.description,
        )

    def run_batch(self, *, strategy: str, payload: dict[str, Any]) -> dict[str, Any]:
        tasks_payload = payload.get("tasks") or []
        tasks = [task if isinstance(task, RouteBriefingTask) else RouteBriefingTask.from_dict(task) for task in tasks_payload]
        briefings = [self.brief_task(task, strategy=strategy) for task in tasks]
        return {
            "strategy": strategy,
            "task_count": len(tasks),
            "briefings": briefings,
            "passed": all(item["recommended_strategy"] == strategy for item in briefings),
        }

    def brief_task(self, task: RouteBriefingTask, *, strategy: str) -> dict[str, Any]:
        route_task = task.to_route_task()
        points = [route_task.start, *route_task.checkpoints, route_task.goal]
        total_steps = 0
        detour_legs = 0
        for start, goal in zip(points, points[1:]):
            leg = NavigationTask(
                task_id=f"{route_task.task_id}_{start}_{goal}",
                width=route_task.width,
                height=route_task.height,
                start=start,
                goal=goal,
                obstacles=list(route_task.obstacles),
                max_steps=route_task.max_steps,
                difficulty=route_task.difficulty,
                description=route_task.description,
            )
            optimal = self.navigation.solve_task(leg, strategy="graph_search")
            optimal_steps = int(optimal["optimal_steps"] or 0)
            total_steps += optimal_steps
            if optimal_steps > self.navigation._manhattan(start, goal):
                detour_legs += 1
        return {
            "task_id": route_task.task_id,
            "recommended_strategy": strategy,
            "leg_count": len(points) - 1,
            "checkpoint_count": len(route_task.checkpoints),
            "detour_legs": detour_legs,
            "total_optimal_steps": total_steps,
        }
