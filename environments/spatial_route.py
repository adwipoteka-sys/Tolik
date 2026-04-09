from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from environments.grounded_navigation import GroundedNavigationLab, NavigationTask


@dataclass(slots=True)
class RouteTask:
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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RouteTask":
        raw = dict(data)
        raw["start"] = tuple(raw["start"])
        raw["goal"] = tuple(raw["goal"])
        raw["checkpoints"] = [tuple(item) for item in raw.get("checkpoints", [])]
        raw["obstacles"] = [tuple(item) for item in raw.get("obstacles", [])]
        return cls(**raw)


class SpatialRouteLab:
    """Deterministic waypoint-composition tasks built from grounded navigation legs."""

    def __init__(self) -> None:
        self.navigation = GroundedNavigationLab()
        self._catalog: dict[str, RouteTask] = {
            "route_train_open_chain": RouteTask(
                task_id="route_train_open_chain",
                width=5,
                height=5,
                start=(0, 0),
                checkpoints=[(2, 0), (2, 2)],
                goal=(4, 2),
                obstacles=[],
                max_steps=18,
                difficulty=1,
                description="Open waypoint chain with no detours.",
            ),
            "route_train_detour_chain": RouteTask(
                task_id="route_train_detour_chain",
                width=6,
                height=5,
                start=(0, 0),
                checkpoints=[(2, 0), (2, 3)],
                goal=(5, 3),
                obstacles=[(1, 0), (1, 1), (1, 2), (3, 1), (3, 2)],
                max_steps=24,
                difficulty=2,
                description="Waypoint chain requiring a grounded detour on early and late legs.",
            ),
            "route_transfer_switchback": RouteTask(
                task_id="route_transfer_switchback",
                width=7,
                height=6,
                start=(0, 5),
                checkpoints=[(2, 5), (2, 1), (4, 1)],
                goal=(6, 1),
                obstacles=[(1, 2), (1, 3), (1, 4), (3, 2), (3, 3), (5, 3), (5, 4)],
                max_steps=32,
                difficulty=3,
                description="Held-out multi-checkpoint route with repeated switchbacks.",
            ),
            "route_transfer_bridge_loop": RouteTask(
                task_id="route_transfer_bridge_loop",
                width=8,
                height=6,
                start=(0, 0),
                checkpoints=[(3, 0), (3, 4), (6, 4)],
                goal=(7, 1),
                obstacles=[(1, 0), (1, 1), (1, 2), (1, 3), (4, 1), (4, 2), (4, 3), (6, 1), (6, 2)],
                max_steps=36,
                difficulty=3,
                description="Held-out bridge-loop route with late-stage detour recovery.",
            ),
        }

    def get_task(self, task_id: str) -> RouteTask:
        try:
            return self._catalog[task_id]
        except KeyError as exc:
            raise KeyError(f"Unknown route task: {task_id}") from exc

    def sample_curriculum(self, *, batch_size: int, max_difficulty: int) -> list[RouteTask]:
        tasks = [task for task in self._catalog.values() if task.difficulty <= max_difficulty]
        tasks.sort(key=lambda item: (item.difficulty, item.task_id))
        if not tasks:
            return []
        repeated: list[RouteTask] = []
        idx = 0
        while len(repeated) < batch_size:
            repeated.append(tasks[idx % len(tasks)])
            idx += 1
        return repeated[:batch_size]

    def run_batch(self, *, strategy: str, payload: dict[str, Any]) -> dict[str, Any]:
        tasks_payload = payload.get("tasks")
        if tasks_payload is not None:
            tasks = [task if isinstance(task, RouteTask) else RouteTask.from_dict(task) for task in tasks_payload]
        else:
            batch_size = int(payload.get("batch_size", 2))
            max_difficulty = int(payload.get("max_difficulty", 2))
            tasks = self.sample_curriculum(batch_size=batch_size, max_difficulty=max_difficulty)

        threshold = float(payload.get("success_threshold", 1.0))
        task_results: list[dict[str, Any]] = []
        successes = 0
        ratios: list[float] = []
        for task in tasks:
            solved = self.solve_task(task, strategy=strategy)
            task_results.append(solved)
            if solved["success"]:
                successes += 1
                if solved["optimal_steps"]:
                    ratios.append(round(solved["steps"] / solved["optimal_steps"], 3))

        total = len(tasks)
        success_rate = round(successes / total, 3) if total else 0.0
        mean_path_ratio = round(sum(ratios) / len(ratios), 3) if ratios else None
        return {
            "strategy": strategy,
            "task_count": total,
            "successes": successes,
            "success_rate": success_rate,
            "mean_path_ratio": mean_path_ratio,
            "passed": success_rate >= threshold,
            "threshold": threshold,
            "task_results": task_results,
        }

    def solve_task(self, task: RouteTask, *, strategy: str) -> dict[str, Any]:
        points = [task.start, *task.checkpoints, task.goal]
        leg_results: list[dict[str, Any]] = []
        all_actions: list[str] = []
        total_steps = 0
        total_optimal_steps = 0
        success = True
        detour_legs = 0
        for idx, (start, goal) in enumerate(zip(points, points[1:]), start=1):
            nav_task = NavigationTask(
                task_id=f"{task.task_id}_leg{idx}",
                width=task.width,
                height=task.height,
                start=start,
                goal=goal,
                obstacles=list(task.obstacles),
                max_steps=task.max_steps,
                difficulty=task.difficulty,
                description=task.description,
            )
            solved = self.navigation.solve_task(nav_task, strategy=strategy)
            optimal = self.navigation.solve_task(nav_task, strategy="graph_search")
            leg = {
                "leg_index": idx,
                "start": start,
                "goal": goal,
                "strategy": strategy,
                "success": bool(solved["success"]),
                "steps": solved["steps"],
                "actions": list(solved.get("actions", [])),
                "optimal_steps": optimal["steps"],
                "requires_detour": bool(optimal.get("requires_detour") and optimal.get("steps") and optimal["steps"] > self.navigation._manhattan(start, goal)),
            }
            leg_results.append(leg)
            if not leg["success"]:
                success = False
                break
            all_actions.extend(leg["actions"])
            total_steps += int(leg["steps"] or 0)
            total_optimal_steps += int(leg["optimal_steps"] or 0)
            if leg["requires_detour"]:
                detour_legs += 1

        return {
            "task_id": task.task_id,
            "difficulty": task.difficulty,
            "strategy": strategy,
            "success": success,
            "steps": total_steps if success else None,
            "actions": all_actions if success else [],
            "optimal_steps": total_optimal_steps if success else None,
            "leg_count": len(points) - 1,
            "checkpoint_count": len(task.checkpoints),
            "detour_legs": detour_legs,
            "task_results": leg_results,
        }
