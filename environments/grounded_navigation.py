from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from typing import Any


Action = str
_COORD = tuple[int, int]
_DELTAS: dict[Action, tuple[int, int]] = {
    "up": (0, -1),
    "down": (0, 1),
    "left": (-1, 0),
    "right": (1, 0),
}


@dataclass(slots=True)
class NavigationTask:
    task_id: str
    width: int
    height: int
    start: _COORD
    goal: _COORD
    obstacles: list[_COORD]
    max_steps: int
    difficulty: int
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "NavigationTask":
        raw = dict(data)
        raw["start"] = tuple(raw["start"])
        raw["goal"] = tuple(raw["goal"])
        raw["obstacles"] = [tuple(item) for item in raw.get("obstacles", [])]
        return cls(**raw)


class GroundedNavigationLab:
    """Deterministic micro-world used to ground autonomous planning practice."""

    def __init__(self) -> None:
        self._catalog: dict[int, list[NavigationTask]] = {
            1: [
                NavigationTask(
                    task_id="nav_easy_open",
                    width=3,
                    height=3,
                    start=(0, 0),
                    goal=(2, 2),
                    obstacles=[],
                    max_steps=8,
                    difficulty=1,
                    description="Open 3x3 grid.",
                ),
                NavigationTask(
                    task_id="nav_easy_corner",
                    width=4,
                    height=4,
                    start=(0, 3),
                    goal=(3, 0),
                    obstacles=[(1, 1)],
                    max_steps=10,
                    difficulty=1,
                    description="Mostly open grid with one obstacle.",
                ),
            ],
            2: [
                NavigationTask(
                    task_id="nav_detour_wall",
                    width=4,
                    height=4,
                    start=(0, 0),
                    goal=(3, 0),
                    obstacles=[(1, 0), (1, 1), (1, 2)],
                    max_steps=12,
                    difficulty=2,
                    description="Requires a temporary move away from goal.",
                ),
                NavigationTask(
                    task_id="nav_detour_channel",
                    width=5,
                    height=4,
                    start=(0, 0),
                    goal=(4, 0),
                    obstacles=[(1, 0), (1, 1), (3, 1), (3, 2)],
                    max_steps=14,
                    difficulty=2,
                    description="Requires a corridor detour.",
                ),
            ],
            3: [
                NavigationTask(
                    task_id="nav_transfer_bridge",
                    width=6,
                    height=5,
                    start=(0, 0),
                    goal=(5, 0),
                    obstacles=[(1, 0), (1, 1), (1, 2), (1, 3)],
                    max_steps=20,
                    difficulty=3,
                    description="Held-out bridge task: a long detour away from the goal is required.",
                ),
                NavigationTask(
                    task_id="nav_transfer_double_wall",
                    width=7,
                    height=5,
                    start=(0, 4),
                    goal=(6, 4),
                    obstacles=[(1, 1), (1, 2), (1, 3), (1, 4), (3, 0), (3, 1), (3, 2), (3, 3), (5, 1), (5, 2), (5, 3), (5, 4)],
                    max_steps=30,
                    difficulty=3,
                    description="Held-out multi-detour task requiring repeated route reversals.",
                ),
            ],
        }


    def get_task(self, task_id: str) -> NavigationTask:
        for tasks in self._catalog.values():
            for task in tasks:
                if task.task_id == task_id:
                    return task
        raise KeyError(f"Unknown navigation task: {task_id}")

    def sample_curriculum(self, *, batch_size: int, max_difficulty: int) -> list[NavigationTask]:
        tasks: list[NavigationTask] = []
        for difficulty in sorted(self._catalog):
            if difficulty > max_difficulty:
                continue
            tasks.extend(self._catalog[difficulty])
        if not tasks:
            return []
        repeated: list[NavigationTask] = []
        idx = 0
        while len(repeated) < batch_size:
            repeated.append(tasks[idx % len(tasks)])
            idx += 1
        return repeated[:batch_size]

    def run_batch(self, *, strategy: str, payload: dict[str, Any]) -> dict[str, Any]:
        tasks_payload = payload.get("tasks")
        if tasks_payload is not None:
            tasks = [task if isinstance(task, NavigationTask) else NavigationTask.from_dict(task) for task in tasks_payload]
        else:
            batch_size = int(payload.get("batch_size", 3))
            max_difficulty = int(payload.get("max_difficulty", 2))
            tasks = self.sample_curriculum(batch_size=batch_size, max_difficulty=max_difficulty)

        threshold = float(payload.get("success_threshold", 1.0))
        case_results: list[dict[str, Any]] = []
        successes = 0
        ratios: list[float] = []
        for task in tasks:
            solved = self.solve_task(task, strategy=strategy)
            case_results.append(solved)
            if solved["success"]:
                successes += 1
                if solved["optimal_steps"]:
                    ratios.append(round(solved["steps"] / solved["optimal_steps"], 3))

        total = len(tasks)
        success_rate = round(successes / total, 3) if total else 0.0
        mean_path_ratio = round(sum(ratios) / len(ratios), 3) if ratios else None
        confidence = round(success_rate if mean_path_ratio is None else max(min(success_rate * (2.0 - mean_path_ratio), 1.0), 0.0), 3)
        failure_reason = None if success_rate >= threshold else ("tool_missing_hint" if strategy == "greedy" and any(item.get("requires_detour") for item in case_results) else "transfer_deficit")
        return {
            "strategy": strategy,
            "task_count": total,
            "successes": successes,
            "success_rate": success_rate,
            "mean_path_ratio": mean_path_ratio,
            "confidence": confidence,
            "generalization_gap": round(max(0.0, threshold - success_rate), 3),
            "failure_reason": failure_reason,
            "passed": success_rate >= threshold,
            "threshold": threshold,
            "task_results": case_results,
        }

    def explain_batch(self, *, strategy: str, payload: dict[str, Any]) -> dict[str, Any]:
        tasks_payload = payload.get("tasks")
        if tasks_payload is not None:
            tasks = [task if isinstance(task, NavigationTask) else NavigationTask.from_dict(task) for task in tasks_payload]
        else:
            batch_size = int(payload.get("batch_size", 2))
            max_difficulty = int(payload.get("max_difficulty", 2))
            tasks = self.sample_curriculum(batch_size=batch_size, max_difficulty=max_difficulty)

        success_threshold = float(payload.get("success_threshold", 1.0))
        detour_threshold = float(payload.get("detour_explanation_threshold", 1.0))
        case_results: list[dict[str, Any]] = []
        successes = 0
        ratios: list[float] = []
        detour_required = 0
        detour_explained = 0
        for task in tasks:
            solved = self.solve_task(task, strategy=strategy)
            explained = self.explain_task(task, solved)
            case_results.append(explained)
            if explained["success"]:
                successes += 1
                if explained["optimal_steps"]:
                    ratios.append(round(explained["steps"] / explained["optimal_steps"], 3))
            if explained["requires_detour"]:
                detour_required += 1
                if explained["mentions_detour"]:
                    detour_explained += 1

        total = len(tasks)
        success_rate = round(successes / total, 3) if total else 0.0
        mean_path_ratio = round(sum(ratios) / len(ratios), 3) if ratios else None
        detour_explanation_rate = round(detour_explained / detour_required, 3) if detour_required else 1.0
        return {
            "strategy": strategy,
            "task_count": total,
            "successes": successes,
            "success_rate": success_rate,
            "mean_path_ratio": mean_path_ratio,
            "detour_required_count": detour_required,
            "detour_explained_count": detour_explained,
            "detour_explanation_rate": detour_explanation_rate,
            "passed": success_rate >= success_threshold and detour_explanation_rate >= detour_threshold,
            "threshold": success_threshold,
            "detour_threshold": detour_threshold,
            "task_results": case_results,
        }

    def explain_task(self, task: NavigationTask, solved: dict[str, Any]) -> dict[str, Any]:
        actions = list(solved.get("actions", []))
        positions = self._trace_positions(task.start, actions)
        took_detour = self._path_requires_detour_reasoning(positions, task.goal)
        if not solved.get("success"):
            explanation = "No valid route was found from the grounded start state."
            mentions_detour = False
        elif solved.get("requires_detour") or took_detour:
            via = positions[1:-1]
            via_preview = ", ".join(str(point) for point in via[:3]) if via else str(task.start)
            explanation = (
                f"The direct path is blocked by obstacles, so the route makes a detour via {via_preview} "
                f"before returning toward the goal {task.goal}."
            )
            mentions_detour = "detour" in explanation and "blocked" in explanation
        else:
            explanation = f"The route moves directly from {task.start} to {task.goal} in {len(actions)} steps."
            mentions_detour = False
        return {
            **solved,
            "path_positions": positions,
            "explanation": explanation,
            "mentions_detour": mentions_detour,
            "took_detour": took_detour,
        }

    def solve_task(self, task: NavigationTask, *, strategy: str) -> dict[str, Any]:
        if strategy == "graph_search":
            actions = self._bfs_actions(task)
        elif strategy == "greedy":
            actions = self._greedy_actions(task)
        else:
            raise ValueError(f"Unsupported navigation strategy: {strategy}")

        optimal_actions = self._bfs_actions(task)
        optimal_steps = len(optimal_actions) if optimal_actions is not None else None
        success = actions is not None
        requires_detour = task.difficulty >= 2
        confidence = 1.0 if success and optimal_steps is not None and len(actions or []) == optimal_steps else (0.45 if success else 0.15)
        failure_reason = None if success else ("planner_overexpansion" if strategy == "greedy" and requires_detour else "path_not_found")
        ambiguity_level = 0.5 if requires_detour else 0.1
        stability_signal = 1.0 if success else 0.0
        return {
            "task_id": task.task_id,
            "difficulty": task.difficulty,
            "strategy": strategy,
            "success": success,
            "steps": len(actions) if actions is not None else None,
            "actions": list(actions) if actions is not None else [],
            "optimal_steps": optimal_steps,
            "requires_detour": requires_detour,
            "confidence": confidence,
            "failure_reason": failure_reason,
            "ambiguity_level": ambiguity_level,
            "stability_signal": stability_signal,
            "confidence_breakdown": {"route_found": 1.0 if success else 0.0, "optimality": 1.0 if success and optimal_steps is not None and len(actions or []) == optimal_steps else 0.0},
        }

    def _greedy_actions(self, task: NavigationTask) -> list[Action] | None:
        current = task.start
        goal = task.goal
        actions: list[Action] = []
        visited: set[_COORD] = {current}
        for _ in range(task.max_steps):
            if current == goal:
                return actions
            current_distance = self._manhattan(current, goal)
            candidates: list[tuple[int, Action, _COORD]] = []
            for action, nxt in self._neighbors(task, current):
                if nxt in visited:
                    continue
                new_distance = self._manhattan(nxt, goal)
                # greedy policy refuses temporary detours that worsen distance
                if new_distance <= current_distance:
                    candidates.append((new_distance, action, nxt))
            if not candidates:
                return None
            candidates.sort(key=lambda item: (item[0], item[1]))
            _dist, action, nxt = candidates[0]
            actions.append(action)
            visited.add(nxt)
            current = nxt
        return None

    def _bfs_actions(self, task: NavigationTask) -> list[Action] | None:
        start, goal = task.start, task.goal
        if start == goal:
            return []
        queue: deque[_COORD] = deque([start])
        parents: dict[_COORD, tuple[_COORD, Action] | None] = {start: None}
        while queue:
            current = queue.popleft()
            if current == goal:
                break
            for action, nxt in self._neighbors(task, current):
                if nxt in parents:
                    continue
                parents[nxt] = (current, action)
                queue.append(nxt)
        if goal not in parents:
            return None
        actions: list[Action] = []
        node = goal
        while parents[node] is not None:
            parent, action = parents[node]  # type: ignore[misc]
            actions.append(action)
            node = parent
        actions.reverse()
        return actions

    @staticmethod
    def _trace_positions(start: _COORD, actions: list[Action]) -> list[_COORD]:
        positions = [start]
        current = start
        for action in actions:
            dx, dy = _DELTAS[action]
            current = (current[0] + dx, current[1] + dy)
            positions.append(current)
        return positions

    @staticmethod
    def _path_requires_detour_reasoning(positions: list[_COORD], goal: _COORD) -> bool:
        if len(positions) < 2:
            return False
        previous = GroundedNavigationLab._manhattan(positions[0], goal)
        for point in positions[1:]:
            current = GroundedNavigationLab._manhattan(point, goal)
            if current > previous:
                return True
            previous = current
        return False

    def _neighbors(self, task: NavigationTask, position: _COORD) -> list[tuple[Action, _COORD]]:
        x, y = position
        blocked = set(task.obstacles)
        neighbors: list[tuple[Action, _COORD]] = []
        for action, (dx, dy) in _DELTAS.items():
            nxt = (x + dx, y + dy)
            if not (0 <= nxt[0] < task.width and 0 <= nxt[1] < task.height):
                continue
            if nxt in blocked:
                continue
            neighbors.append((action, nxt))
        return neighbors

    @staticmethod
    def _manhattan(a: _COORD, b: _COORD) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
