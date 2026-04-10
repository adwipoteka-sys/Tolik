from __future__ import annotations

from typing import Dict, List, Optional


class MotivationModule:
    """Manages external and internal goals."""

    def __init__(self) -> None:
        self.goal_queue: List[str] = []

    def add_goal(self, goal: Optional[str]) -> None:
        if goal and goal not in self.goal_queue:
            self.goal_queue.append(goal)

    def ingest_perception(self, perception: Dict[str, object]) -> None:
        self.add_goal(perception.get("suggested_goal"))

    def ingest_metacognition(self, recommendations: List[str]) -> None:
        for goal in recommendations:
            self.add_goal(goal)

    def next_goal(self) -> Optional[str]:
        if not self.goal_queue:
            return None
        return self.goal_queue.pop(0)
