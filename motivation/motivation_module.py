from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from motivation.autonomous_goal_manager import AutonomousGoalManager
from motivation.goal_schema import Goal


@dataclass
class MotivationModule:
    """Compatibility facade preserving the older motivation API."""

    autonomous_manager: AutonomousGoalManager

    def add_goal(self, goal: str | Goal, internal: bool = False) -> Goal | list[Goal]:
        if isinstance(goal, str):
            return self.autonomous_manager.ingest_external_goal(goal)
        return self.autonomous_manager.admit_candidates([goal], context={})

    def next_goal(self) -> Goal | None:
        return self.autonomous_manager.select_next_goal(context={})

    def has_goals(self) -> bool:
        return bool(self.autonomous_manager.pending or self.autonomous_manager.active)

    def mark_done(self, goal: Goal | str) -> None:
        goal_id = goal.goal_id if isinstance(goal, Goal) else str(goal)
        if goal_id in {item.goal_id for item in self.autonomous_manager.active} | {item.goal_id for item in self.autonomous_manager.pending}:
            self.autonomous_manager.complete_goal(goal_id, {"success": True})
