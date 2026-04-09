from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from memory.goal_ledger import GoalLedger
from motivation.goal_schema import Goal, GoalStatus


def new_schedule_id(prefix: str = "schedule") -> str:
    return f"{prefix}_{uuid4().hex[:12]}"


@dataclass(slots=True)
class ScheduledGoal:
    schedule_id: str
    goal: Goal
    due_cycle: int
    recurrence_interval: int | None = None
    reason: str = ""
    status: str = "scheduled"
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    released_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["goal"] = self.goal.to_dict()
        payload["created_at"] = self.created_at.isoformat()
        payload["released_at"] = self.released_at.isoformat() if self.released_at else None
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ScheduledGoal":
        raw = dict(data)
        raw["goal"] = Goal.from_dict(raw["goal"])
        raw["created_at"] = datetime.fromisoformat(raw["created_at"])
        raw["released_at"] = datetime.fromisoformat(raw["released_at"]) if raw.get("released_at") else None
        return cls(**raw)


class LongHorizonScheduler:
    """Stores future goals and releases them when their cycle becomes due."""

    def __init__(self, ledger: GoalLedger | None = None) -> None:
        self.ledger = ledger
        self._scheduled: dict[str, ScheduledGoal] = {}
        if self.ledger is not None:
            self._rehydrate()

    def _rehydrate(self) -> None:
        for payload in self.ledger.load_scheduled_goals():
            item = ScheduledGoal.from_dict(payload)
            self._scheduled[item.schedule_id] = item

    def _persist(self, item: ScheduledGoal) -> ScheduledGoal:
        self._scheduled[item.schedule_id] = item
        if self.ledger is not None:
            self.ledger.save_scheduled_goal(item.to_dict())
        return item

    def schedule(
        self,
        goal: Goal,
        *,
        due_cycle: int,
        reason: str,
        recurrence_interval: int | None = None,
    ) -> ScheduledGoal:
        if due_cycle <= 0:
            raise ValueError("due_cycle must be positive")
        goal.status = GoalStatus.DEFERRED
        item = ScheduledGoal(
            schedule_id=new_schedule_id(),
            goal=goal,
            due_cycle=due_cycle,
            recurrence_interval=recurrence_interval,
            reason=reason,
        )
        return self._persist(item)

    def pending(self) -> list[ScheduledGoal]:
        return [entry for entry in sorted(self._scheduled.values(), key=lambda entry: entry.due_cycle) if entry.status == "scheduled"]

    def release_due(self, *, current_cycle: int, existing_goal_titles: set[str] | None = None) -> list[Goal]:
        normalized_titles = {" ".join(title.lower().split()) for title in (existing_goal_titles or set())}
        released: list[Goal] = []
        for item in sorted(self._scheduled.values(), key=lambda entry: entry.due_cycle):
            if item.status != "scheduled":
                continue
            if item.due_cycle > current_cycle:
                continue
            normalized_title = " ".join(item.goal.title.lower().split())
            if normalized_title in normalized_titles:
                continue
            normalized_titles.add(normalized_title)
            item.goal.status = GoalStatus.PENDING
            item.released_at = datetime.now(timezone.utc)
            released.append(item.goal)
            if item.recurrence_interval is None:
                item.status = "released"
            else:
                item.due_cycle += item.recurrence_interval
                item.goal.status = GoalStatus.DEFERRED
            self._persist(item)
        return released

    def list_all(self) -> list[ScheduledGoal]:
        return sorted(self._scheduled.values(), key=lambda entry: (entry.due_cycle, entry.schedule_id))
