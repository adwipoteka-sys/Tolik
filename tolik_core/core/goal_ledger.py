from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class GoalRecord:
    id: str
    text: str
    priority: int
    source: str
    status: str = "pending"
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)
    notes: List[str] = field(default_factory=list)


class GoalLedger:
    def __init__(self, storage_dir: str = "data/runtime") -> None:
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.storage_dir / "goals.json"
        self.goals: List[GoalRecord] = []
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            self.goals = [GoalRecord(**item) for item in raw]

    def _save(self) -> None:
        self.path.write_text(
            json.dumps([asdict(g) for g in self.goals], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def add_goal(self, text: str, priority: int = 50, source: str = "user") -> Dict[str, object]:
        rec = GoalRecord(
            id=str(uuid.uuid4())[:8],
            text=text,
            priority=priority,
            source=source,
        )
        self.goals.append(rec)
        self._save()
        return asdict(rec)

    def list_goals(self, status: Optional[str] = None) -> List[Dict[str, object]]:
        items = self.goals
        if status:
            items = [g for g in items if g.status == status]
        items = sorted(items, key=lambda g: (-g.priority, g.created_at))
        return [asdict(g) for g in items]

    def start_next(self) -> Optional[Dict[str, object]]:
        pending = [g for g in self.goals if g.status == "pending"]
        if not pending:
            return None
        pending.sort(key=lambda g: (-g.priority, g.created_at))
        goal = pending[0]
        goal.status = "running"
        goal.updated_at = utc_now()
        self._save()
        return asdict(goal)

    def _mark(self, goal_id: str, status: str, note: str = "") -> None:
        for g in self.goals:
            if g.id == goal_id:
                g.status = status
                g.updated_at = utc_now()
                if note:
                    g.notes.append(note)
                break
        self._save()

    def mark_done(self, goal_id: str, note: str = "") -> None:
        self._mark(goal_id, "done", note)

    def mark_failed(self, goal_id: str, note: str = "") -> None:
        self._mark(goal_id, "failed", note)

    def stats(self) -> Dict[str, int]:
        return {
            "pending": sum(g.status == "pending" for g in self.goals),
            "running": sum(g.status == "running" for g in self.goals),
            "done": sum(g.status == "done" for g in self.goals),
            "failed": sum(g.status == "failed" for g in self.goals),
            "total": len(self.goals),
        }
