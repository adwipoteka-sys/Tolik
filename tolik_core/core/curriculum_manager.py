from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class CurriculumTask:
    id: str
    family: str
    layout: str
    name: str
    difficulty: int
    runs: int = 0
    passes: int = 0
    total_reward: float = 0.0
    last_reward: float = 0.0
    last_ok: bool = False
    novelty_bias: float = 1.0


class CurriculumManager:
    def __init__(self, storage_dir: str = "data/runtime") -> None:
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.storage_dir / "curriculum_tasks.json"
        self.tasks: List[CurriculumTask] = []
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            self.tasks = [CurriculumTask(**item) for item in raw]

    def _save(self) -> None:
        self.path.write_text(
            json.dumps([asdict(t) for t in self.tasks], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def seed_defaults(self) -> None:
        if self.tasks:
            return

        defaults = [
            ("nav_pomdp", "easy", "nav_easy", 1),
            ("nav_pomdp", "detour", "nav_detour", 2),
            ("nav_pomdp", "mirror", "nav_mirror", 2),
            ("nav_pomdp", "corridor", "nav_corridor", 3),
            ("keydoor", "corridor_kd", "kd_corridor", 2),
            ("keydoor", "mirror_kd", "kd_mirror", 2),
            ("keydoor", "detour_kd", "kd_detour", 3),
            ("keydoor", "room_kd", "kd_room", 4),
        ]

        self.tasks = [
            CurriculumTask(
                id=str(uuid.uuid4())[:8],
                family=family,
                layout=layout,
                name=name,
                difficulty=difficulty,
            )
            for family, layout, name, difficulty in defaults
        ]
        self._save()

    def list_tasks(self) -> List[Dict[str, object]]:
        return [asdict(t) for t in self.tasks]

    @staticmethod
    def _mastery(task: CurriculumTask) -> float:
        if task.runs == 0:
            return 0.0
        return task.passes / task.runs

    def pick_next(self) -> Optional[Dict[str, object]]:
        if not self.tasks:
            return None

        # Prefer unseen tasks first
        unseen = [t for t in self.tasks if t.runs == 0]
        if unseen:
            unseen.sort(key=lambda t: (t.difficulty, t.name))
            return asdict(unseen[0])

        def score(t: CurriculumTask) -> float:
            mastery = self._mastery(t)
            avg_reward = t.total_reward / max(1, t.runs)
            failure_pressure = 1.0 - mastery
            reward_gap = max(0.0, 1.0 - avg_reward)
            novelty = t.novelty_bias
            challenge = 0.08 * t.difficulty
            return 0.50 * failure_pressure + 0.20 * reward_gap + 0.20 * novelty + challenge

        ranked = sorted(self.tasks, key=score, reverse=True)
        return asdict(ranked[0])

    def update_after_run(self, task_id: str, ok: bool, reward_total: float) -> Dict[str, float]:
        before_mastery = 0.0
        after_mastery = 0.0

        for t in self.tasks:
            if t.id == task_id:
                before_mastery = self._mastery(t)
                t.runs += 1
                t.passes += int(ok)
                t.total_reward += reward_total
                t.last_reward = reward_total
                t.last_ok = ok
                t.novelty_bias *= 0.7
                after_mastery = self._mastery(t)
                break

        self._save()
        return {
            "before_mastery": before_mastery,
            "after_mastery": after_mastery,
            "progress": max(0.0, after_mastery - before_mastery),
        }

    def summary(self) -> Dict[str, int]:
        return {
            "tasks": len(self.tasks),
            "runs": sum(t.runs for t in self.tasks),
            "passes": sum(t.passes for t in self.tasks),
            "unseen": sum(t.runs == 0 for t in self.tasks),
        }
