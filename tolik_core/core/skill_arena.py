from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List


@dataclass
class ArenaTask:
    id: str
    name: str
    prompt: str
    required: List[str]
    runs: int = 0
    passes: int = 0
    last_ok: bool = False
    last_answer: str = ""


class SkillArena:
    def __init__(self, storage_dir: str = "data/runtime") -> None:
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.storage_dir / "skill_arena.json"
        self.tasks: List[ArenaTask] = []
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            self.tasks = [ArenaTask(**item) for item in raw]

    def _save(self) -> None:
        self.path.write_text(
            json.dumps([asdict(t) for t in self.tasks], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def add_task(self, name: str, prompt: str, required: List[str]) -> Dict[str, object]:
        task = ArenaTask(
            id=str(uuid.uuid4())[:8],
            name=name,
            prompt=prompt,
            required=required,
        )
        self.tasks.append(task)
        self._save()
        return asdict(task)

    def list_tasks(self) -> List[Dict[str, object]]:
        return [asdict(t) for t in self.tasks]

    def run_all(self, agi) -> List[Dict[str, object]]:
        results: List[Dict[str, object]] = []

        for task in self.tasks:
            cycle = agi.run_cycle(task.prompt)
            answer = cycle["answer"]
            answer_l = answer.lower()
            required = [r.lower() for r in task.required if r.strip()]
            ok = all(r in answer_l for r in required)

            task.runs += 1
            task.passes += int(ok)
            task.last_ok = ok
            task.last_answer = answer

            results.append(
                {
                    "task": task.name,
                    "ok": ok,
                    "required": required,
                    "answer": answer,
                }
            )

        self._save()
        return results

    def summary(self) -> Dict[str, int]:
        return {
            "tasks": len(self.tasks),
            "runs": sum(t.runs for t in self.tasks),
            "passes": sum(t.passes for t in self.tasks),
        }
