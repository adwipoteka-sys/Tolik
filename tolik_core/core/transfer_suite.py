from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List


@dataclass
class TransferTask:
    name: str
    layout: str
    success_required: bool = True
    runs: int = 0
    passes: int = 0
    last_ok: bool = False
    last_steps: int = 0


class TransferSuite:
    def __init__(self, storage_dir: str = "data/runtime") -> None:
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.storage_dir / "transfer_suite.json"
        self.tasks: List[TransferTask] = []
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            self.tasks = [TransferTask(**item) for item in raw]

    def _save(self) -> None:
        self.path.write_text(
            json.dumps([asdict(t) for t in self.tasks], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def seed_defaults(self) -> None:
        if self.tasks:
            return
        self.tasks = [
            TransferTask(name="easy_nav", layout="easy"),
            TransferTask(name="detour_nav", layout="detour"),
            TransferTask(name="mirror_nav", layout="mirror"),
            TransferTask(name="corridor_nav", layout="corridor"),
        ]
        self._save()

    def list_tasks(self) -> List[Dict[str, object]]:
        return [asdict(t) for t in self.tasks]

    def run_all(self, agi) -> List[Dict[str, object]]:
        results: List[Dict[str, object]] = []
        for task in self.tasks:
            out = agi.run_env_episode(task.layout)
            ok = bool(out["done"])

            task.runs += 1
            task.passes += int(ok)
            task.last_ok = ok
            task.last_steps = len(out["executed_actions"])

            results.append(
                {
                    "task": task.name,
                    "layout": task.layout,
                    "ok": ok,
                    "steps": len(out["executed_actions"]),
                    "reward_total": out["reward_total"],
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
