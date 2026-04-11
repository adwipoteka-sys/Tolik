from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List


@dataclass
class KeyDoorTask:
    name: str
    layout: str
    runs: int = 0
    passes: int = 0
    last_ok: bool = False
    last_steps: int = 0


class KeyDoorTransferSuite:
    def __init__(self, storage_dir: str = "data/runtime") -> None:
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.storage_dir / "keydoor_transfer_suite.json"
        self.tasks: List[KeyDoorTask] = []
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            self.tasks = [KeyDoorTask(**item) for item in raw]

    def _save(self) -> None:
        self.path.write_text(
            json.dumps([asdict(t) for t in self.tasks], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def seed_defaults(self) -> None:
        if self.tasks:
            return
        self.tasks = [
            KeyDoorTask(name="corridor_keydoor", layout="corridor_kd"),
            KeyDoorTask(name="mirror_keydoor", layout="mirror_kd"),
            KeyDoorTask(name="detour_keydoor", layout="detour_kd"),
            KeyDoorTask(name="room_keydoor", layout="room_kd"),
        ]
        self._save()

    def list_tasks(self) -> List[Dict[str, object]]:
        return [asdict(t) for t in self.tasks]

    def run_all_with(self, runner) -> List[Dict[str, object]]:
        results: List[Dict[str, object]] = []
        for task in self.tasks:
            out = runner(task.layout)
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
