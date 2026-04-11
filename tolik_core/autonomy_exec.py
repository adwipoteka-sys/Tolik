from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional

from core.curriculum_manager import CurriculumManager
from main import TolikAGI
from compositional_exec import TolikCompositionalExec


@dataclass
class RepairRecord:
    family: str
    layout: str
    reason: str


class RepairBacklog:
    def __init__(self, storage_dir: str = "data/runtime") -> None:
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.storage_dir / "repair_backlog.json"
        self.items: List[RepairRecord] = []
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            self.items = [RepairRecord(**item) for item in raw]

    def _save(self) -> None:
        self.path.write_text(
            json.dumps([asdict(x) for x in self.items], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def add(self, family: str, layout: str, reason: str) -> None:
        self.items.append(RepairRecord(family=family, layout=layout, reason=reason))
        self._save()

    def list_items(self) -> List[Dict[str, object]]:
        return [asdict(x) for x in self.items]

    def clear(self) -> None:
        self.items = []
        self._save()


class TolikAutonomyExec:
    def __init__(self) -> None:
        module_root = Path(__file__).resolve().parent
        runtime_dir = module_root / "data" / "runtime"

        self.nav_exec = TolikAGI()
        self.kd_exec = TolikCompositionalExec()

        self.curriculum = CurriculumManager(storage_dir=str(runtime_dir))
        self.curriculum.seed_defaults()

        self.repair = RepairBacklog(storage_dir=str(runtime_dir))

    def _run_task(self, family: str, layout: str) -> Dict[str, object]:
        if family == "nav_pomdp":
            return self.nav_exec.run_partial_env_episode(layout)
        if family == "keydoor":
            return self.kd_exec.run_episode(layout)
        raise ValueError(f"Unknown family: {family}")

    @staticmethod
    def _intrinsic_reward(
        *,
        task_runs_before: int,
        ok: bool,
        reward_total: float,
        progress: float,
        difficulty: int,
    ) -> Dict[str, float]:
        exploration_bonus = 0.25 if task_runs_before == 0 else 0.0
        first_success_bonus = 0.35 if ok and task_runs_before == 0 else 0.0
        competence_progress_bonus = 1.5 * progress
        challenge_bonus = 0.05 * difficulty if ok else 0.0
        failure_penalty = -0.10 if not ok else 0.0

        intrinsic = (
            reward_total
            + exploration_bonus
            + first_success_bonus
            + competence_progress_bonus
            + challenge_bonus
            + failure_penalty
        )

        return {
            "exploration_bonus": exploration_bonus,
            "first_success_bonus": first_success_bonus,
            "competence_progress_bonus": competence_progress_bonus,
            "challenge_bonus": challenge_bonus,
            "failure_penalty": failure_penalty,
            "intrinsic_reward": intrinsic,
        }

    def autonomous_step(self) -> Dict[str, object]:
        task = self.curriculum.pick_next()
        if task is None:
            return {"status": "empty"}

        task_runs_before = int(task["runs"])
        result = self._run_task(task["family"], task["layout"])
        ok = bool(result["done"])
        reward_total = float(result["reward_total"])

        progress_info = self.curriculum.update_after_run(task["id"], ok=ok, reward_total=reward_total)
        intrinsic = self._intrinsic_reward(
            task_runs_before=task_runs_before,
            ok=ok,
            reward_total=reward_total,
            progress=progress_info["progress"],
            difficulty=int(task["difficulty"]),
        )

        if not ok:
            self.repair.add(
                family=task["family"],
                layout=task["layout"],
                reason=str(result["answer"]),
            )

        return {
            "task": task,
            "result": result,
            "progress": progress_info,
            "intrinsic": intrinsic,
        }

    def train(self, steps: int) -> List[Dict[str, object]]:
        out: List[Dict[str, object]] = []
        for _ in range(max(1, steps)):
            out.append(self.autonomous_step())
        return out

    def eval_all(self) -> Dict[str, List[Dict[str, object]]]:
        nav = self.nav_exec.transfer_suite.run_all_with(lambda layout: self.nav_exec.run_partial_env_episode(layout))
        kd = self.kd_exec.transfer.run_all_with(lambda layout: self.kd_exec.run_episode(layout))
        return {"nav": nav, "keydoor": kd}


def print_step(step: Dict[str, object]) -> None:
    if step.get("status") == "empty":
        print("No curriculum tasks.\n")
        return

    task = step["task"]
    result = step["result"]
    progress = step["progress"]
    intrinsic = step["intrinsic"]

    print(f"[TRAIN] {task['name']} family={task['family']} layout={task['layout']} difficulty={task['difficulty']}")
    print(f"done={result['done']} external_reward={result['reward_total']:.3f} intrinsic_reward={intrinsic['intrinsic_reward']:.3f}")
    print(f"progress={progress['progress']:.3f} before_mastery={progress['before_mastery']:.3f} after_mastery={progress['after_mastery']:.3f}")
    print(result["answer"])
    print()


def print_eval(results: Dict[str, List[Dict[str, object]]]) -> None:
    for family, rows in results.items():
        print(f"[EVAL] {family}")
        for row in rows:
            print(f"  - {row['task']} layout={row['layout']} :: {'PASS' if row['ok'] else 'FAIL'} steps={row['steps']} reward={row['reward_total']}")
        print()


def main() -> None:
    agi = TolikAutonomyExec()
    print("Tolik autonomy executive ready.")
    print("Commands: /curriculum_list, /train <n>, /repair_list, /repair_clear, /eval, /status, exit")

    while True:
        user_text = input("you> ").strip()
        if user_text.lower() in {"exit", "quit", "q"}:
            break

        if user_text == "/curriculum_list":
            for task in agi.curriculum.list_tasks():
                mastery = 0.0 if task["runs"] == 0 else task["passes"] / task["runs"]
                print(f"- {task['name']} family={task['family']} layout={task['layout']} diff={task['difficulty']} runs={task['runs']} passes={task['passes']} mastery={mastery:.2f}")
            print()
            continue

        if user_text.startswith("/train"):
            parts = user_text.split()
            steps = 1
            if len(parts) > 1:
                try:
                    steps = int(parts[1])
                except ValueError:
                    steps = 1
            results = agi.train(steps)
            for row in results:
                print_step(row)
            continue

        if user_text == "/repair_list":
            items = agi.repair.list_items()
            if not items:
                print("Repair backlog empty.\n")
                continue
            for item in items:
                print(f"- family={item['family']} layout={item['layout']} reason={item['reason']}")
            print()
            continue

        if user_text == "/repair_clear":
            agi.repair.clear()
            print("Repair backlog cleared.\n")
            continue

        if user_text == "/eval":
            results = agi.eval_all()
            print_eval(results)
            continue

        if user_text == "/status":
            print("Curriculum:", agi.curriculum.summary())
            print("Repairs:", len(agi.repair.list_items()))
            print()
            continue

        print("Unknown command.\n")


if __name__ == "__main__":
    main()
