from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from core.global_workspace import GlobalWorkspace
from core.object_map_memory import ObjectMapMemory
from core.keydoor_transfer_suite import KeyDoorTransferSuite
from memory.memory_module import MemoryModule
from planning.keydoor_planner import KeyDoorPlanner
from sim.keydoor_world import KeyDoorWorld


class TolikCompositionalExec:
    def __init__(self) -> None:
        module_root = Path(__file__).resolve().parent
        runtime_dir = module_root / "data" / "runtime"

        self.workspace = GlobalWorkspace()
        self.memory = MemoryModule(storage_dir=str(runtime_dir))
        self.memory.seed_defaults()

        self.env = KeyDoorWorld()
        self.map_memory = ObjectMapMemory()
        self.planner = KeyDoorPlanner()
        self.transfer = KeyDoorTransferSuite(storage_dir=str(runtime_dir))
        self.transfer.seed_defaults()

        self.reset(self.env.layout_name)

    def reset(self, layout: str) -> None:
        self.env.reset(layout)
        self.map_memory.reset(layout, len(self.env.grid), len(self.env.grid[0]))

        saved = self.memory.recall_fact(f"keydoor_model::{layout}")
        if isinstance(saved, dict):
            self.map_memory.load_from_fact(saved)

    def _publish(self, topic: str, payload: Dict[str, object], source: str) -> None:
        self.workspace.publish(topic, payload, source=source)
        self.memory.remember_event({"type": topic, "data": payload})

    def run_episode(self, layout: Optional[str] = None, max_steps: int = 120) -> Dict[str, object]:
        if layout:
            self.reset(layout)

        executed_actions: List[str] = []
        reward_total = 0.0

        for _ in range(max_steps):
            local_obs = self.env.observe_local(radius=1)
            self.map_memory.update_from_local(local_obs)
            self.memory.store_fact(f"keydoor_model::{self.env.layout_name}", self.map_memory.to_fact())
            self._publish("keydoor_local_obs", local_obs, "perception")

            plan = self.planner.plan(self.map_memory, tuple(local_obs["agent"]))
            self._publish("keydoor_plan", {"layout": self.env.layout_name, "plan": plan}, "planning")

            if not plan:
                answer = f"Не найден план в layout={self.env.layout_name}. Нужно улучшать world model."
                meta = {"cycle_ok": False, "confidence": 0.2, "recommendations": [f"repair_keydoor::{self.env.layout_name}"]}
                return {
                    "goal": f"keydoor_goal@{self.env.layout_name}",
                    "answer": answer,
                    "meta": meta,
                    "done": False,
                    "executed_actions": executed_actions,
                    "reward_total": reward_total,
                    "render_known": self.map_memory.render_known(self.env.agent),
                    "render_world": self.env.render(),
                }

            action = plan[0]
            step = self.env.step(action)
            executed_actions.append(action)
            reward_total += step.reward
            self._publish(
                "keydoor_step",
                {"layout": self.env.layout_name, "action": action, "reward": step.reward, "done": step.done, "info": step.info},
                "agency",
            )

            local_after = self.env.observe_local(radius=1)
            self.map_memory.update_from_local(local_after)
            self.memory.store_fact(f"keydoor_model::{self.env.layout_name}", self.map_memory.to_fact())

            if step.done:
                self.memory.store_fact(f"keydoor_policy::{self.env.layout_name}", " ".join(executed_actions))
                answer = (
                    f"KeyDoor-эпизод завершён успешно в layout={self.env.layout_name}. "
                    f"Шагов: {len(executed_actions)}. "
                    f"Ключ/дверь обработаны, внутренняя карта сохранена."
                )
                meta = {"cycle_ok": True, "confidence": 0.82, "recommendations": []}
                return {
                    "goal": f"keydoor_goal@{self.env.layout_name}",
                    "answer": answer,
                    "meta": meta,
                    "done": True,
                    "executed_actions": executed_actions,
                    "reward_total": reward_total,
                    "render_known": self.map_memory.render_known(self.env.agent),
                    "render_world": self.env.render(),
                }

        answer = f"Достигнут лимит шагов в layout={self.env.layout_name}."
        meta = {"cycle_ok": False, "confidence": 0.3, "recommendations": [f"repair_keydoor::{self.env.layout_name}"]}
        return {
            "goal": f"keydoor_goal@{self.env.layout_name}",
            "answer": answer,
            "meta": meta,
            "done": False,
            "executed_actions": executed_actions,
            "reward_total": reward_total,
            "render_known": self.map_memory.render_known(self.env.agent),
            "render_world": self.env.render(),
        }


def print_result(result: Dict[str, object]) -> None:
    print("\n[GOAL]", result["goal"])
    print("[ANSWER]\n" + str(result["answer"]))
    print("[META]", result["meta"])
    print("[KNOWN MAP]")
    print(result["render_known"])
    print("[WORLD]")
    print(result["render_world"])
    print()


def print_transfer(results: List[Dict[str, object]]) -> None:
    for r in results:
        print(f"[KD-TRANSFER] {r['task']} layout={r['layout']} :: {'PASS' if r['ok'] else 'FAIL'} steps={r['steps']} reward={r['reward_total']}")
    print()


def main() -> None:
    agi = TolikCompositionalExec()
    print("Tolik compositional executive ready.")
    print("Commands: /kd_reset <layout>, /kd_show, /kd_run [layout], /kd_transfer_list, /kd_transfer, /kd_status, exit")

    while True:
        user_text = input("you> ").strip()
        if user_text.lower() in {"exit", "quit", "q"}:
            break

        if user_text.startswith("/kd_reset "):
            layout = user_text[len("/kd_reset ") :].strip()
            agi.reset(layout)
            print(agi.map_memory.render_known(agi.env.agent))
            print()
            continue

        if user_text == "/kd_show":
            print(agi.map_memory.render_known(agi.env.agent))
            print()
            continue

        if user_text.startswith("/kd_run"):
            parts = user_text.split()
            layout = parts[1] if len(parts) > 1 else None
            out = agi.run_episode(layout)
            print_result(out)
            continue

        if user_text == "/kd_transfer_list":
            for task in agi.transfer.list_tasks():
                print(f"- {task['name']} layout={task['layout']} runs={task['runs']} passes={task['passes']}")
            print()
            continue

        if user_text == "/kd_transfer":
            results = agi.transfer.run_all_with(lambda layout: agi.run_episode(layout))
            print_transfer(results)
            continue

        if user_text == "/kd_status":
            print("Layout:", agi.env.layout_name)
            print("Transfer:", agi.transfer.summary())
            print("Has key:", agi.map_memory.has_key)
            print("Door open:", agi.map_memory.door_open)
            print("Known map:")
            print(agi.map_memory.render_known(agi.env.agent))
            print()
            continue

        print("Unknown command.\n")


if __name__ == "__main__":
    main()
