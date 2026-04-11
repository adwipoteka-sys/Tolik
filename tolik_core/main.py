from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, List

from agency.agency_module import AgencyModule
from agency.tools import LocalToolbox
from core.global_workspace import GlobalWorkspace
from core.goal_ledger import GoalLedger
from core.skill_arena import SkillArena
from core.transfer_suite import TransferSuite
from language.language_module import LanguageModule
from memory.memory_module import MemoryModule
from metacognition.metacognition_module import MetacognitionModule
from motivation.motivation_module import MotivationModule
from perception.perception_module import PerceptionModule
from planning.planning_module import PlanningModule
from reasoning.reasoning_module import ReasoningModule
from sim.gridworld import GridWorld


class TolikAGI:
    def __init__(self) -> None:
        module_root = Path(__file__).resolve().parent
        repo_root = module_root.parent.resolve()
        runtime_dir = module_root / "data" / "runtime"

        self.workspace = GlobalWorkspace()
        self.perception = PerceptionModule()
        self.memory = MemoryModule(storage_dir=str(runtime_dir))
        self.memory.seed_defaults()
        self.reasoning = ReasoningModule()
        self.motivation = MotivationModule()
        self.planning = PlanningModule()
        self.language = LanguageModule()
        self.toolbox = LocalToolbox(str(repo_root))
        self.agency = AgencyModule(language=self.language, toolbox=self.toolbox)
        self.metacognition = MetacognitionModule()

        self.goal_ledger = GoalLedger(storage_dir=str(runtime_dir))
        self.skill_arena = SkillArena(storage_dir=str(runtime_dir))
        self.transfer_suite = TransferSuite(storage_dir=str(runtime_dir))
        self.transfer_suite.seed_defaults()

        self.env = GridWorld()

    @staticmethod
    def _to_dict(obj: Any) -> Dict[str, Any]:
        if is_dataclass(obj):
            return asdict(obj)
        if isinstance(obj, dict):
            return obj
        raise TypeError(f"Cannot convert object of type {type(obj)!r} to dict")

    def run_cycle(self, user_text: str) -> Dict[str, Any]:
        perception = self._to_dict(self.perception.process_input(user_text))
        self.workspace.publish("perception", perception, source="perception")
        self.memory.remember_event({"type": "perception", "data": perception})

        self.motivation.ingest_perception(perception)
        goal = self.motivation.next_goal() or "maintain_stability"
        self.workspace.publish("goal", goal, source="motivation")

        reasoning = self.reasoning.analyze(goal, perception, self.memory.recent_context())
        self.workspace.publish("reasoning", reasoning, source="reasoning")

        plan = self.planning.make_plan(goal, reasoning)
        self.workspace.publish("plan", plan, source="planning")

        action_result = self.agency.execute_plan(plan, self.memory, reasoning)
        self.workspace.publish("action_result", action_result, source="agency")
        self.memory.remember_event({"type": "action_result", "data": action_result})

        meta = self.metacognition.review(perception, reasoning, action_result)
        self.workspace.publish("metacognition", meta, source="metacognition")
        self.motivation.ingest_metacognition(meta.get("recommendations", []))

        self.memory.save()

        return {
            "goal": goal,
            "reasoning": reasoning,
            "plan": plan,
            "answer": action_result["answer"],
            "meta": meta,
        }

    def run_goal(self, goal_text: str) -> Dict[str, Any]:
        return self.run_cycle(f"goal: {goal_text}")

    def run_env_episode(self, layout: str | None = None) -> Dict[str, Any]:
        if layout:
            obs = self.env.reset(layout)
        else:
            obs = self.env.observe()

        perception = self.perception.process_state(obs)
        self.workspace.publish("env_perception", perception, source="perception")
        self.memory.remember_event({"type": "env_perception", "data": perception})

        stored = self.memory.recall_fact(f"env_policy::{self.env.layout_name}")
        if isinstance(stored, str) and stored.strip():
            plan = stored.split()
            plan_source = "memory"
        else:
            plan = self.planning.make_navigation_plan(obs)
            plan_source = "planner"

        reasoning = {
            "goal": "reach_target",
            "confidence": 0.8 if plan else 0.2,
            "warnings": [] if plan else ["no_path_found"],
            "inferred_subgoals": [],
        }
        self.workspace.publish("env_plan", {"source": plan_source, "actions": plan}, source="planning")

        result = self.agency.execute_navigation_plan(plan, self.env)
        self.workspace.publish("env_action_result", result, source="agency")
        self.memory.remember_event({"type": "env_action_result", "data": result})

        if result["done"]:
            self.memory.store_fact(f"env_policy::{self.env.layout_name}", " ".join(result["executed_actions"]))
            answer = (
                f"Эпизод успешно завершён в layout={self.env.layout_name}. "
                f"Шагов: {len(result['executed_actions'])}. "
                f"Траектория сохранена в память."
            )
        else:
            answer = (
                f"Эпизод не завершён в layout={self.env.layout_name}. "
                f"Шагов: {len(result['executed_actions'])}. "
                f"Нужен пересмотр плана."
            )

        meta = {
            "cycle_ok": result["done"],
            "recommendations": [] if result["done"] else [f"improve_navigation::{self.env.layout_name}"],
            "confidence": reasoning["confidence"],
        }
        self.workspace.publish("env_metacognition", meta, source="metacognition")

        return {
            "goal": f"env_goal: reach_target@{self.env.layout_name}",
            "reasoning": reasoning,
            "plan": [{"action": "env_move", "input": a} for a in plan],
            "answer": answer,
            "meta": meta,
            "done": result["done"],
            "executed_actions": result["executed_actions"],
            "reward_total": result["reward_total"],
            "render": self.env.render(),
        }


def print_result(result: Dict[str, Any]) -> None:
    print("\n[GOAL]", result["goal"])
    print("[ANSWER]\n" + result["answer"])
    print("[META]", result["meta"])
    print()


def print_goals(goals: List[Dict[str, object]]) -> None:
    if not goals:
        print("No goals.\n")
        return
    for g in goals:
        print(f"- {g['id']} [{g['status']}] p={g['priority']} :: {g['text']}")
    print()


def print_arena_results(results: List[Dict[str, object]]) -> None:
    if not results:
        print("Arena is empty.\n")
        return
    for r in results:
        print(f"[ARENA] {r['task']} :: {'PASS' if r['ok'] else 'FAIL'}")
        print(r["answer"])
        print()


def print_transfer_results(results: List[Dict[str, object]]) -> None:
    for r in results:
        print(f"[TRANSFER] {r['task']} layout={r['layout']} :: {'PASS' if r['ok'] else 'FAIL'} steps={r['steps']} reward={r['reward_total']}")
    print()


def main() -> None:
    agi = TolikAGI()
    print(f"Tolik executive ready. LLM provider: {agi.language.provider_name}")
    print("Commands: /goal <text>, /goals, /run_next, /arena_add name|prompt|required1,required2, /arena_list, /arena_run, /arena_repair, /self_improve [n], /sim_reset <layout>, /sim_show, /sim_run [layout], /transfer_list, /transfer_run, /status, exit")

    while True:
        user_text = input("you> ").strip()
        if user_text.lower() in {"exit", "quit", "q"}:
            break

        if user_text.startswith("/goal "):
            goal_text = user_text[len("/goal ") :].strip()
            rec = agi.goal_ledger.add_goal(goal_text, source="user")
            print(f"Goal added: {rec['id']} :: {rec['text']}\n")
            continue

        if user_text == "/goals":
            print_goals(agi.goal_ledger.list_goals())
            continue

        if user_text == "/run_next":
            rec = agi.goal_ledger.start_next()
            if rec is None:
                print("No pending goals.\n")
                continue
            result = agi.run_goal(rec["text"])
            ok = bool(result["meta"].get("cycle_ok", False))
            if ok:
                agi.goal_ledger.mark_done(rec["id"], note=result["answer"])
            else:
                agi.goal_ledger.mark_failed(rec["id"], note=result["answer"])
            print_result(result)
            continue

        if user_text.startswith("/arena_add "):
            payload = user_text[len("/arena_add ") :].strip()
            try:
                name, prompt, required_raw = payload.split("|", 2)
            except ValueError:
                print("Format: /arena_add name|prompt|required1,required2\n")
                continue

            task = agi.skill_arena.add_task(
                name=name.strip(),
                prompt=prompt.strip(),
                required=[x.strip() for x in required_raw.split(",") if x.strip()],
            )
            print(f"Arena task added: {task['id']} :: {task['name']}\n")
            continue

        if user_text == "/arena_list":
            for task in agi.skill_arena.list_tasks():
                print(f"- {task['id']} {task['name']} required={task['required']}")
            print()
            continue

        if user_text == "/arena_run":
            results = agi.skill_arena.run_all(agi)
            print_arena_results(results)
            continue

        if user_text == "/arena_repair":
            repair_goals = agi.skill_arena.propose_repair_goals()
            if not repair_goals:
                print("No repair goals proposed.\n")
                continue
            for goal_text in repair_goals:
                rec = agi.goal_ledger.add_goal(goal_text, priority=80, source="metacognition")
                print(f"Repair goal added: {rec['id']} :: {rec['text']}")
            print()
            continue

        if user_text.startswith("/self_improve"):
            parts = user_text.split()
            steps = 1
            if len(parts) > 1:
                try:
                    steps = max(1, int(parts[1]))
                except ValueError:
                    steps = 1

            results = agi.skill_arena.run_all(agi)
            print_arena_results(results)

            repair_goals = agi.skill_arena.propose_repair_goals()
            for goal_text in repair_goals:
                agi.goal_ledger.add_goal(goal_text, priority=80, source="metacognition")

            for _ in range(steps):
                rec = agi.goal_ledger.start_next()
                if rec is None:
                    break
                result = agi.run_goal(rec["text"])
                ok = bool(result["meta"].get("cycle_ok", False))
                if ok:
                    agi.goal_ledger.mark_done(rec["id"], note=result["answer"])
                else:
                    agi.goal_ledger.mark_failed(rec["id"], note=result["answer"])
                print_result(result)
            continue

        if user_text.startswith("/sim_reset "):
            layout = user_text[len("/sim_reset ") :].strip()
            agi.env.reset(layout)
            print(agi.env.render())
            print()
            continue

        if user_text == "/sim_show":
            print(agi.env.render())
            print()
            continue

        if user_text.startswith("/sim_run"):
            parts = user_text.split()
            layout = parts[1] if len(parts) > 1 else None
            result = agi.run_env_episode(layout)
            print_result(result)
            print(result["render"])
            print()
            continue

        if user_text == "/transfer_list":
            for task in agi.transfer_suite.list_tasks():
                print(f"- {task['name']} layout={task['layout']} runs={task['runs']} passes={task['passes']}")
            print()
            continue

        if user_text == "/transfer_run":
            results = agi.transfer_suite.run_all(agi)
            print_transfer_results(results)
            continue

        if user_text == "/status":
            print("Provider:", agi.language.provider_name)
            print("Goals:", agi.goal_ledger.stats())
            print("Arena:", agi.skill_arena.summary())
            print("Transfer:", agi.transfer_suite.summary())
            print("Current layout:", agi.env.layout_name)
            print()
            continue

        result = agi.run_cycle(user_text)
        print_result(result)


if __name__ == "__main__":
    main()
