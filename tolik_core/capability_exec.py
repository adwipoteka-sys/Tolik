from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from autonomy_exec import TolikAutonomyExec, print_eval, print_step
from core.goal_ledger import GoalLedger
from core.skill_graph import SkillGraph


class TolikCapabilityExec:
    def __init__(self) -> None:
        module_root = Path(__file__).resolve().parent
        runtime_dir = module_root / "data" / "runtime"

        self.autonomy = TolikAutonomyExec()
        self.graph = SkillGraph(storage_dir=str(runtime_dir))
        self.goals = GoalLedger(storage_dir=str(runtime_dir))

    def eval_sync(self) -> Dict[str, List[Dict[str, object]]]:
        results = self.autonomy.eval_all()
        self.graph.sync_from_eval(
            eval_results=results,
            curriculum_summary=self.autonomy.curriculum.summary(),
            repair_count=len(self.autonomy.repair.list_items()),
        )
        return results

    def add_goals_from_gaps(self, limit: int = 5) -> List[Dict[str, object]]:
        existing = {g["text"] for g in self.goals.list_goals()}
        created: List[Dict[str, object]] = []
        for text in self.graph.propose_goals(limit):
            if text in existing:
                continue
            created.append(self.goals.add_goal(text, priority=70, source="skill_graph"))
        return created


def print_graph(rows: List[Dict[str, object]]) -> None:
    for row in rows:
        print(
            f"- {row['name']} mastery={row['mastery']:.3f} "
            f"runs={row['runs']} passes={row['passes']} parents={row['parents']}"
        )
    print()


def print_gaps(rows: List[Dict[str, object]]) -> None:
    for row in rows:
        print(
            f"- GAP {row['name']} gap={row['gap']:.3f} mastery={row['mastery']:.3f} :: {row['description']}"
        )
    print()


def print_goals(rows: List[Dict[str, object]]) -> None:
    if not rows:
        print("No goals.\n")
        return
    for row in rows:
        print(f"- {row['id']} [{row['status']}] p={row['priority']} :: {row['text']}")
    print()


def main() -> None:
    agi = TolikCapabilityExec()
    print("Tolik capability executive ready.")
    print("Commands: /curriculum_list, /train <n>, /eval_sync, /graph_list, /gaps, /roadmap <skill>, /autogoals <n>, /goals, /status, exit")

    while True:
        user_text = input("you> ").strip()
        if user_text.lower() in {"exit", "quit", "q"}:
            break

        if user_text == "/curriculum_list":
            for task in agi.autonomy.curriculum.list_tasks():
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
            results = agi.autonomy.train(steps)
            for row in results:
                print_step(row)
            continue

        if user_text == "/eval_sync":
            results = agi.eval_sync()
            print_eval(results)
            continue

        if user_text == "/graph_list":
            print_graph(agi.graph.list_nodes())
            continue

        if user_text == "/gaps":
            print_gaps(agi.graph.gaps())
            continue

        if user_text.startswith("/roadmap "):
            skill = user_text[len("/roadmap ") :].strip()
            chain = agi.graph.roadmap(skill)
            if not chain:
                print("Unknown skill.\n")
            else:
                print(" -> ".join(chain))
                print()
            continue

        if user_text.startswith("/autogoals"):
            parts = user_text.split()
            limit = 5
            if len(parts) > 1:
                try:
                    limit = max(1, int(parts[1]))
                except ValueError:
                    limit = 5
            created = agi.add_goals_from_gaps(limit)
            if not created:
                print("No new goals were added.\n")
            else:
                for row in created:
                    print(f"Goal added: {row['id']} :: {row['text']}")
                print()
            continue

        if user_text == "/goals":
            print_goals(agi.goals.list_goals())
            continue

        if user_text == "/status":
            print("Curriculum:", agi.autonomy.curriculum.summary())
            print("Repairs:", len(agi.autonomy.repair.list_items()))
            print("SkillGraph:", agi.graph.summary())
            print("Goals:", agi.goals.stats())
            print()
            continue

        print("Unknown command.\n")


if __name__ == "__main__":
    main()
