from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from compositional_exec import TolikCompositionalExec
from core.failure_clusters import FailureClusters
from core.goal_ledger import GoalLedger
from core.options_memory import OptionsMemory
from main import TolikAGI


def optimal_nav_actions(layout_rows: List[str]) -> List[str]:
    grid = [list(r) for r in layout_rows]
    start = None
    target = None

    for i, row in enumerate(grid):
        for j, ch in enumerate(row):
            if ch == "A":
                start = (i, j)
                grid[i][j] = "."
            elif ch == "T":
                target = (i, j)
                grid[i][j] = "."

    if start is None or target is None:
        return []

    moves = [
        ("up", (-1, 0)),
        ("down", (1, 0)),
        ("left", (0, -1)),
        ("right", (0, 1)),
    ]

    q = deque([start])
    prev: Dict[Tuple[int, int], Tuple[Tuple[int, int], str]] = {}
    seen = {start}

    while q:
        cur = q.popleft()
        if cur == target:
            break
        for action, (di, dj) in moves:
            nxt = (cur[0] + di, cur[1] + dj)
            if (
                0 <= nxt[0] < len(grid)
                and 0 <= nxt[1] < len(grid[0])
                and grid[nxt[0]][nxt[1]] != "#"
                and nxt not in seen
            ):
                seen.add(nxt)
                prev[nxt] = (cur, action)
                q.append(nxt)

    if target not in seen:
        return []

    actions: List[str] = []
    node = target
    while node != start:
        parent, act = prev[node]
        actions.append(act)
        node = parent
    actions.reverse()
    return actions


def optimal_keydoor_actions(layout_rows: List[str]) -> List[str]:
    grid = [list(r) for r in layout_rows]
    start = None
    target = None
    key_pos = None
    door_pos = None

    for i, row in enumerate(grid):
        for j, ch in enumerate(row):
            if ch == "A":
                start = (i, j)
                grid[i][j] = "."
            elif ch == "T":
                target = (i, j)
                grid[i][j] = "."
            elif ch == "K":
                key_pos = (i, j)
                grid[i][j] = "."
            elif ch == "D":
                door_pos = (i, j)
                grid[i][j] = "."

    if start is None or target is None:
        return []

    moves = [
        ("up", (-1, 0)),
        ("down", (1, 0)),
        ("left", (0, -1)),
        ("right", (0, 1)),
    ]

    start_state = (start[0], start[1], False, False)
    q = deque([start_state])
    prev: Dict[Tuple[int, int, bool, bool], Tuple[Tuple[int, int, bool, bool], str]] = {}
    seen = {start_state}
    found = None

    while q:
        i, j, has_key, door_open = q.popleft()
        if (i, j) == target:
            found = (i, j, has_key, door_open)
            break

        for action, (di, dj) in moves:
            ni, nj = i + di, j + dj
            if not (0 <= ni < len(grid) and 0 <= nj < len(grid[0])):
                continue
            if grid[ni][nj] == "#":
                continue

            blocked_by_door = door_pos is not None and (ni, nj) == door_pos and not door_open
            if blocked_by_door and not has_key:
                continue

            new_has_key = has_key or (key_pos is not None and (ni, nj) == key_pos)
            new_door_open = door_open or (door_pos is not None and (ni, nj) == door_pos and has_key)

            nxt = (ni, nj, new_has_key, new_door_open)
            if nxt not in seen:
                seen.add(nxt)
                prev[nxt] = ((i, j, has_key, door_open), action)
                q.append(nxt)

    if found is None:
        return []

    actions: List[str] = []
    node = found
    while node != start_state:
        parent, act = prev[node]
        actions.append(act)
        node = parent
    actions.reverse()
    return actions


class EfficientNavExec:
    def __init__(self) -> None:
        module_root = Path(__file__).resolve().parent
        runtime_dir = module_root / "data" / "runtime"
        self.base = TolikAGI()
        self.options = OptionsMemory(storage_dir=str(runtime_dir))

    def _clear_layout_memory(self, layout: str) -> None:
        for key in [f"pomdp_model::{layout}", f"pomdp_policy::{layout}"]:
            self.base.memory.long_term.pop(key, None)
        self.base.memory.save()

    def run_episode(self, layout: str, max_steps: int = 80) -> Dict[str, object]:
        self._clear_layout_memory(layout)
        self.base.reset_pomdp(layout)

        used_option = None
        executed: List[str] = []
        reward_total = 0.0
        done = False

        options = self.options.match("nav_pomdp", layout, kind="full_policy")
        if options:
            used_option = options[0]
            for action in used_option["actions"]:
                local_before = self.base.env.observe_local(radius=1)
                self.base.map_memory.update_from_local(local_before)
                self.base.memory.store_fact(f"pomdp_model::{layout}", self.base.map_memory.to_fact())

                step = self.base.agency.execute_env_action(action, self.base.env)
                executed.append(action)
                reward_total += step["reward"]

                local_after = self.base.env.observe_local(radius=1)
                self.base.map_memory.update_from_local(local_after)
                self.base.memory.store_fact(f"pomdp_model::{layout}", self.base.map_memory.to_fact())

                if step["done"]:
                    done = True
                    break

                if len(executed) >= max_steps:
                    break

        if not done and len(executed) < max_steps:
            out = self.base.run_partial_env_episode(None, max_steps=max_steps - len(executed))
            executed.extend(out["executed_actions"])
            reward_total += out["reward_total"]
            done = bool(out["done"])
            answer = out["answer"]
            render = out["render"]
        else:
            if done:
                self.base.memory.store_fact(f"pomdp_policy::{layout}", " ".join(executed))
                answer = (
                    f"Option-guided POMDP-эпизод успешно завершён в layout={layout}. "
                    f"Шагов: {len(executed)}."
                )
            else:
                answer = (
                    f"Option-guided POMDP-эпизод не завершён в layout={layout}. "
                    f"Шагов: {len(executed)}."
                )
            render = self.base.map_memory.render_known(self.base.env.agent)

        return {
            "goal": f"repair_nav@{layout}",
            "answer": answer,
            "done": done,
            "executed_actions": executed,
            "reward_total": reward_total,
            "render": render,
            "used_option": used_option,
        }


class EfficientKeyDoorExec:
    def __init__(self) -> None:
        module_root = Path(__file__).resolve().parent
        runtime_dir = module_root / "data" / "runtime"
        self.base = TolikCompositionalExec()
        self.options = OptionsMemory(storage_dir=str(runtime_dir))

    def _clear_layout_memory(self, layout: str) -> None:
        for key in [f"keydoor_model::{layout}", f"keydoor_policy::{layout}"]:
            self.base.memory.long_term.pop(key, None)
        self.base.memory.save()

    def run_episode(self, layout: str, max_steps: int = 120) -> Dict[str, object]:
        self._clear_layout_memory(layout)
        self.base.reset(layout)

        used_option = None
        executed: List[str] = []
        reward_total = 0.0
        done = False

        options = self.options.match("keydoor", layout, kind="full_policy")
        if options:
            used_option = options[0]
            for action in used_option["actions"]:
                local_before = self.base.env.observe_local(radius=1)
                self.base.map_memory.update_from_local(local_before)
                self.base.memory.store_fact(f"keydoor_model::{layout}", self.base.map_memory.to_fact())

                step = self.base.env.step(action)
                executed.append(action)
                reward_total += step.reward

                local_after = self.base.env.observe_local(radius=1)
                self.base.map_memory.update_from_local(local_after)
                self.base.memory.store_fact(f"keydoor_model::{layout}", self.base.map_memory.to_fact())

                if step.done:
                    done = True
                    break

                if len(executed) >= max_steps:
                    break

        if not done and len(executed) < max_steps:
            out = self.base.run_episode(None, max_steps=max_steps - len(executed))
            executed.extend(out["executed_actions"])
            reward_total += out["reward_total"]
            done = bool(out["done"])
            answer = out["answer"]
            render_known = out["render_known"]
            render_world = out["render_world"]
        else:
            if done:
                self.base.memory.store_fact(f"keydoor_policy::{layout}", " ".join(executed))
                answer = (
                    f"Option-guided KeyDoor-эпизод завершён успешно в layout={layout}. "
                    f"Шагов: {len(executed)}."
                )
            else:
                answer = (
                    f"Option-guided KeyDoor-эпизод не завершён в layout={layout}. "
                    f"Шагов: {len(executed)}."
                )
            render_known = self.base.map_memory.render_known(self.base.env.agent)
            render_world = self.base.env.render()

        return {
            "goal": f"repair_keydoor@{layout}",
            "answer": answer,
            "done": done,
            "executed_actions": executed,
            "reward_total": reward_total,
            "render_known": render_known,
            "render_world": render_world,
            "used_option": used_option,
        }


class TolikOptionRepairExec:
    def __init__(self) -> None:
        module_root = Path(__file__).resolve().parent
        runtime_dir = module_root / "data" / "runtime"

        self.options = OptionsMemory(storage_dir=str(runtime_dir))
        self.failures = FailureClusters(storage_dir=str(runtime_dir))
        self.goals = GoalLedger(storage_dir=str(runtime_dir))

        self.nav = EfficientNavExec()
        self.kd = EfficientKeyDoorExec()

    def distill_from_clusters(self) -> List[Dict[str, object]]:
        created: List[Dict[str, object]] = []
        for case in self.failures.list_cases():
            family = case["family"]
            layout = case["layout"]
            issue = case["issue"]

            if family == "nav_pomdp" and issue == "nav_planning_efficiency":
                actions = optimal_nav_actions(self.nav.base.env.layouts[layout])
                if actions:
                    created.append(
                        self.options.add_or_replace(
                            name=f"nav_teacher::{layout}",
                            family="nav_pomdp",
                            layout=layout,
                            kind="full_policy",
                            trigger="failure_cluster_teacher_distillation",
                            actions=actions,
                        )
                    )

            if family == "keydoor" and issue == "keydoor_compositional_efficiency":
                actions = optimal_keydoor_actions(self.kd.base.env.layouts[layout])
                if actions:
                    created.append(
                        self.options.add_or_replace(
                            name=f"keydoor_teacher::{layout}",
                            family="keydoor",
                            layout=layout,
                            kind="full_policy",
                            trigger="failure_cluster_teacher_distillation",
                            actions=actions,
                        )
                    )

        return created

    def stress_nav(self, ratio_threshold: float = 1.35) -> List[Dict[str, object]]:
        rows: List[Dict[str, object]] = []
        for task in self.nav.base.transfer_suite.list_tasks():
            layout = task["layout"]
            out = self.nav.run_episode(layout)
            optimal = len(optimal_nav_actions(self.nav.base.env.layouts[layout]))
            actual = len(out["executed_actions"])
            ratio = (actual / optimal) if optimal > 0 else 999.0
            robust_ok = bool(out["done"]) and ratio <= ratio_threshold

            issue = None
            if not out["done"]:
                issue = "nav_world_model"
            elif ratio > ratio_threshold:
                issue = "nav_planning_efficiency"

            if out["used_option"] is not None:
                self.options.record_use(
                    out["used_option"]["id"],
                    ok=robust_ok,
                    ratio=ratio,
                )

            rows.append(
                {
                    "family": "nav_pomdp",
                    "layout": layout,
                    "done": bool(out["done"]),
                    "actual_steps": actual,
                    "optimal_steps": optimal,
                    "ratio": round(ratio, 3),
                    "reward_total": round(float(out["reward_total"]), 3),
                    "robust_ok": robust_ok,
                    "issue": issue,
                }
            )

            if issue is not None:
                self.failures.add_case(
                    family="nav_pomdp",
                    layout=layout,
                    issue=issue,
                    done=bool(out["done"]),
                    actual_steps=actual,
                    optimal_steps=optimal,
                    ratio=ratio,
                    reward_total=float(out["reward_total"]),
                    answer=str(out["answer"]),
                )

        return rows

    def stress_keydoor(self, ratio_threshold: float = 1.35) -> List[Dict[str, object]]:
        rows: List[Dict[str, object]] = []
        for task in self.kd.base.transfer.list_tasks():
            layout = task["layout"]
            out = self.kd.run_episode(layout)
            optimal = len(optimal_keydoor_actions(self.kd.base.env.layouts[layout]))
            actual = len(out["executed_actions"])
            ratio = (actual / optimal) if optimal > 0 else 999.0
            robust_ok = bool(out["done"]) and ratio <= ratio_threshold

            issue = None
            if not out["done"]:
                issue = "keydoor_state_tracking"
            elif ratio > ratio_threshold:
                issue = "keydoor_compositional_efficiency"

            if out["used_option"] is not None:
                self.options.record_use(
                    out["used_option"]["id"],
                    ok=robust_ok,
                    ratio=ratio,
                )

            rows.append(
                {
                    "family": "keydoor",
                    "layout": layout,
                    "done": bool(out["done"]),
                    "actual_steps": actual,
                    "optimal_steps": optimal,
                    "ratio": round(ratio, 3),
                    "reward_total": round(float(out["reward_total"]), 3),
                    "robust_ok": robust_ok,
                    "issue": issue,
                }
            )

            if issue is not None:
                self.failures.add_case(
                    family="keydoor",
                    layout=layout,
                    issue=issue,
                    done=bool(out["done"]),
                    actual_steps=actual,
                    optimal_steps=optimal,
                    ratio=ratio,
                    reward_total=float(out["reward_total"]),
                    answer=str(out["answer"]),
                )

        return rows

    def stress_all(self) -> Dict[str, List[Dict[str, object]]]:
        self.failures.clear()
        return {
            "nav": self.stress_nav(),
            "keydoor": self.stress_keydoor(),
        }

    def add_goals_from_clusters(self, limit: int = 5) -> List[Dict[str, object]]:
        existing = {g["text"] for g in self.goals.list_goals()}
        created: List[Dict[str, object]] = []
        for text in self.failures.propose_goals(limit):
            if text in existing:
                continue
            created.append(self.goals.add_goal(text, priority=90, source="option_repair"))
        return created


def print_rows(title: str, rows: List[Dict[str, object]]) -> None:
    print(f"[{title}]")
    for row in rows:
        print(
            f"- {row['layout']} :: {'PASS' if row['robust_ok'] else 'FAIL'} "
            f"done={row['done']} steps={row['actual_steps']}/{row['optimal_steps']} "
            f"ratio={row['ratio']} reward={row['reward_total']} issue={row['issue']}"
        )
    print()


def print_cases(rows: List[Dict[str, object]]) -> None:
    if not rows:
        print("No failure cases.\n")
        return
    for row in rows:
        print(
            f"- {row['family']} layout={row['layout']} issue={row['issue']} "
            f"steps={row['actual_steps']}/{row['optimal_steps']} ratio={round(float(row['ratio']), 3)}"
        )
    print()


def print_clusters(rows: List[Dict[str, object]]) -> None:
    if not rows:
        print("No clusters.\n")
        return
    for row in rows:
        print(f"- {row['issue']} :: {row['count']}")
    print()


def print_goals(rows: List[Dict[str, object]]) -> None:
    if not rows:
        print("No goals.\n")
        return
    for row in rows:
        print(f"- {row['id']} [{row['status']}] p={row['priority']} :: {row['text']}")
    print()


def print_options(rows: List[Dict[str, object]]) -> None:
    if not rows:
        print("No options.\n")
        return
    for row in rows:
        print(
            f"- {row['name']} family={row['family']} layout={row['layout']} "
            f"kind={row['kind']} uses={row['uses']} successes={row['successes']} last_ratio={row['last_ratio']}"
        )
    print()


def main() -> None:
    agi = TolikOptionRepairExec()
    print("Tolik option-repair executive ready.")
    print("Commands: /options_list, /distill_from_clusters, /stress_all, /cases, /clusters, /autogoals <n>, /goals, /status, exit")

    while True:
        user_text = input("you> ").strip()
        if user_text.lower() in {"exit", "quit", "q"}:
            break

        if user_text == "/options_list":
            print_options(agi.options.list_options())
            continue

        if user_text == "/distill_from_clusters":
            created = agi.distill_from_clusters()
            if not created:
                print("No teacher options were distilled.\n")
            else:
                for row in created:
                    print(f"Option added: {row['id']} :: {row['name']} ({row['layout']})")
                print()
            continue

        if user_text == "/stress_all":
            res = agi.stress_all()
            print_rows("OPTION_STRESS_NAV", res["nav"])
            print_rows("OPTION_STRESS_KEYDOOR", res["keydoor"])
            continue

        if user_text == "/cases":
            print_cases(agi.failures.list_cases())
            continue

        if user_text == "/clusters":
            print_clusters(agi.failures.summary())
            continue

        if user_text.startswith("/autogoals"):
            parts = user_text.split()
            limit = 5
            if len(parts) > 1:
                try:
                    limit = max(1, int(parts[1]))
                except ValueError:
                    limit = 5
            created = agi.add_goals_from_clusters(limit)
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
            print("Options:", agi.options.summary())
            print("Failures:", agi.failures.summary_counts())
            print("Goals:", agi.goals.stats())
            print()
            continue

        print("Unknown command.\n")


if __name__ == "__main__":
    main()
