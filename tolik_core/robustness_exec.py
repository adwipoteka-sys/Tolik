from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import deque

from core.failure_clusters import FailureClusters
from core.goal_ledger import GoalLedger
from main import TolikAGI
from compositional_exec import TolikCompositionalExec


def bfs_nav_optimal(layout_rows: List[str]) -> Optional[int]:
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
        return None

    q = deque([(start, 0)])
    seen = {start}
    moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while q:
        (i, j), d = q.popleft()
        if (i, j) == target:
            return d
        for di, dj in moves:
            ni, nj = i + di, j + dj
            if 0 <= ni < len(grid) and 0 <= nj < len(grid[0]) and grid[ni][nj] != "#" and (ni, nj) not in seen:
                seen.add((ni, nj))
                q.append(((ni, nj), d + 1))

    return None


def bfs_keydoor_optimal(layout_rows: List[str]) -> Optional[int]:
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
        return None

    q = deque([((start[0], start[1], False, False), 0)])
    seen = {(start[0], start[1], False, False)}
    moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while q:
        (i, j, has_key, door_open), d = q.popleft()

        if (i, j) == target:
            return d

        for di, dj in moves:
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

            state = (ni, nj, new_has_key, new_door_open)
            if state not in seen:
                seen.add(state)
                q.append((state, d + 1))

    return None


class TolikRobustnessExec:
    def __init__(self) -> None:
        module_root = Path(__file__).resolve().parent
        runtime_dir = module_root / "data" / "runtime"

        self.nav = TolikAGI()
        self.kd = TolikCompositionalExec()
        self.clusters = FailureClusters(storage_dir=str(runtime_dir))
        self.goals = GoalLedger(storage_dir=str(runtime_dir))

    def _clear_nav_memory(self, layout: str) -> None:
        for key in [f"pomdp_model::{layout}", f"pomdp_policy::{layout}"]:
            self.nav.memory.long_term.pop(key, None)
        self.nav.memory.save()

    def _clear_kd_memory(self, layout: str) -> None:
        for key in [f"keydoor_model::{layout}", f"keydoor_policy::{layout}"]:
            self.kd.memory.long_term.pop(key, None)
        self.kd.memory.save()

    def stress_nav(self, ratio_threshold: float = 1.35) -> List[Dict[str, object]]:
        rows: List[Dict[str, object]] = []
        seen_layouts = []
        for task in self.nav.transfer_suite.list_tasks():
            layout = task["layout"]
            if layout not in seen_layouts:
                seen_layouts.append(layout)

        for layout in seen_layouts:
            self._clear_nav_memory(layout)
            out = self.nav.run_partial_env_episode(layout)
            optimal = bfs_nav_optimal(self.nav.env.layouts[layout])
            actual = len(out["executed_actions"])
            ratio = (actual / optimal) if optimal and optimal > 0 else 999.0

            robust_ok = bool(out["done"]) and ratio <= ratio_threshold
            issue = None
            if not out["done"]:
                issue = "nav_world_model"
            elif ratio > ratio_threshold:
                issue = "nav_planning_efficiency"

            row = {
                "family": "nav_pomdp",
                "layout": layout,
                "done": bool(out["done"]),
                "actual_steps": actual,
                "optimal_steps": optimal,
                "ratio": round(ratio, 3),
                "reward_total": round(float(out["reward_total"]), 3),
                "robust_ok": robust_ok,
                "answer": out["answer"],
                "issue": issue,
            }
            rows.append(row)

            if issue is not None:
                self.clusters.add_case(
                    family="nav_pomdp",
                    layout=layout,
                    issue=issue,
                    done=bool(out["done"]),
                    actual_steps=actual,
                    optimal_steps=int(optimal or 0),
                    ratio=float(ratio),
                    reward_total=float(out["reward_total"]),
                    answer=str(out["answer"]),
                )

        return rows

    def stress_keydoor(self, ratio_threshold: float = 1.35) -> List[Dict[str, object]]:
        rows: List[Dict[str, object]] = []
        seen_layouts = []
        for task in self.kd.transfer.list_tasks():
            layout = task["layout"]
            if layout not in seen_layouts:
                seen_layouts.append(layout)

        for layout in seen_layouts:
            self._clear_kd_memory(layout)
            out = self.kd.run_episode(layout)
            optimal = bfs_keydoor_optimal(self.kd.env.layouts[layout])
            actual = len(out["executed_actions"])
            ratio = (actual / optimal) if optimal and optimal > 0 else 999.0

            robust_ok = bool(out["done"]) and ratio <= ratio_threshold
            issue = None
            if not out["done"]:
                issue = "keydoor_state_tracking"
            elif ratio > ratio_threshold:
                issue = "keydoor_compositional_efficiency"

            row = {
                "family": "keydoor",
                "layout": layout,
                "done": bool(out["done"]),
                "actual_steps": actual,
                "optimal_steps": optimal,
                "ratio": round(ratio, 3),
                "reward_total": round(float(out["reward_total"]), 3),
                "robust_ok": robust_ok,
                "answer": out["answer"],
                "issue": issue,
            }
            rows.append(row)

            if issue is not None:
                self.clusters.add_case(
                    family="keydoor",
                    layout=layout,
                    issue=issue,
                    done=bool(out["done"]),
                    actual_steps=actual,
                    optimal_steps=int(optimal or 0),
                    ratio=float(ratio),
                    reward_total=float(out["reward_total"]),
                    answer=str(out["answer"]),
                )

        return rows

    def stress_all(self) -> Dict[str, List[Dict[str, object]]]:
        self.clusters.clear()
        return {
            "nav": self.stress_nav(),
            "keydoor": self.stress_keydoor(),
        }

    def add_goals_from_clusters(self, limit: int = 5) -> List[Dict[str, object]]:
        existing = {g["text"] for g in self.goals.list_goals()}
        created: List[Dict[str, object]] = []
        for text in self.clusters.propose_goals(limit):
            if text in existing:
                continue
            created.append(self.goals.add_goal(text, priority=85, source="failure_clusters"))
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


def main() -> None:
    agi = TolikRobustnessExec()
    print("Tolik robustness executive ready.")
    print("Commands: /stress_nav, /stress_kd, /stress_all, /cases, /clusters, /autogoals <n>, /goals, /status, exit")

    while True:
        user_text = input("you> ").strip()
        if user_text.lower() in {"exit", "quit", "q"}:
            break

        if user_text == "/stress_nav":
            rows = agi.stress_nav()
            print_rows("STRESS_NAV", rows)
            continue

        if user_text == "/stress_kd":
            rows = agi.stress_keydoor()
            print_rows("STRESS_KEYDOOR", rows)
            continue

        if user_text == "/stress_all":
            res = agi.stress_all()
            print_rows("STRESS_NAV", res["nav"])
            print_rows("STRESS_KEYDOOR", res["keydoor"])
            continue

        if user_text == "/cases":
            print_cases(agi.clusters.list_cases())
            continue

        if user_text == "/clusters":
            print_clusters(agi.clusters.summary())
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
            print("Failures:", agi.clusters.summary_counts())
            print("Goals:", agi.goals.stats())
            print()
            continue

        print("Unknown command.\n")


if __name__ == "__main__":
    main()
