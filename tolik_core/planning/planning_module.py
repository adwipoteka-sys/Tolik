from __future__ import annotations

from collections import deque
from typing import Any, Dict, List, Tuple


class PlanningModule:
    """Turns goals into executable steps."""

    def make_plan(self, goal: str, reasoning: Dict[str, Any]) -> List[Dict[str, str]]:
        steps: List[Dict[str, str]] = []
        inferred = reasoning.get("inferred_subgoals", [])

        if goal.startswith("repair_skill:"):
            steps = [
                {"action": "repair_skill_knowledge", "input": goal},
                {"action": "compose_answer", "input": f"Repair applied: {goal}"},
            ]

        elif goal.startswith("answer_user:"):
            question = goal.split(":", 1)[1].strip()
            steps = [
                {"action": "search_memory", "input": question},
                {"action": "compose_answer", "input": question},
            ]

        elif goal.startswith("execute_request:"):
            request = goal.split(":", 1)[1].strip()
            request_lower = request.lower()

            if request_lower.startswith("remember "):
                payload = request[len("remember ") :].strip()
                steps = [
                    {"action": "store_fact", "input": payload},
                    {"action": "compose_answer", "input": f"Stored fact: {payload}"},
                ]
            elif request_lower.startswith("recall "):
                key = request[len("recall ") :].strip()
                steps = [
                    {"action": "recall_fact", "input": key},
                    {"action": "compose_answer", "input": f"Recall: {key}"},
                ]
            elif request_lower.startswith("list files"):
                rel = request[len("list files") :].strip() or "."
                steps = [
                    {"action": "list_files", "input": rel},
                    {"action": "compose_answer", "input": f"Files under: {rel}"},
                ]
            elif request_lower.startswith("read "):
                rel = request[len("read ") :].strip()
                steps = [
                    {"action": "read_file", "input": rel},
                    {"action": "compose_answer", "input": f"Read file: {rel}"},
                ]
            elif request_lower.startswith("write note "):
                text = request[len("write note ") :].strip()
                steps = [
                    {"action": "write_note", "input": text},
                    {"action": "compose_answer", "input": "Note written"},
                ]
            else:
                steps = [
                    {"action": "search_memory", "input": request},
                    {"action": "decompose_task", "input": request},
                    {"action": "compose_answer", "input": f"Plan for task: {request}"},
                ]

        elif goal == "stabilize_context":
            steps = [{"action": "compose_answer", "input": "Пустой ввод. Нужна цель или запрос."}]

        else:
            goal_lower = goal.lower()
            if any(word in goal_lower for word in ["улучш", "исслед", "разработ", "постро", "спроект", "оптимиз"]):
                steps = [
                    {"action": "search_memory", "input": goal},
                    {"action": "decompose_task", "input": goal},
                    {"action": "write_note", "input": f"# Goal session\n\n{goal}\n"},
                    {"action": "compose_answer", "input": f"Goal execution draft: {goal}"},
                ]
            else:
                steps = [
                    {"action": "reflect", "input": goal},
                    {"action": "compose_answer", "input": f"Internal analysis for: {goal}"},
                ]

        for subgoal in inferred:
            steps.append({"action": "note_subgoal", "input": subgoal})

        return steps

    def make_navigation_plan(self, env_state: Dict[str, object]) -> List[str]:
        grid = env_state["grid"]
        start = tuple(env_state["agent"])
        target = tuple(env_state["target"])

        rows = len(grid)
        cols = len(grid[0])

        def free(i: int, j: int) -> bool:
            return 0 <= i < rows and 0 <= j < cols and grid[i][j] != "#"

        deltas: List[Tuple[str, Tuple[int, int]]] = [
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
            for action, (di, dj) in deltas:
                nxt = (cur[0] + di, cur[1] + dj)
                if nxt not in seen and free(nxt[0], nxt[1]):
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
