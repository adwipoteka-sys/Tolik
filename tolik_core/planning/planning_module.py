from __future__ import annotations

from typing import Any, Dict, List


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
