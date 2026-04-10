from __future__ import annotations

from typing import Any, Dict, List


class PlanningModule:
    """Turns goals into executable steps."""

    def make_plan(self, goal: str, reasoning: Dict[str, Any]) -> List[Dict[str, str]]:
        steps: List[Dict[str, str]] = []
        inferred = reasoning.get("inferred_subgoals", [])

        if goal.startswith("answer_user:"):
            question = goal.split(":", 1)[1].strip()
            steps = [
                {"action": "search_memory", "input": question},
                {"action": "compose_answer", "input": question},
            ]
        elif goal.startswith("execute_request:"):
            task = goal.split(":", 1)[1].strip()
            steps = [
                {"action": "decompose_task", "input": task},
                {"action": "compose_answer", "input": f"Plan for task: {task}"},
            ]
        elif goal == "stabilize_context":
            steps = [{"action": "compose_answer", "input": "Пустой ввод. Нужна цель или запрос."}]
        else:
            steps = [
                {"action": "reflect", "input": goal},
                {"action": "compose_answer", "input": f"Internal analysis for: {goal}"},
            ]

        for subgoal in inferred:
            steps.append({"action": "note_subgoal", "input": subgoal})

        return steps
