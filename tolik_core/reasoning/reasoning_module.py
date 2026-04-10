from __future__ import annotations

from typing import Any, Dict, List


class ReasoningModule:
    """Basic reasoning / consistency check layer."""

    def analyze(
        self,
        goal: str,
        perception: Dict[str, Any],
        recent_context: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        confidence = 0.55
        warnings: List[str] = []
        inferred_subgoals: List[str] = []

        if not goal:
            warnings.append("goal_missing")
            confidence = 0.2
        if perception.get("intent") == "empty":
            warnings.append("empty_input")
            confidence = min(confidence, 0.2)
        if "answer_user:" in goal:
            inferred_subgoals.extend(["retrieve_relevant_memory", "compose_response"])
            confidence = max(confidence, 0.65)
        if "execute_request:" in goal:
            inferred_subgoals.extend(["decompose_request", "draft_execution_plan"])
            confidence = max(confidence, 0.7)

        if recent_context:
            last = recent_context[-1]
            if last.get("type") == "failure":
                warnings.append("previous_cycle_failed")
                inferred_subgoals.append("adapt_strategy")
                confidence -= 0.1

        return {
            "goal": goal,
            "confidence": round(max(0.0, min(1.0, confidence)), 2),
            "warnings": warnings,
            "inferred_subgoals": inferred_subgoals,
        }
