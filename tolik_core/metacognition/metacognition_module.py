from __future__ import annotations

from typing import Any, Dict, List


class MetacognitionModule:
    """Monitors cycles, logs issues, proposes internal improvements."""

    def __init__(self) -> None:
        self.history: List[Dict[str, Any]] = []

    def review(
        self,
        perception: Dict[str, Any],
        reasoning: Dict[str, Any],
        action_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        recommendations: List[str] = []

        if reasoning.get("confidence", 0) < 0.4:
            recommendations.append("improve_reasoning_on_low_confidence_cases")
        if not action_result.get("answer"):
            recommendations.append("improve_response_generation")
        if perception.get("intent") == "empty":
            recommendations.append("clarify_user_goal")

        report = {
            "cycle_ok": action_result.get("status") == "ok",
            "recommendations": recommendations,
            "confidence": reasoning.get("confidence", 0.0),
        }
        self.history.append(report)
        return report
