from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class PerceptionResult:
    raw_text: str
    cleaned_text: str
    intent: str
    entities: Dict[str, str]
    suggested_goal: Optional[str]


class PerceptionModule:
    """Converts raw user input into a structured internal representation."""

    def process_input(self, user_text: str) -> PerceptionResult:
        cleaned = " ".join(user_text.strip().split())
        lowered = cleaned.lower()

        intent = "statement"
        suggested_goal: Optional[str] = None
        entities: Dict[str, str] = {}

        if not cleaned:
            intent = "empty"
            suggested_goal = "stabilize_context"
        elif lowered.startswith("goal:"):
            intent = "explicit_goal"
            suggested_goal = cleaned.split(":", 1)[1].strip() or "clarify_goal"
        elif "?" in cleaned:
            intent = "question"
            suggested_goal = f"answer_user: {cleaned}"
        elif any(word in lowered for word in ["сделай", "создай", "реализуй", "напиши"]):
            intent = "task_request"
            suggested_goal = f"execute_request: {cleaned}"
        else:
            suggested_goal = f"analyze_input: {cleaned}"

        if "agi" in lowered:
            entities["domain"] = "agi"
        if "толик" in lowered:
            entities["project"] = "толик"

        return PerceptionResult(
            raw_text=user_text,
            cleaned_text=cleaned,
            intent=intent,
            entities=entities,
            suggested_goal=suggested_goal,
        )
