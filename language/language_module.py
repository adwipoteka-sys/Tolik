from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class LanguageModule:
    """Dependency-free language helper for local demos."""

    def interpret(self, text: str) -> dict[str, Any]:
        return {
            "intent": "question" if "?" in text else "statement",
            "length": len(text),
        }

    def generate_response(self, context: Any) -> str:
        if isinstance(context, dict) and "answer" in context:
            return str(context["answer"])
        if isinstance(context, dict) and "fact" in context:
            return f"Ответ: {context['fact']}"
        if context in (None, "", {}):
            return "Ответ: данных пока недостаточно."
        return f"Ответ: {context}"

    def chain_of_thought(self, prompt: str) -> list[str]:
        return [
            f"Шаг 1: анализирую запрос — {prompt}",
            "Шаг 2: выбираю наиболее правдоподобное объяснение или действие.",
            "Шаг 3: формирую ответ.",
        ]
