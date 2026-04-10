from __future__ import annotations

from typing import Any, Dict, List


class LanguageModule:
    """Lightweight text generation layer for the bootstrap version."""

    def compose_answer(
        self,
        user_prompt: str,
        memory_hits: List[str],
        reasoning: Dict[str, Any],
        plan: List[Dict[str, str]],
    ) -> str:
        lines: List[str] = []
        lines.append(f"Цель: {user_prompt}")

        if memory_hits:
            lines.append("Память:")
            lines.extend(f"- {item}" for item in memory_hits)

        warnings = reasoning.get("warnings", [])
        if warnings:
            lines.append(f"Предупреждения: {', '.join(warnings)}")

        lines.append(f"Уверенность: {reasoning.get('confidence', 0.0)}")
        lines.append("План:")
        for idx, step in enumerate(plan, start=1):
            lines.append(f"{idx}. {step['action']} -> {step['input']}")

        return "\n".join(lines)
