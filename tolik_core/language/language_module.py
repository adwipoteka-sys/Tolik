from __future__ import annotations

import os
from typing import Any, Dict, List


class LanguageModule:
    """LLM is optional; deterministic cognitive mode is default."""

    def __init__(self) -> None:
        self.external_enabled = os.getenv("TOLIK_DISABLE_LLM", "1").lower() not in {"1", "true", "yes", "on"}
        self.provider = None
        self.provider_name = "disabled"

        if self.external_enabled:
            try:
                from language.llm_provider import OpenAIResponsesProvider
                self.provider = OpenAIResponsesProvider()
                if self.provider.available():
                    self.provider_name = self.provider.provider_name
                else:
                    self.provider = None
                    self.provider_name = "disabled"
            except Exception:
                self.provider = None
                self.provider_name = "disabled"

    @staticmethod
    def _fallback_answer(
        user_prompt: str,
        memory_hits: List[str],
        reasoning: Dict[str, Any],
        plan: List[Dict[str, str]],
    ) -> str:
        lines: List[str] = [f"Цель: {user_prompt}"]

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

    def compose_answer(
        self,
        user_prompt: str,
        memory_hits: List[str],
        reasoning: Dict[str, Any],
        plan: List[Dict[str, str]],
    ) -> str:
        if self.provider is not None:
            try:
                result = self.provider.complete(
                    system_prompt=(
                        "Ты языковой модуль AGI-системы 'Толик'. "
                        "Отвечай по-русски, кратко, структурно и по делу."
                    ),
                    user_prompt=(
                        f"Задача: {user_prompt}\n"
                        f"Память: {memory_hits}\n"
                        f"Рассуждение: {reasoning}\n"
                        f"План: {plan}\n"
                    ),
                )
                return result.text
            except Exception:
                self.provider = None
                self.provider_name = "disabled"

        return self._fallback_answer(
            user_prompt=user_prompt,
            memory_hits=memory_hits,
            reasoning=reasoning,
            plan=plan,
        )
