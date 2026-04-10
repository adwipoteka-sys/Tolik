from __future__ import annotations

from typing import Any, Dict, List

from language.llm_provider import OpenAIResponsesProvider


class LanguageModule:
    """Language layer with OpenAI adapter + deterministic fallback."""

    def __init__(self) -> None:
        self.provider = OpenAIResponsesProvider()
        self.provider_name = self.provider.provider_name

    @staticmethod
    def _fallback_answer(
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

    @staticmethod
    def _system_prompt() -> str:
        return (
            "Ты языковой модуль AGI-системы 'Толик'. "
            "Отвечай по-русски, кратко, структурно и по делу. "
            "Учитывай память, текущую цель, предупреждения рассуждения и план. "
            "Не выдумывай факты. Если уверенность низкая — прямо укажи это. "
            "Если пользователь просил действие, сначала опиши результат или план выполнения."
        )

    @staticmethod
    def _user_prompt(
        user_prompt: str,
        memory_hits: List[str],
        reasoning: Dict[str, Any],
        plan: List[Dict[str, str]],
    ) -> str:
        memory_block = "\n".join(f"- {item}" for item in memory_hits) if memory_hits else "- нет"
        plan_block = "\n".join(
            f"{idx}. {step['action']} -> {step['input']}"
            for idx, step in enumerate(plan, start=1)
        ) or "- нет"

        return (
            f"Текущая задача:\n{user_prompt}\n\n"
            f"Память:\n{memory_block}\n\n"
            f"Рассуждение:\n"
            f"- confidence: {reasoning.get('confidence', 0.0)}\n"
            f"- warnings: {reasoning.get('warnings', [])}\n"
            f"- inferred_subgoals: {reasoning.get('inferred_subgoals', [])}\n\n"
            f"План:\n{plan_block}\n\n"
            f"Сформируй финальный ответ пользователю."
        )

    def compose_answer(
        self,
        user_prompt: str,
        memory_hits: List[str],
        reasoning: Dict[str, Any],
        plan: List[Dict[str, str]],
    ) -> str:
        if self.provider.available():
            try:
                result = self.provider.complete(
                    system_prompt=self._system_prompt(),
                    user_prompt=self._user_prompt(
                        user_prompt=user_prompt,
                        memory_hits=memory_hits,
                        reasoning=reasoning,
                        plan=plan,
                    ),
                )
                return result.text
            except Exception as exc:
                fallback = self._fallback_answer(
                    user_prompt=user_prompt,
                    memory_hits=memory_hits,
                    reasoning=reasoning,
                    plan=plan,
                )
                return f"{fallback}\n\n[LLM fallback after error: {exc}]"

        return self._fallback_answer(
            user_prompt=user_prompt,
            memory_hits=memory_hits,
            reasoning=reasoning,
            plan=plan,
        )
