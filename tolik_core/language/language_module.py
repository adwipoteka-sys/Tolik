from __future__ import annotations

import os
import re
from typing import Any, Dict, List


class LanguageModule:
    """LLM optional. Deterministic synthesis is default."""

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
            except Exception:
                self.provider = None
                self.provider_name = "disabled"

    @staticmethod
    def _clean_hit(hit: str) -> str:
        hit = re.sub(r"\s*\[local-semantic\s+[0-9.]+\]\s*$", "", hit)
        return hit.strip()

    @staticmethod
    def _canonicalize(text: str) -> str:
        mapping = {
            "metacognition:": "метакогниция:",
            "planning:": "планирование:",
            "motivation:": "мотивация:",
            "memory:": "память:",
            "reasoning:": "рассуждение:",
        }
        out = text
        for k, v in mapping.items():
            if out.lower().startswith(k):
                return v + out[len(k):].strip()
        return out

    def _direct_answer_from_memory(self, user_prompt: str, memory_hits: List[str]) -> str | None:
        if not memory_hits:
            return None

        cleaned = [self._canonicalize(self._clean_hit(x)) for x in memory_hits]
        prompt_l = user_prompt.lower().replace("ё", "е")
        top = cleaned[0]

        if "метаког" in prompt_l or "метапозн" in prompt_l:
            return f"Метакогниция в Толике отвечает за самоанализ и мониторинг системы. По памяти: {top}"

        if "планиров" in prompt_l or "план" in prompt_l:
            return f"Планировщик отвечает за построение последовательности действий для достижения цели. По памяти: {top}"

        if "цели" in prompt_l or "цель" in prompt_l or "мотивац" in prompt_l:
            return f"Система ставит себе цели через модуль мотивации, который формирует внутренние цели и приоритеты. По памяти: {top}"

        return f"По памяти Толика: {top}"

    @staticmethod
    def _fallback_answer(
        user_prompt: str,
        memory_hits: List[str],
        reasoning: Dict[str, Any],
        plan: List[Dict[str, str]],
        direct_answer: str | None = None,
    ) -> str:
        lines: List[str] = [f"Цель: {user_prompt}"]

        if direct_answer:
            lines.append("Ответ:")
            lines.append(direct_answer)

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
        direct_answer = self._direct_answer_from_memory(user_prompt, memory_hits)

        if self.provider is not None:
            try:
                result = self.provider.complete(
                    system_prompt="Ты языковой модуль AGI-системы 'Толик'. Отвечай по-русски, кратко, структурно и по делу.",
                    user_prompt=f"Задача: {user_prompt}\nПамять: {memory_hits}\nРассуждение: {reasoning}\nПлан: {plan}\n",
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
            direct_answer=direct_answer,
        )
