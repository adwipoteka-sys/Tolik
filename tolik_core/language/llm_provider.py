from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class LLMResult:
    text: str
    provider: str


class OpenAIResponsesProvider:
    def __init__(self) -> None:
        self.api_key = os.getenv("OPENAI_API_KEY", "").strip()
        self.model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini").strip()
        self._client = None
        self.provider_name = "rule_based"

        if self.api_key:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=self.api_key)
                self.provider_name = f"openai:{self.model}"
            except Exception:
                self._client = None
                self.provider_name = "rule_based"

    def available(self) -> bool:
        return self._client is not None

    def complete(self, system_prompt: str, user_prompt: str, max_output_tokens: int = 700) -> LLMResult:
        if not self._client:
            raise RuntimeError("OpenAI provider is not available")

        response = self._client.responses.create(
            model=self.model,
            instructions=system_prompt,
            input=user_prompt,
            max_output_tokens=max_output_tokens,
        )

        text = getattr(response, "output_text", "") or ""
        text = text.strip()
        if not text:
            raise RuntimeError("OpenAI response was empty")

        return LLMResult(text=text, provider=self.provider_name)
