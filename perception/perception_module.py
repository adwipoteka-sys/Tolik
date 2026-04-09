from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class PerceptionModule:
    """Minimal perception stub for local desktop demos."""

    def process_input(
        self,
        user_text: str | None = None,
        observation: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {}

        if user_text is not None:
            result["text"] = user_text
            result["intent"] = "question" if "?" in user_text else "statement"

        if observation is not None:
            result["observation"] = observation

        return result
