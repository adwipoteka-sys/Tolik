from __future__ import annotations

from collections import deque
from typing import Any, Deque, Dict, List, Optional


class MemoryModule:
    """Short-term + long-term memory skeleton."""

    def __init__(self, short_term_limit: int = 25) -> None:
        self.short_term: Deque[Dict[str, Any]] = deque(maxlen=short_term_limit)
        self.long_term: Dict[str, Any] = {}

    def remember_event(self, event: Dict[str, Any]) -> None:
        self.short_term.append(event)

    def store_fact(self, key: str, value: Any) -> None:
        self.long_term[key] = value

    def recall_fact(self, key: str) -> Optional[Any]:
        return self.long_term.get(key)

    def recent_context(self, limit: int = 5) -> List[Dict[str, Any]]:
        return list(self.short_term)[-limit:]

    def seed_defaults(self) -> None:
        self.store_fact(
            "architecture_principle",
            "AGI loop: perception -> memory -> reasoning -> goal selection -> planning -> action -> feedback -> metacognition",
        )
        self.store_fact(
            "project_modules",
            [
                "global_workspace",
                "perception",
                "memory",
                "reasoning",
                "planning",
                "language",
                "motivation",
                "agency",
                "metacognition",
            ],
        )
