from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional


class MemoryModule:
    """Short-term + long-term memory with JSON persistence."""

    def __init__(self, short_term_limit: int = 25, storage_dir: str = "data/runtime") -> None:
        self.short_term_limit = short_term_limit
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self.facts_path = self.storage_dir / "facts.json"
        self.events_path = self.storage_dir / "events.json"

        self.short_term: Deque[Dict[str, Any]] = deque(maxlen=short_term_limit)
        self.long_term: Dict[str, Any] = {}

        self._load()

    def _load(self) -> None:
        if self.facts_path.exists():
            self.long_term = json.loads(self.facts_path.read_text(encoding="utf-8"))

        if self.events_path.exists():
            items = json.loads(self.events_path.read_text(encoding="utf-8"))
            self.short_term = deque(items[-self.short_term_limit :], maxlen=self.short_term_limit)

    def _write_json(self, path: Path, data: Any) -> None:
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def save(self) -> None:
        self._write_json(self.facts_path, self.long_term)
        self._write_json(self.events_path, list(self.short_term))

    def remember_event(self, event: Dict[str, Any]) -> None:
        self.short_term.append(event)
        self.save()

    def store_fact(self, key: str, value: Any) -> None:
        self.long_term[key] = value
        self.save()

    def recall_fact(self, key: str) -> Optional[Any]:
        return self.long_term.get(key)

    def search_facts(self, query: str, limit: int = 5) -> List[str]:
        q = query.lower().strip()
        if not q:
            return [f"{k}: {v}" for k, v in list(self.long_term.items())[:limit]]

        hits: List[str] = []
        for key, value in self.long_term.items():
            text = f"{key}: {value}"
            if q in key.lower() or q in str(value).lower():
                hits.append(text)
            if len(hits) >= limit:
                break
        return hits

    def recent_context(self, limit: int = 5) -> List[Dict[str, Any]]:
        return list(self.short_term)[-limit:]

    def seed_defaults(self) -> None:
        defaults = {
            "architecture_principle": "AGI loop: perception -> memory -> reasoning -> goal selection -> planning -> action -> feedback -> metacognition",
            "project_modules": [
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
            "project_name": "Толик",
        }
        changed = False
        for key, value in defaults.items():
            if key not in self.long_term:
                self.long_term[key] = value
                changed = True
        if changed:
            self.save()
