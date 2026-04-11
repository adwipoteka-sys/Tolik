from __future__ import annotations

import json
import re
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Set


class MemoryModule:
    """Persistent memory with local semantic fallback and alias expansion."""

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

    def _save_json(self, path: Path, data: Any) -> None:
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def save(self) -> None:
        self._save_json(self.facts_path, self.long_term)
        self._save_json(self.events_path, list(self.short_term))

    def remember_event(self, event: Dict[str, Any]) -> None:
        self.short_term.append(event)
        self.save()

    def store_fact(self, key: str, value: Any) -> None:
        self.long_term[key] = value

        alias_map = {
            "motivation": "мотивация",
            "planning": "планирование",
            "metacognition": "метакогниция",
            "reasoning": "рассуждение",
            "memory": "память",
        }

        alias = alias_map.get(key)
        if alias:
            self.long_term[alias] = value

        self.save()

    def recall_fact(self, key: str) -> Optional[Any]:
        return self.long_term.get(key)

    @staticmethod
    def _normalize_text(text: str) -> str:
        text = text.lower().replace("ё", "е")
        text = re.sub(r"[^a-zA-Zа-яА-Я0-9_\s]+", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    @classmethod
    def _base_tokens(cls, text: str) -> Set[str]:
        return {tok for tok in cls._normalize_text(text).split() if len(tok) >= 2}

    @classmethod
    def _expand_tokens(cls, text: str) -> Set[str]:
        tokens = cls._base_tokens(text)
        norm = cls._normalize_text(text)

        alias_rules = {
            "метаког": {"metacognition", "метакогниция", "метапознание", "самоанализ"},
            "метапозн": {"metacognition", "метакогниция", "метапознание", "самоанализ"},
            "самоанализ": {"metacognition", "метакогниция", "метапознание"},
            "планиров": {"planning", "планирование", "планировщик", "план"},
            "планировщик": {"planning", "планирование", "планировщик", "план"},
            "мотивац": {"motivation", "мотивация", "цель", "цели", "внутренние_цели"},
            "внутрен": {"motivation", "мотивация", "внутренние_цели", "цель", "цели"},
            "цел": {"motivation", "мотивация", "цель", "цели", "goal", "goals"},
            "себе цел": {"motivation", "мотивация", "внутренние_цели", "цель", "цели"},
            "памят": {"memory", "память"},
            "рассужд": {"reasoning", "логика", "рассуждение"},
            "логик": {"reasoning", "логика", "рассуждение"},
        }

        for marker, expanded in alias_rules.items():
            if marker in norm:
                tokens.update(expanded)

        if "metacognition" in tokens:
            tokens.update({"метакогниция", "метапознание", "самоанализ"})
        if "planning" in tokens:
            tokens.update({"планирование", "планировщик", "план"})
        if "motivation" in tokens:
            tokens.update({"мотивация", "цель", "цели", "внутренние_цели"})

        return tokens

    @classmethod
    def _local_similarity(cls, query: str, text: str) -> float:
        q = cls._expand_tokens(query)
        t = cls._expand_tokens(text)
        if not q or not t:
            return 0.0
        overlap = len(q & t)
        if overlap == 0:
            return 0.0
        return overlap / max(1, len(q))

    def search_facts(self, query: str, limit: int = 5) -> List[str]:
        q = query.strip()
        if not q:
            return [f"{k}: {v}" for k, v in list(self.long_term.items())[:limit]]

        hits: List[str] = []

        q_norm = self._normalize_text(q)
        for key, value in self.long_term.items():
            text = f"{key}: {value}"
            if q_norm in self._normalize_text(text):
                hits.append(text)
            if len(hits) >= limit:
                return hits[:limit]

        scored: List[tuple[float, str]] = []
        for key, value in self.long_term.items():
            text = f"{key}: {value}"
            score = self._local_similarity(q, text)
            if score > 0:
                scored.append((score, text))

        scored.sort(key=lambda x: x[0], reverse=True)
        for score, text in scored[:limit]:
            candidate = f"{text} [local-semantic {score:.3f}]"
            if all(text not in existing for existing in hits):
                hits.append(candidate)
            if len(hits) >= limit:
                break

        return hits[:limit]

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
