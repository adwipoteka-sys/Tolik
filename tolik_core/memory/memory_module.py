from __future__ import annotations

import json
import math
import os
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional


class MemoryModule:
    """Short-term + long-term memory with JSON persistence + semantic search."""

    def __init__(self, short_term_limit: int = 25, storage_dir: str = "data/runtime") -> None:
        self.short_term_limit = short_term_limit
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self.facts_path = self.storage_dir / "facts.json"
        self.events_path = self.storage_dir / "events.json"
        self.semantic_path = self.storage_dir / "semantic_memory.json"

        self.short_term: Deque[Dict[str, Any]] = deque(maxlen=short_term_limit)
        self.long_term: Dict[str, Any] = {}
        self.semantic_items: List[Dict[str, Any]] = []

        self.embed_model = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small").strip()
        self.api_key = os.getenv("OPENAI_API_KEY", "").strip()
        self._client = None

        if self.api_key:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=self.api_key)
            except Exception:
                self._client = None

        self._load()

        if self._client and (not self.semantic_items or len(self.semantic_items) < len(self.long_term)):
            self.reindex_all()

    def _load(self) -> None:
        if self.facts_path.exists():
            self.long_term = json.loads(self.facts_path.read_text(encoding="utf-8"))

        if self.events_path.exists():
            items = json.loads(self.events_path.read_text(encoding="utf-8"))
            self.short_term = deque(items[-self.short_term_limit :], maxlen=self.short_term_limit)

        if self.semantic_path.exists():
            self.semantic_items = json.loads(self.semantic_path.read_text(encoding="utf-8"))

    def _write_json(self, path: Path, data: Any) -> None:
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def save(self) -> None:
        self._write_json(self.facts_path, self.long_term)
        self._write_json(self.events_path, list(self.short_term))
        self._write_json(self.semantic_path, self.semantic_items)

    def _embed_text(self, text: str) -> Optional[List[float]]:
        if not self._client or not text.strip():
            return None
        try:
            response = self._client.embeddings.create(
                model=self.embed_model,
                input=text,
                encoding_format="float",
            )
            return list(response.data[0].embedding)
        except Exception:
            return None

    @staticmethod
    def _cosine_similarity(a: List[float], b: List[float]) -> float:
        if not a or not b or len(a) != len(b):
            return -1.0

        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))

        if norm_a == 0 or norm_b == 0:
            return -1.0

        return dot / (norm_a * norm_b)

    def _upsert_semantic_item(self, item_id: str, text: str) -> None:
        vector = self._embed_text(text)
        if vector is None:
            return

        self.semantic_items = [item for item in self.semantic_items if item.get("id") != item_id]
        self.semantic_items.append(
            {
                "id": item_id,
                "text": text,
                "vector": vector,
            }
        )

    def reindex_all(self) -> None:
        self.semantic_items = []
        for key, value in self.long_term.items():
            self._upsert_semantic_item(str(key), f"{key}: {value}")
        self.save()

    def remember_event(self, event: Dict[str, Any]) -> None:
        self.short_term.append(event)
        self.save()

    def store_fact(self, key: str, value: Any) -> None:
        self.long_term[key] = value
        self._upsert_semantic_item(str(key), f"{key}: {value}")
        self.save()

    def recall_fact(self, key: str) -> Optional[Any]:
        return self.long_term.get(key)

    def search_facts(self, query: str, limit: int = 5) -> List[str]:
        q = query.lower().strip()
        hits: List[str] = []

        # Exact / substring matches first
        for key, value in self.long_term.items():
            text = f"{key}: {value}"
            if not q or q in key.lower() or q in str(value).lower():
                hits.append(text)
            if len(hits) >= limit:
                return hits[:limit]

        # Semantic matches next
        if self._client and self.semantic_items and q:
            q_vec = self._embed_text(query)
            if q_vec is not None:
                scored: List[tuple[float, str]] = []
                for item in self.semantic_items:
                    score = self._cosine_similarity(q_vec, item["vector"])
                    scored.append((score, item["text"]))

                scored.sort(key=lambda x: x[0], reverse=True)

                for score, text in scored[:limit]:
                    candidate = f"{text} [semantic {score:.3f}]"
                    if text not in hits:
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
            if self._client:
                self.reindex_all()
            else:
                self.save()
