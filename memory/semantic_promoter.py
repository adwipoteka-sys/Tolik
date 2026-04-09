from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from memory.episodic_memory import EpisodeRecord, EpisodicMemory
from memory.goal_ledger import GoalLedger
from memory.memory_module import MemoryModule
from reasoning.reasoning_module import ReasoningModule


def new_promotion_id(prefix: str = "semantic") -> str:
    return f"{prefix}_{uuid4().hex[:12]}"


@dataclass(slots=True)
class SemanticPromotion:
    promotion_id: str
    pattern_key: str
    fact_key: str
    summary: str
    support_count: int
    confidence: float
    source_episode_ids: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["created_at"] = self.created_at.isoformat()
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SemanticPromotion":
        raw = dict(data)
        raw["created_at"] = datetime.fromisoformat(raw["created_at"])
        return cls(**raw)


class SemanticPromoter:
    """Promotes repeated successful episode patterns into reusable semantic facts."""

    def __init__(
        self,
        *,
        reasoning: ReasoningModule | None = None,
        ledger: GoalLedger | None = None,
        min_support: int = 2,
    ) -> None:
        self.reasoning = reasoning or ReasoningModule()
        self.ledger = ledger
        self.min_support = min_support
        self._promotions: dict[str, SemanticPromotion] = {}
        if self.ledger is not None:
            self._rehydrate()

    def _rehydrate(self) -> None:
        for payload in self.ledger.load_semantic_promotions():
            promotion = SemanticPromotion.from_dict(payload)
            self._promotions[promotion.pattern_key] = promotion

    def list_promotions(self) -> list[SemanticPromotion]:
        return sorted(self._promotions.values(), key=lambda item: item.created_at)

    def get(self, pattern_key: str) -> SemanticPromotion | None:
        return self._promotions.get(pattern_key)

    def consolidate(
        self,
        *,
        memory: MemoryModule,
        episodic_memory: EpisodicMemory,
        pattern_key: str | None = None,
    ) -> list[SemanticPromotion]:
        groups: dict[str, list[EpisodeRecord]] = {}
        if pattern_key is not None:
            groups[pattern_key] = episodic_memory.successful_by_pattern(pattern_key)
        else:
            for key in episodic_memory.pattern_keys():
                groups[key] = episodic_memory.successful_by_pattern(key)

        promotions: list[SemanticPromotion] = []
        for key, episodes in groups.items():
            if key in self._promotions:
                continue
            if len(episodes) < self.min_support:
                continue
            proposal = self._build_promotion(key, episodes)
            assessment = self.reasoning.assess(proposal.summary, context={"support_count": len(episodes)})
            if not assessment["consistent"] or assessment["insufficient_evidence"]:
                continue
            proposal.confidence = round(min(0.99, 0.45 + 0.10 * len(episodes) + 0.30 * assessment["confidence"]), 3)
            if hasattr(memory, "store_semantic_promotion"):
                memory.store_semantic_promotion(proposal.fact_key, proposal.to_dict())
            else:
                memory.store_fact(proposal.fact_key, proposal.to_dict())
            self._promotions[key] = proposal
            promotions.append(proposal)
            if self.ledger is not None:
                self.ledger.save_semantic_promotion(proposal.to_dict())
        return promotions

    def _build_promotion(self, pattern_key: str, episodes: list[EpisodeRecord]) -> SemanticPromotion:
        lessons = [lesson for lesson in (record.lesson for record in episodes) if lesson]
        if lessons:
            summary, _count = Counter(lessons).most_common(1)[0]
        else:
            capability = next((record.capability for record in episodes if record.capability), "unknown capability")
            summary = f"Repeated successful episodes indicate a stable operating procedure for {capability}."
        tags = sorted({tag for record in episodes for tag in record.tags})
        fact_key = f"semantic:{_sanitize_pattern_key(pattern_key)}"
        return SemanticPromotion(
            promotion_id=new_promotion_id(),
            pattern_key=pattern_key,
            fact_key=fact_key,
            summary=summary,
            support_count=len(episodes),
            confidence=0.0,
            source_episode_ids=[record.episode_id for record in episodes],
            tags=tags,
        )



def _sanitize_pattern_key(pattern_key: str) -> str:
    return pattern_key.replace("/", "_").replace(":", "_").replace("|", "__")
