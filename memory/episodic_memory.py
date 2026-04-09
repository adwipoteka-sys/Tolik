from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from memory.goal_ledger import GoalLedger
from motivation.goal_schema import Goal


def new_episode_id(prefix: str = "episode") -> str:
    return f"{prefix}_{uuid4().hex[:12]}"


@dataclass(slots=True)
class EpisodeRecord:
    episode_id: str
    goal_id: str
    title: str
    source: str
    kind: str
    success: bool
    cycle: int
    capability: str | None = None
    pattern_key: str | None = None
    lesson: str | None = None
    tags: list[str] = field(default_factory=list)
    goal_payload: dict[str, Any] | None = None
    trace: list[dict[str, Any]] = field(default_factory=list)
    outcome: dict[str, Any] = field(default_factory=dict)
    workspace_excerpt: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["created_at"] = self.created_at.isoformat()
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EpisodeRecord":
        raw = dict(data)
        raw.setdefault("goal_payload", None)
        raw["created_at"] = datetime.fromisoformat(raw["created_at"])
        return cls(**raw)


class EpisodicMemory:
    """Persistent episodic memory for completed goals and execution traces."""

    def __init__(self, ledger: GoalLedger | None = None) -> None:
        self.ledger = ledger
        self._episodes: dict[str, EpisodeRecord] = {}
        if self.ledger is not None:
            self._rehydrate()

    def _rehydrate(self) -> None:
        for payload in self.ledger.load_episodes():
            record = EpisodeRecord.from_dict(payload)
            self._episodes[record.episode_id] = record

    def _persist(self, record: EpisodeRecord) -> EpisodeRecord:
        self._episodes[record.episode_id] = record
        if self.ledger is not None:
            self.ledger.save_episode(record.to_dict())
        return record

    def record_goal_episode(
        self,
        goal: Goal,
        *,
        cycle: int,
        trace: list[dict[str, Any]],
        outcome: dict[str, Any],
        workspace_excerpt: dict[str, Any] | None = None,
        capability: str | None = None,
        pattern_key: str | None = None,
        lesson: str | None = None,
        tags: list[str] | None = None,
    ) -> EpisodeRecord:
        if capability is None:
            capability = _infer_capability(goal, trace)
        record = EpisodeRecord(
            episode_id=new_episode_id(),
            goal_id=goal.goal_id,
            title=goal.title,
            source=goal.source.value,
            kind=goal.kind.value,
            success=bool(outcome.get("success", False)),
            cycle=cycle,
            capability=capability,
            pattern_key=pattern_key,
            lesson=lesson,
            tags=list(tags or goal.tags),
            goal_payload=goal.to_dict(),
            trace=[dict(item) for item in trace],
            outcome=dict(outcome),
            workspace_excerpt=dict(workspace_excerpt or {}),
        )
        return self._persist(record)

    def list_episodes(self) -> list[EpisodeRecord]:
        return sorted(self._episodes.values(), key=lambda item: item.created_at)

    def recent(self, limit: int = 5) -> list[EpisodeRecord]:
        if limit <= 0:
            return []
        return self.list_episodes()[-limit:]

    def get_by_id(self, episode_id: str) -> EpisodeRecord | None:
        return self._episodes.get(episode_id)

    def by_goal(self, goal_id: str) -> list[EpisodeRecord]:
        return [record for record in self.list_episodes() if record.goal_id == goal_id]

    def by_capability(self, capability: str) -> list[EpisodeRecord]:
        return [record for record in self.list_episodes() if record.capability == capability]

    def by_pattern(self, pattern_key: str) -> list[EpisodeRecord]:
        return [record for record in self.list_episodes() if record.pattern_key == pattern_key]

    def successful_by_pattern(self, pattern_key: str) -> list[EpisodeRecord]:
        return [record for record in self.by_pattern(pattern_key) if record.success]

    def support_count(self, pattern_key: str) -> int:
        return len(self.successful_by_pattern(pattern_key))

    def list_lessons(self, pattern_key: str) -> list[str]:
        lessons: list[str] = []
        for record in self.successful_by_pattern(pattern_key):
            if record.lesson:
                lessons.append(record.lesson)
        return lessons

    def pattern_keys(self) -> list[str]:
        keys = {record.pattern_key for record in self.list_episodes() if record.pattern_key}
        return sorted(key for key in keys if key is not None)



def _infer_capability(goal: Goal, trace: list[dict[str, Any]]) -> str | None:
    explicit = str(goal.evidence.get("target_capability", "")).strip()
    if explicit:
        return explicit
    for capability in goal.required_capabilities:
        if capability not in {"classical_planning", "classical_simulation", "local_llm"}:
            return capability
    for item in trace:
        result = dict(item.get("result", {}))
        capability = result.get("capability")
        if isinstance(capability, str) and capability:
            return capability
    return None
