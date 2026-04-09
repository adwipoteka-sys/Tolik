from __future__ import annotations

from dataclasses import asdict, dataclass, field

from memory.goal_ledger import GoalLedger
from metacognition.failure_miner import FailureCase
from metacognition.postmortem_clusterer import FailureCluster


@dataclass(slots=True)
class CurriculumPattern:
    signature: str
    capability: str
    status: str = "open"
    occurrence_count: int = 0
    latest_goal_id: str | None = None
    latest_tool_name: str | None = None
    remediation_goals: list[str] = field(default_factory=list)
    closed_by_tool: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class CurriculumRegistry:
    """Tracks open and closed failure patterns for self-improvement."""

    def __init__(self, ledger: GoalLedger | None = None) -> None:
        self.ledger = ledger
        self._patterns: dict[str, CurriculumPattern] = {}
        if self.ledger is not None:
            for raw in self.ledger.load_failure_patterns():
                pattern = CurriculumPattern(**raw)
                self._patterns[pattern.signature] = pattern

    def _persist(self, pattern: CurriculumPattern) -> None:
        if self.ledger is not None:
            self.ledger.save_failure_pattern(pattern.to_dict())

    def register_failure(self, failure: FailureCase, cluster: FailureCluster) -> CurriculumPattern:
        pattern = self._patterns.get(failure.signature)
        if pattern is None:
            pattern = CurriculumPattern(signature=failure.signature, capability=failure.capability)
            self._patterns[failure.signature] = pattern

        pattern.status = "open"
        pattern.closed_by_tool = None
        pattern.occurrence_count = cluster.occurrence_count
        pattern.latest_goal_id = failure.goal_id
        pattern.latest_tool_name = failure.tool_name
        self._persist(pattern)
        return pattern

    def attach_remediation_goal(self, signature: str, title: str) -> None:
        pattern = self._patterns.setdefault(signature, CurriculumPattern(signature=signature, capability="unknown"))
        if title not in pattern.remediation_goals:
            pattern.remediation_goals.append(title)
            self._persist(pattern)

    def mark_closed(self, signature: str, stable_tool_name: str) -> None:
        pattern = self._patterns.get(signature)
        if pattern is None:
            return
        pattern.status = "closed"
        pattern.closed_by_tool = stable_tool_name
        self._persist(pattern)

    def is_open(self, signature: str) -> bool:
        pattern = self._patterns.get(signature)
        return bool(pattern and pattern.status == "open")

    def get(self, signature: str) -> CurriculumPattern | None:
        return self._patterns.get(signature)

    def all_patterns(self) -> list[CurriculumPattern]:
        return sorted(self._patterns.values(), key=lambda item: item.signature)

    def open_patterns(self) -> list[CurriculumPattern]:
        return [item for item in self.all_patterns() if item.status == "open"]

    def closed_patterns(self) -> list[CurriculumPattern]:
        return [item for item in self.all_patterns() if item.status == "closed"]
