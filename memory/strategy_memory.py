from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from memory.goal_ledger import GoalLedger


def new_strategy_id(prefix: str = "strategy") -> str:
    return f"{prefix}_{uuid4().hex[:12]}"


@dataclass(slots=True)
class StrategyPattern:
    strategy_id: str
    capability: str
    signature: str
    title: str
    template_parameters: dict[str, Any]
    remediation_targets: list[str] = field(default_factory=list)
    source_goal_id: str | None = None
    source_tool_name: str | None = None
    evidence: dict[str, Any] = field(default_factory=dict)
    uses: int = 0
    wins: int = 0
    status: str = "active"
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def touch(self) -> None:
        self.updated_at = datetime.now(timezone.utc)

    def success_rate(self) -> float:
        return self.wins / self.uses if self.uses else 0.0

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["created_at"] = self.created_at.isoformat()
        payload["updated_at"] = self.updated_at.isoformat()
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StrategyPattern":
        raw = dict(data)
        raw["created_at"] = datetime.fromisoformat(raw["created_at"])
        raw["updated_at"] = datetime.fromisoformat(raw["updated_at"])
        return cls(**raw)


class StrategyMemory:
    """Persistent procedural memory for successful remediation strategies."""

    def __init__(self, ledger: GoalLedger | None = None) -> None:
        self.ledger = ledger
        self._patterns_by_id: dict[str, StrategyPattern] = {}
        self._patterns_by_signature: dict[str, StrategyPattern] = {}
        if self.ledger is not None:
            self._rehydrate()

    def _rehydrate(self) -> None:
        for payload in self.ledger.load_strategy_patterns():
            pattern = StrategyPattern.from_dict(payload)
            self._patterns_by_id[pattern.strategy_id] = pattern
            self._patterns_by_signature[pattern.signature] = pattern

    def _persist(self, pattern: StrategyPattern) -> StrategyPattern:
        pattern.touch()
        if self.ledger is not None:
            self.ledger.save_strategy_pattern(pattern.to_dict())
        self._patterns_by_id[pattern.strategy_id] = pattern
        self._patterns_by_signature[pattern.signature] = pattern
        return pattern

    def register_patch_strategy(
        self,
        *,
        failure_signature: str,
        capability: str,
        template_parameters: dict[str, Any],
        remediation_targets: list[str] | None = None,
        source_goal_id: str | None = None,
        source_tool_name: str | None = None,
        evidence: dict[str, Any] | None = None,
    ) -> StrategyPattern:
        if failure_signature in self._patterns_by_signature:
            pattern = self._patterns_by_signature[failure_signature]
            pattern.template_parameters = dict(template_parameters)
            pattern.remediation_targets = list(remediation_targets or pattern.remediation_targets)
            pattern.source_goal_id = source_goal_id or pattern.source_goal_id
            pattern.source_tool_name = source_tool_name or pattern.source_tool_name
            if evidence:
                merged_evidence = dict(pattern.evidence)
                merged_evidence.update(evidence)
                pattern.evidence = merged_evidence
            title = f"Learned strategy for {capability} / {failure_signature}"
            pattern.title = title
            return self._persist(pattern)

        pattern = StrategyPattern(
            strategy_id=new_strategy_id(),
            capability=capability,
            signature=failure_signature,
            title=f"Learned strategy for {capability} / {failure_signature}",
            template_parameters=dict(template_parameters),
            remediation_targets=list(remediation_targets or []),
            source_goal_id=source_goal_id,
            source_tool_name=source_tool_name,
            evidence=dict(evidence or {}),
        )
        return self._persist(pattern)

    def get_by_signature(self, signature: str) -> StrategyPattern | None:
        return self._patterns_by_signature.get(signature)

    def get_by_id(self, strategy_id: str) -> StrategyPattern | None:
        return self._patterns_by_id.get(strategy_id)

    def patterns_for_capability(self, capability: str) -> list[StrategyPattern]:
        return [pattern for pattern in self._patterns_by_id.values() if pattern.capability == capability and pattern.status == "active"]

    def best_for_capability(self, capability: str) -> StrategyPattern | None:
        candidates = self.patterns_for_capability(capability)
        if not candidates:
            return None
        return sorted(
            candidates,
            key=lambda item: (
                item.success_rate(),
                item.wins,
                -item.uses,
                item.updated_at.isoformat(),
                item.strategy_id,
            ),
            reverse=True,
        )[0]

    def select_parameters(
        self,
        capability: str,
        *,
        preferred_signature: str | None = None,
        allow_capability_fallback: bool = False,
    ) -> tuple[dict[str, Any] | None, StrategyPattern | None]:
        pattern: StrategyPattern | None = None
        if preferred_signature:
            pattern = self.get_by_signature(preferred_signature)
        if pattern is None and allow_capability_fallback:
            pattern = self.best_for_capability(capability)
        if pattern is None:
            return None, None
        return dict(pattern.template_parameters), pattern

    def record_outcome(self, strategy_id: str, *, passed: bool) -> StrategyPattern | None:
        pattern = self.get_by_id(strategy_id)
        if pattern is None:
            return None
        pattern.uses += 1
        if passed:
            pattern.wins += 1
        return self._persist(pattern)

    def list_patterns(self) -> list[StrategyPattern]:
        return list(self._patterns_by_id.values())
