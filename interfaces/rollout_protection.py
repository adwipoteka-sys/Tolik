from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from motivation.operator_charter import OperatorCharter


@dataclass(slots=True)
class ProviderProtectionStatus:
    provider: str
    blocked_until_rollout_index: int
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class RolloutProtectionRecord:
    protection_id: str
    adapter_name: str
    affected_provider: str
    fallback_provider: str | None
    trigger_type: str
    rollout_index: int
    cooldown_rollouts: int
    cooldown_until_rollout_index: int
    provider_recent_failure_count: int
    anti_flap_window_rollouts: int
    anti_flap_repeat_failures: int
    anti_flap_active: bool
    anti_flap_until_rollout_index: int | None
    reasons: list[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @classmethod
    def new(
        cls,
        *,
        adapter_name: str,
        affected_provider: str,
        fallback_provider: str | None,
        trigger_type: str,
        rollout_index: int,
        cooldown_rollouts: int,
        cooldown_until_rollout_index: int,
        provider_recent_failure_count: int,
        anti_flap_window_rollouts: int,
        anti_flap_repeat_failures: int,
        anti_flap_active: bool,
        anti_flap_until_rollout_index: int | None,
        reasons: list[str] | None = None,
    ) -> "RolloutProtectionRecord":
        return cls(
            protection_id=f"protect_{uuid4().hex[:12]}",
            adapter_name=adapter_name,
            affected_provider=affected_provider,
            fallback_provider=fallback_provider,
            trigger_type=str(trigger_type),
            rollout_index=int(rollout_index),
            cooldown_rollouts=int(cooldown_rollouts),
            cooldown_until_rollout_index=int(cooldown_until_rollout_index),
            provider_recent_failure_count=int(provider_recent_failure_count),
            anti_flap_window_rollouts=int(anti_flap_window_rollouts),
            anti_flap_repeat_failures=int(anti_flap_repeat_failures),
            anti_flap_active=bool(anti_flap_active),
            anti_flap_until_rollout_index=None if anti_flap_until_rollout_index is None else int(anti_flap_until_rollout_index),
            reasons=list(reasons or []),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RolloutProtectionRecord":
        return cls(**payload)


@dataclass(slots=True)
class RolloutProtectionEvaluation:
    adapter_name: str
    current_rollout_index: int
    blocked_providers: dict[str, ProviderProtectionStatus] = field(default_factory=dict)
    allow_canary: bool = True
    anti_flap_active: bool = False
    anti_flap_until_rollout_index: int | None = None
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "adapter_name": self.adapter_name,
            "current_rollout_index": self.current_rollout_index,
            "blocked_providers": {name: status.to_dict() for name, status in self.blocked_providers.items()},
            "allow_canary": self.allow_canary,
            "anti_flap_active": self.anti_flap_active,
            "anti_flap_until_rollout_index": self.anti_flap_until_rollout_index,
            "reasons": list(self.reasons),
        }


class RolloutProtectionAdvisor:
    """Tracks rollback/demotion cooldowns and suppresses rollout flapping."""

    def __init__(self, *, adapter_name: str, charter: OperatorCharter, ledger: Any | None = None) -> None:
        self.adapter_name = adapter_name
        self.charter = charter
        self.ledger = ledger

    def evaluate(self, *, candidate_providers: list[str] | None = None) -> RolloutProtectionEvaluation:
        current_rollout_index = self._next_rollout_index()
        provider_filter = None if candidate_providers is None else set(candidate_providers)
        blocked: dict[str, ProviderProtectionStatus] = {}
        anti_flap_until: int | None = None
        reasons: list[str] = []
        for record in self._load_records():
            if record.adapter_name != self.adapter_name:
                continue
            if provider_filter is not None and record.affected_provider not in provider_filter:
                continue
            block_until: int | None = None
            provider_reasons: list[str] = []
            if current_rollout_index <= record.cooldown_until_rollout_index:
                block_until = record.cooldown_until_rollout_index
                provider_reasons.append(
                    f"cooldown_active:{record.trigger_type}:until_rollout_{record.cooldown_until_rollout_index}"
                )
            if (
                record.anti_flap_until_rollout_index is not None
                and current_rollout_index <= record.anti_flap_until_rollout_index
            ):
                anti_flap_until = max(anti_flap_until or 0, record.anti_flap_until_rollout_index)
                block_until = max(block_until or 0, record.anti_flap_until_rollout_index)
                provider_reasons.append(
                    f"anti_flap_hold:{record.trigger_type}:until_rollout_{record.anti_flap_until_rollout_index}"
                )
            if block_until is None:
                continue
            existing = blocked.get(record.affected_provider)
            if existing is None or block_until > existing.blocked_until_rollout_index:
                blocked[record.affected_provider] = ProviderProtectionStatus(
                    provider=record.affected_provider,
                    blocked_until_rollout_index=block_until,
                    reasons=provider_reasons,
                )
        anti_flap_active = anti_flap_until is not None
        if anti_flap_active:
            reasons.append(f"anti_flap_canary_freeze:until_rollout_{anti_flap_until}")
        if blocked:
            reasons.extend(
                f"provider_blocked:{provider}:until_rollout_{status.blocked_until_rollout_index}"
                for provider, status in sorted(blocked.items())
            )
        return RolloutProtectionEvaluation(
            adapter_name=self.adapter_name,
            current_rollout_index=current_rollout_index,
            blocked_providers=blocked,
            allow_canary=not anti_flap_active,
            anti_flap_active=anti_flap_active,
            anti_flap_until_rollout_index=anti_flap_until,
            reasons=reasons,
        )

    def record_protective_event(
        self,
        *,
        affected_provider: str,
        fallback_provider: str | None,
        trigger_type: str,
        reasons: list[str] | None = None,
    ) -> RolloutProtectionRecord:
        rollout_index = self._current_rollout_index()
        records = self._load_records()
        lookback_floor = max(1, rollout_index - self.charter.anti_flap_window_rollouts + 1)
        recent_provider_events = [
            record
            for record in records
            if record.adapter_name == self.adapter_name
            and record.affected_provider == affected_provider
            and record.rollout_index >= lookback_floor
        ]
        provider_recent_failure_count = len(recent_provider_events) + 1
        cooldown_rollouts = int(self.charter.rollback_cooldown_rollouts)
        cooldown_until_rollout_index = rollout_index + cooldown_rollouts
        anti_flap_active = provider_recent_failure_count >= self.charter.anti_flap_repeat_failures
        anti_flap_until_rollout_index = (
            rollout_index + int(self.charter.anti_flap_freeze_rollouts) if anti_flap_active else None
        )
        record_reasons = list(reasons or [])
        record_reasons.append(f"cooldown_until_rollout:{cooldown_until_rollout_index}")
        if anti_flap_active and anti_flap_until_rollout_index is not None:
            record_reasons.append(f"anti_flap_until_rollout:{anti_flap_until_rollout_index}")
        record = RolloutProtectionRecord.new(
            adapter_name=self.adapter_name,
            affected_provider=affected_provider,
            fallback_provider=fallback_provider,
            trigger_type=trigger_type,
            rollout_index=rollout_index,
            cooldown_rollouts=cooldown_rollouts,
            cooldown_until_rollout_index=cooldown_until_rollout_index,
            provider_recent_failure_count=provider_recent_failure_count,
            anti_flap_window_rollouts=self.charter.anti_flap_window_rollouts,
            anti_flap_repeat_failures=self.charter.anti_flap_repeat_failures,
            anti_flap_active=anti_flap_active,
            anti_flap_until_rollout_index=anti_flap_until_rollout_index,
            reasons=record_reasons,
        )
        self._persist(record)
        return record

    def _current_rollout_index(self) -> int:
        if self.ledger is None or not hasattr(self.ledger, "load_interface_rollout_decisions"):
            return 0
        decisions = [
            payload
            for payload in self.ledger.load_interface_rollout_decisions()
            if payload.get("adapter_name") == self.adapter_name
        ]
        return len(decisions)

    def _next_rollout_index(self) -> int:
        return self._current_rollout_index() + 1

    def _load_records(self) -> list[RolloutProtectionRecord]:
        if self.ledger is None or not hasattr(self.ledger, "load_interface_rollout_protections"):
            return []
        return [
            RolloutProtectionRecord.from_dict(payload)
            for payload in self.ledger.load_interface_rollout_protections()
        ]

    def _persist(self, record: RolloutProtectionRecord) -> None:
        if self.ledger is not None and hasattr(self.ledger, "save_interface_rollout_protection"):
            self.ledger.save_interface_rollout_protection(record.to_dict())
