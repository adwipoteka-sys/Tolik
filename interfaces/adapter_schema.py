from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4


class AdapterMode(str, Enum):
    DISABLED = "disabled"
    STUB = "stub"
    SIMULATED = "simulated"
    LIVE = "live"


@dataclass(slots=True)
class AdapterSafetyPolicy:
    """Safety guardrails for future-facing external adapters."""

    allow_live_calls: bool = False
    max_prompt_chars: int = 4000
    max_response_chars: int = 4000
    max_problem_size: int = 256
    max_numeric_value: int = 1_000_000
    timeout_seconds: float = 5.0
    redact_emails: bool = True
    redact_long_numbers: bool = False
    blocked_terms: list[str] = field(default_factory=list)
    reject_blocked_terms: bool = True
    truncate_inputs: bool = True
    truncate_outputs: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "AdapterSafetyPolicy":
        if not data:
            return cls()
        return cls(**data)


@dataclass(slots=True)
class AdapterCallRecord:
    call_id: str
    adapter_name: str
    capability: str
    operation: str
    mode: str
    provider: str | None
    status: str
    request_summary: dict[str, Any]
    response_summary: dict[str, Any] | None = None
    reason: str | None = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @classmethod
    def new(
        cls,
        *,
        adapter_name: str,
        capability: str,
        operation: str,
        mode: str,
        provider: str | None,
        status: str,
        request_summary: dict[str, Any],
        response_summary: dict[str, Any] | None = None,
        reason: str | None = None,
    ) -> "AdapterCallRecord":
        return cls(
            call_id=f"adaptercall_{uuid4().hex[:12]}",
            adapter_name=adapter_name,
            capability=capability,
            operation=operation,
            mode=mode,
            provider=provider,
            status=status,
            request_summary=request_summary,
            response_summary=response_summary,
            reason=reason,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class AdapterRuntimeState:
    adapter_name: str
    capability: str
    mode: str
    provider: str | None
    live_ready: bool
    deferred_count: int
    call_count: int
    reason: str | None = None
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class InterfaceRuntimeSpec:
    mode: str = AdapterMode.STUB.value
    provider: str | None = None
    endpoint: str | None = None
    headers: dict[str, str] = field(default_factory=dict)
    headers_env: dict[str, str] = field(default_factory=dict)
    estimated_cost_per_call_usd: float | None = None
    policy: AdapterSafetyPolicy = field(default_factory=AdapterSafetyPolicy)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "InterfaceRuntimeSpec":
        if not data:
            return cls()
        payload = dict(data)
        payload["policy"] = AdapterSafetyPolicy.from_dict(payload.get("policy"))
        return cls(**payload)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["policy"] = self.policy.to_dict()
        return result
