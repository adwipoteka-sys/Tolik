from __future__ import annotations

import json
import re
from typing import Any, Callable

from interfaces.adapter_schema import AdapterCallRecord, AdapterMode, AdapterRuntimeState, AdapterSafetyPolicy


_UNSET = object()


class SafeExternalAdapter:
    """Shared safety/audit logic for future live external interfaces."""

    def __init__(
        self,
        *,
        adapter_name: str,
        capability: str,
        mode: str = AdapterMode.STUB.value,
        provider: str | None = None,
        policy: AdapterSafetyPolicy | None = None,
        ledger: Any | None = None,
        live_transport: Callable[..., Any] | None = None,
    ) -> None:
        self.adapter_name = adapter_name
        self.capability = capability
        self.mode = AdapterMode(mode).value
        self.provider = provider
        self.policy = policy or AdapterSafetyPolicy()
        self.ledger = ledger
        self.live_transport = live_transport
        self.call_history: list[AdapterCallRecord] = []
        self.deferred_queue: list[dict[str, Any]] = []
        self.canary_guard: Any | None = None
        self.post_promotion_guard: Any | None = None
        self._persist_state(self._live_readiness_reason())

    def attach_canary_guard(self, guard: Any | None) -> None:
        self.canary_guard = guard
        self._persist_state(self._live_readiness_reason())

    def attach_post_promotion_guard(self, guard: Any | None) -> None:
        self.post_promotion_guard = guard
        self._persist_state(self._live_readiness_reason())

    def set_mode(
        self,
        mode: str,
        *,
        provider: str | None | object = _UNSET,
        live_transport: Callable[..., Any] | None | object = _UNSET,
    ) -> None:
        self.mode = AdapterMode(mode).value
        if provider is not _UNSET:
            self.provider = None if provider is None else str(provider)
        if live_transport is not _UNSET:
            self.live_transport = live_transport if callable(live_transport) else None
        self._persist_state(self._live_readiness_reason())

    def summary(self) -> dict[str, Any]:
        summary: dict[str, Any] = {
            "adapter_name": self.adapter_name,
            "capability": self.capability,
            "mode": self.mode,
            "provider": self.provider,
            "live_ready": self._live_ready(),
            "deferred_count": len(self.deferred_queue),
            "call_count": len(self.call_history),
            "reason": self._live_readiness_reason(),
        }
        if self.canary_guard is not None and hasattr(self.canary_guard, "summary"):
            summary["canary_guard"] = self.canary_guard.summary()
        if self.post_promotion_guard is not None and hasattr(self.post_promotion_guard, "summary"):
            summary["post_promotion_guard"] = self.post_promotion_guard.summary()
        return summary

    def replay_deferred(self, *, limit: int | None = None) -> list[dict[str, Any]]:
        replayed: list[dict[str, Any]] = []
        if not self.deferred_queue:
            return replayed
        if not self._live_ready():
            self._persist_state(self._live_readiness_reason())
            return replayed
        remaining: list[dict[str, Any]] = []
        budget = limit if limit is not None else len(self.deferred_queue)
        for index, ticket in enumerate(self.deferred_queue):
            if index >= budget:
                remaining.append(ticket)
                continue
            result = self._dispatch_ticket(ticket)
            replayed.append(result)
            if result.get("status") != "ok":
                remaining.append(ticket)
        self.deferred_queue = remaining
        self._persist_state(self._live_readiness_reason())
        return replayed

    def _dispatch_ticket(self, ticket: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    def _live_ready(self) -> bool:
        return self.mode == AdapterMode.LIVE.value and self.policy.allow_live_calls and self.live_transport is not None

    def _live_readiness_reason(self) -> str | None:
        if self.mode != AdapterMode.LIVE.value:
            return f"mode:{self.mode}"
        if not self.policy.allow_live_calls:
            return "live_calls_not_allowed"
        if self.live_transport is None:
            return "live_transport_missing"
        return None

    def _persist_state(self, reason: str | None) -> None:
        if self.ledger is None:
            return
        if hasattr(self.ledger, "save_interface_state"):
            state = AdapterRuntimeState(
                adapter_name=self.adapter_name,
                capability=self.capability,
                mode=self.mode,
                provider=self.provider,
                live_ready=self._live_ready(),
                deferred_count=len(self.deferred_queue),
                call_count=len(self.call_history),
                reason=reason,
            )
            self.ledger.save_interface_state(state.to_dict())

    def _record_call(
        self,
        *,
        operation: str,
        status: str,
        request_summary: dict[str, Any],
        response_summary: dict[str, Any] | None = None,
        reason: str | None = None,
    ) -> AdapterCallRecord:
        record = AdapterCallRecord.new(
            adapter_name=self.adapter_name,
            capability=self.capability,
            operation=operation,
            mode=self.mode,
            provider=self.provider,
            status=status,
            request_summary=request_summary,
            response_summary=response_summary,
            reason=reason,
        )
        self.call_history.append(record)
        if self.ledger is not None and hasattr(self.ledger, "save_interface_call"):
            self.ledger.save_interface_call(record.to_dict())
        self._persist_state(reason)
        return record

    def _defer(self, *, operation: str, payload: dict[str, Any], reason: str, request_summary: dict[str, Any]) -> dict[str, Any]:
        ticket = {
            "adapter_name": self.adapter_name,
            "capability": self.capability,
            "operation": operation,
            "payload": payload,
            "reason": reason,
        }
        self.deferred_queue.append(ticket)
        record = self._record_call(operation=operation, status="deferred", request_summary=request_summary, reason=reason)
        return {"status": "deferred", "reason": reason, "ticket": ticket, "call_id": record.call_id}

    def _reject(self, *, operation: str, request_summary: dict[str, Any], reason: str) -> dict[str, Any]:
        record = self._record_call(operation=operation, status="rejected", request_summary=request_summary, reason=reason)
        return {"status": "rejected", "reason": reason, "call_id": record.call_id}

    def _ok(
        self,
        *,
        operation: str,
        request_summary: dict[str, Any],
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        record = self._record_call(
            operation=operation,
            status="ok",
            request_summary=request_summary,
            response_summary=self._summarize_payload(payload),
        )
        return {**payload, "call_id": record.call_id}

    def _summarize_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        summary: dict[str, Any] = {}
        for key, value in payload.items():
            if key == "text":
                summary["text_preview"] = str(value)[:120]
                summary["text_length"] = len(str(value))
            elif isinstance(value, (str, int, float, bool)) or value is None:
                summary[key] = value
            elif isinstance(value, list):
                summary[key] = {"count": len(value), "preview": value[:3]}
            elif isinstance(value, dict):
                summary[key] = {"keys": sorted(value.keys())[:10]}
            else:
                summary[key] = str(type(value).__name__)
        return summary

    def _sanitize_output_text(self, text: str, *, max_chars: int) -> tuple[str, dict[str, Any]]:
        cleaned = " ".join(str(text).split())
        redactions: list[str] = []
        if self.policy.redact_emails:
            new_cleaned = re.sub(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "[redacted-email]", cleaned)
            if new_cleaned != cleaned:
                redactions.append("email")
            cleaned = new_cleaned
        if self.policy.redact_long_numbers:
            new_cleaned = re.sub(r"\b\d{8,}\b", "[redacted-number]", cleaned)
            if new_cleaned != cleaned:
                redactions.append("long_number")
            cleaned = new_cleaned
        blocked = [term for term in self.policy.blocked_terms if term and term.lower() in cleaned.lower()]
        if blocked and self.policy.reject_blocked_terms:
            raise ValueError(f"blocked_output_terms:{blocked}")
        truncated = False
        if len(cleaned) > max_chars:
            if not self.policy.truncate_outputs:
                raise ValueError("response_too_large")
            cleaned = cleaned[:max_chars]
            truncated = True
        meta = {
            "preview": cleaned[:160],
            "length": len(cleaned),
            "redactions": redactions,
            "truncated": truncated,
            "blocked_terms": blocked,
        }
        return cleaned, meta

    def _sanitize_text(self, text: str, *, max_chars: int) -> tuple[str, dict[str, Any]]:
        cleaned = " ".join(str(text).split())
        redactions: list[str] = []
        if self.policy.redact_emails:
            new_cleaned = re.sub(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "[redacted-email]", cleaned)
            if new_cleaned != cleaned:
                redactions.append("email")
            cleaned = new_cleaned
        if self.policy.redact_long_numbers:
            new_cleaned = re.sub(r"\b\d{8,}\b", "[redacted-number]", cleaned)
            if new_cleaned != cleaned:
                redactions.append("long_number")
            cleaned = new_cleaned
        blocked = [term for term in self.policy.blocked_terms if term and term.lower() in cleaned.lower()]
        if blocked and self.policy.reject_blocked_terms:
            raise ValueError(f"blocked_terms:{blocked}")
        truncated = False
        if len(cleaned) > max_chars:
            if not self.policy.truncate_inputs:
                raise ValueError("input_too_large")
            cleaned = cleaned[:max_chars]
            truncated = True
        meta = {
            "preview": cleaned[:160],
            "length": len(cleaned),
            "redactions": redactions,
            "truncated": truncated,
            "blocked_terms": blocked,
        }
        return cleaned, meta

    def _ensure_jsonable(self, value: Any) -> Any:
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        if isinstance(value, list):
            return [self._ensure_jsonable(item) for item in value]
        if isinstance(value, dict):
            return {str(key): self._ensure_jsonable(item) for key, item in value.items()}
        if hasattr(value, "to_dict"):
            return self._ensure_jsonable(value.to_dict())
        try:
            json.dumps(value)
            return value
        except TypeError:
            return str(value)

    def _prepare_live_call(self) -> str | None:
        reason = self._live_readiness_reason()
        if reason is not None:
            return reason
        return None
