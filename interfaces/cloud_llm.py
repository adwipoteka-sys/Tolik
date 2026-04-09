from __future__ import annotations

import time
from typing import Any, Callable

from interfaces.adapter_schema import AdapterMode, AdapterSafetyPolicy
from interfaces.safe_adapter_base import SafeExternalAdapter


class CloudLLMClient(SafeExternalAdapter):
    """Safe future-facing cloud-LLM adapter."""

    def __init__(
        self,
        mode: str = AdapterMode.STUB.value,
        *,
        provider: str | None = None,
        policy: AdapterSafetyPolicy | None = None,
        ledger: Any | None = None,
        live_transport: Callable[..., Any] | None = None,
    ) -> None:
        super().__init__(
            adapter_name="cloud_llm",
            capability="cloud_llm",
            mode=mode,
            provider=provider,
            policy=policy,
            ledger=ledger,
            live_transport=live_transport,
        )
        self.deferred_requests = self.deferred_queue

    def generate(self, prompt: str, **kwargs: Any) -> dict[str, Any]:
        try:
            sanitized_prompt, prompt_meta = self._sanitize_text(prompt, max_chars=self.policy.max_prompt_chars)
        except ValueError as exc:
            request_summary = {"prompt_preview": str(prompt)[:160], "length": len(str(prompt)), "kwargs_keys": sorted(kwargs.keys())}
            return self._reject(operation="generate", request_summary=request_summary, reason=str(exc))

        request_summary = {
            "prompt_preview": prompt_meta["preview"],
            "length": prompt_meta["length"],
            "redactions": list(prompt_meta["redactions"]),
            "truncated": prompt_meta["truncated"],
            "kwargs_keys": sorted(kwargs.keys()),
            "task": kwargs.get("task"),
        }
        payload = {"prompt": sanitized_prompt, "kwargs": self._ensure_jsonable(dict(kwargs))}

        if self.mode == AdapterMode.DISABLED.value:
            return self._defer(operation="generate", payload=payload, reason="cloud_llm_disabled", request_summary=request_summary)
        if self.mode == AdapterMode.STUB.value:
            return self._ok(
                operation="generate",
                request_summary=request_summary,
                payload={"status": "ok", "mode": "stub", "provider": self.provider, "text": f"[stub-cloud-llm] {sanitized_prompt[:80]}"},
            )
        if self.mode == AdapterMode.SIMULATED.value:
            return self._ok(
                operation="generate",
                request_summary=request_summary,
                payload={"status": "ok", "mode": "simulated", "provider": self.provider, "text": self._simulate_text(sanitized_prompt, kwargs=dict(kwargs))},
            )
        if self.mode != AdapterMode.LIVE.value:
            raise ValueError(f"Unsupported CloudLLMClient mode: {self.mode}")

        readiness_reason = self._prepare_live_call()
        if readiness_reason is not None:
            return self._defer(operation="generate", payload=payload, reason=readiness_reason, request_summary=request_summary)

        route = None
        active_policy = self.policy
        live_transport = self.live_transport
        routed_provider = self.provider
        routed_stage = "full_live"
        if self.canary_guard is not None:
            route = self.canary_guard.select_route()
            active_policy = AdapterSafetyPolicy.from_dict(route["policy"].to_dict())
            active_policy.allow_live_calls = True
            live_transport = route["transport"]
            routed_provider = route["provider"]
            routed_stage = route["rollout_stage"]

        started = time.perf_counter()
        try:
            raw_result = live_transport(sanitized_prompt, **dict(kwargs)) if live_transport is not None else None
        except Exception as exc:  # pragma: no cover - defensive path
            return self._reject(
                operation="generate",
                request_summary=request_summary,
                reason=f"live_transport_error:{exc.__class__.__name__}",
            )
        live_latency_ms = (time.perf_counter() - started) * 1000.0
        live_result = self._finalize_live_result(raw_result, request_summary=request_summary, active_policy=active_policy, fallback_provider=routed_provider)

        if self.canary_guard is not None:
            live_result["canary_rollout"] = self.canary_guard.observe(
                routed_provider=str(routed_provider),
                routed_stage=routed_stage,
                operation="generate",
                request_summary=request_summary,
                live_result=live_result,
                live_latency_ms=live_latency_ms,
                shadow_runner=lambda provider, transport, policy: self._run_shadow_generate(sanitized_prompt, dict(kwargs), transport, provider, policy),
            )
        elif self.post_promotion_guard is not None:
            live_result["post_promotion_monitoring"] = self.post_promotion_guard.observe(
                operation="generate",
                request_summary=request_summary,
                live_result=live_result,
                live_latency_ms=live_latency_ms,
                shadow_runner=lambda provider, transport: self._run_shadow_generate(
                    sanitized_prompt,
                    dict(kwargs),
                    transport,
                    provider,
                    active_policy,
                )[0],
            )
        return live_result

    def _dispatch_ticket(self, ticket: dict[str, Any]) -> dict[str, Any]:
        payload = dict(ticket.get("payload", {}))
        kwargs = dict(payload.get("kwargs", {}))
        return self.generate(str(payload.get("prompt", "")), **kwargs)

    def _simulate_text(self, prompt: str, *, kwargs: dict[str, Any]) -> str:
        text = " ".join(str(prompt).split())
        if kwargs.get("task") == "summarize":
            sentences = [item.strip() for item in text.replace("!", ".").replace("?", ".").split(".") if item.strip()]
            return ". ".join(sentences[:2]) + ("." if sentences else "")
        return text[:120]

    def _extract_live_text(self, raw_result: Any) -> str:
        if isinstance(raw_result, str):
            return raw_result
        if isinstance(raw_result, dict):
            if "text" in raw_result:
                return str(raw_result["text"])
            if "response" in raw_result:
                return str(raw_result["response"])
        return str(raw_result)

    def _finalize_live_result(
        self,
        raw_result: Any,
        *,
        request_summary: dict[str, Any],
        active_policy: AdapterSafetyPolicy,
        fallback_provider: str | None,
    ) -> dict[str, Any]:
        previous_policy = self.policy
        self.policy = active_policy
        try:
            text = self._extract_live_text(raw_result)
            if not text.strip():
                return self._reject(operation="generate", request_summary=request_summary, reason="empty_live_response")
            try:
                text, output_meta = self._sanitize_output_text(text, max_chars=self.policy.max_response_chars)
            except ValueError as exc:
                return self._reject(operation="generate", request_summary=request_summary, reason=str(exc))
            merged_request_summary = {
                **request_summary,
                "output_redactions": list(output_meta["redactions"]),
                "output_truncated": output_meta["truncated"],
            }
            provider = None
            if isinstance(raw_result, dict):
                provider = raw_result.get("provider")
            return self._ok(
                operation="generate",
                request_summary=merged_request_summary,
                payload={"status": "ok", "mode": "live", "provider": provider or fallback_provider or self.provider, "text": text},
            )
        finally:
            self.policy = previous_policy

    def _run_shadow_generate(
        self,
        prompt: str,
        kwargs: dict[str, Any],
        transport: Callable[..., Any],
        provider: str,
        policy: AdapterSafetyPolicy,
    ) -> tuple[dict[str, Any], float]:
        previous_policy = self.policy
        self.policy = AdapterSafetyPolicy.from_dict(policy.to_dict())
        self.policy.allow_live_calls = True
        started = time.perf_counter()
        try:
            try:
                raw_result = transport(prompt, **kwargs)
            except Exception as exc:  # pragma: no cover - defensive path
                return {"status": "rejected", "reason": f"shadow_transport_error:{exc.__class__.__name__}", "provider": provider}, (time.perf_counter() - started) * 1000.0
            text = self._extract_live_text(raw_result)
            if not text.strip():
                return {"status": "rejected", "reason": "empty_shadow_response", "provider": provider}, (time.perf_counter() - started) * 1000.0
            try:
                text, _ = self._sanitize_output_text(text, max_chars=self.policy.max_response_chars)
            except ValueError as exc:
                return {"status": "rejected", "reason": str(exc), "provider": provider}, (time.perf_counter() - started) * 1000.0
            return {"status": "ok", "provider": provider, "text": text}, (time.perf_counter() - started) * 1000.0
        finally:
            self.policy = previous_policy
