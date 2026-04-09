from __future__ import annotations

import time
from typing import Any, Callable

from interfaces.adapter_schema import AdapterMode, AdapterSafetyPolicy
from interfaces.safe_adapter_base import SafeExternalAdapter


class QuantumSolver(SafeExternalAdapter):
    """Safe future-facing quantum adapter with live/mock switching."""

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
            adapter_name="quantum_solver",
            capability="quantum_solver",
            mode=mode,
            provider=provider,
            policy=policy,
            ledger=ledger,
            live_transport=live_transport,
        )

    def solve_optimization(self, problem: dict[str, Any]) -> dict[str, Any]:
        values = [float(value) for value in problem.get("values", [])]
        request_summary = {"problem_keys": sorted(problem.keys()), "value_count": len(values), "value_preview": values[:5]}
        payload = {"values": list(values), "problem": self._ensure_jsonable(problem)}
        if len(values) > self.policy.max_problem_size:
            return self._reject(operation="solve_optimization", request_summary=request_summary, reason="problem_too_large")

        if self.mode == AdapterMode.DISABLED.value:
            return self._defer(operation="solve_optimization", payload=payload, reason="quantum_solver_disabled", request_summary=request_summary)
        if self.mode == AdapterMode.STUB.value:
            best = min(values) if values else None
            return self._ok(
                operation="solve_optimization",
                request_summary=request_summary,
                payload={"status": "ok", "mode": "stub", "provider": self.provider, "best_value": best},
            )
        if self.mode == AdapterMode.SIMULATED.value:
            best = min(values) if values else None
            best_index = values.index(best) if best is not None else None
            return self._ok(
                operation="solve_optimization",
                request_summary=request_summary,
                payload={"status": "ok", "mode": "simulated", "provider": self.provider, "best_value": best, "best_index": best_index},
            )
        if self.mode != AdapterMode.LIVE.value:
            raise ValueError(f"Unsupported QuantumSolver mode: {self.mode}")
        readiness_reason = self._prepare_live_call()
        if readiness_reason is not None:
            return self._defer(operation="solve_optimization", payload=payload, reason=readiness_reason, request_summary=request_summary)

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
            raw_result = live_transport("solve_optimization", payload) if live_transport is not None else None
        except Exception as exc:  # pragma: no cover - defensive path
            return self._reject(operation="solve_optimization", request_summary=request_summary, reason=f"live_transport_error:{exc.__class__.__name__}")
        live_latency_ms = (time.perf_counter() - started) * 1000.0
        optimization_result = self._extract_optimization_result(raw_result, values)
        if optimization_result is None:
            live_result = self._reject(operation="solve_optimization", request_summary=request_summary, reason="invalid_live_result")
        else:
            best_value, best_index = optimization_result
            live_result = self._ok(
                operation="solve_optimization",
                request_summary=request_summary,
                payload={"status": "ok", "mode": "live", "provider": self._extract_provider(raw_result) or routed_provider, "best_value": best_value, "best_index": best_index},
            )
        if self.canary_guard is not None:
            live_result["canary_rollout"] = self.canary_guard.observe(
                routed_provider=str(routed_provider),
                routed_stage=routed_stage,
                operation="solve_optimization",
                request_summary=request_summary,
                live_result=live_result,
                live_latency_ms=live_latency_ms,
                shadow_runner=lambda provider, transport, policy: self._run_shadow_quantum(
                    "solve_optimization",
                    payload,
                    transport,
                    provider,
                    values=values,
                ),
            )
        elif self.post_promotion_guard is not None:
            live_result["post_promotion_monitoring"] = self.post_promotion_guard.observe(
                operation="solve_optimization",
                request_summary=request_summary,
                live_result=live_result,
                live_latency_ms=live_latency_ms,
                shadow_runner=lambda provider, transport: self._run_shadow_quantum(
                    "solve_optimization",
                    payload,
                    transport,
                    provider,
                    values=values,
                )[0],
            )
        return live_result

    def factorize(self, value: int) -> dict[str, Any]:
        request_summary = {"value": int(value)}
        payload = {"value": int(value)}
        if abs(int(value)) > self.policy.max_numeric_value:
            return self._reject(operation="factorize", request_summary=request_summary, reason="value_too_large")
        if self.mode == AdapterMode.DISABLED.value:
            return self._defer(operation="factorize", payload=payload, reason="quantum_solver_disabled", request_summary=request_summary)
        if self.mode in {AdapterMode.STUB.value, AdapterMode.SIMULATED.value}:
            return self._ok(
                operation="factorize",
                request_summary=request_summary,
                payload={"status": "ok", "mode": self.mode, "provider": self.provider, "factors": self._classical_factorize(int(value))},
            )
        if self.mode != AdapterMode.LIVE.value:
            raise ValueError(f"Unsupported QuantumSolver mode: {self.mode}")
        readiness_reason = self._prepare_live_call()
        if readiness_reason is not None:
            return self._defer(operation="factorize", payload=payload, reason=readiness_reason, request_summary=request_summary)

        live_transport = self.live_transport
        routed_provider = self.provider
        routed_stage = "full_live"
        if self.canary_guard is not None:
            route = self.canary_guard.select_route()
            live_transport = route["transport"]
            routed_provider = route["provider"]
            routed_stage = route["rollout_stage"]

        started = time.perf_counter()
        try:
            raw_result = live_transport("factorize", payload) if live_transport is not None else None
        except Exception as exc:  # pragma: no cover - defensive path
            return self._reject(operation="factorize", request_summary=request_summary, reason=f"live_transport_error:{exc.__class__.__name__}")
        live_latency_ms = (time.perf_counter() - started) * 1000.0
        factors = self._extract_factor_result(raw_result, int(value))
        if factors is None:
            live_result = self._reject(operation="factorize", request_summary=request_summary, reason="invalid_live_result")
        else:
            live_result = self._ok(
                operation="factorize",
                request_summary=request_summary,
                payload={"status": "ok", "mode": "live", "provider": self._extract_provider(raw_result) or routed_provider, "factors": factors},
            )
        if self.canary_guard is not None:
            live_result["canary_rollout"] = self.canary_guard.observe(
                routed_provider=str(routed_provider),
                routed_stage=routed_stage,
                operation="factorize",
                request_summary=request_summary,
                live_result=live_result,
                live_latency_ms=live_latency_ms,
                shadow_runner=lambda provider, transport, policy: self._run_shadow_quantum(
                    "factorize",
                    payload,
                    transport,
                    provider,
                    original_value=int(value),
                ),
            )
        elif self.post_promotion_guard is not None:
            live_result["post_promotion_monitoring"] = self.post_promotion_guard.observe(
                operation="factorize",
                request_summary=request_summary,
                live_result=live_result,
                live_latency_ms=live_latency_ms,
                shadow_runner=lambda provider, transport: self._run_shadow_quantum(
                    "factorize",
                    payload,
                    transport,
                    provider,
                    original_value=int(value),
                )[0],
            )
        return live_result

    def _dispatch_ticket(self, ticket: dict[str, Any]) -> dict[str, Any]:
        payload = dict(ticket.get("payload", {}))
        operation = str(ticket.get("operation", ""))
        if operation == "solve_optimization":
            return self.solve_optimization(dict(payload.get("problem", {})))
        if operation == "factorize":
            return self.factorize(int(payload.get("value", 1)))
        raise ValueError(f"Unsupported deferred quantum operation: {operation}")

    def _classical_factorize(self, value: int) -> list[int]:
        if value <= 1:
            return [value]
        factor = None
        for candidate in range(2, int(value ** 0.5) + 1):
            if value % candidate == 0:
                factor = candidate
                break
        if factor is None:
            return [1, value]
        return [factor, value // factor]

    def _extract_factor_result(self, raw_result: Any, value: int) -> list[int] | None:
        if isinstance(raw_result, dict) and "factors" in raw_result:
            try:
                factors = [int(item) for item in raw_result["factors"]]
            except Exception:
                return None
            if not factors:
                return None
            product = 1
            for factor in factors:
                product *= factor
            if product != value:
                return None
            return factors
        return self._classical_factorize(value)

    def _extract_optimization_result(self, raw_result: Any, values: list[float]) -> tuple[float | None, int | None] | None:
        if isinstance(raw_result, dict):
            best_value = raw_result.get("best_value")
            best_index = raw_result.get("best_index")
            if best_value is not None:
                try:
                    best_value = float(best_value)
                except Exception:
                    return None
                if best_index is not None:
                    try:
                        best_index = int(best_index)
                    except Exception:
                        return None
                    if best_index < 0 or best_index >= len(values):
                        return None
                    if values and float(values[best_index]) != best_value:
                        return None
                return best_value, best_index if best_index is not None else None
        if not values:
            return None, None
        best = min(values)
        return best, values.index(best)

    def _extract_provider(self, raw_result: Any) -> str | None:
        if isinstance(raw_result, dict) and raw_result.get("provider") is not None:
            return str(raw_result.get("provider"))
        return self.provider

    def _run_shadow_quantum(
        self,
        operation: str,
        payload: dict[str, Any],
        transport: Callable[..., Any],
        provider: str,
        *,
        original_value: int | None = None,
        values: list[float] | None = None,
    ) -> tuple[dict[str, Any], float]:
        started = time.perf_counter()
        try:
            try:
                raw_result = transport(operation, payload)
            except Exception as exc:  # pragma: no cover - defensive path
                return {"status": "rejected", "reason": f"shadow_transport_error:{exc.__class__.__name__}", "provider": provider}, (time.perf_counter() - started) * 1000.0
            if operation == "factorize":
                factors = self._extract_factor_result(raw_result, int(original_value or payload.get("value", 1)))
                if factors is None:
                    return {"status": "rejected", "reason": "invalid_shadow_result", "provider": provider}, (time.perf_counter() - started) * 1000.0
                return {"status": "ok", "provider": provider, "factors": factors}, (time.perf_counter() - started) * 1000.0
            if operation == "solve_optimization":
                optimization_result = self._extract_optimization_result(raw_result, list(values or payload.get("values", [])))
                if optimization_result is None:
                    return {"status": "rejected", "reason": "invalid_shadow_result", "provider": provider}, (time.perf_counter() - started) * 1000.0
                best_value, best_index = optimization_result
                return {"status": "ok", "provider": provider, "best_value": best_value, "best_index": best_index}, (time.perf_counter() - started) * 1000.0
            return {"status": "rejected", "reason": f"unsupported_operation:{operation}", "provider": provider}, (time.perf_counter() - started) * 1000.0
        finally:
            pass
