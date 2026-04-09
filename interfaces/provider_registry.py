from __future__ import annotations

import json
import os
import time
from typing import Any, Callable
from urllib import request

from interfaces.adapter_schema import InterfaceRuntimeSpec




def _risk_label_from_prompt(prompt: str) -> str:
    lowered = str(prompt).lower()
    high_markers = ("urgent", "fraud", "breach", "attack", "exploit", "malware", "poison", "critical")
    return "high" if any(marker in lowered for marker in high_markers) else "low"

def _mock_live_cloud_llm(prompt: str, **kwargs: Any) -> dict[str, Any]:
    text = " ".join(str(prompt).split())
    task = kwargs.get("task")
    if task == "summarize":
        sentences = [item.strip() for item in text.replace("!", ".").replace("?", ".").split(".") if item.strip()]
        return {"text": ". ".join(sentences[:2]) + ("." if sentences else ""), "provider": "mock_live"}
    if task == "classify_risk":
        label = _risk_label_from_prompt(text)
        return {"text": f"label:{label}\nconfidence:0.90", "provider": "mock_live"}
    if task in {"structured_extract", "json_extract"}:
        label = _risk_label_from_prompt(text)
        return {"text": json.dumps({"label": label, "source": "mock_live"}), "provider": "mock_live"}
    return {"text": f"[mock-live-cloud-llm] {text[:120]}", "provider": "mock_live"}


def _mock_live_cloud_llm_fast(prompt: str, **kwargs: Any) -> dict[str, Any]:
    text = " ".join(str(prompt).split())
    task = kwargs.get("task")
    if task == "summarize":
        sentences = [item.strip() for item in text.replace("!", ".").replace("?", ".").split(".") if item.strip()]
        return {"text": ". ".join(sentences[:2]) + ("." if sentences else ""), "provider": "mock_live_fast"}
    if task == "classify_risk":
        label = _risk_label_from_prompt(text)
        return {"text": f"label:{label}\nconfidence:0.92", "provider": "mock_live_fast"}
    if task in {"structured_extract", "json_extract"}:
        label = _risk_label_from_prompt(text)
        return {"text": json.dumps({"label": label, "source": "mock_live_fast"}), "provider": "mock_live_fast"}
    return {"text": text[:100], "provider": "mock_live_fast"}


def _mock_live_cloud_llm_safe(prompt: str, **kwargs: Any) -> dict[str, Any]:
    time.sleep(0.01)
    text = " ".join(str(prompt).split())
    task = kwargs.get("task")
    if task == "summarize":
        sentences = [item.strip() for item in text.replace("!", ".").replace("?", ".").split(".") if item.strip()]
        return {"text": ". ".join(sentences[:2]) + ("." if sentences else ""), "provider": "mock_live_safe"}
    if task == "classify_risk":
        label = _risk_label_from_prompt(text)
        return {"text": f"label:{label}\nconfidence:0.91", "provider": "mock_live_safe"}
    if task in {"structured_extract", "json_extract"}:
        label = _risk_label_from_prompt(text)
        return {"text": json.dumps({"label": label, "source": "mock_live_safe"}), "provider": "mock_live_safe"}
    return {"text": f"safe:{text[:96]}", "provider": "mock_live_safe"}


def _mock_live_cloud_llm_safe_alt(prompt: str, **kwargs: Any) -> dict[str, Any]:
    time.sleep(0.012)
    text = " ".join(str(prompt).split())
    task = kwargs.get("task")
    if task == "summarize":
        sentences = [item.strip() for item in text.replace("!", ".").replace("?", ".").split(".") if item.strip()]
        return {"text": ". ".join(sentences[:2]) + ("." if sentences else ""), "provider": "mock_live_safe_alt"}
    if task == "classify_risk":
        label = _risk_label_from_prompt(text)
        return {"text": json.dumps({"label": label, "confidence": 0.89}), "provider": "mock_live_safe_alt"}
    if task in {"structured_extract", "json_extract"}:
        label = _risk_label_from_prompt(text)
        return {"text": json.dumps({"label": label, "source": "mock_live_safe_alt", "verdict": label}), "provider": "mock_live_safe_alt"}
    return {"text": f"alt-safe:{text[:96]}", "provider": "mock_live_safe_alt"}


def _mock_live_cloud_llm_risky(prompt: str, **_: Any) -> dict[str, Any]:
    time.sleep(0.005)
    return {
        "text": "malware instructions should never be returned by a qualified provider",
        "provider": "mock_live_risky",
    }


def _build_mock_live_cloud_llm_regressing() -> Callable[..., dict[str, Any]]:
    state = {"calls": 0}

    def _call(prompt: str, **kwargs: Any) -> dict[str, Any]:
        state["calls"] += 1
        text = " ".join(str(prompt).split())
        task = kwargs.get("task")
        if state["calls"] <= 2:
            if task == "summarize":
                sentences = [item.strip() for item in text.replace("!", ".").replace("?", ".").split(".") if item.strip()]
                return {"text": ". ".join(sentences[:2]) + ("." if sentences else ""), "provider": "mock_live_regressing"}
            if task == "classify_risk":
                label = _risk_label_from_prompt(text)
                return {"text": f"label:{label}\nconfidence:0.93", "provider": "mock_live_regressing"}
            if task in {"structured_extract", "json_extract"}:
                label = _risk_label_from_prompt(text)
                return {"text": json.dumps({"label": label, "source": "mock_live_regressing"}), "provider": "mock_live_regressing"}
            return {"text": f"stable:{text[:96]}", "provider": "mock_live_regressing"}
        time.sleep(0.08)
        if task == "classify_risk":
            bad_label = "low" if _risk_label_from_prompt(text) == "high" else "high"
            return {"text": f"label:{bad_label}\nconfidence:0.92", "provider": "mock_live_regressing"}
        if task in {"structured_extract", "json_extract"}:
            bad_label = "low" if _risk_label_from_prompt(text) == "high" else "high"
            return {"text": json.dumps({"label": bad_label, "source": "mock_live_regressing"}), "provider": "mock_live_regressing"}
        return {
            "text": "malware instructions appear after rollout regression",
            "provider": "mock_live_regressing",
        }

    return _call


def _mock_live_quantum(operation: str, payload: dict[str, Any]) -> dict[str, Any]:
    if operation == "solve_optimization":
        values = list(payload.get("values", []))
        best = min(values) if values else None
        best_index = values.index(best) if best is not None else None
        return {"best_value": best, "best_index": best_index, "provider": "mock_live"}
    if operation == "factorize":
        value = int(payload.get("value", 1))
        if value <= 1:
            return {"factors": [value], "provider": "mock_live"}
        factor = None
        for candidate in range(2, int(value ** 0.5) + 1):
            if value % candidate == 0:
                factor = candidate
                break
        factors = [1, value] if factor is None else [factor, value // factor]
        return {"factors": factors, "provider": "mock_live"}
    raise ValueError(f"Unsupported quantum mock operation: {operation}")


def _mock_live_quantum_precise(operation: str, payload: dict[str, Any]) -> dict[str, Any]:
    time.sleep(0.008)
    base = _mock_live_quantum(operation, payload)
    base["provider"] = "mock_live_precise"
    return base


def _mock_live_quantum_slow(operation: str, payload: dict[str, Any]) -> dict[str, Any]:
    time.sleep(0.05)
    base = _mock_live_quantum(operation, payload)
    base["provider"] = "mock_live_slow"
    return base


def _mock_live_quantum_noisy(operation: str, payload: dict[str, Any]) -> dict[str, Any]:
    time.sleep(0.006)
    if operation == "factorize":
        return {"factors": [2, 10], "provider": "mock_live_noisy"}
    if operation == "solve_optimization":
        values = list(payload.get("values", []))
        if len(values) >= 2:
            return {"best_value": values[0], "best_index": 0, "provider": "mock_live_noisy"}
        return {"best_value": values[0] if values else None, "best_index": 0 if values else None, "provider": "mock_live_noisy"}
    raise ValueError(f"Unsupported quantum mock operation: {operation}")


def _build_mock_live_quantum_regressing() -> Callable[..., dict[str, Any]]:
    state = {"calls": 0}

    def _call(operation: str, payload: dict[str, Any]) -> dict[str, Any]:
        state["calls"] += 1
        if state["calls"] <= 2:
            base = _mock_live_quantum(operation, payload)
            base["provider"] = "mock_live_regressing"
            return base
        time.sleep(0.08)
        if operation == "factorize":
            return {"factors": [2, 10], "provider": "mock_live_regressing"}
        if operation == "solve_optimization":
            values = list(payload.get("values", []))
            return {"best_value": values[0] if values else None, "best_index": 0 if values else None, "provider": "mock_live_regressing"}
        raise ValueError(f"Unsupported quantum mock operation: {operation}")

    return _call


def _build_http_json_transport(spec: InterfaceRuntimeSpec) -> Callable[..., dict[str, Any]]:
    if not spec.endpoint:
        raise ValueError("endpoint is required for http_json provider")

    def _call(operation_or_prompt: str, payload: dict[str, Any] | None = None, **kwargs: Any) -> dict[str, Any]:
        if payload is None:
            body = {"prompt": operation_or_prompt, **kwargs}
        else:
            body = {"operation": operation_or_prompt, "payload": payload, **kwargs}
        headers = {"Content-Type": "application/json", **spec.headers}
        for header_name, env_name in spec.headers_env.items():
            if os.getenv(env_name):
                headers[header_name] = os.environ[env_name]
        req = request.Request(
            spec.endpoint,
            data=json.dumps(body).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with request.urlopen(req, timeout=spec.policy.timeout_seconds) as response:
            content = response.read().decode("utf-8")
        if not content.strip():
            return {}
        return json.loads(content)

    return _call


def build_cloud_transport(spec: InterfaceRuntimeSpec) -> Callable[..., dict[str, Any]] | None:
    match spec.provider:
        case "mock_live":
            return _mock_live_cloud_llm
        case "mock_live_fast":
            return _mock_live_cloud_llm_fast
        case "mock_live_safe":
            return _mock_live_cloud_llm_safe
        case "mock_live_safe_alt":
            return _mock_live_cloud_llm_safe_alt
        case "mock_live_risky":
            return _mock_live_cloud_llm_risky
        case "mock_live_regressing":
            return _build_mock_live_cloud_llm_regressing()
        case "http_json":
            return _build_http_json_transport(spec)
        case _:
            return None


def build_quantum_transport(spec: InterfaceRuntimeSpec) -> Callable[..., dict[str, Any]] | None:
    match spec.provider:
        case "mock_live":
            return _mock_live_quantum
        case "mock_live_precise":
            return _mock_live_quantum_precise
        case "mock_live_slow":
            return _mock_live_quantum_slow
        case "mock_live_noisy":
            return _mock_live_quantum_noisy
        case "mock_live_regressing":
            return _build_mock_live_quantum_regressing()
        case "http_json":
            return _build_http_json_transport(spec)
        case _:
            return None
