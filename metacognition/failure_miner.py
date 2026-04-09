from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from motivation.goal_schema import Goal
from tooling.tool_spec import GeneratedTool
from tooling.runtime_guard import RuntimeAssessment


def _normalized_texts(payload: dict[str, Any]) -> list[str]:
    texts = payload.get("texts", [])
    if not isinstance(texts, list):
        return []
    normalized: list[str] = []
    for item in texts:
        if item is None:
            continue
        text = str(item).strip()
        if text:
            normalized.append(text)
    return normalized


def _signature(capability: str, violations: list[str], input_shape: dict[str, Any]) -> str:
    ordered_violations = "+".join(sorted(set(violations))) or "unknown"
    blank_bucket = "blank_inputs" if input_shape.get("blank_texts_count", 0) else "no_blank_inputs"
    requested_limit = input_shape.get("requested_max_sentences", "na")
    return f"{capability}|{ordered_violations}|{blank_bucket}|limit={requested_limit}"


@dataclass(slots=True)
class FailureCase:
    case_id: str
    goal_id: str
    capability: str
    tool_name: str
    tool_version: str
    rollout_stage: str
    payload: dict[str, Any]
    input_shape: dict[str, Any]
    violation_types: list[str]
    expected: dict[str, Any]
    actual: dict[str, Any]
    rollback_target: str | None = None
    signature: str = ""
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class FailureMiner:
    """Converts failed canary evaluations into structured regression cases."""

    def mine_canary_failure(
        self,
        *,
        goal: Goal,
        tool: GeneratedTool,
        payload: dict[str, Any],
        assessment: RuntimeAssessment,
        output: dict[str, Any] | None,
        rollback_target: str | None,
    ) -> FailureCase:
        capability = tool.capability
        normalized = _normalized_texts(payload)
        input_shape: dict[str, Any] = {
            "payload_keys": sorted(payload.keys()),
            "requested_max_sentences": int(payload.get("max_sentences", 3)) if payload.get("max_sentences") is not None else None,
        }

        if capability == "text_summarizer":
            raw_texts = payload.get("texts", [])
            raw_count = len(raw_texts) if isinstance(raw_texts, list) else 0
            input_shape.update(
                {
                    "texts_count": raw_count,
                    "nonempty_texts_count": len(normalized),
                    "blank_texts_count": max(raw_count - len(normalized), 0),
                    "characters": sum(len(item) for item in normalized),
                }
            )

        expected: dict[str, Any]
        actual: dict[str, Any]
        notes: list[str] = []

        if capability == "text_summarizer":
            requested_max = input_shape.get("requested_max_sentences") or 3
            expected = {
                "source_count": len(normalized),
                "max_sentences": requested_max,
                "required_summary_nonempty": bool(normalized),
            }
            actual = {
                "summary": "" if output is None else str(output.get("summary", "")),
                "source_count": None if output is None else output.get("source_count"),
                "sentences_used": None if output is None else output.get("sentences_used"),
                **assessment.details,
            }
            if "source_count_mismatch" in assessment.violations:
                notes.append("count_normalized_inputs_only")
            if "sentence_limit_exceeded" in assessment.violations:
                notes.append("respect_runtime_sentence_limit")
            if input_shape.get("blank_texts_count", 0):
                notes.append("ignore_blank_inputs")
        else:
            expected = dict(assessment.details)
            actual = {} if output is None else dict(output)
            if assessment.violations:
                notes.append("repair_runtime_contract")

        signature = _signature(capability, assessment.violations, input_shape)
        return FailureCase(
            case_id=f"failure_{goal.goal_id}",
            goal_id=goal.goal_id,
            capability=capability,
            tool_name=tool.name,
            tool_version=tool.version,
            rollout_stage="canary",
            payload=dict(payload),
            input_shape=input_shape,
            violation_types=list(assessment.violations),
            expected=expected,
            actual=actual,
            rollback_target=rollback_target,
            signature=signature,
            notes=notes,
        )
