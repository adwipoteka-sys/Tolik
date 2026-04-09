from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class RuntimeAssessment:
    capability: str
    tool_name: str
    passed: bool
    score: float
    reason: str | None = None
    violations: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "capability": self.capability,
            "tool_name": self.tool_name,
            "passed": self.passed,
            "score": round(self.score, 4),
            "reason": self.reason,
            "violations": list(self.violations),
            "details": dict(self.details),
        }


@dataclass(slots=True)
class RuntimeHealthState:
    capability: str
    active_tool_name: str | None = None
    total_calls: int = 0
    consecutive_failures: int = 0
    recent_scores: list[float] = field(default_factory=list)
    auto_rollbacks: int = 0

    def rolling_mean(self) -> float:
        if not self.recent_scores:
            return 0.0
        return sum(self.recent_scores) / len(self.recent_scores)

    def to_dict(self) -> dict[str, Any]:
        return {
            "capability": self.capability,
            "active_tool_name": self.active_tool_name,
            "total_calls": self.total_calls,
            "consecutive_failures": self.consecutive_failures,
            "recent_scores": [round(item, 4) for item in self.recent_scores],
            "rolling_mean": round(self.rolling_mean(), 4),
            "auto_rollbacks": self.auto_rollbacks,
        }


class RuntimeGuard:
    """Lightweight live-runtime checks for active or canary tools."""

    DEFAULT_THRESHOLDS = {
        "text_summarizer": 0.75,
        "keyword_extractor": 0.70,
        "numeric_stats": 0.90,
    }

    def __init__(self, thresholds: dict[str, float] | None = None) -> None:
        merged = dict(self.DEFAULT_THRESHOLDS)
        if thresholds:
            merged.update(thresholds)
        self.thresholds = merged

    def assess(
        self,
        *,
        capability: str,
        tool_name: str,
        payload: dict[str, Any],
        output: dict[str, Any] | None = None,
        error: Exception | None = None,
    ) -> RuntimeAssessment:
        if error is not None:
            return RuntimeAssessment(
                capability=capability,
                tool_name=tool_name,
                passed=False,
                score=0.0,
                reason=str(error),
                violations=["exception"],
                details={"exception": str(error)},
            )

        if capability == "text_summarizer":
            return self._assess_text_summarizer(tool_name=tool_name, payload=payload, output=output or {})
        if capability == "keyword_extractor":
            return self._assess_keyword_extractor(tool_name=tool_name, payload=payload, output=output or {})
        if capability == "numeric_stats":
            return self._assess_numeric_stats(tool_name=tool_name, payload=payload, output=output or {})

        return RuntimeAssessment(
            capability=capability,
            tool_name=tool_name,
            passed=True,
            score=1.0,
            details={"mode": "generic"},
        )

    def _assess_text_summarizer(self, *, tool_name: str, payload: dict[str, Any], output: dict[str, Any]) -> RuntimeAssessment:
        normalized = []
        for item in payload.get("texts", []):
            if item is None:
                continue
            text = str(item).strip()
            if text:
                normalized.append(text)

        requested_max = int(payload.get("max_sentences", 3))
        summary = str(output.get("summary", "")).strip()
        source_count = output.get("source_count")
        sentences_used = output.get("sentences_used")

        violations: list[str] = []
        if normalized and not summary:
            violations.append("empty_summary")
        if source_count != len(normalized):
            violations.append("source_count_mismatch")
        if not isinstance(sentences_used, int) or sentences_used < 0:
            violations.append("invalid_sentences_used")
        elif sentences_used > requested_max:
            violations.append("sentence_limit_exceeded")
        elif summary and sentences_used == 0:
            violations.append("missing_sentence_count")

        summary_score = 1.0 if (summary or not normalized) else 0.0
        source_score = 1.0 if source_count == len(normalized) else 0.0
        sentence_score = 1.0 if isinstance(sentences_used, int) and 0 <= sentences_used <= requested_max else 0.0
        consistency_score = 1.0
        if summary:
            consistency_score = 1.0 if isinstance(sentences_used, int) and sentences_used >= 1 else 0.0
        elif normalized:
            consistency_score = 0.0

        score = 0.35 * summary_score + 0.30 * source_score + 0.20 * sentence_score + 0.15 * consistency_score
        threshold = self.thresholds["text_summarizer"]
        return RuntimeAssessment(
            capability="text_summarizer",
            tool_name=tool_name,
            passed=not violations and score >= threshold,
            score=score,
            reason=", ".join(violations) if violations else None,
            violations=violations,
            details={
                "expected_sources": len(normalized),
                "source_count": source_count,
                "requested_max_sentences": requested_max,
                "sentences_used": sentences_used,
            },
        )

    def _assess_keyword_extractor(self, *, tool_name: str, payload: dict[str, Any], output: dict[str, Any]) -> RuntimeAssessment:
        keywords = output.get("keywords")
        unique_terms = output.get("unique_terms")
        violations: list[str] = []
        if not isinstance(keywords, list):
            violations.append("keywords_not_list")
        if not isinstance(unique_terms, int) or unique_terms < 0:
            violations.append("invalid_unique_terms")
        score = 1.0 if not violations else 0.0
        return RuntimeAssessment(
            capability="keyword_extractor",
            tool_name=tool_name,
            passed=not violations and score >= self.thresholds["keyword_extractor"],
            score=score,
            reason=", ".join(violations) if violations else None,
            violations=violations,
            details={"payload_length": len(str(payload.get("text", "")))},
        )

    def _assess_numeric_stats(self, *, tool_name: str, payload: dict[str, Any], output: dict[str, Any]) -> RuntimeAssessment:
        values = payload.get("values", [])
        expected_count = len(values) if isinstance(values, list) else None
        violations: list[str] = []
        if expected_count is None:
            violations.append("values_not_list")
        elif output.get("count") != expected_count:
            violations.append("count_mismatch")
        for key in ("mean", "min", "max"):
            if key not in output:
                violations.append(f"missing_{key}")
        score = 1.0 if not violations else 0.0
        return RuntimeAssessment(
            capability="numeric_stats",
            tool_name=tool_name,
            passed=not violations and score >= self.thresholds["numeric_stats"],
            score=score,
            reason=", ".join(violations) if violations else None,
            violations=violations,
            details={"expected_count": expected_count},
        )


class RuntimeMonitor:
    """Tracks recent live-runtime health for active tools."""

    def __init__(self, *, window_size: int = 4, rollback_after_failures: int = 1) -> None:
        self.window_size = window_size
        self.rollback_after_failures = rollback_after_failures
        self._state: dict[str, RuntimeHealthState] = {}

    def record(self, assessment: RuntimeAssessment) -> RuntimeHealthState:
        state = self._state.setdefault(
            assessment.capability,
            RuntimeHealthState(capability=assessment.capability),
        )
        if state.active_tool_name != assessment.tool_name:
            state.active_tool_name = assessment.tool_name
            state.total_calls = 0
            state.consecutive_failures = 0
            state.recent_scores = []

        state.total_calls += 1
        state.recent_scores.append(float(assessment.score))
        if len(state.recent_scores) > self.window_size:
            state.recent_scores.pop(0)
        if assessment.passed:
            state.consecutive_failures = 0
        else:
            state.consecutive_failures += 1
        return state

    def should_rollback(self, capability: str) -> bool:
        state = self._state.get(capability)
        if state is None:
            return False
        return state.consecutive_failures >= self.rollback_after_failures

    def mark_rollback(self, capability: str, restored_tool_name: str | None = None) -> RuntimeHealthState:
        state = self._state.setdefault(
            capability,
            RuntimeHealthState(capability=capability),
        )
        state.auto_rollbacks += 1
        state.consecutive_failures = 0
        state.recent_scores = []
        state.total_calls = 0
        state.active_tool_name = restored_tool_name
        return state

    def snapshot(self, capability: str) -> RuntimeHealthState | None:
        state = self._state.get(capability)
        if state is None:
            return None
        return RuntimeHealthState(
            capability=state.capability,
            active_tool_name=state.active_tool_name,
            total_calls=state.total_calls,
            consecutive_failures=state.consecutive_failures,
            recent_scores=list(state.recent_scores),
            auto_rollbacks=state.auto_rollbacks,
        )
