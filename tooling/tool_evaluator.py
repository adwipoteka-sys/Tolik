from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from benchmarks.benchmark_expander import BenchmarkExpander


@dataclass(slots=True)
class BenchmarkCase:
    case_id: str
    payload: dict[str, Any]
    expected: dict[str, Any]


@dataclass(slots=True)
class CaseEvaluation:
    case_id: str
    score: float
    passed: bool
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "score": self.score,
            "passed": self.passed,
            "details": dict(self.details),
        }


@dataclass(slots=True)
class ToolEvaluationReport:
    tool_name: str
    capability: str
    threshold: float
    mean_score: float
    passed: bool
    cases: list[CaseEvaluation] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "capability": self.capability,
            "threshold": self.threshold,
            "mean_score": self.mean_score,
            "passed": self.passed,
            "cases": [case.to_dict() for case in self.cases],
            "errors": list(self.errors),
        }


class ToolEvaluator:
    """Runs deterministic offline benchmarks before a generated tool is promoted."""

    DEFAULT_THRESHOLDS = {
        "text_summarizer": 0.80,
        "keyword_extractor": 0.80,
        "numeric_stats": 0.95,
    }

    def __init__(
        self,
        thresholds: dict[str, float] | None = None,
        benchmark_expander: BenchmarkExpander | None = None,
    ) -> None:
        merged = dict(self.DEFAULT_THRESHOLDS)
        if thresholds:
            merged.update(thresholds)
        self.thresholds = merged
        self.benchmark_expander = benchmark_expander

    def evaluate(
        self,
        *,
        tool_name: str,
        capability: str,
        runtime_callable: Callable[[dict[str, Any]], dict[str, Any]],
    ) -> ToolEvaluationReport:
        threshold = self.thresholds.get(capability, 0.8)
        cases = self._cases_for(capability)
        if self.benchmark_expander is not None:
            cases = cases + [BenchmarkCase(case_id=item.case_id, payload=dict(item.payload), expected=dict(item.expected)) for item in self.benchmark_expander.get_cases(capability)]
        case_results: list[CaseEvaluation] = []
        errors: list[str] = []

        for case in cases:
            try:
                output = runtime_callable(dict(case.payload))
            except Exception as exc:  # pragma: no cover - exercised via report.errors path
                errors.append(f"{case.case_id}: {exc}")
                case_results.append(CaseEvaluation(case.case_id, score=0.0, passed=False, details={"exception": str(exc)}))
                continue

            score, details = self._score(capability, output, case)
            case_results.append(
                CaseEvaluation(
                    case_id=case.case_id,
                    score=score,
                    passed=score >= threshold,
                    details=details,
                )
            )

        mean_score = sum(item.score for item in case_results) / len(case_results) if case_results else 0.0
        passed = bool(case_results) and not errors and all(item.passed for item in case_results) and mean_score >= threshold
        return ToolEvaluationReport(
            tool_name=tool_name,
            capability=capability,
            threshold=threshold,
            mean_score=round(mean_score, 4),
            passed=passed,
            cases=case_results,
            errors=errors,
        )

    def _cases_for(self, capability: str) -> list[BenchmarkCase]:
        if capability == "text_summarizer":
            return [
                BenchmarkCase(
                    case_id="summary_stability",
                    payload={
                        "texts": [
                            "Stable behavior followed policy tuning.",
                            "Shorter plans reduced retry costs.",
                            "Operator review became faster.",
                        ],
                        "max_sentences": 2,
                    },
                    expected={
                        "keywords": ["stable", "shorter plans"],
                        "max_sentences": 2,
                        "source_count": 3,
                    },
                ),
                BenchmarkCase(
                    case_id="summary_noise_handling",
                    payload={
                        "texts": [
                            "Noise handling improved after switching strategy.",
                            "Reports were compressed into concise summaries.",
                            "Operators needed fewer manual checks.",
                        ],
                        "max_sentences": 2,
                    },
                    expected={
                        "keywords": ["noise", "concise"],
                        "max_sentences": 2,
                        "source_count": 3,
                    },
                ),
            ]
        if capability == "keyword_extractor":
            return [
                BenchmarkCase(
                    case_id="keywords_basic",
                    payload={"text": "alpha beta beta gamma gamma gamma delta"},
                    expected={"keywords": {"gamma", "beta"}},
                )
            ]
        if capability == "numeric_stats":
            return [
                BenchmarkCase(
                    case_id="stats_basic",
                    payload={"values": [2, 4, 6]},
                    expected={"count": 3, "mean": 4.0, "min": 2.0, "max": 6.0},
                )
            ]
        raise ValueError(f"No benchmark suite defined for capability: {capability}")

    def _score(self, capability: str, output: dict[str, Any], case: BenchmarkCase) -> tuple[float, dict[str, Any]]:
        if capability == "text_summarizer":
            summary = str(output.get("summary", "")).lower()
            if "keywords" in case.expected:
                keywords = case.expected["keywords"]
                hits = sum(1 for keyword in keywords if keyword in summary)
                keyword_score = hits / len(keywords)
                source_score = 1.0 if int(output.get("source_count", -1)) == int(case.expected["source_count"]) else 0.0
                sentence_score = 1.0 if int(output.get("sentences_used", 9999)) <= int(case.expected["max_sentences"]) else 0.0
                non_empty_score = 1.0 if summary.strip() else 0.0
                score = 0.5 * keyword_score + 0.2 * source_score + 0.2 * sentence_score + 0.1 * non_empty_score
                return round(score, 4), {
                    "keywords_hit": hits,
                    "keywords_total": len(keywords),
                    "source_count": output.get("source_count"),
                    "sentences_used": output.get("sentences_used"),
                }

            components: dict[str, float] = {}
            if "source_count" in case.expected:
                components["source_score"] = 1.0 if int(output.get("source_count", -1)) == int(case.expected["source_count"]) else 0.0
            if "max_sentences" in case.expected:
                components["sentence_score"] = 1.0 if int(output.get("sentences_used", 9999)) <= int(case.expected["max_sentences"]) else 0.0
            if "required_summary_nonempty" in case.expected:
                required = bool(case.expected["required_summary_nonempty"])
                components["non_empty_score"] = 1.0 if bool(summary.strip()) == required else 0.0
            if not components:
                components["fallback_score"] = 1.0 if summary.strip() else 0.0
            score = sum(components.values()) / len(components)
            return round(score, 4), {
                **components,
                "source_count": output.get("source_count"),
                "sentences_used": output.get("sentences_used"),
            }

        if capability == "keyword_extractor":
            actual = set(output.get("keywords", []))
            expected = set(case.expected["keywords"])
            overlap = len(actual & expected) / max(1, len(expected))
            unique_terms = float(output.get("unique_terms", 0) > 0)
            score = 0.8 * overlap + 0.2 * unique_terms
            return round(score, 4), {"actual": sorted(actual), "expected": sorted(expected)}

        if capability == "numeric_stats":
            exact = 0
            for key in ("count", "mean", "min", "max"):
                if output.get(key) == case.expected[key]:
                    exact += 1
            score = exact / 4.0
            return round(score, 4), {"actual": dict(output), "expected": dict(case.expected)}

        raise ValueError(f"Unsupported capability: {capability}")
