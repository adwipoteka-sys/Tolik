from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from memory.goal_ledger import GoalLedger

from metacognition.failure_miner import FailureCase


@dataclass(slots=True)
class RegressionCase:
    case_id: str
    capability: str
    payload: dict[str, Any]
    expected: dict[str, Any]
    source_signature: str
    rationale: str
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RegressionCase":
        return cls(**data)


class BenchmarkExpander:
    """Promotes live canary failures into persistent regression cases."""

    def __init__(self, ledger: GoalLedger | None = None) -> None:
        self.ledger = ledger
        self._cases_by_capability: dict[str, list[RegressionCase]] = {}
        self._signatures: set[str] = set()
        if self.ledger is not None:
            for case in self.ledger.load_regression_cases():
                self._remember(case)

    def _remember(self, case: RegressionCase) -> None:
        if case.source_signature in self._signatures:
            return
        self._signatures.add(case.source_signature)
        self._cases_by_capability.setdefault(case.capability, []).append(case)

    def expand_from_failure(self, failure: FailureCase) -> RegressionCase:
        existing = next((case for case in self._cases_by_capability.get(failure.capability, []) if case.source_signature == failure.signature), None)
        if existing is not None:
            return existing

        if failure.capability == "text_summarizer":
            expected = {
                "source_count": failure.expected.get("source_count", 0),
                "max_sentences": failure.expected.get("max_sentences", 3),
                "required_summary_nonempty": failure.expected.get("required_summary_nonempty", False),
            }
            rationale = "Added after live canary failure: blank-input handling and sentence-limit compliance must be preserved."
            tags = ["runtime_regression", *failure.violation_types]
        else:
            expected = dict(failure.expected)
            rationale = "Added after live canary failure."
            tags = ["runtime_regression"]

        case = RegressionCase(
            case_id=f"regression_{failure.capability}_{len(self._cases_by_capability.get(failure.capability, [])) + 1}",
            capability=failure.capability,
            payload=dict(failure.payload),
            expected=expected,
            source_signature=failure.signature,
            rationale=rationale,
            tags=tags,
        )
        self._remember(case)
        if self.ledger is not None:
            self.ledger.save_regression_case(case)
        return case

    def get_cases(self, capability: str) -> list[RegressionCase]:
        return list(self._cases_by_capability.get(capability, []))
