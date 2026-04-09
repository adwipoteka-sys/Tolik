from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from metacognition.curriculum_builder import CurriculumBuilder
from metacognition.postmortem import PostmortemAnalyzer, PostmortemReport


@dataclass
class MetacognitionModule:
    """Simple self-monitoring layer for v3.116."""

    postmortem_analyzer: PostmortemAnalyzer = field(default_factory=PostmortemAnalyzer)
    curriculum_builder: CurriculumBuilder = field(default_factory=CurriculumBuilder)
    logger: list[dict[str, Any]] = field(default_factory=list)

    def log_event(self, module: str, event: str, info: Any = None) -> None:
        self.logger.append({"module": module, "event": event, "info": info})

    def analyze(self) -> dict[str, Any]:
        recent = self.logger[-20:]
        failure_events = 0
        insufficient_evidence = False
        for entry in recent:
            info = entry.get("info")
            if isinstance(info, dict):
                if info.get("success") is False:
                    failure_events += 1
                if info.get("knowledge_gap") or info.get("insufficient_evidence"):
                    insufficient_evidence = True
        regression_pressure = min(failure_events / 5.0, 1.0)
        return {
            "event_count": len(self.logger),
            "failure_events": failure_events,
            "insufficient_evidence": insufficient_evidence,
            "regression_pressure": regression_pressure,
        }

    def run_postmortem(
        self,
        goal: Any,
        trace: list[dict[str, Any]],
        expected: dict[str, Any],
        observed: dict[str, Any],
    ) -> tuple[PostmortemReport, list[Any]]:
        report = self.postmortem_analyzer.analyze(goal, trace, expected, observed)
        tasks = self.curriculum_builder.build(report)
        return report, tasks

    def notify(self, event: dict[str, Any]) -> None:
        self.logger.append({"module": "notify", "event": event.get("type", "external_event"), "info": event})
