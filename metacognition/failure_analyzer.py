from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any
from uuid import uuid4

from benchmarks.report_types import NavigationConfidenceReport, SkillArenaReport, TransferEvaluationReport
from memory.improvement_ledger import CapabilityGap, CurriculumItem, ImprovementGoal
from config.self_improvement import MAX_CURRICULUM_ITEMS


@dataclass(slots=True)
class FailureSignature:
    capability_id: str
    reason: str
    confidence: float
    needs_tool_proposal: bool = False
    needs_external_backend: bool = False
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class FailureAnalyzer:
    def analyze_gap(self, gap: CapabilityGap, latest_reports: list[Any]) -> FailureSignature:
        reason = gap.gap_type
        confidence = gap.confidence
        needs_tool = False
        needs_external = False
        tags = [gap.gap_type, gap.capability_id]
        for report in latest_reports:
            failure_reason = getattr(report, 'failure_reason', None)
            if failure_reason:
                reason = failure_reason
                tags.append(failure_reason)
                if 'tool' in failure_reason:
                    needs_tool = True
                if 'external' in failure_reason or 'quantum' in failure_reason or 'cloud' in failure_reason:
                    needs_external = True
        if gap.gap_type == 'insufficient_evidence':
            reason = 'insufficient_evidence'
        return FailureSignature(capability_id=gap.capability_id, reason=reason, confidence=confidence, needs_tool_proposal=needs_tool, needs_external_backend=needs_external, tags=sorted(set(tags)))

    def build_curriculum(self, goal: ImprovementGoal, signature: FailureSignature) -> list[CurriculumItem]:
        action_hint = {
            'low_confidence': 'replay_easy_cases',
            'transfer_deficit': 'run_transfer_drills',
            'performance_regression': 'run_regression_suite',
            'insufficient_evidence': 'collect_more_evidence',
            'stale_capability': 'run_maintenance_eval',
        }.get(signature.reason, 'run_transfer_drills')
        scenario_ids = ['anchor_easy', 'anchor_detour', 'heldout_transfer']
        items: list[CurriculumItem] = []
        for idx, scenario in enumerate(scenario_ids[:MAX_CURRICULUM_ITEMS], start=1):
            items.append(CurriculumItem(
                item_id=f'cur_{uuid4().hex[:8]}',
                goal_id=goal.goal_id,
                capability_id=goal.capability_id,
                benchmark_name='skill_arena' if idx < 3 else 'transfer_suite',
                scenario_id=scenario,
                action_hint=action_hint,
                expected_signal='improved_confidence' if goal.goal_type == 'confidence_recovery' else 'improved_transfer',
                max_attempts=1,
            ))
        return items
