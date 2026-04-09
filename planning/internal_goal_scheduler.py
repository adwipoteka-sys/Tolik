from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from config.self_improvement import COOLDOWN_CYCLES
from memory.improvement_ledger import CapabilityGap, ImprovementGoal


@dataclass(slots=True)
class GoalPriorityWeights:
    utility_weight: float = 0.30
    uncertainty_weight: float = 0.30
    transfer_weight: float = 0.20
    risk_weight: float = 0.10
    cost_weight: float = 0.10


class InternalGoalScheduler:
    def __init__(self, weights: GoalPriorityWeights | None = None) -> None:
        self.weights = weights or GoalPriorityWeights()

    @staticmethod
    def _normalized_stability_cost(episodes_to_stable: int | None) -> float:
        if episodes_to_stable is None:
            return 0.0
        from config.self_improvement import MAX_EPISODES_TO_STABLE
        return min(episodes_to_stable / MAX_EPISODES_TO_STABLE, 1.0)

    def score_gap(self, gap: CapabilityGap, *, episodes_to_stable: int | None = None, missing_tool: bool = False) -> float:
        missing_tool_bonus = 1.0 if missing_tool else 0.0
        normalized_regression = min(max(gap.regression_delta, 0.0), 1.0)
        stability_cost = self._normalized_stability_cost(episodes_to_stable)
        return (
            self.weights.utility_weight * (1.0 - gap.confidence)
            + self.weights.uncertainty_weight * gap.severity
            + self.weights.transfer_weight * (1.0 - gap.transfer_score)
            + self.weights.risk_weight * normalized_regression
            + self.weights.cost_weight * stability_cost
            + 0.05 * missing_tool_bonus
        )

    def select_gap(self, gaps: Iterable[CapabilityGap]) -> CapabilityGap | None:
        ranked = sorted(gaps, key=lambda gap: (-gap.severity, gap.capability_id, gap.gap_type))
        return ranked[0] if ranked else None

    def materialize_goal(self, gap: CapabilityGap, *, priority: float | None = None, goal_id: str | None = None) -> ImprovementGoal:
        mapping = {
            'low_confidence': ('confidence_recovery', 'confidence', 0.75),
            'insufficient_evidence': ('collect_evidence', 'evidence_count', 3.0),
            'transfer_deficit': ('improve_transfer', 'transfer_score', 0.8),
            'performance_regression': ('recover_performance', 'transfer_score', 0.75),
            'stale_capability': ('maintenance_eval', 'confidence', 0.7),
        }
        goal_type, target_metric, target_value = mapping.get(gap.gap_type, ('maintenance_eval', 'confidence', 0.7))
        return ImprovementGoal(
            goal_id=goal_id or f'impr_{gap.capability_id}_{gap.gap_type}',
            capability_id=gap.capability_id,
            goal_type=goal_type,
            target_metric=target_metric,
            target_value=float(target_value),
            priority=priority if priority is not None else gap.severity,
            budget_steps=3,
            cooldown_cycles=COOLDOWN_CYCLES,
        )
