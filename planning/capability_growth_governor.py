from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from memory.capability_graph import CapabilityGraph
from memory.capability_portfolio import CapabilityPortfolio
from memory.goal_ledger import GoalLedger
from motivation.goal_schema import GoalBudget
from planning.capability_growth_planner import CapabilityGrowthPlan


_STAGE_GAP = {
    "latent": 1.00,
    "unknown": 0.92,
    "emerging": 0.84,
    "available": 0.36,
    "stable": 0.14,
    "transfer_validated": 0.00,
}


@dataclass(slots=True)
class GrowthPlanAssessment:
    assessment_id: str
    plan_id: str
    root_capability: str
    path_targets: list[str]
    expected_utility: float
    estimated_risk: float
    estimated_cost: float
    confidence: float
    unlock_bonus: float
    portfolio_gap_bonus: float
    composite_score: float
    admissible: bool
    rationale: str
    evidence: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["created_at"] = self.created_at.isoformat()
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GrowthPlanAssessment":
        raw = dict(data)
        raw["created_at"] = datetime.fromisoformat(raw["created_at"])
        return cls(**raw)


class CapabilityGrowthGovernor:
    """Utility/risk gate for multi-edge capability growth plans."""

    def __init__(
        self,
        ledger: GoalLedger | None = None,
        *,
        min_composite_score: float = 0.28,
        hard_risk_ceiling: float = 0.24,
    ) -> None:
        self.ledger = ledger
        self.min_composite_score = min_composite_score
        self.hard_risk_ceiling = hard_risk_ceiling
        self._assessments: dict[str, GrowthPlanAssessment] = {}
        if self.ledger is not None:
            self._rehydrate()

    def _rehydrate(self) -> None:
        for payload in self.ledger.load_capability_growth_assessments():
            assessment = GrowthPlanAssessment.from_dict(payload)
            self._assessments[assessment.assessment_id] = assessment

    def _persist(self, assessment: GrowthPlanAssessment) -> GrowthPlanAssessment:
        self._assessments[assessment.assessment_id] = assessment
        if self.ledger is not None:
            self.ledger.save_capability_growth_assessment(assessment.to_dict())
        return assessment

    def list_assessments(self) -> list[GrowthPlanAssessment]:
        return sorted(self._assessments.values(), key=lambda item: (item.created_at, item.assessment_id))

    def assess_candidates(
        self,
        plans: list[CapabilityGrowthPlan],
        *,
        graph: CapabilityGraph,
        portfolio: CapabilityPortfolio,
        context: dict[str, Any] | None = None,
    ) -> list[GrowthPlanAssessment]:
        context = context or {}
        assessments = [
            self._persist(self._assess_plan(plan, graph=graph, portfolio=portfolio, context=context))
            for plan in plans
        ]
        return sorted(assessments, key=lambda item: (-item.composite_score, item.estimated_risk, item.plan_id))

    def select(
        self,
        plans: list[CapabilityGrowthPlan],
        *,
        graph: CapabilityGraph,
        portfolio: CapabilityPortfolio,
        context: dict[str, Any] | None = None,
    ) -> tuple[CapabilityGrowthPlan | None, GrowthPlanAssessment | None, list[GrowthPlanAssessment]]:
        ranked = self.assess_candidates(plans, graph=graph, portfolio=portfolio, context=context)
        plans_by_id = {plan.plan_id: plan for plan in plans}
        for assessment in ranked:
            if assessment.admissible:
                return plans_by_id.get(assessment.plan_id), assessment, ranked
        return None, None, ranked

    def goal_overrides_for(self, assessment: GrowthPlanAssessment, plan: CapabilityGrowthPlan) -> dict[str, Any]:
        horizon = max(1, len(assessment.path_targets))
        return {
            "expected_gain": round(max(0.55, min(0.98, assessment.expected_utility)), 3),
            "novelty": round(max(0.30, min(0.78, 0.24 + (0.12 * horizon) + (0.18 * assessment.portfolio_gap_bonus))), 3),
            "uncertainty_reduction": round(max(0.55, min(0.97, 0.40 + (0.45 * assessment.confidence))), 3),
            "strategic_fit": round(max(0.72, min(0.99, 0.48 + (0.30 * assessment.expected_utility) + (0.20 * assessment.unlock_bonus))), 3),
            "risk_estimate": round(assessment.estimated_risk, 3),
            "priority": round(max(0.67, min(0.99, 0.60 + (assessment.composite_score * 0.55))), 3),
            "risk_budget": round(max(0.14, min(0.30, assessment.estimated_risk + 0.06)), 3),
            "resource_budget": GoalBudget(
                max_steps=max(4, 2 + horizon),
                max_seconds=round(12.0 + (6.0 * horizon) + (assessment.estimated_cost * 8.0), 1),
                max_tool_calls=0,
                max_api_calls=0,
            ),
            "tags": [],
            "evidence": {
                "growth_assessment_id": assessment.assessment_id,
                "growth_expected_utility": assessment.expected_utility,
                "growth_estimated_risk": assessment.estimated_risk,
                "growth_estimated_cost": assessment.estimated_cost,
                "growth_confidence": assessment.confidence,
                "growth_composite_score": assessment.composite_score,
            },
        }

    def _assess_plan(
        self,
        plan: CapabilityGrowthPlan,
        *,
        graph: CapabilityGraph,
        portfolio: CapabilityPortfolio,
        context: dict[str, Any],
    ) -> GrowthPlanAssessment:
        remaining_steps = plan.steps[plan.active_step_index:] or plan.steps
        assessment_root = remaining_steps[0].source_capability if remaining_steps else plan.root_capability
        mean_strength = sum(step.strength for step in remaining_steps) / max(1, len(remaining_steps))
        mean_strategic = sum(step.strategic_value for step in remaining_steps) / max(1, len(remaining_steps))
        support_total = sum(len(step.evidence.get("support_capabilities", [])) for step in remaining_steps)
        support_ready = sum(
            1
            for step in remaining_steps
            for support in step.evidence.get("support_capabilities", [])
            if graph.is_ready(support)
        )
        support_ready_ratio = 1.0 if support_total == 0 else support_ready / support_total
        portfolio_gap_bonus = self._portfolio_gap_bonus(plan, portfolio)
        unlock_bonus = self._unlock_bonus(plan, graph)
        queue_pressure = float(context.get("queue_pressure", 0.0))
        regression_pressure = float(context.get("regression_pressure", 0.0))
        llm_steps = sum(1 for step in remaining_steps if "local_llm" in step.evidence.get("support_capabilities", []))
        compounding_bonus = min(0.18, 0.12 * max(0, len(remaining_steps) - 1))

        expected_utility = round(
            min(
                1.0,
                (0.40 * mean_strategic)
                + (0.30 * mean_strength)
                + (0.15 * portfolio_gap_bonus)
                + (0.15 * unlock_bonus)
                + compounding_bonus,
            ),
            3,
        )

        root_state = portfolio.get(plan.root_capability)
        root_transfer = root_state.latest_transfer_score if root_state and root_state.latest_transfer_score is not None else 0.55
        confidence = round(min(1.0, (0.55 * float(root_transfer)) + (0.25 * mean_strength) + (0.20 * support_ready_ratio)), 3)
        estimated_cost = round(min(1.0, 0.08 + (0.09 * len(remaining_steps)) + (0.04 * support_total) + (0.06 * queue_pressure)), 3)
        estimated_risk = round(
            min(
                1.0,
                0.04
                + (0.16 * (1.0 - mean_strength))
                + (0.07 * (1.0 - support_ready_ratio))
                + (0.03 * len(remaining_steps))
                + (0.03 * llm_steps)
                + (0.03 * queue_pressure)
                + (0.04 * regression_pressure),
            ),
            3,
        )
        composite_score = round((expected_utility * confidence) - (0.60 * estimated_risk) - (0.45 * estimated_cost), 3)
        admissible = composite_score >= self.min_composite_score and estimated_risk <= self.hard_risk_ceiling
        path = " -> ".join([assessment_root, *[step.target_capability for step in remaining_steps]])
        rationale = (
            f"{path}: utility {expected_utility:.3f}, confidence {confidence:.3f}, "
            f"risk {estimated_risk:.3f}, cost {estimated_cost:.3f}, composite {composite_score:.3f}."
        )
        return GrowthPlanAssessment(
            assessment_id=f"growth_assess_{uuid4().hex[:12]}",
            plan_id=plan.plan_id,
            root_capability=assessment_root,
            path_targets=[step.target_capability for step in remaining_steps],
            expected_utility=expected_utility,
            estimated_risk=estimated_risk,
            estimated_cost=estimated_cost,
            confidence=confidence,
            unlock_bonus=round(unlock_bonus, 3),
            portfolio_gap_bonus=round(portfolio_gap_bonus, 3),
            composite_score=composite_score,
            admissible=admissible,
            rationale=rationale,
            evidence={
                "mean_strength": round(mean_strength, 3),
                "mean_strategic_value": round(mean_strategic, 3),
                "support_total": support_total,
                "support_ready_ratio": round(support_ready_ratio, 3),
                "queue_pressure": queue_pressure,
                "regression_pressure": regression_pressure,
                "llm_steps": llm_steps,
                "compounding_bonus": round(compounding_bonus, 3),
            },
        )

    def _portfolio_gap_bonus(self, plan: CapabilityGrowthPlan, portfolio: CapabilityPortfolio) -> float:
        remaining_targets = [step.target_capability for step in (plan.steps[plan.active_step_index:] or plan.steps)]
        gaps: list[float] = []
        for capability in remaining_targets:
            state = portfolio.get(capability)
            stage = state.maturity_stage if state is not None else "unknown"
            gaps.append(_STAGE_GAP.get(stage, 0.75))
        return 0.0 if not gaps else sum(gaps) / len(gaps)

    def _unlock_bonus(self, plan: CapabilityGrowthPlan, graph: CapabilityGraph) -> float:
        remaining_steps = plan.steps[plan.active_step_index:] or plan.steps
        remaining_targets = [step.target_capability for step in remaining_steps]
        outgoing = 0
        for capability in remaining_targets:
            node = graph.get(capability)
            outgoing += len(node.outgoing) if node is not None else 0
        return min(1.0, (0.20 if len(remaining_steps) > 1 else 0.0) + (0.08 * outgoing))
