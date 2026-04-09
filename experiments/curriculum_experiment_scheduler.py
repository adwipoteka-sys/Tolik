from __future__ import annotations

from typing import Any
from uuid import uuid4

from experiments.experiment_schema import (
    ExperimentAssessment,
    ExperimentKind,
    ExperimentProposal,
    ExperimentStatus,
    new_experiment_proposal_id,
)
from memory.goal_ledger import GoalLedger
from motivation.operator_charter import OperatorCharter


class CurriculumExperimentScheduler:
    """Ranks and selects safe self-improvement experiments under curriculum, risk, and budget constraints."""

    def __init__(
        self,
        ledger: GoalLedger | None = None,
        *,
        min_composite_score: float = 0.24,
        max_risk_per_cycle: float = 0.18,
        max_cost_per_cycle: float = 0.62,
        cooldown_cycles: int = 2,
        max_experiments_per_cycle: int = 1,
    ) -> None:
        self.ledger = ledger
        self.min_composite_score = min_composite_score
        self.max_risk_per_cycle = max_risk_per_cycle
        self.max_cost_per_cycle = max_cost_per_cycle
        self.cooldown_cycles = cooldown_cycles
        self.max_experiments_per_cycle = max_experiments_per_cycle
        self._proposals: dict[str, ExperimentProposal] = {}
        self._assessments: dict[str, ExperimentAssessment] = {}
        if self.ledger is not None:
            self._rehydrate()

    def _rehydrate(self) -> None:
        for payload in self.ledger.load_experiment_proposals():
            proposal = ExperimentProposal.from_dict(payload)
            self._proposals[proposal.proposal_id] = proposal
        for payload in self.ledger.load_experiment_assessments():
            assessment = ExperimentAssessment.from_dict(payload)
            self._assessments[assessment.assessment_id] = assessment

    def _persist_proposal(self, proposal: ExperimentProposal) -> ExperimentProposal:
        proposal.touch()
        self._proposals[proposal.proposal_id] = proposal
        if self.ledger is not None:
            self.ledger.save_experiment_proposal(proposal.to_dict())
        return proposal

    def _persist_assessment(self, assessment: ExperimentAssessment) -> ExperimentAssessment:
        self._assessments[assessment.assessment_id] = assessment
        if self.ledger is not None:
            self.ledger.save_experiment_assessment(assessment.to_dict())
        return assessment

    def list_proposals(self) -> list[ExperimentProposal]:
        return sorted(self._proposals.values(), key=lambda item: (item.created_at, item.proposal_id))

    def list_assessments(self) -> list[ExperimentAssessment]:
        return sorted(self._assessments.values(), key=lambda item: (item.created_at, item.assessment_id))

    def latest_for_signature(self, experiment_kind: ExperimentKind, source_signature: str) -> ExperimentProposal | None:
        matches = [
            proposal
            for proposal in self._proposals.values()
            if proposal.experiment_kind == experiment_kind and proposal.source_signature == source_signature
        ]
        if not matches:
            return None
        return sorted(matches, key=lambda item: item.updated_at)[-1]

    def _canonicalize(self, candidate: ExperimentProposal) -> ExperimentProposal:
        latest = self.latest_for_signature(candidate.experiment_kind, candidate.source_signature)
        if latest is None or latest.status in {ExperimentStatus.COMPLETED, ExperimentStatus.FAILED, ExperimentStatus.SKIPPED}:
            return self._persist_proposal(candidate)
        latest.title = candidate.title
        latest.description = candidate.description
        latest.expected_utility = candidate.expected_utility
        latest.estimated_risk = candidate.estimated_risk
        latest.estimated_cost = candidate.estimated_cost
        latest.confidence = candidate.confidence
        latest.curriculum_signals = list(candidate.curriculum_signals)
        latest.tags = list(candidate.tags)
        latest.required_capabilities = list(candidate.required_capabilities)
        latest.evidence = dict(candidate.evidence)
        if latest.status == ExperimentStatus.BLOCKED:
            latest.status = ExperimentStatus.PROPOSED
        return self._persist_proposal(latest)

    def _count_selected_in_cycle(self, current_cycle: int) -> int:
        return sum(1 for proposal in self._proposals.values() if proposal.selected_in_cycle == current_cycle)

    def _compute_curriculum_alignment(self, proposal: ExperimentProposal, context: dict[str, Any]) -> float:
        alignment = 0.24 + (0.12 * len(proposal.curriculum_signals))
        latest_pattern = str(context.get("latest_pattern_key", ""))
        if latest_pattern and any(latest_pattern in signal for signal in proposal.curriculum_signals):
            alignment += 0.22
        if context.get("goal_queue_size", 0) == 0:
            alignment += 0.10
        if proposal.experiment_kind == ExperimentKind.POLICY_CHANGE and context.get("latest_self_mod_status") in {"rolled_back", "regression_rejected"}:
            alignment += 0.10
        if proposal.experiment_kind == ExperimentKind.MODEL_UPGRADE and context.get("latest_model_status") in {"rolled_back", "regression_rejected"}:
            alignment += 0.10
        return round(min(1.0, alignment), 3)

    def _compute_strategic_fit(self, proposal: ExperimentProposal, context: dict[str, Any]) -> float:
        available = set(context.get("available_capabilities", []))
        required = set(proposal.required_capabilities)
        ready_ratio = 1.0 if not required else len(required & available) / max(1, len(required))
        fit = 0.40 + (0.30 * proposal.expected_utility) + (0.20 * ready_ratio)
        if proposal.experiment_kind == ExperimentKind.MODEL_UPGRADE:
            fit += 0.06
        return round(min(1.0, fit), 3)

    def _compute_urgency(self, proposal: ExperimentProposal, context: dict[str, Any]) -> float:
        urgency = 0.18
        if context.get("goal_queue_size", 0) == 0:
            urgency += 0.22
        if context.get("scheduled_goal_count", 0) == 0:
            urgency += 0.12
        if float(context.get("regression_pressure", 0.0)) > 0.0:
            urgency += 0.18
        if proposal.experiment_kind == ExperimentKind.POLICY_CHANGE and context.get("latest_self_mod_status") in {"rolled_back", "regression_rejected"}:
            urgency += 0.12
        if proposal.experiment_kind == ExperimentKind.MODEL_UPGRADE and context.get("latest_model_status") in {None, "rolled_back", "regression_rejected"}:
            urgency += 0.10
        return round(min(1.0, urgency), 3)

    def assess_candidates(
        self,
        candidates: list[ExperimentProposal],
        *,
        context: dict[str, Any] | None = None,
        current_cycle: int,
        charter: OperatorCharter,
    ) -> list[ExperimentAssessment]:
        context = dict(context or {})
        assessments: list[ExperimentAssessment] = []
        selected_in_cycle = self._count_selected_in_cycle(current_cycle)
        budget_exhausted = selected_in_cycle >= self.max_experiments_per_cycle
        for raw_candidate in candidates:
            historical_latest = self.latest_for_signature(raw_candidate.experiment_kind, raw_candidate.source_signature)
            candidate = self._canonicalize(raw_candidate)
            curriculum_alignment = self._compute_curriculum_alignment(candidate, context)
            strategic_fit = self._compute_strategic_fit(candidate, context)
            urgency = self._compute_urgency(candidate, context)
            charter_ok, charter_reason = charter.goal_allowed(tags=candidate.tags, required_capabilities=candidate.required_capabilities)
            latest_for_signature = historical_latest
            cooldown_until = candidate.cooldown_until_cycle
            if latest_for_signature is not None and latest_for_signature.cooldown_until_cycle is not None:
                cooldown_until = max(cooldown_until or 0, latest_for_signature.cooldown_until_cycle)
            cooldown_blocked = cooldown_until is not None and current_cycle <= cooldown_until
            budget_blocked = budget_exhausted
            composite_score = round(
                (0.36 * candidate.expected_utility)
                + (0.22 * curriculum_alignment)
                + (0.16 * candidate.confidence)
                + (0.12 * urgency)
                + (0.14 * strategic_fit)
                - (0.46 * candidate.estimated_risk)
                - (0.28 * candidate.estimated_cost),
                3,
            )
            admissible = (
                composite_score >= self.min_composite_score
                and candidate.estimated_risk <= self.max_risk_per_cycle
                and candidate.estimated_cost <= self.max_cost_per_cycle
                and not cooldown_blocked
                and not budget_blocked
                and charter_ok
            )
            rationale_parts = [
                f"utility={candidate.expected_utility:.3f}",
                f"curriculum={curriculum_alignment:.3f}",
                f"confidence={candidate.confidence:.3f}",
                f"urgency={urgency:.3f}",
                f"fit={strategic_fit:.3f}",
                f"risk={candidate.estimated_risk:.3f}",
                f"cost={candidate.estimated_cost:.3f}",
                f"composite={composite_score:.3f}",
            ]
            if cooldown_blocked:
                rationale_parts.append(f"cooldown_until_cycle={candidate.cooldown_until_cycle}")
            if budget_blocked:
                rationale_parts.append("cycle_budget_exhausted")
            if not charter_ok and charter_reason is not None:
                rationale_parts.append(f"charter={charter_reason}")
            assessment = ExperimentAssessment(
                assessment_id=f"exp_assess_{uuid4().hex[:12]}",
                proposal_id=candidate.proposal_id,
                experiment_kind=candidate.experiment_kind,
                source_signature=candidate.source_signature,
                curriculum_alignment=curriculum_alignment,
                strategic_fit=strategic_fit,
                urgency=urgency,
                expected_utility=candidate.expected_utility,
                estimated_risk=candidate.estimated_risk,
                estimated_cost=candidate.estimated_cost,
                confidence=candidate.confidence,
                composite_score=composite_score,
                admissible=admissible,
                rationale=", ".join(rationale_parts),
                evidence={
                    "charter_ok": charter_ok,
                    "charter_reason": charter_reason,
                    "cooldown_blocked": cooldown_blocked,
                    "cooldown_until": cooldown_until,
                    "budget_blocked": budget_blocked,
                    "selected_in_cycle": selected_in_cycle,
                },
            )
            assessments.append(self._persist_assessment(assessment))
            if not admissible:
                candidate.status = ExperimentStatus.BLOCKED
                candidate.cooldown_until_cycle = cooldown_until
                self._persist_proposal(candidate)
        return sorted(assessments, key=lambda item: (-item.composite_score, item.estimated_risk, item.proposal_id))

    def select(
        self,
        candidates: list[ExperimentProposal],
        *,
        context: dict[str, Any] | None = None,
        current_cycle: int,
        charter: OperatorCharter,
    ) -> tuple[ExperimentProposal | None, ExperimentAssessment | None, list[ExperimentAssessment]]:
        ranked = self.assess_candidates(candidates, context=context, current_cycle=current_cycle, charter=charter)
        for assessment in ranked:
            if not assessment.admissible:
                continue
            proposal = self._proposals[assessment.proposal_id]
            proposal.status = ExperimentStatus.SELECTED
            proposal.selected_in_cycle = current_cycle
            self._persist_proposal(proposal)
            if self.ledger is not None:
                self.ledger.append_event(
                    {
                        "event_type": "experiment_selected",
                        "proposal_id": proposal.proposal_id,
                        "experiment_kind": proposal.experiment_kind.value,
                        "source_signature": proposal.source_signature,
                        "composite_score": assessment.composite_score,
                        "selected_in_cycle": current_cycle,
                    }
                )
            return proposal, assessment, ranked
        return None, None, ranked

    def record_materialized(self, proposal_id: str, *, current_cycle: int, goal_id: str) -> ExperimentProposal:
        proposal = self._proposals[proposal_id]
        proposal.status = ExperimentStatus.MATERIALIZED
        proposal.cooldown_until_cycle = current_cycle + self.cooldown_cycles
        proposal.materialized_goal_id = goal_id
        persisted = self._persist_proposal(proposal)
        if self.ledger is not None:
            self.ledger.append_event(
                {
                    "event_type": "experiment_materialized",
                    "proposal_id": proposal_id,
                    "goal_id": goal_id,
                    "cooldown_until_cycle": proposal.cooldown_until_cycle,
                }
            )
        return persisted

    def record_outcome(self, proposal_id: str, *, success: bool, current_cycle: int, note: str | None = None) -> ExperimentProposal:
        proposal = self._proposals[proposal_id]
        proposal.status = ExperimentStatus.COMPLETED if success else ExperimentStatus.FAILED
        proposal.cooldown_until_cycle = current_cycle + self.cooldown_cycles
        if note is not None:
            evidence_notes = list(proposal.evidence.get("scheduler_notes", []))
            evidence_notes.append(note)
            proposal.evidence["scheduler_notes"] = evidence_notes
        persisted = self._persist_proposal(proposal)
        if self.ledger is not None:
            self.ledger.append_event(
                {
                    "event_type": "experiment_completed" if success else "experiment_failed",
                    "proposal_id": proposal_id,
                    "experiment_kind": proposal.experiment_kind.value,
                    "cooldown_until_cycle": proposal.cooldown_until_cycle,
                    "note": note,
                }
            )
        return persisted

    def make_proposal(
        self,
        *,
        experiment_kind: ExperimentKind,
        source_signature: str,
        title: str,
        description: str,
        expected_utility: float,
        estimated_risk: float,
        estimated_cost: float,
        confidence: float,
        curriculum_signals: list[str] | None = None,
        tags: list[str] | None = None,
        required_capabilities: list[str] | None = None,
        evidence: dict[str, Any] | None = None,
    ) -> ExperimentProposal:
        return ExperimentProposal(
            proposal_id=new_experiment_proposal_id(),
            experiment_kind=experiment_kind,
            source_signature=source_signature,
            title=title,
            description=description,
            expected_utility=expected_utility,
            estimated_risk=estimated_risk,
            estimated_cost=estimated_cost,
            confidence=confidence,
            curriculum_signals=list(curriculum_signals or []),
            tags=list(tags or []),
            required_capabilities=list(required_capabilities or []),
            evidence=dict(evidence or {}),
        )
