from __future__ import annotations

from typing import Any

from experiments.experiment_board_schema import (
    ExperimentCampaign,
    ExperimentCampaignStatus,
    ExperimentCycleBudgetSnapshot,
    new_experiment_campaign_id,
)
from experiments.experiment_schema import ExperimentAssessment, ExperimentKind, ExperimentProposal
from memory.goal_ledger import GoalLedger
from motivation.operator_charter import OperatorCharter


class ExperimentBoard:
    """Persistent board for experiment campaigns across multiple autonomous cycles."""

    def __init__(
        self,
        *,
        ledger: GoalLedger,
        default_retry_after_cycles: int = 1,
        default_campaign_budgets: dict[ExperimentKind, dict[str, float | int]] | None = None,
    ) -> None:
        self.ledger = ledger
        self.default_retry_after_cycles = max(1, int(default_retry_after_cycles))
        self.default_campaign_budgets = default_campaign_budgets or {
            ExperimentKind.POLICY_CHANGE: {"max_total_cost": 0.90, "max_total_risk": 0.22, "max_attempts": 2},
            ExperimentKind.MODEL_UPGRADE: {"max_total_cost": 1.30, "max_total_risk": 0.30, "max_attempts": 3},
        }
        self._campaigns: dict[str, ExperimentCampaign] = {}
        self._cycle_budgets: dict[int, ExperimentCycleBudgetSnapshot] = {}
        self._rehydrate()

    def _rehydrate(self) -> None:
        for payload in self.ledger.load_experiment_campaigns():
            campaign = ExperimentCampaign.from_dict(payload)
            self._campaigns[campaign.campaign_id] = campaign
        for payload in self.ledger.load_experiment_cycle_budgets():
            snapshot = ExperimentCycleBudgetSnapshot.from_dict(payload)
            self._cycle_budgets[snapshot.cycle] = snapshot

    def _persist_campaign(self, campaign: ExperimentCampaign) -> ExperimentCampaign:
        campaign.touch()
        self._campaigns[campaign.campaign_id] = campaign
        self.ledger.save_experiment_campaign(campaign.to_dict())
        return campaign

    def _persist_cycle_budget(self, snapshot: ExperimentCycleBudgetSnapshot) -> ExperimentCycleBudgetSnapshot:
        snapshot.touch()
        self._cycle_budgets[snapshot.cycle] = snapshot
        self.ledger.save_experiment_cycle_budget(snapshot.to_dict())
        return snapshot

    def list_campaigns(self) -> list[ExperimentCampaign]:
        return sorted(self._campaigns.values(), key=lambda item: (item.created_at, item.campaign_id))

    def list_cycle_budgets(self) -> list[ExperimentCycleBudgetSnapshot]:
        return sorted(self._cycle_budgets.values(), key=lambda item: item.cycle)

    def summary(self) -> dict[str, Any]:
        campaigns = self.list_campaigns()
        return {
            "campaign_count": len(campaigns),
            "deferred_campaign_count": sum(1 for item in campaigns if item.status == ExperimentCampaignStatus.DEFERRED),
            "queued_campaign_count": sum(1 for item in campaigns if item.status == ExperimentCampaignStatus.QUEUED),
            "completed_campaign_count": sum(1 for item in campaigns if item.status == ExperimentCampaignStatus.COMPLETED),
            "budget_exhausted_campaign_count": sum(1 for item in campaigns if item.status == ExperimentCampaignStatus.BUDGET_EXHAUSTED),
        }

    def campaign_for_signature(self, experiment_kind: ExperimentKind, source_signature: str) -> ExperimentCampaign | None:
        matches = [
            campaign
            for campaign in self._campaigns.values()
            if campaign.experiment_kind == experiment_kind and campaign.source_signature == source_signature
        ]
        if not matches:
            return None
        return sorted(matches, key=lambda item: item.updated_at)[-1]

    def _budget_policy_for(self, proposal: ExperimentProposal) -> dict[str, float | int]:
        defaults = dict(self.default_campaign_budgets.get(proposal.experiment_kind, {}))
        overrides = dict(proposal.evidence.get("campaign_budget", {}))
        defaults.update(overrides)
        return {
            "max_total_cost": float(defaults.get("max_total_cost", 1.0)),
            "max_total_risk": float(defaults.get("max_total_risk", 0.30)),
            "max_attempts": int(defaults.get("max_attempts", 3)),
        }

    def upsert_campaign(self, proposal: ExperimentProposal, assessment: ExperimentAssessment | None = None) -> ExperimentCampaign:
        existing = self.campaign_for_signature(proposal.experiment_kind, proposal.source_signature)
        if existing is None:
            budget = self._budget_policy_for(proposal)
            campaign = ExperimentCampaign(
                campaign_id=new_experiment_campaign_id(),
                experiment_kind=proposal.experiment_kind,
                source_signature=proposal.source_signature,
                title=proposal.title,
                description=proposal.description,
                tags=list(proposal.tags),
                required_capabilities=list(proposal.required_capabilities),
                proposal_ids=[proposal.proposal_id],
                assessment_ids=[assessment.assessment_id] if assessment is not None else [],
                latest_proposal=proposal.to_dict(),
                max_total_cost=float(budget["max_total_cost"]),
                max_total_risk=float(budget["max_total_risk"]),
                max_attempts=int(budget["max_attempts"]),
                last_composite_score=assessment.composite_score if assessment is not None else None,
                last_rationale=assessment.rationale if assessment is not None else None,
                evidence=dict(proposal.evidence),
            )
            return self._persist_campaign(campaign)

        existing.title = proposal.title
        existing.description = proposal.description
        existing.tags = sorted({*existing.tags, *proposal.tags})
        existing.required_capabilities = sorted({*existing.required_capabilities, *proposal.required_capabilities})
        if proposal.proposal_id not in existing.proposal_ids:
            existing.proposal_ids.append(proposal.proposal_id)
        if assessment is not None and assessment.assessment_id not in existing.assessment_ids:
            existing.assessment_ids.append(assessment.assessment_id)
        existing.latest_proposal = proposal.to_dict()
        existing.evidence.update(dict(proposal.evidence))
        budget = self._budget_policy_for(proposal)
        existing.max_total_cost = max(existing.max_total_cost, float(budget["max_total_cost"]))
        existing.max_total_risk = max(existing.max_total_risk, float(budget["max_total_risk"]))
        existing.max_attempts = max(existing.max_attempts, int(budget["max_attempts"]))
        if assessment is not None:
            existing.last_composite_score = assessment.composite_score
            existing.last_rationale = assessment.rationale
        return self._persist_campaign(existing)

    def _budget_exhaustion_reasons(self, campaign: ExperimentCampaign, proposal: ExperimentProposal) -> list[str]:
        reasons: list[str] = []
        if campaign.attempt_count >= campaign.max_attempts:
            reasons.append("attempt_budget_exhausted")
        if campaign.spent_cost + proposal.estimated_cost > campaign.max_total_cost:
            reasons.append("cost_budget_exhausted")
        if campaign.spent_risk + proposal.estimated_risk > campaign.max_total_risk:
            reasons.append("risk_budget_exhausted")
        return reasons

    def _execution_reasons(
        self,
        proposal: ExperimentProposal,
        *,
        current_cycle: int,
        context: dict[str, Any],
        charter: OperatorCharter,
    ) -> tuple[list[str], int | None]:
        reasons: list[str] = []
        defer_until: int | None = None
        retry_after = max(1, int(proposal.evidence.get("retry_after_cycles", self.default_retry_after_cycles)))

        if proposal.cooldown_until_cycle is not None and current_cycle <= proposal.cooldown_until_cycle:
            reasons.append(f"cooldown_until_cycle:{proposal.cooldown_until_cycle}")
            defer_until = proposal.cooldown_until_cycle + 1

        if bool(proposal.evidence.get("requires_live_interface", False)):
            required_interface = str(proposal.evidence.get("required_interface", "")).strip()
            interface_modes = dict(context.get("future_interfaces", {}))
            mode = interface_modes.get(required_interface)
            if mode not in {"live", "enabled"}:
                reasons.append(f"awaiting_live_interface:{required_interface}:{mode or 'unknown'}")
                defer_until = max(defer_until or 0, current_cycle + retry_after)

        allowed, charter_reason = charter.goal_allowed(tags=proposal.tags, required_capabilities=proposal.required_capabilities)
        if not allowed and charter_reason is not None:
            reasons.append(f"charter:{charter_reason}")
            defer_until = max(defer_until or 0, current_cycle + retry_after)

        return reasons, defer_until

    def _current_cycle_budget(self, cycle: int) -> ExperimentCycleBudgetSnapshot:
        return self._cycle_budgets.get(cycle, ExperimentCycleBudgetSnapshot(cycle=cycle))

    def _note_deferred(self, cycle: int, campaign_id: str, reason: str) -> None:
        snapshot = self._current_cycle_budget(cycle)
        if campaign_id not in snapshot.deferred_campaign_ids:
            snapshot.deferred_campaign_ids.append(campaign_id)
        if reason not in snapshot.notes:
            snapshot.notes.append(reason)
        self._persist_cycle_budget(snapshot)

    def _reserve_budget(self, cycle: int, campaign_id: str, proposal: ExperimentProposal) -> None:
        snapshot = self._current_cycle_budget(cycle)
        snapshot.reserved_cost = round(snapshot.reserved_cost + proposal.estimated_cost, 3)
        snapshot.reserved_risk = round(snapshot.reserved_risk + proposal.estimated_risk, 3)
        if campaign_id not in snapshot.selected_campaign_ids:
            snapshot.selected_campaign_ids.append(campaign_id)
        self._persist_cycle_budget(snapshot)

    def refresh_candidate_pool(
        self,
        proposals: list[ExperimentProposal],
        assessments: list[ExperimentAssessment],
        *,
        current_cycle: int,
        selected_proposal_id: str | None,
        context: dict[str, Any],
        charter: OperatorCharter,
    ) -> list[ExperimentCampaign]:
        assessment_by_id = {assessment.proposal_id: assessment for assessment in assessments}
        updated: list[ExperimentCampaign] = []
        for proposal in proposals:
            assessment = assessment_by_id.get(proposal.proposal_id)
            campaign = self.upsert_campaign(proposal, assessment)
            if proposal.proposal_id == selected_proposal_id:
                updated.append(campaign)
                continue
            budget_reasons = self._budget_exhaustion_reasons(campaign, proposal)
            if budget_reasons:
                campaign.status = ExperimentCampaignStatus.BUDGET_EXHAUSTED
                campaign.defer_reason = "; ".join(budget_reasons)
                campaign.defer_until_cycle = None
                updated.append(self._persist_campaign(campaign))
                continue
            if assessment is not None and assessment.admissible:
                campaign.status = ExperimentCampaignStatus.QUEUED
                campaign.defer_reason = "not_selected_this_cycle"
                campaign.defer_until_cycle = current_cycle + 1
                updated.append(self._persist_campaign(campaign))
                self._note_deferred(current_cycle, campaign.campaign_id, campaign.defer_reason)
                continue
            reasons, defer_until = self._execution_reasons(proposal, current_cycle=current_cycle, context=context, charter=charter)
            campaign.status = ExperimentCampaignStatus.DEFERRED
            campaign.defer_reason = "; ".join(reasons) if reasons else "assessment_blocked"
            campaign.defer_until_cycle = defer_until or (current_cycle + self.default_retry_after_cycles)
            updated.append(self._persist_campaign(campaign))
            self._note_deferred(current_cycle, campaign.campaign_id, campaign.defer_reason)
        return updated

    def release_due_candidates(
        self,
        *,
        current_cycle: int,
        context: dict[str, Any],
        charter: OperatorCharter,
    ) -> list[ExperimentProposal]:
        released: list[ExperimentProposal] = []
        for campaign in self.list_campaigns():
            if campaign.status not in {ExperimentCampaignStatus.QUEUED, ExperimentCampaignStatus.DEFERRED, ExperimentCampaignStatus.FAILED}:
                continue
            proposal = campaign.latest_proposal_object()
            if proposal is None:
                continue
            if campaign.status == ExperimentCampaignStatus.DEFERRED and campaign.defer_until_cycle is not None and current_cycle < campaign.defer_until_cycle:
                continue
            budget_reasons = self._budget_exhaustion_reasons(campaign, proposal)
            if budget_reasons:
                campaign.status = ExperimentCampaignStatus.BUDGET_EXHAUSTED
                campaign.defer_reason = "; ".join(budget_reasons)
                campaign.defer_until_cycle = None
                self._persist_campaign(campaign)
                continue
            reasons, defer_until = self._execution_reasons(proposal, current_cycle=current_cycle, context=context, charter=charter)
            if reasons:
                campaign.status = ExperimentCampaignStatus.DEFERRED
                campaign.defer_reason = "; ".join(reasons)
                campaign.defer_until_cycle = defer_until or (current_cycle + self.default_retry_after_cycles)
                self._persist_campaign(campaign)
                continue
            campaign.status = ExperimentCampaignStatus.QUEUED
            campaign.defer_reason = None
            campaign.defer_until_cycle = None
            self._persist_campaign(campaign)
            released.append(proposal)
        return released

    def stage_selected_execution(
        self,
        proposal: ExperimentProposal,
        *,
        current_cycle: int,
        context: dict[str, Any],
        charter: OperatorCharter,
    ) -> tuple[ExperimentCampaign, dict[str, Any]]:
        campaign = self.upsert_campaign(proposal)
        budget_reasons = self._budget_exhaustion_reasons(campaign, proposal)
        if budget_reasons:
            campaign.status = ExperimentCampaignStatus.BUDGET_EXHAUSTED
            campaign.defer_reason = "; ".join(budget_reasons)
            campaign.defer_until_cycle = None
            persisted = self._persist_campaign(campaign)
            return persisted, {"materialize_now": False, "reason": campaign.defer_reason}
        reasons, defer_until = self._execution_reasons(proposal, current_cycle=current_cycle, context=context, charter=charter)
        if reasons:
            campaign.status = ExperimentCampaignStatus.DEFERRED
            campaign.defer_reason = "; ".join(reasons)
            campaign.defer_until_cycle = defer_until or (current_cycle + self.default_retry_after_cycles)
            persisted = self._persist_campaign(campaign)
            self._note_deferred(current_cycle, campaign.campaign_id, campaign.defer_reason)
            return persisted, {
                "materialize_now": False,
                "reason": campaign.defer_reason,
                "defer_until_cycle": campaign.defer_until_cycle,
            }
        campaign.status = ExperimentCampaignStatus.SELECTED
        campaign.selection_count += 1
        campaign.attempt_count += 1
        campaign.last_selected_cycle = current_cycle
        campaign.defer_reason = None
        campaign.defer_until_cycle = None
        campaign.spent_cost = round(campaign.spent_cost + proposal.estimated_cost, 3)
        campaign.spent_risk = round(campaign.spent_risk + proposal.estimated_risk, 3)
        persisted = self._persist_campaign(campaign)
        self._reserve_budget(current_cycle, campaign.campaign_id, proposal)
        return persisted, {"materialize_now": True, "reason": None}

    def _find_campaign_by_proposal_id(self, proposal_id: str) -> ExperimentCampaign:
        for campaign in self._campaigns.values():
            if proposal_id in campaign.proposal_ids or campaign.latest_proposal.get("proposal_id") == proposal_id:
                return campaign
        raise KeyError(f"Campaign for proposal_id={proposal_id!r} not found")

    def record_materialized(self, proposal_id: str, *, goal_id: str) -> ExperimentCampaign:
        campaign = self._find_campaign_by_proposal_id(proposal_id)
        campaign.status = ExperimentCampaignStatus.MATERIALIZED
        if goal_id not in campaign.materialized_goal_ids:
            campaign.materialized_goal_ids.append(goal_id)
        return self._persist_campaign(campaign)

    def record_outcome(
        self,
        proposal_id: str,
        *,
        success: bool,
        current_cycle: int,
        note: str | None = None,
        cooldown_until_cycle: int | None = None,
    ) -> ExperimentCampaign:
        campaign = self._find_campaign_by_proposal_id(proposal_id)
        campaign.last_rationale = note or campaign.last_rationale
        if success:
            campaign.status = ExperimentCampaignStatus.COMPLETED
            campaign.defer_reason = None
            campaign.defer_until_cycle = None
            return self._persist_campaign(campaign)
        if campaign.attempt_count >= campaign.max_attempts:
            campaign.status = ExperimentCampaignStatus.BUDGET_EXHAUSTED
            campaign.defer_reason = "attempt_budget_exhausted"
            campaign.defer_until_cycle = None
            return self._persist_campaign(campaign)
        campaign.status = ExperimentCampaignStatus.FAILED
        campaign.defer_reason = note or "retry_after_failure"
        campaign.defer_until_cycle = max(cooldown_until_cycle or 0, current_cycle + self.default_retry_after_cycles)
        return self._persist_campaign(campaign)
