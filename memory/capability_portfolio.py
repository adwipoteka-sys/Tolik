from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from config.self_improvement import COOLDOWN_CYCLES, MAX_EPISODES_TO_STABLE, MIN_CONFIDENCE, MIN_TRANSFER
from memory.goal_ledger import GoalLedger
from memory.improvement_ledger import CapabilityGap


@dataclass(slots=True)
class CapabilityState:
    capability: str
    maturity_stage: str = "emerging"
    latest_strategy: str | None = None
    latest_skill_run_id: str | None = None
    latest_skill_score: float | None = None
    latest_transfer_run_id: str | None = None
    latest_transfer_score: float | None = None
    promotion_fact_keys: list[str] = field(default_factory=list)
    maintenance_schedule_ids: list[str] = field(default_factory=list)
    latest_pattern_key: str | None = None
    evidence: dict[str, Any] = field(default_factory=dict)
    confidence: float | None = None
    episodes_to_stable: int | None = None
    regression_delta: float = 0.0
    cooldown_until: int | None = None
    last_goal_id: str | None = None
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["updated_at"] = self.updated_at.isoformat()
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CapabilityState":
        raw = dict(data)
        raw["updated_at"] = datetime.fromisoformat(raw["updated_at"])
        return cls(**raw)


class CapabilityPortfolio:
    """Persistent view of which capabilities are merely stable versus transfer-validated."""

    def __init__(self, ledger: GoalLedger | None = None) -> None:
        self.ledger = ledger
        self._states: dict[str, CapabilityState] = {}
        if self.ledger is not None:
            self._rehydrate()

    def _rehydrate(self) -> None:
        for payload in self.ledger.load_capability_states():
            state = CapabilityState.from_dict(payload)
            self._states[state.capability] = state

    def _persist(self, state: CapabilityState) -> CapabilityState:
        state.updated_at = datetime.now(timezone.utc)
        self._states[state.capability] = state
        if self.ledger is not None:
            self.ledger.save_capability_state(state.to_dict())
        return state

    def get(self, capability: str) -> CapabilityState | None:
        return self._states.get(capability)

    def list_states(self) -> list[CapabilityState]:
        return sorted(self._states.values(), key=lambda item: item.capability)

    def upsert_metrics(
        self,
        *,
        capability: str,
        confidence: float | None = None,
        transfer_score: float | None = None,
        episodes_to_stable: int | None = None,
        regression_delta: float | None = None,
        maturity_stage: str | None = None,
    ) -> CapabilityState:
        state = self._states.get(capability, CapabilityState(capability=capability))
        if confidence is not None:
            state.confidence = confidence
        if transfer_score is not None:
            state.latest_transfer_score = transfer_score
        if episodes_to_stable is not None:
            state.episodes_to_stable = episodes_to_stable
        if regression_delta is not None:
            state.regression_delta = regression_delta
        if maturity_stage is not None:
            state.maturity_stage = maturity_stage
        return self._persist(state)

    def register_skill_validation(
        self,
        *,
        capability: str,
        strategy: str,
        run_id: str,
        mean_score: float,
        passed: bool,
        pattern_key: str | None = None,
    ) -> CapabilityState:
        state = self._states.get(capability, CapabilityState(capability=capability))
        state.latest_strategy = strategy
        state.latest_skill_run_id = run_id
        state.latest_skill_score = mean_score
        state.latest_pattern_key = pattern_key
        state.confidence = mean_score if passed else max(min(mean_score * 0.5, 1.0), 0.0)
        state.evidence["stable_skill_validation"] = {
            "run_id": run_id,
            "passed": passed,
            "mean_score": mean_score,
        }
        if passed:
            if state.maturity_stage == "emerging":
                state.maturity_stage = "stable"
        else:
            state.maturity_stage = "emerging"
        return self._persist(state)

    def register_semantic_promotions(
        self,
        *,
        capability: str,
        fact_keys: list[str],
        pattern_key: str | None = None,
    ) -> CapabilityState:
        state = self._states.get(capability, CapabilityState(capability=capability))
        for fact_key in fact_keys:
            if fact_key not in state.promotion_fact_keys:
                state.promotion_fact_keys.append(fact_key)
        if pattern_key is not None:
            state.latest_pattern_key = pattern_key
        state.evidence["semantic_support"] = {
            "promotion_count": len(state.promotion_fact_keys),
            "pattern_key": state.latest_pattern_key,
        }
        if state.promotion_fact_keys and state.maturity_stage == "emerging":
            state.maturity_stage = "stable"
        return self._persist(state)

    def register_transfer_validation(
        self,
        *,
        capability: str,
        strategy: str,
        run_id: str,
        mean_score: float,
        passed: bool,
    ) -> CapabilityState:
        state = self._states.get(capability, CapabilityState(capability=capability))
        state.latest_strategy = strategy
        state.latest_transfer_run_id = run_id
        state.latest_transfer_score = mean_score
        state.evidence["transfer_validation"] = {
            "run_id": run_id,
            "passed": passed,
            "mean_score": mean_score,
        }
        if passed:
            state.maturity_stage = "transfer_validated"
            state.confidence = max(state.confidence or 0.0, mean_score)
        elif state.latest_skill_score is not None:
            state.maturity_stage = "stable"
        else:
            state.maturity_stage = "emerging"
        return self._persist(state)

    def register_scheduled_maintenance(self, *, capability: str, schedule_id: str) -> CapabilityState:
        state = self._states.get(capability, CapabilityState(capability=capability))
        if schedule_id not in state.maintenance_schedule_ids:
            state.maintenance_schedule_ids.append(schedule_id)
        state.evidence["maintenance_count"] = len(state.maintenance_schedule_ids)
        return self._persist(state)

    def ready_for_unattended_use(self, capability: str) -> bool:
        state = self.get(capability)
        return bool(state and state.maturity_stage == "transfer_validated")

    def summary(self) -> dict[str, dict[str, Any]]:
        return {
            state.capability: {
                "maturity_stage": state.maturity_stage,
                "latest_strategy": state.latest_strategy,
                "latest_skill_score": state.latest_skill_score,
                "latest_transfer_score": state.latest_transfer_score,
                "promotion_count": len(state.promotion_fact_keys),
                "scheduled_maintenance": len(state.maintenance_schedule_ids),
                "confidence": state.confidence,
                "episodes_to_stable": state.episodes_to_stable,
                "regression_delta": state.regression_delta,
                "cooldown_until": state.cooldown_until,
                "last_goal_id": state.last_goal_id,
            }
            for state in self.list_states()
        }

    def next_training_focus(self) -> str | None:
        if not self._states:
            return None
        ordered = sorted(
            self._states.values(),
            key=lambda item: (
                0 if item.maturity_stage == "emerging" else 1 if item.maturity_stage == "stable" else 2,
                item.capability,
            ),
        )
        return ordered[0].capability if ordered else None

    def snapshot(self) -> dict[str, dict[str, Any]]:
        return {state.capability: state.to_dict() for state in self.list_states()}

    def compute_regression(self, previous_snapshot: dict[str, dict[str, Any]]) -> dict[str, float]:
        deltas: dict[str, float] = {}
        for state in self.list_states():
            previous = previous_snapshot.get(state.capability, {})
            previous_score = previous.get("latest_transfer_score")
            if previous_score is None:
                previous_score = previous.get("latest_skill_score")
            current_score = state.latest_transfer_score if state.latest_transfer_score is not None else state.latest_skill_score
            if previous_score is None or current_score is None:
                continue
            delta = round(float(previous_score) - float(current_score), 4)
            state.regression_delta = max(delta, 0.0)
            deltas[state.capability] = state.regression_delta
            self._persist(state)
        return deltas

    def identify_gaps(self, *, current_step: int = 0) -> list[CapabilityGap]:
        gaps: list[CapabilityGap] = []
        for state in self.list_states():
            if state.cooldown_until is not None and current_step < state.cooldown_until:
                continue
            confidence = state.confidence
            if confidence is None:
                confidence = state.latest_transfer_score if state.latest_transfer_score is not None else state.latest_skill_score or 0.0
            transfer_score = state.latest_transfer_score if state.latest_transfer_score is not None else 0.0
            evidence_count = len(state.promotion_fact_keys)
            if state.latest_skill_run_id:
                evidence_count += 1
            if state.latest_transfer_run_id:
                evidence_count += 1
            episodes_to_stable = state.episodes_to_stable if state.episodes_to_stable is not None else (MAX_EPISODES_TO_STABLE + 1 if state.maturity_stage == 'emerging' else 1)
            if confidence < MIN_CONFIDENCE:
                gaps.append(CapabilityGap(state.capability, 'low_confidence', round(1.0 - confidence, 3), confidence, evidence_count, transfer_score, state.regression_delta, current_step))
            if evidence_count < 2:
                gaps.append(CapabilityGap(state.capability, 'insufficient_evidence', round(min((2 - evidence_count) / 2, 1.0), 3), confidence, evidence_count, transfer_score, state.regression_delta, current_step))
            if state.latest_transfer_run_id is not None and transfer_score < MIN_TRANSFER:
                gaps.append(CapabilityGap(state.capability, 'transfer_deficit', round(1.0 - transfer_score, 3), confidence, evidence_count, transfer_score, state.regression_delta, current_step))
            if state.regression_delta > 0.05:
                gaps.append(CapabilityGap(state.capability, 'performance_regression', round(min(state.regression_delta, 1.0), 3), confidence, evidence_count, transfer_score, state.regression_delta, current_step))
            if episodes_to_stable > MAX_EPISODES_TO_STABLE:
                gaps.append(CapabilityGap(state.capability, 'stale_capability', round(min(episodes_to_stable / MAX_EPISODES_TO_STABLE - 1.0, 1.0), 3), confidence, evidence_count, transfer_score, state.regression_delta, current_step))
        # deduplicate by strongest severity for same capability/gap
        dedup: dict[tuple[str, str], CapabilityGap] = {}
        for gap in gaps:
            key = (gap.capability_id, gap.gap_type)
            if key not in dedup or gap.severity > dedup[key].severity:
                dedup[key] = gap
        return sorted(dedup.values(), key=lambda gap: (-gap.severity, gap.capability_id, gap.gap_type))

    def apply_cooldown(self, capability_id: str, cycles: int = COOLDOWN_CYCLES, *, current_cycle: int = 0) -> CapabilityState:
        state = self._states.get(capability_id, CapabilityState(capability=capability_id))
        state.cooldown_until = current_cycle + cycles
        return self._persist(state)

    def mark_improvement_result(self, goal_id: str, outcome: dict[str, Any]) -> CapabilityState | None:
        capability_id = outcome.get('capability_id')
        if capability_id is None:
            return None
        state = self._states.get(str(capability_id), CapabilityState(capability=str(capability_id)))
        state.last_goal_id = goal_id
        if 'confidence' in outcome and outcome['confidence'] is not None:
            state.confidence = float(outcome['confidence'])
        if 'transfer_score' in outcome and outcome['transfer_score'] is not None:
            state.latest_transfer_score = float(outcome['transfer_score'])
        if 'episodes_to_stable' in outcome and outcome['episodes_to_stable'] is not None:
            state.episodes_to_stable = int(outcome['episodes_to_stable'])
        if outcome.get('reset_cooldown'):
            state.cooldown_until = None
        return self._persist(state)
