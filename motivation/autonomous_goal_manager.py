from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from core.event_types import GoalEventType
from memory.goal_ledger import GoalLedger
from motivation.goal_arbitrator import GoalArbitrator
from motivation.goal_schema import (
    Goal,
    GoalBudget,
    GoalKind,
    GoalSource,
    GoalStatus,
    SuccessCriterion,
    new_goal_id,
)
from motivation.internal_goal_sources import ALL_DETECTORS


class AutonomousGoalManager:
    """Generates, admits, tracks, recovers, and unblocks goals."""

    def __init__(
        self,
        ledger: GoalLedger,
        arbitrator: GoalArbitrator,
        max_active_goals: int = 3,
    ) -> None:
        self.ledger = ledger
        self.arbitrator = arbitrator
        self.max_active_goals = max_active_goals
        self.pending: list[Goal] = []
        self.active: list[Goal] = []
        self._goals: dict[str, Goal] = {}
        self._recover_goals()


    def _recover_goals(self) -> None:
        for goal in self.ledger.load_active_goals():
            if goal.goal_id in self._goals:
                continue
            self._track(goal)
            if goal.status == GoalStatus.ACTIVE:
                self.active.append(goal)
            else:
                self.pending.append(goal)

    def _track(self, goal: Goal) -> Goal:
        self._goals[goal.goal_id] = goal
        return goal

    def _save(self, goal: Goal, event_type: GoalEventType, extra: dict[str, Any] | None = None) -> None:
        goal.touch()
        self.ledger.save_goal_snapshot(goal)
        event = {
            "event_type": event_type.value,
            "goal_id": goal.goal_id,
            "title": goal.title,
            "status": goal.status.value,
            "source": goal.source.value,
        }
        if extra:
            event.update(extra)
        self.ledger.append_event(event)

    def all_goals(self) -> list[Goal]:
        return list(self._goals.values())

    def ingest_external_goal(
        self,
        goal_text: str,
        *,
        required_capabilities: list[str] | None = None,
        tags: list[str] | None = None,
        evidence: dict[str, Any] | None = None,
    ) -> Goal:
        goal = Goal(
            goal_id=new_goal_id("user"),
            title=goal_text.strip() or "User goal",
            description=goal_text.strip() or "User goal",
            source=GoalSource.USER,
            kind=GoalKind.USER_TASK,
            expected_gain=0.95,
            novelty=0.20,
            uncertainty_reduction=0.45,
            strategic_fit=1.00,
            risk_estimate=0.05,
            priority=1.00,
            risk_budget=0.40,
            resource_budget=GoalBudget(max_steps=6, max_seconds=30.0, max_tool_calls=1, max_api_calls=0),
            success_criteria=[SuccessCriterion(metric="status", comparator="==", target="done")],
            required_capabilities=required_capabilities or ["classical_planning"],
            tags=tags or ["user"],
            evidence=evidence or {"raw_text": goal_text},
        )
        self._track(goal)
        self.pending.append(goal)
        self._save(goal, GoalEventType.GOAL_INGESTED, {"origin": "external"})
        return goal

    def generate_candidates(
        self,
        workspace_state: dict[str, Any],
        memory_summary: dict[str, Any],
        meta_summary: dict[str, Any],
    ) -> list[Goal]:
        candidates: list[Goal] = []
        for detector in ALL_DETECTORS:
            candidates.extend(detector(workspace_state, memory_summary, meta_summary))
        return candidates

    def validate_candidate(self, goal: Goal) -> tuple[bool, str | None]:
        try:
            Goal.from_dict(goal.to_dict())
        except Exception as exc:  # pragma: no cover
            return False, str(exc)

        forbidden = {"cloud_llm", "quantum_solver"}
        if forbidden & set(goal.required_capabilities):
            return False, "forbidden capability in v3.117"
        return True, None

    def admit_candidates(
        self,
        candidates: list[Goal],
        context: dict[str, Any],
    ) -> list[Goal]:
        validated: list[Goal] = []
        for goal in candidates:
            ok, reason = self.validate_candidate(goal)
            if ok:
                validated.append(goal)
            else:
                goal.status = GoalStatus.REJECTED
                self._track(goal)
                self._save(goal, GoalEventType.GOAL_REJECTED, {"reason": reason or "validation_failed"})

        deduped = self.arbitrator.deduplicate(validated)
        available = set(context.get("available_capabilities", []))
        admissible, deferred = self.arbitrator.filter_by_capability(deduped, available)

        for goal in deferred:
            self._track(goal)
            if goal not in self.pending:
                self.pending.append(goal)
            self._save(
                goal,
                GoalEventType.GOAL_DEFERRED,
                {"reason": "missing_capability", "missing_capabilities": goal.evidence.get("missing_capabilities", [])},
            )

        active_titles = {_title(goal) for goal in self.active}
        already_present = {_title(goal) for goal in self.pending}
        context_for_arbitrator = dict(context)
        context_for_arbitrator["active_titles"] = active_titles

        accepted = []
        for goal in self.arbitrator.admit(admissible, context_for_arbitrator):
            title_key = _title(goal)
            if title_key in already_present:
                continue
            already_present.add(title_key)
            self._track(goal)
            self.pending.append(goal)
            self._save(goal, GoalEventType.GOAL_ADMITTED)
            accepted.append(goal)

        return accepted

    def create_tooling_goals_for_deferred(self, supported_capabilities: set[str]) -> list[Goal]:
        existing_titles = {_title(goal) for goal in self.pending + self.active + self.all_goals()}
        tool_goals: list[Goal] = []
        for goal in self.pending:
            if goal.status != GoalStatus.DEFERRED:
                continue
            missing = set(goal.evidence.get("missing_capabilities", []))
            for capability in sorted(missing & supported_capabilities):
                title = f"Create safe tool for {capability}"
                if " ".join(title.lower().split()) in existing_titles:
                    continue
                tool_goal = Goal(
                    goal_id=new_goal_id("tool"),
                    title=title,
                    description=f"Generate and register a controlled tool for capability {capability}.",
                    source=GoalSource.TOOLING_GAP,
                    kind=GoalKind.TOOL_CREATION,
                    expected_gain=0.88,
                    novelty=0.52,
                    uncertainty_reduction=0.82,
                    strategic_fit=0.93,
                    risk_estimate=0.08,
                    priority=0.90,
                    risk_budget=0.20,
                    resource_budget=GoalBudget(max_steps=7, max_seconds=25.0, max_tool_calls=1, max_api_calls=0),
                    success_criteria=[SuccessCriterion(metric="tool_registered", comparator="==", target=True)],
                    required_capabilities=["classical_planning"],
                    tags=["tooling", capability],
                    evidence={
                        "blocked_goal_id": goal.goal_id,
                        "blocked_goal_title": goal.title,
                        "target_capability": capability,
                    },
                )
                existing_titles.add(_title(tool_goal))
                tool_goals.append(tool_goal)
        return tool_goals

    def reactivate_deferred_goals_for_capability(self, capability: str) -> list[Goal]:
        reactivated: list[Goal] = []
        for goal in self.pending:
            if goal.status != GoalStatus.DEFERRED:
                continue
            missing = set(goal.evidence.get("missing_capabilities", []))
            if capability not in missing:
                continue
            remaining = sorted(missing - {capability})
            if remaining:
                goal.evidence["missing_capabilities"] = remaining
                continue
            goal.evidence["missing_capabilities"] = []
            goal.status = GoalStatus.PENDING
            self._save(goal, GoalEventType.GOAL_UNBLOCKED, {"capability": capability})
            reactivated.append(goal)
        return reactivated

    def select_next_goal(self, context: dict[str, Any]) -> Goal | None:
        if len(self.active) >= self.max_active_goals:
            return None

        available = set(context.get("available_capabilities", []))
        pending_ready = [
            goal
            for goal in self.pending
            if goal.status == GoalStatus.PENDING and set(goal.required_capabilities).issubset(available)
        ]
        if not pending_ready:
            return None

        ranked = self.arbitrator.rank(pending_ready, context)
        goal = ranked[0]
        self.activate_goal(goal.goal_id)
        return goal

    def activate_goal(self, goal_id: str) -> None:
        goal = self._goals[goal_id]
        goal.status = GoalStatus.ACTIVE
        goal.updated_at = datetime.now(timezone.utc)
        self.pending = [item for item in self.pending if item.goal_id != goal_id]
        if goal not in self.active:
            self.active.append(goal)
        self._save(goal, GoalEventType.GOAL_ACTIVATED)

    def update_progress(self, goal_id: str, event: dict[str, Any]) -> None:
        goal = self._goals[goal_id]
        goal.touch()
        self._save(goal, GoalEventType.GOAL_PROGRESS, {"progress": event})

    def complete_goal(self, goal_id: str, outcome: dict[str, Any]) -> None:
        goal = self._goals[goal_id]
        goal.status = GoalStatus.DONE if outcome.get("success", False) else GoalStatus.FAILED
        goal.evidence["outcome"] = outcome
        self.active = [item for item in self.active if item.goal_id != goal_id]
        self.pending = [item for item in self.pending if item.goal_id != goal_id]
        self._save(goal, GoalEventType.GOAL_COMPLETED, {"success": outcome.get("success", False)})

    def defer_goal(self, goal_id: str, reason: str) -> None:
        goal = self._goals[goal_id]
        goal.status = GoalStatus.DEFERRED
        self.active = [item for item in self.active if item.goal_id != goal_id]
        if goal not in self.pending:
            self.pending.append(goal)
        self._save(goal, GoalEventType.GOAL_DEFERRED, {"reason": reason})

    def restore_from_ledger(self) -> None:
        self.pending.clear()
        self.active.clear()
        self._goals.clear()

        for goal in self.ledger.load_active_goals():
            self._track(goal)
            if goal.status == GoalStatus.ACTIVE:
                self.active.append(goal)
            else:
                self.pending.append(goal)
            self._save(goal, GoalEventType.GOAL_RESTORED)


def _title(goal: Goal) -> str:
    return " ".join(goal.title.lower().split())
