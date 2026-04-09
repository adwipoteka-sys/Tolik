from __future__ import annotations

from typing import Any

from motivation.goal_schema import Goal, GoalStatus


def _title_key(goal: Goal) -> str:
    return " ".join(goal.title.lower().split())


class GoalArbitrator:
    """Risk-aware admission and ranking of candidate goals."""

    def __init__(
        self,
        min_score_to_admit: float = 0.25,
        hard_risk_ceiling: float = 0.55,
    ) -> None:
        self.min_score_to_admit = min_score_to_admit
        self.hard_risk_ceiling = hard_risk_ceiling

    def deduplicate(self, goals: list[Goal]) -> list[Goal]:
        unique: list[Goal] = []
        seen: set[tuple[str, tuple[str, ...], str | None]] = set()
        for goal in goals:
            fingerprint = (_title_key(goal), tuple(sorted(goal.tags)), goal.parent_goal_id)
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            unique.append(goal)
        return unique

    def filter_by_capability(
        self,
        goals: list[Goal],
        available: set[str],
    ) -> tuple[list[Goal], list[Goal]]:
        admissible: list[Goal] = []
        deferred: list[Goal] = []
        for goal in goals:
            required = set(goal.required_capabilities)
            missing = sorted(required - available)
            if not missing:
                admissible.append(goal)
            else:
                goal.status = GoalStatus.DEFERRED
                goal.evidence.setdefault("missing_capabilities", missing)
                deferred.append(goal)
        return admissible, deferred

    def compute_resource_cost(self, goal: Goal) -> float:
        budget = goal.resource_budget
        step_cost = min(budget.max_steps / 20.0, 1.0)
        time_cost = min(budget.max_seconds / 60.0, 1.0)
        tool_cost = min((budget.max_tool_calls + budget.max_api_calls) / 5.0, 1.0)
        token_cost = 0.0
        if budget.max_tokens is not None:
            token_cost = min(budget.max_tokens / 8192.0, 1.0)
        return (step_cost + time_cost + tool_cost + token_cost) / 4.0

    def score(self, goal: Goal, context: dict[str, Any]) -> float:
        resource_cost = self.compute_resource_cost(goal)
        duplication_penalty = float(context.get("duplication_penalty", 0.0))
        regression_pressure = float(context.get("regression_pressure", 0.0))
        return (
            0.30 * goal.expected_gain
            + 0.20 * goal.novelty
            + 0.20 * goal.uncertainty_reduction
            + 0.15 * goal.strategic_fit
            + 0.15 * regression_pressure
            + 0.10 * goal.priority
            - 0.30 * goal.risk_estimate
            - 0.20 * resource_cost
            - 0.20 * duplication_penalty
        )

    def rank(self, goals: list[Goal], context: dict[str, Any]) -> list[Goal]:
        return sorted(
            goals,
            key=lambda goal: (self.score(goal, context), goal.priority, goal.created_at),
            reverse=True,
        )

    def admit(self, goals: list[Goal], context: dict[str, Any]) -> list[Goal]:
        active_titles = set(context.get("active_titles", set()))
        accepted: list[Goal] = []
        for goal in self.rank(self.deduplicate(goals), context):
            score = self.score(goal, context)
            if _title_key(goal) in active_titles:
                continue
            if score < self.min_score_to_admit:
                continue
            if goal.risk_estimate > goal.risk_budget:
                continue
            if goal.risk_estimate > self.hard_risk_ceiling:
                continue
            accepted.append(goal)
        return accepted
