from __future__ import annotations

from typing import Any

from motivation.goal_schema import (
    Goal,
    GoalBudget,
    GoalKind,
    GoalSource,
    SuccessCriterion,
    new_goal_id,
)


def _internal_goal(
    *,
    title: str,
    description: str,
    source: GoalSource,
    kind: GoalKind,
    expected_gain: float,
    novelty: float,
    uncertainty_reduction: float,
    strategic_fit: float,
    risk_estimate: float,
    priority: float,
    risk_budget: float,
    required_capabilities: list[str] | None = None,
    tags: list[str] | None = None,
    evidence: dict[str, Any] | None = None,
) -> Goal:
    return Goal(
        goal_id=new_goal_id(source.value),
        title=title,
        description=description,
        source=source,
        kind=kind,
        expected_gain=expected_gain,
        novelty=novelty,
        uncertainty_reduction=uncertainty_reduction,
        strategic_fit=strategic_fit,
        risk_estimate=risk_estimate,
        priority=priority,
        risk_budget=risk_budget,
        resource_budget=GoalBudget(max_steps=4, max_seconds=20.0, max_tool_calls=0, max_api_calls=0),
        success_criteria=[
            SuccessCriterion(metric="status", comparator="==", target="done"),
        ],
        required_capabilities=required_capabilities or ["classical_planning"],
        tags=tags or [],
        evidence=evidence or {},
    )


def detect_curiosity(
    workspace_state: dict[str, Any],
    memory_summary: dict[str, Any],
    meta_summary: dict[str, Any],
) -> list[Goal]:
    novelty_score = float(workspace_state.get("novelty_score", 0.0))
    queue_pressure = float(workspace_state.get("queue_pressure", workspace_state.get("goal_queue_size", 0.0)))
    risk_estimate = float(workspace_state.get("risk_estimate", 1.0))

    if novelty_score > 0.70 and queue_pressure < 3 and risk_estimate < 0.35:
        return [
            _internal_goal(
                title="Investigate novel observation",
                description="Explore a new observation to reduce uncertainty and collect useful knowledge.",
                source=GoalSource.CURIOSITY,
                kind=GoalKind.EXPLORATION,
                expected_gain=0.58,
                novelty=novelty_score,
                uncertainty_reduction=0.62,
                strategic_fit=0.54,
                risk_estimate=risk_estimate,
                priority=0.55,
                risk_budget=0.35,
                tags=["curiosity", "novelty"],
                evidence={
                    "novelty_score": novelty_score,
                    "queue_pressure": queue_pressure,
                },
            )
        ]
    return []


def detect_drift_alarm(
    workspace_state: dict[str, Any],
    memory_summary: dict[str, Any],
    meta_summary: dict[str, Any],
) -> list[Goal]:
    ood_score = float(workspace_state.get("ood_score", 0.0))
    holdout_loss_delta = float(workspace_state.get("holdout_loss_delta", 0.0))
    effective_update_rate = float(workspace_state.get("effective_update_rate", 1.0))

    if ood_score > 0.60 or holdout_loss_delta > 0.15 or effective_update_rate < 0.30:
        return [
            _internal_goal(
                title="Diagnose model drift",
                description="Inspect drift indicators and refresh local calibration baselines.",
                source=GoalSource.DRIFT_ALARM,
                kind=GoalKind.MAINTENANCE,
                expected_gain=0.72,
                novelty=0.36,
                uncertainty_reduction=0.80,
                strategic_fit=0.82,
                risk_estimate=0.18,
                priority=0.78,
                risk_budget=0.45,
                tags=["drift", "maintenance"],
                evidence={
                    "ood_score": ood_score,
                    "holdout_loss_delta": holdout_loss_delta,
                    "effective_update_rate": effective_update_rate,
                },
            )
        ]
    return []


def detect_memory_gap(
    workspace_state: dict[str, Any],
    memory_summary: dict[str, Any],
    meta_summary: dict[str, Any],
) -> list[Goal]:
    retrieval_confidence = float(workspace_state.get("retrieval_confidence", 1.0))
    query_miss = bool(memory_summary.get("query_miss") or memory_summary.get("last_query_miss"))
    insufficient_evidence = bool(meta_summary.get("insufficient_evidence"))

    if retrieval_confidence < 0.45 or query_miss or insufficient_evidence:
        return [
            _internal_goal(
                title="Fill missing knowledge",
                description="Retrieve or learn the missing fact needed for robust reasoning.",
                source=GoalSource.MEMORY_GAP,
                kind=GoalKind.LEARNING,
                expected_gain=0.80,
                novelty=0.28,
                uncertainty_reduction=0.90,
                strategic_fit=0.88,
                risk_estimate=0.10,
                priority=0.84,
                risk_budget=0.30,
                required_capabilities=["classical_planning", "local_llm"],
                tags=["knowledge_gap", "learning"],
                evidence={
                    "retrieval_confidence": retrieval_confidence,
                    "query_miss": query_miss,
                    "insufficient_evidence": insufficient_evidence,
                },
            )
        ]
    return []


def detect_regression_failure(
    workspace_state: dict[str, Any],
    memory_summary: dict[str, Any],
    meta_summary: dict[str, Any],
) -> list[Goal]:
    anchor_suite_failed = bool(workspace_state.get("anchor_suite_failed", False))
    regression_score = float(workspace_state.get("regression_score", 1.0))

    if anchor_suite_failed or regression_score < 0.65:
        return [
            _internal_goal(
                title="Recover regressed skill",
                description="Replay anchor tasks and recover a previously working capability.",
                source=GoalSource.REGRESSION_FAILURE,
                kind=GoalKind.REGRESSION_RECOVERY,
                expected_gain=0.92,
                novelty=0.16,
                uncertainty_reduction=0.86,
                strategic_fit=0.94,
                risk_estimate=0.12,
                priority=0.96,
                risk_budget=0.35,
                tags=["regression", "anchor_suite"],
                evidence={
                    "anchor_suite_failed": anchor_suite_failed,
                    "regression_score": regression_score,
                },
            )
        ]
    return []


ALL_DETECTORS = [
    detect_curiosity,
    detect_drift_alarm,
    detect_memory_gap,
    detect_regression_failure,
]
