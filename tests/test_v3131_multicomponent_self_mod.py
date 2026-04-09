from __future__ import annotations

from metacognition.postmortem import PostmortemReport
from motivation.goal_schema import Goal, GoalBudget, GoalKind, GoalSource, SuccessCriterion
from self_modification.experience_self_mod_proposer import (
    ExperienceSelfModificationProposer,
    MEMORY_RETRIEVAL_SIGNATURE,
    RESPONSE_PLANNING_SIGNATURE,
)
from main import build_system


def _response_goal() -> Goal:
    return Goal(
        goal_id="goal_response_failure",
        title="Answer a critical user question",
        description="Respond to a user task that requires explicit verification.",
        source=GoalSource.USER,
        kind=GoalKind.USER_TASK,
        expected_gain=0.6,
        novelty=0.2,
        uncertainty_reduction=0.4,
        strategic_fit=0.7,
        risk_estimate=0.1,
        priority=0.7,
        risk_budget=0.1,
        resource_budget=GoalBudget(max_steps=4, max_seconds=10.0),
        success_criteria=[SuccessCriterion(metric="status", comparator="==", target="done")],
        required_capabilities=["classical_planning"],
        tags=["user", "critical_answer"],
        evidence={"requires_verification": True},
    )


def _memory_goal() -> Goal:
    return Goal(
        goal_id="goal_memory_failure",
        title="Recover a missing fact from memory",
        description="Resolve a knowledge gap from memory without external help.",
        source=GoalSource.MEMORY_GAP,
        kind=GoalKind.MAINTENANCE,
        expected_gain=0.6,
        novelty=0.2,
        uncertainty_reduction=0.5,
        strategic_fit=0.7,
        risk_estimate=0.1,
        priority=0.7,
        risk_budget=0.1,
        resource_budget=GoalBudget(max_steps=3, max_seconds=10.0),
        success_criteria=[SuccessCriterion(metric="status", comparator="==", target="done")],
        required_capabilities=["classical_planning"],
        tags=["knowledge_gap", "memory"],
        evidence={},
    )


def test_multicomponent_self_mod_proposer_infers_planning_and_memory_patches(tmp_path) -> None:
    system = build_system(tmp_path / "runtime")
    ledger = system["ledger"]
    episodic_memory = system["episodic_memory"]
    planning = system["planning"]
    memory = system["memory"]
    agency = system["agency"]

    response_goal = _response_goal()
    episodic_memory.record_goal_episode(
        response_goal,
        cycle=1,
        trace=[{"step": "form_response", "result": {"success": False, "capability": "response_planning"}}],
        outcome={"success": False},
        workspace_excerpt={"response_planning_policy": planning.response_planning_policy},
        capability="response_planning",
        pattern_key=RESPONSE_PLANNING_SIGNATURE,
        lesson="verification should precede high-risk answers",
        tags=list(response_goal.tags),
    )
    ledger.save_postmortem(
        PostmortemReport(
            goal_id=response_goal.goal_id,
            success=False,
            expectation_gap=1.0,
            model_error=0.0,
            plan_error=1.0,
            execution_error=0.0,
            knowledge_gap=0.0,
            regression_flag=False,
            drift_flag=False,
            root_causes=["plan_error"],
            recommendations=["insert verification step"],
            tags=list(response_goal.tags),
        )
    )

    memory_goal = _memory_goal()
    episodic_memory.record_goal_episode(
        memory_goal,
        cycle=2,
        trace=[{"step": "retrieve_missing_knowledge", "result": {"success": False, "capability": "memory_retrieval", "knowledge_gap": True}}],
        outcome={"success": False},
        workspace_excerpt={"retrieval_policy": memory.retrieval_policy},
        capability="memory_retrieval",
        pattern_key=MEMORY_RETRIEVAL_SIGNATURE,
        lesson="consult working memory and semantic aliases before declaring a gap",
        tags=list(memory_goal.tags),
    )
    ledger.save_postmortem(
        PostmortemReport(
            goal_id=memory_goal.goal_id,
            success=False,
            expectation_gap=1.0,
            model_error=0.0,
            plan_error=0.0,
            execution_error=0.0,
            knowledge_gap=1.0,
            regression_flag=False,
            drift_flag=False,
            root_causes=["knowledge_gap"],
            recommendations=["promote semantic backoff retrieval"],
            tags=list(memory_goal.tags),
        )
    )

    proposer = ExperienceSelfModificationProposer(
        ledger=ledger,
        episodic_memory=episodic_memory,
        components={"agency": agency, "planning": planning, "memory": memory},
    )
    proposals = proposer.propose_from_memory()

    signatures = {proposal.signature for proposal in proposals}
    assert RESPONSE_PLANNING_SIGNATURE in signatures
    assert MEMORY_RETRIEVAL_SIGNATURE in signatures

    planning_proposal = next(item for item in proposals if item.signature == RESPONSE_PLANNING_SIGNATURE)
    assert planning_proposal.target_component == "planning"
    assert planning_proposal.parameter_name == "response_planning_policy"
    assert planning_proposal.candidate_value == "verify_before_answer"

    memory_proposal = next(item for item in proposals if item.signature == MEMORY_RETRIEVAL_SIGNATURE)
    assert memory_proposal.target_component == "memory"
    assert memory_proposal.parameter_name == "retrieval_policy"
    assert memory_proposal.candidate_value == "working_then_semantic_backoff"


def test_safe_self_mod_manager_finalizes_planning_and_memory_changes(tmp_path) -> None:
    system = build_system(tmp_path / "runtime")
    manager = system["self_mod_manager"]
    proposer = system["self_mod_proposer"]
    planning = system["planning"]
    memory = system["memory"]

    planning_anchor, planning_transfer, planning_canary = proposer._proposal_cases_for_signature(RESPONSE_PLANNING_SIGNATURE)
    planning_spec = manager.stage_attribute_change(
        goal_id="goal_response_patch",
        title="Enable verification-aware response planning",
        target_component="planning",
        capability="response_planning",
        parameter_name="response_planning_policy",
        candidate_value="verify_before_answer",
        anchor_cases=planning_anchor,
        transfer_cases=planning_transfer,
        canary_cases=planning_canary,
        rationale="insert verification before answering flagged user tasks",
        threshold=0.99,
    )
    assert manager.run_regression_gate(planning_spec.change_id).passed is True
    manager.promote_canary(planning_spec.change_id)
    canary = manager.evaluate_canary(planning_spec.change_id)
    assert canary.passed is True and canary.rolled_back is False
    manager.finalize_change(planning_spec.change_id)
    assert planning.response_planning_policy == "verify_before_answer"

    memory_anchor, memory_transfer, memory_canary = proposer._proposal_cases_for_signature(MEMORY_RETRIEVAL_SIGNATURE)
    memory_spec = manager.stage_attribute_change(
        goal_id="goal_memory_patch",
        title="Enable semantic-backoff retrieval",
        target_component="memory",
        capability="memory_retrieval",
        parameter_name="retrieval_policy",
        candidate_value="working_then_semantic_backoff",
        anchor_cases=memory_anchor,
        transfer_cases=memory_transfer,
        canary_cases=memory_canary,
        rationale="route lookups through working memory and aliases before declaring failure",
        threshold=0.99,
    )
    assert manager.run_regression_gate(memory_spec.change_id).passed is True
    manager.promote_canary(memory_spec.change_id)
    canary = manager.evaluate_canary(memory_spec.change_id)
    assert canary.passed is True and canary.rolled_back is False
    manager.finalize_change(memory_spec.change_id)
    assert memory.retrieval_policy == "working_then_semantic_backoff"
