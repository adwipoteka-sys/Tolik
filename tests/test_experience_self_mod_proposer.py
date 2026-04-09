from __future__ import annotations

from agency.agency_module import AgencyModule
from memory.episodic_memory import EpisodicMemory
from memory.goal_ledger import GoalLedger
from metacognition.postmortem import PostmortemReport
from motivation.goal_schema import Goal, GoalBudget, GoalKind, GoalSource, SuccessCriterion
from self_modification.experience_self_mod_proposer import (
    ExperienceSelfModificationProposer,
    GROUNDED_NAVIGATION_SIGNATURE,
)


def _goal() -> Goal:
    return Goal(
        goal_id="goal_detour_failure",
        title="Practice grounded navigation in detour tasks",
        description="Run grounded navigation practice.",
        source=GoalSource.CURIOSITY,
        kind=GoalKind.LEARNING,
        expected_gain=0.8,
        novelty=0.4,
        uncertainty_reduction=0.7,
        strategic_fit=0.9,
        risk_estimate=0.1,
        priority=0.9,
        risk_budget=0.2,
        resource_budget=GoalBudget(max_steps=3, max_seconds=10.0),
        success_criteria=[SuccessCriterion(metric="success_rate", comparator=">=", target=1.0)],
        required_capabilities=["classical_planning", "grounded_navigation"],
        tags=["grounded_self_training", "grounded_navigation", "local_only"],
        evidence={},
    )


def test_experience_self_mod_proposer_infers_navigation_upgrade_from_memory(tmp_path) -> None:
    ledger = GoalLedger(tmp_path / "ledger")
    episodic_memory = EpisodicMemory(ledger=ledger)
    agency = AgencyModule()

    goal = _goal()
    episodic_memory.record_goal_episode(
        goal,
        cycle=1,
        trace=[{"step": "run_capability:grounded_navigation", "result": {"success": False, "capability": "grounded_navigation"}}],
        outcome={"success": False},
        workspace_excerpt={"navigation_strategy": "greedy"},
        capability="grounded_navigation",
        pattern_key=GROUNDED_NAVIGATION_SIGNATURE,
        lesson="graph search should replace greedy routing on detours",
        tags=list(goal.tags),
    )
    ledger.save_postmortem(
        PostmortemReport(
            goal_id=goal.goal_id,
            success=False,
            expectation_gap=1.0,
            model_error=0.0,
            plan_error=1.0,
            execution_error=0.0,
            knowledge_gap=0.0,
            regression_flag=False,
            drift_flag=False,
            root_causes=["plan_error"],
            recommendations=["switch routing strategy"],
            tags=list(goal.tags),
        )
    )

    proposer = ExperienceSelfModificationProposer(
        ledger=ledger,
        episodic_memory=episodic_memory,
        components={"agency": agency},
    )
    proposals = proposer.propose_from_memory()

    assert len(proposals) == 1
    proposal = proposals[0]
    assert proposal.parameter_name == "grounded_navigation_strategy"
    assert proposal.baseline_value == "greedy"
    assert proposal.candidate_value == "graph_search"
    assert proposal.failure_support == 2
    assert proposal.confidence >= 0.8
    assert proposal.supporting_root_causes == ["plan_error"]
    assert proposal.anchor_cases and proposal.transfer_cases and proposal.canary_cases


def test_experience_self_mod_proposer_skips_when_component_already_upgraded(tmp_path) -> None:
    ledger = GoalLedger(tmp_path / "ledger")
    episodic_memory = EpisodicMemory(ledger=ledger)
    agency = AgencyModule(grounded_navigation_strategy="graph_search")

    goal = _goal()
    episodic_memory.record_goal_episode(
        goal,
        cycle=1,
        trace=[{"step": "run_capability:grounded_navigation", "result": {"success": False, "capability": "grounded_navigation"}}],
        outcome={"success": False},
        workspace_excerpt={"navigation_strategy": "graph_search"},
        capability="grounded_navigation",
        pattern_key=GROUNDED_NAVIGATION_SIGNATURE,
        lesson="no-op because already upgraded",
        tags=list(goal.tags),
    )
    ledger.save_postmortem(
        PostmortemReport(
            goal_id=goal.goal_id,
            success=False,
            expectation_gap=1.0,
            model_error=0.0,
            plan_error=1.0,
            execution_error=0.0,
            knowledge_gap=0.0,
            regression_flag=False,
            drift_flag=False,
            root_causes=["plan_error"],
            recommendations=["switch routing strategy"],
            tags=list(goal.tags),
        )
    )

    proposer = ExperienceSelfModificationProposer(
        ledger=ledger,
        episodic_memory=episodic_memory,
        components={"agency": agency},
    )

    assert proposer.propose_from_memory() == []
