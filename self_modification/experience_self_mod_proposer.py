from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from benchmarks.skill_arena import SkillArenaCase
from benchmarks.transfer_suite import TransferCase
from environments.grounded_navigation import GroundedNavigationLab
from memory.episodic_memory import EpisodeRecord, EpisodicMemory
from memory.goal_ledger import GoalLedger
from motivation.goal_schema import Goal, GoalBudget, GoalKind, GoalSource, SuccessCriterion, new_goal_id
from self_modification.proposal_schema import SelfModificationProposal, new_proposal_id


GROUNDED_NAVIGATION_SIGNATURE = "grounded_navigation|graph_search_patch"
RESPONSE_PLANNING_SIGNATURE = "response_planning|verify_before_answer_patch"
MEMORY_RETRIEVAL_SIGNATURE = "memory_retrieval|working_then_semantic_backoff_patch"


@dataclass(slots=True)
class ProposalTemplate:
    signature: str
    capability: str
    target_component: str
    parameter_name: str
    candidate_value: Any
    title: str
    description: str
    min_evidence: int = 2
    root_cause_hints: tuple[str, ...] = ()


class ExperienceSelfModificationProposer:
    """Infers internal self-modification candidates from episodic and postmortem memory."""

    def __init__(
        self,
        *,
        ledger: GoalLedger,
        episodic_memory: EpisodicMemory,
        components: dict[str, Any],
    ) -> None:
        self.ledger = ledger
        self.episodic_memory = episodic_memory
        self.components = dict(components)
        self.templates = {
            GROUNDED_NAVIGATION_SIGNATURE: ProposalTemplate(
                signature=GROUNDED_NAVIGATION_SIGNATURE,
                capability="grounded_navigation",
                target_component="agency",
                parameter_name="grounded_navigation_strategy",
                candidate_value="graph_search",
                title="Promote graph-search grounded navigation",
                description="Infer a safer internal navigation strategy from repeated detour failures.",
                min_evidence=2,
                root_cause_hints=("plan_error", "regression_failure"),
            ),
            RESPONSE_PLANNING_SIGNATURE: ProposalTemplate(
                signature=RESPONSE_PLANNING_SIGNATURE,
                capability="response_planning",
                target_component="planning",
                parameter_name="response_planning_policy",
                candidate_value="verify_before_answer",
                title="Insert answer verification in high-risk user tasks",
                description="Promote a safer response-planning policy when memory shows repeated unverified answer failures.",
                min_evidence=2,
                root_cause_hints=("plan_error", "regression_failure"),
            ),
            MEMORY_RETRIEVAL_SIGNATURE: ProposalTemplate(
                signature=MEMORY_RETRIEVAL_SIGNATURE,
                capability="memory_retrieval",
                target_component="memory",
                parameter_name="retrieval_policy",
                candidate_value="working_then_semantic_backoff",
                title="Promote semantic-backoff retrieval with working-memory routing",
                description="Promote a more robust retrieval policy when exact lookup repeatedly misses recoverable facts.",
                min_evidence=2,
                root_cause_hints=("knowledge_gap", "plan_error"),
            ),
        }
        self._proposals: dict[str, SelfModificationProposal] = {}
        self._rehydrate()

    def _rehydrate(self) -> None:
        for payload in self.ledger.load_self_modification_proposals():
            proposal = SelfModificationProposal.from_dict(payload)
            self._proposals[proposal.proposal_id] = proposal

    def _persist(self, proposal: SelfModificationProposal) -> SelfModificationProposal:
        proposal.touch()
        self._proposals[proposal.proposal_id] = proposal
        self.ledger.save_self_modification_proposal(proposal.to_dict())
        return proposal

    def list_proposals(self) -> list[SelfModificationProposal]:
        return sorted(self._proposals.values(), key=lambda item: item.created_at)

    def get(self, proposal_id: str) -> SelfModificationProposal | None:
        return self._proposals.get(proposal_id)

    def latest_for_signature(self, signature: str) -> SelfModificationProposal | None:
        matches = [item for item in self._proposals.values() if item.signature == signature]
        if not matches:
            return None
        return sorted(matches, key=lambda item: item.updated_at)[-1]

    def _resolve_component(self, name: str) -> Any:
        if name not in self.components:
            raise KeyError(f"Unknown self-modification component: {name}")
        return self.components[name]

    def _matching_failure_episodes(self, signature: str) -> list[EpisodeRecord]:
        return [episode for episode in self.episodic_memory.by_pattern(signature) if not episode.success]

    def _supporting_postmortems(self, goal_ids: set[str]) -> list[dict[str, Any]]:
        postmortems = getattr(self.ledger, "load_postmortems", lambda: [])()
        return [payload for payload in postmortems if payload.get("goal_id") in goal_ids and not payload.get("success", False)]

    def _has_live_proposal(self, signature: str) -> bool:
        latest = self.latest_for_signature(signature)
        return latest is not None and latest.status in {"proposed", "goal_materialized", "in_review", "finalized"}

    def _filter_postmortems(self, postmortems: list[dict[str, Any]], template: ProposalTemplate) -> list[dict[str, Any]]:
        if not template.root_cause_hints:
            return list(postmortems)
        selected: list[dict[str, Any]] = []
        for payload in postmortems:
            causes = set(payload.get("root_causes", []))
            if causes.intersection(template.root_cause_hints):
                selected.append(payload)
        return selected

    def _build_proposal(
        self,
        *,
        template: ProposalTemplate,
        baseline_value: Any,
        episodes: list[EpisodeRecord],
        postmortems: list[dict[str, Any]],
        anchor_cases: list[SkillArenaCase],
        transfer_cases: list[TransferCase],
        canary_cases: list[SkillArenaCase],
        rationale: str,
        tags: list[str],
    ) -> SelfModificationProposal | None:
        if baseline_value == template.candidate_value:
            latest = self.latest_for_signature(template.signature)
            if latest is not None and latest.status != "finalized":
                latest.status = "superseded"
                self._persist(latest)
            return None
        if self._has_live_proposal(template.signature):
            return None

        goal_ids = {episode.goal_id for episode in episodes}
        supporting_postmortems = self._filter_postmortems(postmortems, template)
        evidence_count = len(episodes) + len(supporting_postmortems)
        if evidence_count < template.min_evidence:
            return None

        confidence = min(0.55 + 0.15 * len(episodes) + 0.10 * len(supporting_postmortems), 0.99)
        proposal = SelfModificationProposal(
            proposal_id=new_proposal_id(),
            signature=template.signature,
            capability=template.capability,
            title=template.title,
            description=template.description,
            target_component=template.target_component,
            parameter_name=template.parameter_name,
            baseline_value=baseline_value,
            candidate_value=template.candidate_value,
            rationale=rationale,
            confidence=round(confidence, 3),
            failure_support=evidence_count,
            supporting_episode_ids=[episode.episode_id for episode in episodes],
            supporting_goal_ids=sorted(goal_ids),
            supporting_root_causes=sorted({cause for payload in postmortems for cause in payload.get("root_causes", [])}),
            supporting_postmortem_ids=[str(payload.get("goal_id")) for payload in supporting_postmortems],
            tags=list(tags),
            anchor_cases=anchor_cases,
            transfer_cases=transfer_cases,
            canary_cases=canary_cases,
        )
        return self._persist(proposal)

    def _proposal_cases_for_signature(self, signature: str) -> tuple[list[SkillArenaCase], list[TransferCase], list[SkillArenaCase]]:
        if signature == GROUNDED_NAVIGATION_SIGNATURE:
            lab = GroundedNavigationLab()
            anchor_cases = [
                SkillArenaCase(
                    case_id="nav_anchor_easy_open",
                    payload={"tasks": [lab.get_task("nav_easy_open").to_dict()], "success_threshold": 1.0},
                    expected={"min_success_rate": 1.0, "max_mean_path_ratio": 1.0},
                    description="Easy open grounded task must remain solved.",
                ),
                SkillArenaCase(
                    case_id="nav_anchor_easy_corner",
                    payload={"tasks": [lab.get_task("nav_easy_corner").to_dict()], "success_threshold": 1.0},
                    expected={"min_success_rate": 1.0, "max_mean_path_ratio": 1.0},
                    description="Easy corner grounded task must remain solved.",
                ),
            ]
            transfer_cases = [
                TransferCase(
                    case_id="nav_transfer_bridge",
                    payload={"tasks": [lab.get_task("nav_transfer_bridge").to_dict()], "success_threshold": 1.0},
                    expected={"min_success_rate": 1.0, "max_mean_path_ratio": 1.0, "strategy": "graph_search", "min_task_count": 1},
                    description="Held-out bridge detour should generalize under graph search.",
                ),
                TransferCase(
                    case_id="nav_transfer_double_wall",
                    payload={"tasks": [lab.get_task("nav_transfer_double_wall").to_dict()], "success_threshold": 1.0},
                    expected={"min_success_rate": 1.0, "max_mean_path_ratio": 1.0, "strategy": "graph_search", "min_task_count": 1},
                    description="Held-out multi-wall detour should generalize under graph search.",
                ),
            ]
            canary_cases = [
                SkillArenaCase(
                    case_id="nav_detour_curriculum",
                    payload={"tasks": [lab.get_task("nav_detour_wall").to_dict(), lab.get_task("nav_detour_channel").to_dict()], "success_threshold": 1.0},
                    expected={"min_success_rate": 1.0, "max_mean_path_ratio": 1.0, "strategy": "graph_search"},
                    description="Detour curriculum must pass in guarded canary.",
                ),
                SkillArenaCase(
                    case_id="nav_easy_regression_guard",
                    payload={"tasks": [lab.get_task("nav_easy_open").to_dict(), lab.get_task("nav_easy_corner").to_dict()], "success_threshold": 1.0},
                    expected={"min_success_rate": 1.0, "max_mean_path_ratio": 1.0, "strategy": "graph_search"},
                    description="Easy cases must stay stable in canary.",
                ),
            ]
            return anchor_cases, transfer_cases, canary_cases

        if signature == RESPONSE_PLANNING_SIGNATURE:
            anchor_cases = [
                SkillArenaCase(
                    case_id="planning_anchor_simple_question",
                    payload={"goal": _make_response_goal("What is 2+2?", requires_verification=False).to_dict()},
                    expected={"required_steps": ["understand_request", "form_response"], "forbidden_steps": ["verify_outcome"]},
                    description="Simple answers should remain lightweight.",
                ),
                SkillArenaCase(
                    case_id="planning_anchor_summarizer",
                    payload={"goal": _make_response_goal("Summarize notes", requires_verification=False, uses_summarizer=True).to_dict()},
                    expected={"required_steps": ["understand_request", "run_capability:text_summarizer", "form_response"], "forbidden_steps": ["verify_outcome"]},
                    description="Stable summarizer requests should keep the regular fast path.",
                ),
            ]
            transfer_cases = [
                TransferCase(
                    case_id="planning_transfer_verified_answer",
                    payload={"goal": _make_response_goal("Answer with verification", requires_verification=True).to_dict()},
                    expected={"required_steps": ["understand_request", "verify_outcome", "form_response"], "forbidden_steps": [], "policy": "verify_before_answer"},
                    description="High-risk answer tasks should insert verification before responding.",
                ),
                TransferCase(
                    case_id="planning_transfer_verified_summarizer",
                    payload={"goal": _make_response_goal("Summarize with verification", requires_verification=True, uses_summarizer=True).to_dict()},
                    expected={"required_steps": ["understand_request", "run_capability:text_summarizer", "verify_outcome", "form_response"], "forbidden_steps": [], "policy": "verify_before_answer"},
                    description="Held-out verified summarizer requests should insert verification.",
                ),
            ]
            canary_cases = [
                SkillArenaCase(
                    case_id="planning_canary_verified_answer",
                    payload={"goal": _make_response_goal("Critical response path", requires_verification=True).to_dict()},
                    expected={"required_steps": ["understand_request", "verify_outcome", "form_response"], "forbidden_steps": [], "policy": "verify_before_answer"},
                    description="Canary should verify answers on flagged user tasks.",
                ),
                SkillArenaCase(
                    case_id="planning_canary_simple_question",
                    payload={"goal": _make_response_goal("Routine answer", requires_verification=False).to_dict()},
                    expected={"required_steps": ["understand_request", "form_response"], "forbidden_steps": ["verify_outcome"]},
                    description="Canary must preserve the lightweight simple-answer path.",
                ),
            ]
            return anchor_cases, transfer_cases, canary_cases

        if signature == MEMORY_RETRIEVAL_SIGNATURE:
            france_fact = {"fact": "Paris", "aliases": ["capital of france"]}
            berlin_fact = {"fact": "Berlin", "aliases": ["capital of germany"]}
            anchor_cases = [
                SkillArenaCase(
                    case_id="memory_anchor_exact_lookup",
                    payload={"long_term_facts": {"France": france_fact}, "query": "France"},
                    expected={"retrieved": france_fact},
                    description="Exact fact retrieval must remain intact.",
                ),
            ]
            transfer_cases = [
                TransferCase(
                    case_id="memory_transfer_alias_lookup",
                    payload={"long_term_facts": {"France": france_fact}, "query": "capital of france"},
                    expected={"retrieved": france_fact},
                    description="Alias queries should back off to semantic matches.",
                ),
                TransferCase(
                    case_id="memory_transfer_working_memory_lookup",
                    payload={"working_facts": {"capital of germany": berlin_fact}, "query": "capital of germany"},
                    expected={"retrieved": berlin_fact},
                    description="Recent working-memory facts should be retrievable before long-term misses.",
                ),
            ]
            canary_cases = [
                SkillArenaCase(
                    case_id="memory_canary_alias_lookup",
                    payload={"long_term_facts": {"France": france_fact}, "query": "capital of france"},
                    expected={"retrieved": france_fact},
                    description="Canary should resolve semantic alias lookups.",
                ),
                SkillArenaCase(
                    case_id="memory_canary_working_memory_lookup",
                    payload={"working_facts": {"capital of germany": berlin_fact}, "query": "capital of germany"},
                    expected={"retrieved": berlin_fact},
                    description="Canary should consult working memory before reporting a gap.",
                ),
            ]
            return anchor_cases, transfer_cases, canary_cases

        return [], [], []

    def _build_grounded_navigation_proposal(self, template: ProposalTemplate) -> SelfModificationProposal | None:
        component = self._resolve_component(template.target_component)
        baseline_value = getattr(component, template.parameter_name)
        episodes = [
            episode
            for episode in self._matching_failure_episodes(template.signature)
            if episode.workspace_excerpt.get("navigation_strategy") == baseline_value
        ]
        postmortems = self._supporting_postmortems({episode.goal_id for episode in episodes})
        anchor_cases, transfer_cases, canary_cases = self._proposal_cases_for_signature(template.signature)
        rationale = (
            f"Observed {len(episodes)} failed episode(s) with strategy='{baseline_value}' on detour-heavy grounded navigation "
            f"and {len(self._filter_postmortems(postmortems, template))} supporting postmortem(s) pointing to a planning/strategy failure. "
            f"Candidate '{template.candidate_value}' is predicted to reduce detour regressions while preserving anchor behavior."
        )
        return self._build_proposal(
            template=template,
            baseline_value=baseline_value,
            episodes=episodes,
            postmortems=postmortems,
            anchor_cases=anchor_cases,
            transfer_cases=transfer_cases,
            canary_cases=canary_cases,
            rationale=rationale,
            tags=["grounded_navigation_patch", "grounded_navigation", "maintenance", "local_only"],
        )

    def _build_response_planning_proposal(self, template: ProposalTemplate) -> SelfModificationProposal | None:
        component = self._resolve_component(template.target_component)
        baseline_value = getattr(component, template.parameter_name)
        episodes = [
            episode
            for episode in self._matching_failure_episodes(template.signature)
            if episode.workspace_excerpt.get("response_planning_policy") == baseline_value
        ]
        postmortems = self._supporting_postmortems({episode.goal_id for episode in episodes})
        anchor_cases, transfer_cases, canary_cases = self._proposal_cases_for_signature(template.signature)
        rationale = (
            f"Observed {len(episodes)} failed response-planning episode(s) under policy='{baseline_value}' "
            f"and {len(self._filter_postmortems(postmortems, template))} supporting postmortem(s) with plan-level failure signals. "
            f"Candidate '{template.candidate_value}' adds verification on flagged user tasks while preserving the lightweight path for routine requests."
        )
        return self._build_proposal(
            template=template,
            baseline_value=baseline_value,
            episodes=episodes,
            postmortems=postmortems,
            anchor_cases=anchor_cases,
            transfer_cases=transfer_cases,
            canary_cases=canary_cases,
            rationale=rationale,
            tags=["response_planning_patch", "planning", "maintenance", "local_only"],
        )

    def _build_memory_retrieval_proposal(self, template: ProposalTemplate) -> SelfModificationProposal | None:
        component = self._resolve_component(template.target_component)
        baseline_value = getattr(component, template.parameter_name)
        episodes = [
            episode
            for episode in self._matching_failure_episodes(template.signature)
            if episode.workspace_excerpt.get("retrieval_policy") == baseline_value
        ]
        postmortems = self._supporting_postmortems({episode.goal_id for episode in episodes})
        anchor_cases, transfer_cases, canary_cases = self._proposal_cases_for_signature(template.signature)
        rationale = (
            f"Observed {len(episodes)} recoverable retrieval miss(es) under policy='{baseline_value}' "
            f"and {len(self._filter_postmortems(postmortems, template))} supporting postmortem(s) with knowledge-gap signals. "
            f"Candidate '{template.candidate_value}' routes lookups through working memory and semantic aliases before declaring a miss."
        )
        return self._build_proposal(
            template=template,
            baseline_value=baseline_value,
            episodes=episodes,
            postmortems=postmortems,
            anchor_cases=anchor_cases,
            transfer_cases=transfer_cases,
            canary_cases=canary_cases,
            rationale=rationale,
            tags=["memory_retrieval_patch", "memory", "maintenance", "local_only"],
        )

    def propose_from_memory(self) -> list[SelfModificationProposal]:
        proposals: list[SelfModificationProposal] = []
        for signature, builder in [
            (GROUNDED_NAVIGATION_SIGNATURE, self._build_grounded_navigation_proposal),
            (RESPONSE_PLANNING_SIGNATURE, self._build_response_planning_proposal),
            (MEMORY_RETRIEVAL_SIGNATURE, self._build_memory_retrieval_proposal),
        ]:
            template = self.templates[signature]
            if template.target_component not in self.components:
                continue
            proposal = builder(template)
            if proposal is not None:
                self.ledger.append_event(
                    {
                        "event_type": "self_mod_proposed",
                        "proposal_id": proposal.proposal_id,
                        "signature": proposal.signature,
                        "capability": proposal.capability,
                        "confidence": proposal.confidence,
                        "failure_support": proposal.failure_support,
                    }
                )
                proposals.append(proposal)
        return proposals

    def materialize_goal(self, proposal_id: str) -> Goal:
        proposal = self._proposals[proposal_id]
        proposal.status = "goal_materialized"
        self._persist(proposal)
        self.ledger.append_event(
            {
                "event_type": "self_mod_goal_materialized",
                "proposal_id": proposal.proposal_id,
                "signature": proposal.signature,
                "capability": proposal.capability,
            }
        )
        return Goal(
            goal_id=new_goal_id("selfmodgoal"),
            title=proposal.title,
            description=proposal.description,
            source=GoalSource.METACOGNITION,
            kind=GoalKind.MAINTENANCE,
            expected_gain=min(0.70 + 0.10 * proposal.confidence, 0.95),
            novelty=0.28,
            uncertainty_reduction=min(0.50 + 0.20 * proposal.confidence, 0.95),
            strategic_fit=0.95,
            risk_estimate=0.07,
            priority=min(0.78 + 0.10 * proposal.confidence, 0.97),
            risk_budget=0.15,
            resource_budget=GoalBudget(max_steps=6, max_seconds=20.0, max_tool_calls=0, max_api_calls=0),
            success_criteria=[SuccessCriterion(metric="status", comparator="==", target="finalized")],
            required_capabilities=["classical_planning", proposal.capability],
            tags=list(proposal.tags),
            evidence={
                "self_modification": True,
                "self_mod_proposal_id": proposal.proposal_id,
                "self_mod_signature": proposal.signature,
                "target_capability": proposal.capability,
                "self_mod_target_component": proposal.target_component,
                "self_mod_parameter_name": proposal.parameter_name,
                "target_strategy": proposal.candidate_value,
                "proposal_confidence": proposal.confidence,
            },
        )

    def mark_finalized(self, proposal_id: str) -> SelfModificationProposal | None:
        proposal = self._proposals.get(proposal_id)
        if proposal is None:
            return None
        proposal.status = "finalized"
        persisted = self._persist(proposal)
        self.ledger.append_event(
            {
                "event_type": "self_mod_proposal_finalized",
                "proposal_id": proposal_id,
                "signature": proposal.signature,
                "candidate_value": proposal.candidate_value,
            }
        )
        return persisted

    def mark_rolled_back(self, proposal_id: str, *, reason: str) -> SelfModificationProposal | None:
        proposal = self._proposals.get(proposal_id)
        if proposal is None:
            return None
        proposal.status = "rolled_back"
        persisted = self._persist(proposal)
        self.ledger.append_event(
            {
                "event_type": "self_mod_proposal_rolled_back",
                "proposal_id": proposal_id,
                "signature": proposal.signature,
                "reason": reason,
            }
        )
        return persisted


def _make_response_goal(title: str, *, requires_verification: bool, uses_summarizer: bool = False) -> Goal:
    required_capabilities = ["classical_planning"]
    if uses_summarizer:
        required_capabilities.append("text_summarizer")
    return Goal(
        goal_id=new_goal_id("usertask"),
        title=title,
        description=title,
        source=GoalSource.USER,
        kind=GoalKind.USER_TASK,
        expected_gain=0.6,
        novelty=0.2,
        uncertainty_reduction=0.4,
        strategic_fit=0.7,
        risk_estimate=0.1,
        priority=0.7,
        risk_budget=0.1,
        resource_budget=GoalBudget(max_steps=4, max_seconds=10.0, max_tool_calls=0, max_api_calls=0),
        success_criteria=[SuccessCriterion(metric="status", comparator="==", target="done")],
        required_capabilities=required_capabilities,
        tags=["user", "critical_answer"] if requires_verification else ["user"],
        evidence={"requires_verification": requires_verification},
    )
