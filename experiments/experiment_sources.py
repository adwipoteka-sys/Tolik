from __future__ import annotations

from automl.model_registry import ModelRegistry
from automl.safe_automl_manager import SafeAutoMLManager
from automl.response_risk_data_pipeline import ResponseRiskDataAcquisitionPipeline
from automl.response_risk_curriculum import audit_response_risk_model, response_risk_signature
from automl.response_risk_model import RESPONSE_RISK_FAMILY
from experiments.curriculum_experiment_scheduler import CurriculumExperimentScheduler
from experiments.experiment_schema import ExperimentKind, ExperimentProposal
from interfaces.cloud_llm import CloudLLMClient
from interfaces.quantum_solver import QuantumSolver
from self_modification.experience_self_mod_proposer import ExperienceSelfModificationProposer


_RESPONSE_RISK_UPGRADE_SIGNATURE = response_risk_signature()


def build_self_mod_experiment_candidates(
    proposer: ExperienceSelfModificationProposer,
    *,
    scheduler: CurriculumExperimentScheduler,
) -> list[ExperimentProposal]:
    candidates: list[ExperimentProposal] = []
    for proposal in proposer.propose_from_memory():
        expected_utility = min(0.98, 0.50 + (0.30 * float(proposal.confidence)) + (0.04 * min(5, proposal.failure_support)))
        estimated_cost = min(0.80, 0.18 + (0.03 * len(proposal.anchor_cases)) + (0.04 * len(proposal.transfer_cases)))
        curriculum_signals = [proposal.signature, *[f"root_cause:{cause}" for cause in proposal.supporting_root_causes[:3]]]
        candidate = scheduler.make_proposal(
            experiment_kind=ExperimentKind.POLICY_CHANGE,
            source_signature=proposal.signature,
            title=proposal.title,
            description=proposal.description,
            expected_utility=round(expected_utility, 3),
            estimated_risk=0.08,
            estimated_cost=round(estimated_cost, 3),
            confidence=proposal.confidence,
            curriculum_signals=curriculum_signals,
            tags=list(proposal.tags),
            required_capabilities=["classical_planning", proposal.capability],
            evidence={
                "self_mod_proposal_id": proposal.proposal_id,
                "proposal_confidence": proposal.confidence,
                "failure_support": proposal.failure_support,
                "supporting_goal_ids": list(proposal.supporting_goal_ids),
            },
        )
        candidates.append(candidate)
    return candidates


def build_response_risk_model_upgrade_candidate(
    registry: ModelRegistry,
    *,
    automl_manager: SafeAutoMLManager,
    scheduler: CurriculumExperimentScheduler,
    data_pipeline: ResponseRiskDataAcquisitionPipeline,
    min_accuracy: float = 0.99,
) -> ExperimentProposal | None:
    ongoing = [spec for spec in automl_manager.list_specs() if spec.model_family == RESPONSE_RISK_FAMILY and spec.status not in {"finalized", "rolled_back", "regression_rejected"}]
    if ongoing:
        return None
    active_model = registry.get_active_model(RESPONSE_RISK_FAMILY)
    audit = audit_response_risk_model(active_model)
    if float(audit["accuracy"]) >= min_accuracy and int(audit["false_negative_count"]) == 0:
        return None

    bundle = data_pipeline.latest_training_bundle()
    if bundle is None:
        return None

    train_count = len(bundle.training_examples)
    error_pressure = 1.0 - float(audit["accuracy"])
    expected_utility = min(0.98, 0.62 + (0.30 * error_pressure) + (0.04 * int(audit["false_negative_count"])))
    confidence = min(0.97, 0.58 + (0.08 * int(audit["case_count"]) / 5.0) + (0.02 * min(10, train_count)))
    estimated_cost = min(0.62, 0.28 + (0.01 * train_count) + (0.02 * len(bundle.transfer_cases)))
    curriculum_signals = [
        _RESPONSE_RISK_UPGRADE_SIGNATURE,
        f"dataset_snapshot:{bundle.snapshot.snapshot_id}",
        *[f"false_negative:{case_id}" for case_id in audit["false_negatives"][:3]],
    ]
    return scheduler.make_proposal(
        experiment_kind=ExperimentKind.MODEL_UPGRADE,
        source_signature=_RESPONSE_RISK_UPGRADE_SIGNATURE,
        title="Upgrade response risk model from autonomous curriculum data",
        description="Train a guarded response-risk model candidate from autonomously collected curriculum data when the active model misses verification cases.",
        expected_utility=round(expected_utility, 3),
        estimated_risk=0.09,
        estimated_cost=round(estimated_cost, 3),
        confidence=round(confidence, 3),
        curriculum_signals=curriculum_signals,
        tags=["automl_model_upgrade", "planning", "maintenance", "local_only", "response_risk_model"],
        required_capabilities=["classical_planning"],
        evidence={
            "model_family": RESPONSE_RISK_FAMILY,
            "audit": audit,
            "dataset_snapshot_id": bundle.snapshot.snapshot_id,
            "training_examples": [example.to_dict() for example in bundle.training_examples],
            "anchor_cases": [case.to_dict() for case in bundle.anchor_cases],
            "transfer_cases": [case.to_dict() for case in bundle.transfer_cases],
            "canary_cases": [case.to_dict() for case in bundle.canary_cases],
            "search_space": bundle.search_space,
            "dataset_stats": dict(bundle.snapshot.stats),
        },
    )


_FUTURE_QUANTUM_SIGNATURE = "future_interface|quantum_solver_probe"
_FUTURE_CLOUD_SIGNATURE = "future_interface|cloud_llm_probe"


def build_future_interface_candidates(
    *,
    scheduler: CurriculumExperimentScheduler,
    quantum_solver: QuantumSolver,
    cloud_llm: CloudLLMClient,
) -> list[ExperimentProposal]:
    return [
        scheduler.make_proposal(
            experiment_kind=ExperimentKind.POLICY_CHANGE,
            source_signature=_FUTURE_QUANTUM_SIGNATURE,
            title="Probe deferred quantum solver campaign when live access is available",
            description="Keep a deferred quantum-optimization probe on the experiment board until live access and charter approval are available.",
            expected_utility=0.56,
            estimated_risk=0.05,
            estimated_cost=0.18,
            confidence=0.72,
            curriculum_signals=[_FUTURE_QUANTUM_SIGNATURE, f"mode:{quantum_solver.mode}"],
            tags=["maintenance", "future_integration", "deferred_execution", "local_only"],
            required_capabilities=["classical_planning", "quantum_solver"],
            evidence={
                "requires_live_interface": True,
                "required_interface": "quantum_solver",
                "retry_after_cycles": 2,
                "experiment_action": "probe_quantum_interface",
                "campaign_budget": {"max_total_cost": 0.54, "max_total_risk": 0.16, "max_attempts": 2},
            },
        ),
        scheduler.make_proposal(
            experiment_kind=ExperimentKind.POLICY_CHANGE,
            source_signature=_FUTURE_CLOUD_SIGNATURE,
            title="Probe deferred cloud LLM campaign when live access is available",
            description="Keep a deferred cloud-LLM probe on the experiment board until live access and charter approval are available.",
            expected_utility=0.58,
            estimated_risk=0.06,
            estimated_cost=0.20,
            confidence=0.74,
            curriculum_signals=[_FUTURE_CLOUD_SIGNATURE, f"mode:{cloud_llm.mode}"],
            tags=["maintenance", "future_integration", "deferred_execution", "local_only"],
            required_capabilities=["classical_planning", "cloud_llm"],
            evidence={
                "requires_live_interface": True,
                "required_interface": "cloud_llm",
                "retry_after_cycles": 2,
                "experiment_action": "probe_cloud_llm_interface",
                "campaign_budget": {"max_total_cost": 0.60, "max_total_risk": 0.18, "max_attempts": 2},
            },
        ),
    ]
