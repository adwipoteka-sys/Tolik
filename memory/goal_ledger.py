from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmarks.benchmark_expander import RegressionCase
from metacognition.failure_miner import FailureCase
from metacognition.postmortem import PostmortemReport
from motivation.goal_schema import Goal


def _json_default(obj: Any) -> Any:
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    if isinstance(obj, datetime):
        return obj.isoformat()
    if hasattr(obj, "value"):
        return obj.value
    if is_dataclass(obj):
        return asdict(obj)
    raise TypeError(f"Object of type {type(obj)!r} is not JSON serializable")


class GoalLedger:
    """Append-only goal/event ledger with snapshots for crash recovery."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.audit_path = self.root / "goal_events.jsonl"
        self.snapshot_dir = self.root / "snapshots"
        self.postmortem_dir = self.root / "postmortems"
        self.failure_dir = self.root / "failure_cases"
        self.regression_case_dir = self.root / "regression_cases"
        self.failure_pattern_dir = self.root / "failure_patterns"
        self.strategy_pattern_dir = self.root / "strategy_patterns"
        self.episode_dir = self.root / "episodes"
        self.semantic_dir = self.root / "semantic_promotions"
        self.skill_dir = self.root / "skill_arena"
        self.schedule_dir = self.root / "scheduled_goals"
        self.transfer_dir = self.root / "transfer_runs"
        self.capability_dir = self.root / "capability_portfolio"
        self.capability_graph_dir = self.root / "capability_graph"
        self.capability_growth_dir = self.root / "capability_growth_plans"
        self.capability_growth_assessment_dir = self.root / "capability_growth_assessments"
        self.self_mod_proposal_dir = self.root / "self_modification_proposals"
        self.self_mod_spec_dir = self.root / "self_modification_specs"
        self.self_mod_regression_dir = self.root / "self_modification_regressions"
        self.self_mod_canary_dir = self.root / "self_modification_canaries"
        self.model_registry_dir = self.root / "model_registry"
        self.automl_spec_dir = self.root / "automl_specs"
        self.automl_training_dir = self.root / "automl_training_reports"
        self.automl_regression_dir = self.root / "automl_regression_reports"
        self.automl_canary_dir = self.root / "automl_canary_reports"
        self.experiment_proposal_dir = self.root / "experiment_proposals"
        self.experiment_assessment_dir = self.root / "experiment_assessments"
        self.experiment_campaign_dir = self.root / "experiment_campaigns"
        self.experiment_cycle_budget_dir = self.root / "experiment_cycle_budgets"
        self.interface_call_dir = self.root / "interface_calls"
        self.interface_state_dir = self.root / "interface_states"
        self.interface_qualification_dir = self.root / "interface_qualifications"
        self.interface_rollout_dir = self.root / "interface_rollout_decisions"
        self.interface_provider_routing_dir = self.root / "interface_provider_routing"
        self.interface_canary_sample_dir = self.root / "interface_canary_samples"
        self.interface_requalification_dir = self.root / "interface_requalification_reports"
        self.interface_canary_decision_dir = self.root / "interface_canary_decisions"
        self.interface_shadow_dir = self.root / "interface_shadow_runs"
        self.interface_shadow_consensus_dir = self.root / "interface_shadow_consensus"
        self.interface_drift_dir = self.root / "interface_drift_reports"
        self.interface_demotion_dir = self.root / "interface_demotion_decisions"
        self.interface_rollout_protection_dir = self.root / "interface_rollout_protections"
        self.dataset_example_dir = self.root / "dataset_examples"
        self.dataset_snapshot_dir = self.root / "dataset_snapshots"
        for directory in (
            self.snapshot_dir,
            self.postmortem_dir,
            self.failure_dir,
            self.regression_case_dir,
            self.failure_pattern_dir,
            self.strategy_pattern_dir,
            self.episode_dir,
            self.semantic_dir,
            self.skill_dir,
            self.schedule_dir,
            self.transfer_dir,
            self.capability_dir,
            self.capability_graph_dir,
            self.capability_growth_dir,
            self.capability_growth_assessment_dir,
            self.self_mod_proposal_dir,
            self.self_mod_spec_dir,
            self.self_mod_regression_dir,
            self.self_mod_canary_dir,
            self.model_registry_dir,
            self.automl_spec_dir,
            self.automl_training_dir,
            self.automl_regression_dir,
            self.automl_canary_dir,
            self.experiment_proposal_dir,
            self.experiment_assessment_dir,
            self.experiment_campaign_dir,
            self.experiment_cycle_budget_dir,
            self.interface_call_dir,
            self.interface_state_dir,
            self.interface_qualification_dir,
            self.interface_rollout_dir,
            self.interface_provider_routing_dir,
            self.interface_canary_sample_dir,
            self.interface_requalification_dir,
            self.interface_canary_decision_dir,
            self.interface_shadow_dir,
            self.interface_shadow_consensus_dir,
            self.interface_drift_dir,
            self.interface_demotion_dir,
            self.interface_rollout_protection_dir,
            self.dataset_example_dir,
            self.dataset_snapshot_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    def append_event(self, event: dict[str, Any]) -> None:
        payload = dict(event)
        payload.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
        with self.audit_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False, default=_json_default) + "\n")

    def save_goal_snapshot(self, goal: Goal) -> None:
        snapshot_path = self.snapshot_dir / f"{goal.goal_id}.json"
        with snapshot_path.open("w", encoding="utf-8") as f:
            json.dump(goal.to_dict(), f, ensure_ascii=False, indent=2)

    def load_active_goals(self) -> list[Goal]:
        goals: list[Goal] = []
        for snapshot_path in sorted(self.snapshot_dir.glob("*.json")):
            data = json.loads(snapshot_path.read_text(encoding="utf-8"))
            goal = Goal.from_dict(data)
            if goal.status.value in {"pending", "active", "blocked", "deferred"}:
                goals.append(goal)
        return goals

    def load_history(self, limit: int = 1000) -> list[dict[str, Any]]:
        if not self.audit_path.exists():
            return []
        lines = self.audit_path.read_text(encoding="utf-8").splitlines()
        selected = lines[-limit:]
        return [json.loads(line) for line in selected if line.strip()]

    def save_postmortem(self, report: PostmortemReport) -> Path:
        safe_goal_id = report.goal_id.replace("/", "_")
        path = self.postmortem_dir / f"{safe_goal_id}_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)
        self.append_event(
            {
                "event_type": "postmortem_stored",
                "goal_id": report.goal_id,
                "path": str(path.name),
                "success": report.success,
            }
        )
        return path

    def load_postmortems(self) -> list[dict[str, Any]]:
        reports: list[dict[str, Any]] = []
        for path in sorted(self.postmortem_dir.glob("*.json")):
            reports.append(json.loads(path.read_text(encoding="utf-8")))
        return reports

    def save_failure_case(self, case: FailureCase) -> Path:
        path = self.failure_dir / f"{case.case_id}.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(case.to_dict(), f, ensure_ascii=False, indent=2)
        self.append_event(
            {
                "event_type": "canary_failure_mined",
                "goal_id": case.goal_id,
                "capability": case.capability,
                "signature": case.signature,
                "path": path.name,
            }
        )
        return path

    def load_failure_cases(self) -> list[FailureCase]:
        cases: list[FailureCase] = []
        for path in sorted(self.failure_dir.glob("*.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            cases.append(FailureCase(**data))
        return cases

    def save_regression_case(self, case: RegressionCase) -> Path:
        path = self.regression_case_dir / f"{case.case_id}.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(case.to_dict(), f, ensure_ascii=False, indent=2)
        self.append_event(
            {
                "event_type": "benchmark_expanded",
                "capability": case.capability,
                "case_id": case.case_id,
                "signature": case.source_signature,
                "path": path.name,
            }
        )
        return path

    def load_regression_cases(self) -> list[RegressionCase]:
        cases: list[RegressionCase] = []
        for path in sorted(self.regression_case_dir.glob("*.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            cases.append(RegressionCase.from_dict(data))
        return cases

    def save_failure_pattern(self, pattern: dict[str, Any]) -> Path:
        signature = str(pattern.get("signature", "unknown"))
        safe_signature = signature.replace("/", "_").replace(":", "_").replace("|", "__")
        path = self.failure_pattern_dir / f"{safe_signature}.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(pattern, f, ensure_ascii=False, indent=2)
        self.append_event(
            {
                "event_type": "failure_pattern_updated",
                "signature": signature,
                "status": pattern.get("status"),
                "path": path.name,
            }
        )
        return path

    def load_failure_patterns(self) -> list[dict[str, Any]]:
        patterns: list[dict[str, Any]] = []
        for path in sorted(self.failure_pattern_dir.glob("*.json")):
            patterns.append(json.loads(path.read_text(encoding="utf-8")))
        return patterns

    def save_strategy_pattern(self, pattern: dict[str, Any]) -> Path:
        strategy_id = str(pattern.get("strategy_id", "unknown"))
        path = self.strategy_pattern_dir / f"{strategy_id}.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(pattern, f, ensure_ascii=False, indent=2)
        self.append_event(
            {
                "event_type": "strategy_pattern_stored",
                "strategy_id": strategy_id,
                "signature": pattern.get("signature"),
                "capability": pattern.get("capability"),
                "path": path.name,
            }
        )
        return path

    def load_strategy_patterns(self) -> list[dict[str, Any]]:
        patterns: list[dict[str, Any]] = []
        for path in sorted(self.strategy_pattern_dir.glob("*.json")):
            patterns.append(json.loads(path.read_text(encoding="utf-8")))
        return patterns

    def save_episode(self, record: dict[str, Any]) -> Path:
        episode_id = str(record.get("episode_id", "unknown"))
        path = self.episode_dir / f"{episode_id}.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)
        self.append_event(
            {
                "event_type": "episode_stored",
                "episode_id": episode_id,
                "goal_id": record.get("goal_id"),
                "title": record.get("title"),
                "path": path.name,
            }
        )
        return path

    def load_episodes(self) -> list[dict[str, Any]]:
        payloads: list[dict[str, Any]] = []
        for path in sorted(self.episode_dir.glob("*.json")):
            payloads.append(json.loads(path.read_text(encoding="utf-8")))
        return payloads

    def save_semantic_promotion(self, promotion: dict[str, Any]) -> Path:
        promotion_id = str(promotion.get("promotion_id", "unknown"))
        path = self.semantic_dir / f"{promotion_id}.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(promotion, f, ensure_ascii=False, indent=2)
        self.append_event(
            {
                "event_type": "semantic_promoted",
                "promotion_id": promotion_id,
                "pattern_key": promotion.get("pattern_key"),
                "fact_key": promotion.get("fact_key"),
                "path": path.name,
            }
        )
        return path

    def load_semantic_promotions(self) -> list[dict[str, Any]]:
        payloads: list[dict[str, Any]] = []
        for path in sorted(self.semantic_dir.glob("*.json")):
            payloads.append(json.loads(path.read_text(encoding="utf-8")))
        return payloads

    def save_skill_run(self, run: dict[str, Any]) -> Path:
        run_id = str(run.get("run_id", "unknown"))
        path = self.skill_dir / f"{run_id}.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(run, f, ensure_ascii=False, indent=2)
        self.append_event(
            {
                "event_type": "skill_arena_recorded",
                "run_id": run_id,
                "capability": run.get("capability"),
                "mean_score": run.get("mean_score"),
                "path": path.name,
            }
        )
        return path

    def load_skill_runs(self) -> list[dict[str, Any]]:
        payloads: list[dict[str, Any]] = []
        for path in sorted(self.skill_dir.glob("*.json")):
            payloads.append(json.loads(path.read_text(encoding="utf-8")))
        return payloads

    def save_scheduled_goal(self, scheduled: dict[str, Any]) -> Path:
        schedule_id = str(scheduled.get("schedule_id", "unknown"))
        path = self.schedule_dir / f"{schedule_id}.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(scheduled, f, ensure_ascii=False, indent=2)
        self.append_event(
            {
                "event_type": "goal_scheduled",
                "schedule_id": schedule_id,
                "goal_id": scheduled.get("goal", {}).get("goal_id"),
                "title": scheduled.get("goal", {}).get("title"),
                "due_cycle": scheduled.get("due_cycle"),
                "status": scheduled.get("status"),
                "path": path.name,
            }
        )
        return path

    def load_scheduled_goals(self) -> list[dict[str, Any]]:
        payloads: list[dict[str, Any]] = []
        for path in sorted(self.schedule_dir.glob("*.json")):
            payloads.append(json.loads(path.read_text(encoding="utf-8")))
        return payloads

    def save_experiment_proposal(self, proposal: dict[str, Any]) -> Path:
        proposal_id = str(proposal.get("proposal_id", "unknown"))
        path = self.experiment_proposal_dir / f"{proposal_id}.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(proposal, f, ensure_ascii=False, indent=2)
        self.append_event(
            {
                "event_type": "experiment_proposed",
                "proposal_id": proposal_id,
                "experiment_kind": proposal.get("experiment_kind"),
                "source_signature": proposal.get("source_signature"),
                "status": proposal.get("status"),
                "path": path.name,
            }
        )
        return path

    def load_experiment_proposals(self) -> list[dict[str, Any]]:
        payloads: list[dict[str, Any]] = []
        for path in sorted(self.experiment_proposal_dir.glob("*.json")):
            payloads.append(json.loads(path.read_text(encoding="utf-8")))
        return payloads

    def save_experiment_assessment(self, assessment: dict[str, Any]) -> Path:
        assessment_id = str(assessment.get("assessment_id", "unknown"))
        path = self.experiment_assessment_dir / f"{assessment_id}.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(assessment, f, ensure_ascii=False, indent=2)
        self.append_event(
            {
                "event_type": "experiment_assessed",
                "assessment_id": assessment_id,
                "proposal_id": assessment.get("proposal_id"),
                "experiment_kind": assessment.get("experiment_kind"),
                "admissible": assessment.get("admissible"),
                "composite_score": assessment.get("composite_score"),
                "path": path.name,
            }
        )
        return path

    def load_experiment_assessments(self) -> list[dict[str, Any]]:
        payloads: list[dict[str, Any]] = []
        for path in sorted(self.experiment_assessment_dir.glob("*.json")):
            payloads.append(json.loads(path.read_text(encoding="utf-8")))
        return payloads


    def save_experiment_campaign(self, campaign: dict[str, Any]) -> Path:
        campaign_id = str(campaign.get("campaign_id", "unknown"))
        path = self.experiment_campaign_dir / f"{campaign_id}.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(campaign, f, ensure_ascii=False, indent=2)
        self.append_event(
            {
                "event_type": "experiment_campaign_updated",
                "campaign_id": campaign_id,
                "experiment_kind": campaign.get("experiment_kind"),
                "source_signature": campaign.get("source_signature"),
                "status": campaign.get("status"),
                "path": path.name,
            }
        )
        return path

    def load_experiment_campaigns(self) -> list[dict[str, Any]]:
        payloads: list[dict[str, Any]] = []
        for path in sorted(self.experiment_campaign_dir.glob("*.json")):
            payloads.append(json.loads(path.read_text(encoding="utf-8")))
        return payloads

    def save_experiment_cycle_budget(self, budget: dict[str, Any]) -> Path:
        cycle = int(budget.get("cycle", 0))
        path = self.experiment_cycle_budget_dir / f"cycle_{cycle:04d}.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(budget, f, ensure_ascii=False, indent=2)
        self.append_event(
            {
                "event_type": "experiment_cycle_budget_updated",
                "cycle": cycle,
                "reserved_cost": budget.get("reserved_cost"),
                "reserved_risk": budget.get("reserved_risk"),
                "path": path.name,
            }
        )
        return path

    def load_experiment_cycle_budgets(self) -> list[dict[str, Any]]:
        payloads: list[dict[str, Any]] = []
        for path in sorted(self.experiment_cycle_budget_dir.glob("*.json")):
            payloads.append(json.loads(path.read_text(encoding="utf-8")))
        return payloads



    def save_interface_call(self, record: dict[str, Any]) -> Path:
        call_id = str(record.get("call_id", "unknown"))
        path = self.interface_call_dir / f"{call_id}.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)
        self.append_event(
            {
                "event_type": "external_interface_call_recorded",
                "call_id": call_id,
                "adapter_name": record.get("adapter_name"),
                "operation": record.get("operation"),
                "status": record.get("status"),
                "path": path.name,
            }
        )
        return path

    def load_interface_calls(self) -> list[dict[str, Any]]:
        payloads: list[dict[str, Any]] = []
        for path in sorted(self.interface_call_dir.glob("*.json")):
            payloads.append(json.loads(path.read_text(encoding="utf-8")))
        return payloads

    def save_interface_state(self, state: dict[str, Any]) -> Path:
        adapter_name = str(state.get("adapter_name", "unknown"))
        path = self.interface_state_dir / f"{adapter_name}.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
        self.append_event(
            {
                "event_type": "external_interface_state_updated",
                "adapter_name": adapter_name,
                "mode": state.get("mode"),
                "live_ready": state.get("live_ready"),
                "path": path.name,
            }
        )
        return path

    def load_interface_states(self) -> list[dict[str, Any]]:
        payloads: list[dict[str, Any]] = []
        for path in sorted(self.interface_state_dir.glob("*.json")):
            payloads.append(json.loads(path.read_text(encoding="utf-8")))
        return payloads

    def save_interface_qualification(self, report: dict[str, Any]) -> Path:
        qualification_id = str(report.get("qualification_id", "unknown"))
        path = self.interface_qualification_dir / f"{qualification_id}.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        self.append_event(
            {
                "event_type": "external_interface_provider_qualified",
                "qualification_id": qualification_id,
                "adapter_name": report.get("adapter_name"),
                "provider": report.get("provider"),
                "eligible": report.get("eligible"),
                "score": report.get("score"),
                "path": path.name,
            }
        )
        return path

    def load_interface_qualifications(self) -> list[dict[str, Any]]:
        payloads: list[dict[str, Any]] = []
        for path in sorted(self.interface_qualification_dir.glob("*.json")):
            payloads.append(json.loads(path.read_text(encoding="utf-8")))
        return payloads

    def save_interface_rollout_decision(self, decision: dict[str, Any]) -> Path:
        decision_id = str(decision.get("decision_id", "unknown"))
        path = self.interface_rollout_dir / f"{decision_id}.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(decision, f, ensure_ascii=False, indent=2)
        self.append_event(
            {
                "event_type": "external_interface_rollout_decided",
                "decision_id": decision_id,
                "adapter_name": decision.get("adapter_name"),
                "provider": decision.get("provider"),
                "decision": decision.get("decision"),
                "mode": decision.get("mode"),
                "path": path.name,
            }
        )
        return path

    def load_interface_rollout_decisions(self) -> list[dict[str, Any]]:
        payloads: list[dict[str, Any]] = []
        for path in sorted(self.interface_rollout_dir.glob("*.json")):
            payloads.append(json.loads(path.read_text(encoding="utf-8")))
        return payloads

    def save_interface_provider_routing(self, decision: dict[str, Any]) -> Path:
        routing_id = str(decision.get("routing_id", "unknown"))
        path = self.interface_provider_routing_dir / f"{routing_id}.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(decision, f, ensure_ascii=False, indent=2)
        self.append_event(
            {
                "event_type": "external_interface_provider_routed",
                "routing_id": routing_id,
                "adapter_name": decision.get("adapter_name"),
                "role": decision.get("role"),
                "selected_provider": decision.get("selected_provider"),
                "strategy": decision.get("strategy"),
                "path": path.name,
            }
        )
        return path

    def load_interface_provider_routing(self) -> list[dict[str, Any]]:
        payloads: list[dict[str, Any]] = []
        for path in sorted(self.interface_provider_routing_dir.glob("*.json")):
            payloads.append(json.loads(path.read_text(encoding="utf-8")))
        return payloads


    def save_interface_canary_sample(self, sample: dict[str, Any]) -> Path:
        sample_id = str(sample.get("sample_id", "unknown"))
        path = self.interface_canary_sample_dir / f"{sample_id}.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(sample, f, ensure_ascii=False, indent=2)
        self.append_event(
            {
                "event_type": "external_interface_canary_sampled",
                "sample_id": sample_id,
                "adapter_name": sample.get("adapter_name"),
                "candidate_provider": sample.get("candidate_provider"),
                "routed_provider": sample.get("routed_provider"),
                "rollout_stage": sample.get("rollout_stage"),
                "path": path.name,
            }
        )
        return path

    def load_interface_canary_samples(self) -> list[dict[str, Any]]:
        payloads: list[dict[str, Any]] = []
        for path in sorted(self.interface_canary_sample_dir.glob("*.json")):
            payloads.append(json.loads(path.read_text(encoding="utf-8")))
        return payloads

    def save_interface_requalification_report(self, report: dict[str, Any]) -> Path:
        report_id = str(report.get("report_id", "unknown"))
        path = self.interface_requalification_dir / f"{report_id}.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        self.append_event(
            {
                "event_type": "external_interface_requalified",
                "report_id": report_id,
                "adapter_name": report.get("adapter_name"),
                "candidate_provider": report.get("candidate_provider"),
                "eligible": report.get("eligible"),
                "path": path.name,
            }
        )
        return path

    def load_interface_requalification_reports(self) -> list[dict[str, Any]]:
        payloads: list[dict[str, Any]] = []
        for path in sorted(self.interface_requalification_dir.glob("*.json")):
            payloads.append(json.loads(path.read_text(encoding="utf-8")))
        return payloads

    def save_interface_canary_decision(self, decision: dict[str, Any]) -> Path:
        decision_id = str(decision.get("decision_id", "unknown"))
        path = self.interface_canary_decision_dir / f"{decision_id}.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(decision, f, ensure_ascii=False, indent=2)
        self.append_event(
            {
                "event_type": "external_interface_canary_decided",
                "decision_id": decision_id,
                "adapter_name": decision.get("adapter_name"),
                "candidate_provider": decision.get("candidate_provider"),
                "fallback_provider": decision.get("fallback_provider"),
                "action": decision.get("action"),
                "path": path.name,
            }
        )
        return path

    def load_interface_canary_decisions(self) -> list[dict[str, Any]]:
        payloads: list[dict[str, Any]] = []
        for path in sorted(self.interface_canary_decision_dir.glob("*.json")):
            payloads.append(json.loads(path.read_text(encoding="utf-8")))
        return payloads


    def save_interface_shadow_run(self, record: dict[str, Any]) -> Path:
        shadow_run_id = str(record.get("shadow_run_id", "unknown"))
        path = self.interface_shadow_dir / f"{shadow_run_id}.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)
        self.append_event(
            {
                "event_type": "external_interface_shadow_recorded",
                "shadow_run_id": shadow_run_id,
                "adapter_name": record.get("adapter_name"),
                "primary_provider": record.get("primary_provider"),
                "shadow_provider": record.get("shadow_provider"),
                "correctness_pass": record.get("correctness_pass"),
                "safety_pass": record.get("safety_pass"),
                "path": path.name,
            }
        )
        return path

    def load_interface_shadow_runs(self) -> list[dict[str, Any]]:
        payloads: list[dict[str, Any]] = []
        for path in sorted(self.interface_shadow_dir.glob("*.json")):
            payloads.append(json.loads(path.read_text(encoding="utf-8")))
        return payloads

    def save_interface_shadow_consensus(self, record: dict[str, Any]) -> Path:
        consensus_id = str(record.get("consensus_id", "unknown"))
        path = self.interface_shadow_consensus_dir / f"{consensus_id}.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)
        self.append_event(
            {
                "event_type": "external_interface_shadow_consensus_recorded",
                "consensus_id": consensus_id,
                "adapter_name": record.get("adapter_name"),
                "comparison_profile": record.get("comparison_profile"),
                "consensus_provider": record.get("consensus_provider"),
                "correctness_pass": record.get("correctness_pass"),
                "path": path.name,
            }
        )
        return path

    def load_interface_shadow_consensus(self) -> list[dict[str, Any]]:
        payloads: list[dict[str, Any]] = []
        for path in sorted(self.interface_shadow_consensus_dir.glob("*.json")):
            payloads.append(json.loads(path.read_text(encoding="utf-8")))
        return payloads

    def save_interface_drift_report(self, report: dict[str, Any]) -> Path:
        drift_report_id = str(report.get("drift_report_id", "unknown"))
        path = self.interface_drift_dir / f"{drift_report_id}.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        self.append_event(
            {
                "event_type": "external_interface_drift_reported",
                "drift_report_id": drift_report_id,
                "adapter_name": report.get("adapter_name"),
                "provider": report.get("provider"),
                "demotion_triggered": report.get("demotion_triggered"),
                "path": path.name,
            }
        )
        return path

    def load_interface_drift_reports(self) -> list[dict[str, Any]]:
        payloads: list[dict[str, Any]] = []
        for path in sorted(self.interface_drift_dir.glob("*.json")):
            payloads.append(json.loads(path.read_text(encoding="utf-8")))
        return payloads

    def save_interface_demotion_decision(self, decision: dict[str, Any]) -> Path:
        demotion_id = str(decision.get("demotion_id", "unknown"))
        path = self.interface_demotion_dir / f"{demotion_id}.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(decision, f, ensure_ascii=False, indent=2)
        self.append_event(
            {
                "event_type": "external_interface_demoted",
                "demotion_id": demotion_id,
                "adapter_name": decision.get("adapter_name"),
                "previous_provider": decision.get("previous_provider"),
                "fallback_provider": decision.get("fallback_provider"),
                "action": decision.get("action"),
                "path": path.name,
            }
        )
        return path

    def load_interface_demotion_decisions(self) -> list[dict[str, Any]]:
        payloads: list[dict[str, Any]] = []
        for path in sorted(self.interface_demotion_dir.glob("*.json")):
            payloads.append(json.loads(path.read_text(encoding="utf-8")))
        return payloads

    def save_interface_rollout_protection(self, record: dict[str, Any]) -> Path:
        protection_id = str(record.get("protection_id", "unknown"))
        path = self.interface_rollout_protection_dir / f"{protection_id}.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)
        self.append_event(
            {
                "event_type": "external_interface_rollout_protected",
                "protection_id": protection_id,
                "adapter_name": record.get("adapter_name"),
                "affected_provider": record.get("affected_provider"),
                "trigger_type": record.get("trigger_type"),
                "cooldown_until_rollout_index": record.get("cooldown_until_rollout_index"),
                "anti_flap_until_rollout_index": record.get("anti_flap_until_rollout_index"),
                "path": path.name,
            }
        )
        return path

    def load_interface_rollout_protections(self) -> list[dict[str, Any]]:
        payloads: list[dict[str, Any]] = []
        for path in self.interface_rollout_protection_dir.glob("*.json"):
            payloads.append(json.loads(path.read_text(encoding="utf-8")))
        return sorted(payloads, key=lambda item: str(item.get("timestamp", "")))

    def save_dataset_example(self, example: dict[str, Any]) -> Path:
        example_id = str(example.get("example_id", "unknown"))
        path = self.dataset_example_dir / f"{example_id}.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(example, f, ensure_ascii=False, indent=2)
        self.append_event(
            {
                "event_type": "dataset_example_stored",
                "example_id": example_id,
                "model_family": example.get("model_family"),
                "split": example.get("split"),
                "source_type": example.get("source_type"),
                "path": path.name,
            }
        )
        return path

    def load_dataset_examples(self) -> list[dict[str, Any]]:
        payloads: list[dict[str, Any]] = []
        for path in sorted(self.dataset_example_dir.glob("*.json")):
            payloads.append(json.loads(path.read_text(encoding="utf-8")))
        return payloads

    def save_dataset_snapshot(self, snapshot: dict[str, Any]) -> Path:
        snapshot_id = str(snapshot.get("snapshot_id", "unknown"))
        path = self.dataset_snapshot_dir / f"{snapshot_id}.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(snapshot, f, ensure_ascii=False, indent=2)
        self.append_event(
            {
                "event_type": "dataset_snapshot_stored",
                "snapshot_id": snapshot_id,
                "model_family": snapshot.get("model_family"),
                "status": snapshot.get("status"),
                "example_count": snapshot.get("stats", {}).get("example_count"),
                "path": path.name,
            }
        )
        return path

    def load_dataset_snapshots(self) -> list[dict[str, Any]]:
        payloads: list[dict[str, Any]] = []
        for path in sorted(self.dataset_snapshot_dir.glob("*.json")):
            payloads.append(json.loads(path.read_text(encoding="utf-8")))
        return payloads


    def save_transfer_run(self, run: dict[str, Any]) -> Path:
        run_id = str(run.get("run_id", "unknown"))
        path = self.transfer_dir / f"{run_id}.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(run, f, ensure_ascii=False, indent=2)
        self.append_event(
            {
                "event_type": "transfer_run_recorded",
                "run_id": run_id,
                "capability": run.get("capability"),
                "mean_score": run.get("mean_score"),
                "path": path.name,
            }
        )
        return path

    def load_transfer_runs(self) -> list[dict[str, Any]]:
        payloads: list[dict[str, Any]] = []
        for path in sorted(self.transfer_dir.glob("*.json")):
            payloads.append(json.loads(path.read_text(encoding="utf-8")))
        return payloads

    def save_capability_state(self, state: dict[str, Any]) -> Path:
        capability = str(state.get("capability", "unknown"))
        safe_capability = capability.replace("/", "_").replace(":", "_")
        path = self.capability_dir / f"{safe_capability}.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
        self.append_event(
            {
                "event_type": "capability_state_updated",
                "capability": capability,
                "maturity_stage": state.get("maturity_stage"),
                "path": path.name,
            }
        )
        return path

    def load_capability_states(self) -> list[dict[str, Any]]:
        payloads: list[dict[str, Any]] = []
        for path in sorted(self.capability_dir.glob("*.json")):
            payloads.append(json.loads(path.read_text(encoding="utf-8")))
        return payloads

    def save_capability_graph_node(self, node: dict[str, Any]) -> Path:
        capability = str(node.get("capability", "unknown"))
        safe_capability = capability.replace("/", "_").replace(":", "_")
        path = self.capability_graph_dir / f"{safe_capability}.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(node, f, ensure_ascii=False, indent=2)
        self.append_event(
            {
                "event_type": "capability_graph_updated",
                "capability": capability,
                "stage": node.get("stage"),
                "path": path.name,
            }
        )
        return path

    def load_capability_graph_nodes(self) -> list[dict[str, Any]]:
        payloads: list[dict[str, Any]] = []
        for path in sorted(self.capability_graph_dir.glob("*.json")):
            payloads.append(json.loads(path.read_text(encoding="utf-8")))
        return payloads


    def save_capability_growth_plan(self, plan: dict[str, Any]) -> Path:
        plan_id = str(plan.get("plan_id", "unknown"))
        path = self.capability_growth_dir / f"{plan_id}.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(plan, f, ensure_ascii=False, indent=2)
        self.append_event(
            {
                "event_type": "capability_growth_plan_updated",
                "plan_id": plan_id,
                "root_capability": plan.get("root_capability"),
                "path_targets": plan.get("path_targets"),
                "status": plan.get("status"),
                "path": path.name,
            }
        )
        return path

    def load_capability_growth_plans(self) -> list[dict[str, Any]]:
        payloads: list[dict[str, Any]] = []
        for path in sorted(self.capability_growth_dir.glob("*.json")):
            payloads.append(json.loads(path.read_text(encoding="utf-8")))
        return payloads

    def save_capability_growth_assessment(self, assessment: dict[str, Any]) -> Path:
        assessment_id = str(assessment.get("assessment_id", "unknown"))
        path = self.capability_growth_assessment_dir / f"{assessment_id}.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(assessment, f, ensure_ascii=False, indent=2)
        self.append_event(
            {
                "event_type": "capability_growth_assessed",
                "assessment_id": assessment_id,
                "plan_id": assessment.get("plan_id"),
                "path_targets": assessment.get("path_targets"),
                "admissible": assessment.get("admissible"),
                "composite_score": assessment.get("composite_score"),
                "path": path.name,
            }
        )
        return path

    def load_capability_growth_assessments(self) -> list[dict[str, Any]]:
        payloads: list[dict[str, Any]] = []
        for path in sorted(self.capability_growth_assessment_dir.glob("*.json")):
            payloads.append(json.loads(path.read_text(encoding="utf-8")))
        return payloads

    def save_self_modification_proposal(self, proposal: dict[str, Any]) -> Path:
        proposal_id = str(proposal.get("proposal_id", "unknown"))
        path = self.self_mod_proposal_dir / f"{proposal_id}.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(proposal, f, ensure_ascii=False, indent=2)
        self.append_event(
            {
                "event_type": "self_mod_proposal_stored",
                "proposal_id": proposal_id,
                "signature": proposal.get("signature"),
                "capability": proposal.get("capability"),
                "status": proposal.get("status"),
                "path": path.name,
            }
        )
        return path

    def load_self_modification_proposals(self) -> list[dict[str, Any]]:
        payloads: list[dict[str, Any]] = []
        for path in sorted(self.self_mod_proposal_dir.glob("*.json")):
            payloads.append(json.loads(path.read_text(encoding="utf-8")))
        return payloads

    def save_self_modification_spec(self, spec: dict[str, Any]) -> Path:
        change_id = str(spec.get("change_id", "unknown"))
        path = self.self_mod_spec_dir / f"{change_id}.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(spec, f, ensure_ascii=False, indent=2)
        self.append_event(
            {
                "event_type": "self_mod_spec_stored",
                "change_id": change_id,
                "goal_id": spec.get("goal_id"),
                "capability": spec.get("capability"),
                "status": spec.get("status"),
                "path": path.name,
            }
        )
        return path

    def load_self_modification_specs(self) -> list[dict[str, Any]]:
        payloads: list[dict[str, Any]] = []
        for path in sorted(self.self_mod_spec_dir.glob("*.json")):
            payloads.append(json.loads(path.read_text(encoding="utf-8")))
        return payloads

    def save_self_modification_regression(self, report: dict[str, Any]) -> Path:
        change_id = str(report.get("change_id", "unknown"))
        path = self.self_mod_regression_dir / f"{change_id}.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        self.append_event(
            {
                "event_type": "self_mod_regression_stored",
                "change_id": change_id,
                "capability": report.get("capability"),
                "passed": report.get("passed"),
                "path": path.name,
            }
        )
        return path

    def load_self_modification_regressions(self) -> list[dict[str, Any]]:
        payloads: list[dict[str, Any]] = []
        for path in sorted(self.self_mod_regression_dir.glob("*.json")):
            payloads.append(json.loads(path.read_text(encoding="utf-8")))
        return payloads

    def save_self_modification_canary(self, report: dict[str, Any]) -> Path:
        change_id = str(report.get("change_id", "unknown"))
        path = self.self_mod_canary_dir / f"{change_id}.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        self.append_event(
            {
                "event_type": "self_mod_canary_stored",
                "change_id": change_id,
                "capability": report.get("capability"),
                "passed": report.get("passed"),
                "rolled_back": report.get("rolled_back"),
                "path": path.name,
            }
        )
        return path

    def load_self_modification_canaries(self) -> list[dict[str, Any]]:
        payloads: list[dict[str, Any]] = []
        for path in sorted(self.self_mod_canary_dir.glob("*.json")):
            payloads.append(json.loads(path.read_text(encoding="utf-8")))
        return payloads

    def save_model_record(self, model: dict[str, Any]) -> Path:
        model_id = str(model.get("model_id", "unknown"))
        path = self.model_registry_dir / f"{model_id}.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(model, f, ensure_ascii=False, indent=2)
        self.append_event({
            "event_type": "model_record_stored",
            "model_id": model_id,
            "family": model.get("family"),
            "status": model.get("status"),
            "path": path.name,
        })
        return path

    def load_model_records(self) -> list[dict[str, Any]]:
        payloads: list[dict[str, Any]] = []
        for path in sorted(self.model_registry_dir.glob("*.json")):
            payloads.append(json.loads(path.read_text(encoding="utf-8")))
        return payloads

    def save_automl_spec(self, spec: dict[str, Any]) -> Path:
        change_id = str(spec.get("change_id", "unknown"))
        path = self.automl_spec_dir / f"{change_id}.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(spec, f, ensure_ascii=False, indent=2)
        self.append_event({
            "event_type": "automl_spec_stored",
            "change_id": change_id,
            "model_family": spec.get("model_family"),
            "status": spec.get("status"),
            "path": path.name,
        })
        return path

    def load_automl_specs(self) -> list[dict[str, Any]]:
        payloads: list[dict[str, Any]] = []
        for path in sorted(self.automl_spec_dir.glob("*.json")):
            payloads.append(json.loads(path.read_text(encoding="utf-8")))
        return payloads

    def save_automl_training_report(self, report: dict[str, Any]) -> Path:
        change_id = str(report.get("change_id", "unknown"))
        path = self.automl_training_dir / f"{change_id}.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        self.append_event({
            "event_type": "automl_training_stored",
            "change_id": change_id,
            "model_family": report.get("model_family"),
            "path": path.name,
        })
        return path

    def load_automl_training_reports(self) -> list[dict[str, Any]]:
        payloads: list[dict[str, Any]] = []
        for path in sorted(self.automl_training_dir.glob("*.json")):
            payloads.append(json.loads(path.read_text(encoding="utf-8")))
        return payloads

    def save_automl_regression_report(self, report: dict[str, Any]) -> Path:
        change_id = str(report.get("change_id", "unknown"))
        path = self.automl_regression_dir / f"{change_id}.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        self.append_event({
            "event_type": "automl_regression_stored",
            "change_id": change_id,
            "model_family": report.get("model_family"),
            "passed": report.get("passed"),
            "path": path.name,
        })
        return path

    def load_automl_regression_reports(self) -> list[dict[str, Any]]:
        payloads: list[dict[str, Any]] = []
        for path in sorted(self.automl_regression_dir.glob("*.json")):
            payloads.append(json.loads(path.read_text(encoding="utf-8")))
        return payloads

    def save_automl_canary_report(self, report: dict[str, Any]) -> Path:
        change_id = str(report.get("change_id", "unknown"))
        path = self.automl_canary_dir / f"{change_id}.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        self.append_event({
            "event_type": "automl_canary_stored",
            "change_id": change_id,
            "model_family": report.get("model_family"),
            "passed": report.get("passed"),
            "rolled_back": report.get("rolled_back"),
            "path": path.name,
        })
        return path

    def load_automl_canary_reports(self) -> list[dict[str, Any]]:
        payloads: list[dict[str, Any]] = []
        for path in sorted(self.automl_canary_dir.glob("*.json")):
            payloads.append(json.loads(path.read_text(encoding="utf-8")))
        return payloads

    def query_goal_history(self, goal_id: str) -> list[dict[str, Any]]:
        return [event for event in self.load_history(limit=10000) if event.get("goal_id") == goal_id]

    def retrieve_similar_failures(self, tags: list[str]) -> list[dict[str, Any]]:
        tags_set = set(tags)
        results: list[dict[str, Any]] = []
        for path in sorted(self.postmortem_dir.glob("*.json")):
            report = json.loads(path.read_text(encoding="utf-8"))
            report_tags = set(report.get("tags", []))
            if report.get("success") is False and (not tags_set or tags_set & report_tags):
                results.append(report)
        return results
