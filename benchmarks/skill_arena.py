from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable
from uuid import uuid4

from memory.goal_ledger import GoalLedger


def new_skill_run_id(prefix: str = "skillrun") -> str:
    return f"{prefix}_{uuid4().hex[:12]}"


@dataclass(slots=True)
class SkillArenaCase:
    case_id: str
    payload: dict[str, Any]
    expected: dict[str, Any]
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SkillArenaCase":
        return cls(**dict(data))


@dataclass(slots=True)
class SkillArenaCaseResult:
    case_id: str
    score: float
    passed: bool
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SkillArenaCaseResult":
        return cls(**dict(data))


@dataclass(slots=True)
class SkillArenaRun:
    run_id: str
    capability: str
    label: str
    threshold: float
    cases: list[SkillArenaCaseResult]
    mean_score: float
    passed: bool
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["created_at"] = self.created_at.isoformat()
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SkillArenaRun":
        raw = dict(data)
        raw["created_at"] = datetime.fromisoformat(raw["created_at"])
        raw["cases"] = [SkillArenaCaseResult.from_dict(item) for item in raw["cases"]]
        return cls(**raw)


class SkillArena:
    """Deterministic post-rollout evaluation harness for stable skills."""

    def __init__(self, ledger: GoalLedger | None = None) -> None:
        self.ledger = ledger
        self._runs: list[SkillArenaRun] = []
        if self.ledger is not None:
            self._rehydrate()

    def _rehydrate(self) -> None:
        for payload in self.ledger.load_skill_runs():
            self._runs.append(SkillArenaRun.from_dict(payload))

    def list_runs(self, capability: str | None = None) -> list[SkillArenaRun]:
        runs = list(self._runs)
        if capability is not None:
            runs = [run for run in runs if run.capability == capability]
        return sorted(runs, key=lambda item: item.created_at)

    def latest_run(self, capability: str) -> SkillArenaRun | None:
        runs = self.list_runs(capability)
        return runs[-1] if runs else None

    def run(
        self,
        *,
        capability: str,
        label: str,
        cases: list[SkillArenaCase],
        execute: Callable[[dict[str, Any]], Any],
        threshold: float = 0.80,
    ) -> SkillArenaRun:
        results: list[SkillArenaCaseResult] = []
        for case in cases:
            raw_output = execute(case.payload)
            output = raw_output.get("output", raw_output) if isinstance(raw_output, dict) else raw_output
            result = self._evaluate_case(capability, case, output)
            results.append(result)
        mean_score = round(sum(item.score for item in results) / len(results), 3) if results else 0.0
        run = SkillArenaRun(
            run_id=new_skill_run_id(),
            capability=capability,
            label=label,
            threshold=threshold,
            cases=results,
            mean_score=mean_score,
            passed=mean_score >= threshold,
        )
        self._runs.append(run)
        if self.ledger is not None:
            self.ledger.save_skill_run(run.to_dict())
        return run

    def _evaluate_case(self, capability: str, case: SkillArenaCase, output: Any) -> SkillArenaCaseResult:
        if capability == "text_summarizer":
            return self._evaluate_text_summarizer(case, output)
        if capability in {"grounded_navigation", "spatial_route_composition"}:
            return self._evaluate_grounded_navigation(case, output)
        if capability == "navigation_route_explanation":
            return self._evaluate_navigation_route_explanation(case, output)
        if capability == "route_mission_briefing":
            return self._evaluate_structured_report(case, output)
        if capability == "response_planning":
            return self._evaluate_response_planning(case, output)
        if capability == "response_risk_model":
            return self._evaluate_response_risk_model(case, output)
        if capability == "memory_retrieval":
            return self._evaluate_memory_retrieval(case, output)
        return self._evaluate_generic(case, output)

    def _evaluate_text_summarizer(self, case: SkillArenaCase, output: Any) -> SkillArenaCaseResult:
        details: dict[str, Any] = {}
        if not isinstance(output, dict):
            return SkillArenaCaseResult(case_id=case.case_id, score=0.0, passed=False, details={"error": "non_mapping_output"})
        summary = str(output.get("summary", "")).strip()
        source_count = output.get("source_count")
        sentences_used = output.get("sentences_used")

        score = 0.0
        weight = 1.0 / 3.0
        required_nonempty = bool(case.expected.get("required_summary_nonempty", True))
        if (not required_nonempty) or summary:
            score += weight
            details["summary_nonempty"] = True
        else:
            details["summary_nonempty"] = False

        expected_sources = case.expected.get("source_count")
        if expected_sources is None or source_count == expected_sources:
            score += weight
            details["source_count_match"] = True
        else:
            details["source_count_match"] = False
            details["expected_source_count"] = expected_sources
            details["actual_source_count"] = source_count

        max_sentences = case.expected.get("max_sentences")
        if max_sentences is None or (isinstance(sentences_used, int) and sentences_used <= int(max_sentences)):
            score += weight
            details["sentence_limit_respected"] = True
        else:
            details["sentence_limit_respected"] = False
            details["expected_max_sentences"] = max_sentences
            details["actual_sentences_used"] = sentences_used

        final_score = round(score, 3)
        return SkillArenaCaseResult(case_id=case.case_id, score=final_score, passed=final_score >= 0.99, details=details)


    def _evaluate_grounded_navigation(self, case: SkillArenaCase, output: Any) -> SkillArenaCaseResult:
        details: dict[str, Any] = {}
        if not isinstance(output, dict):
            return SkillArenaCaseResult(case_id=case.case_id, score=0.0, passed=False, details={"error": "non_mapping_output"})

        success_rate = output.get("success_rate")
        mean_path_ratio = output.get("mean_path_ratio")
        strategy = output.get("strategy")

        score = 0.0
        weight = 1.0 / 3.0
        min_success_rate = float(case.expected.get("min_success_rate", 1.0))
        max_ratio = float(case.expected.get("max_mean_path_ratio", 1.0))
        expected_strategy = case.expected.get("strategy")

        if isinstance(success_rate, (float, int)) and float(success_rate) >= min_success_rate:
            score += weight
            details["success_rate_ok"] = True
        else:
            details["success_rate_ok"] = False
            details["expected_min_success_rate"] = min_success_rate
            details["actual_success_rate"] = success_rate

        if mean_path_ratio is None:
            ratio_ok = False
        else:
            ratio_ok = float(mean_path_ratio) <= max_ratio
        if ratio_ok:
            score += weight
            details["path_efficiency_ok"] = True
        else:
            details["path_efficiency_ok"] = False
            details["expected_max_mean_path_ratio"] = max_ratio
            details["actual_mean_path_ratio"] = mean_path_ratio

        if expected_strategy is None or strategy == expected_strategy:
            score += weight
            details["strategy_match"] = True
        else:
            details["strategy_match"] = False
            details["expected_strategy"] = expected_strategy
            details["actual_strategy"] = strategy

        final_score = round(score, 3)
        return SkillArenaCaseResult(case_id=case.case_id, score=final_score, passed=final_score >= 0.99, details=details)

    def _evaluate_navigation_route_explanation(self, case: SkillArenaCase, output: Any) -> SkillArenaCaseResult:
        details: dict[str, Any] = {}
        if not isinstance(output, dict):
            return SkillArenaCaseResult(case_id=case.case_id, score=0.0, passed=False, details={"error": "non_mapping_output"})

        success_rate = output.get("success_rate")
        mean_path_ratio = output.get("mean_path_ratio")
        detour_explanation_rate = output.get("detour_explanation_rate")
        strategy = output.get("strategy")

        score = 0.0
        weight = 1.0 / 4.0
        min_success_rate = float(case.expected.get("min_success_rate", 1.0))
        max_ratio = float(case.expected.get("max_mean_path_ratio", 1.0))
        min_detour_rate = float(case.expected.get("min_detour_explanation_rate", 1.0))
        expected_strategy = case.expected.get("strategy")

        if isinstance(success_rate, (float, int)) and float(success_rate) >= min_success_rate:
            score += weight
            details["success_rate_ok"] = True
        else:
            details["success_rate_ok"] = False
            details["expected_min_success_rate"] = min_success_rate
            details["actual_success_rate"] = success_rate

        if mean_path_ratio is not None and float(mean_path_ratio) <= max_ratio:
            score += weight
            details["path_efficiency_ok"] = True
        else:
            details["path_efficiency_ok"] = False
            details["expected_max_mean_path_ratio"] = max_ratio
            details["actual_mean_path_ratio"] = mean_path_ratio

        if isinstance(detour_explanation_rate, (float, int)) and float(detour_explanation_rate) >= min_detour_rate:
            score += weight
            details["detour_explanation_ok"] = True
        else:
            details["detour_explanation_ok"] = False
            details["expected_min_detour_explanation_rate"] = min_detour_rate
            details["actual_detour_explanation_rate"] = detour_explanation_rate

        if expected_strategy is None or strategy == expected_strategy:
            score += weight
            details["strategy_match"] = True
        else:
            details["strategy_match"] = False
            details["expected_strategy"] = expected_strategy
            details["actual_strategy"] = strategy

        final_score = round(score, 3)
        return SkillArenaCaseResult(case_id=case.case_id, score=final_score, passed=final_score >= 0.99, details=details)

    def _evaluate_structured_report(self, case: SkillArenaCase, output: Any) -> SkillArenaCaseResult:
        details: dict[str, Any] = {}
        if not isinstance(output, dict):
            return SkillArenaCaseResult(case_id=case.case_id, score=0.0, passed=False, details={"error": "non_mapping_output"})

        reports = output.get("explanations") or output.get("briefings") or []
        task_count = output.get("task_count")
        strategy = output.get("strategy")
        score = 0.0
        weight = 1.0 / 4.0
        expected_reports = case.expected.get("reports")
        expected_task_count = int(case.expected.get("task_count", len(expected_reports or [])))
        expected_strategy = case.expected.get("strategy")

        if isinstance(task_count, int) and task_count >= expected_task_count:
            score += weight
            details["task_count_ok"] = True
        else:
            details["task_count_ok"] = False
            details["expected_task_count"] = expected_task_count
            details["actual_task_count"] = task_count

        if expected_strategy is None or strategy == expected_strategy:
            score += weight
            details["strategy_match"] = True
        else:
            details["strategy_match"] = False
            details["expected_strategy"] = expected_strategy
            details["actual_strategy"] = strategy

        reports_ok = bool(reports) if expected_reports is None else reports == expected_reports
        if reports_ok:
            score += weight
            details["reports_match"] = True
        else:
            details["reports_match"] = False
            details["expected_reports"] = expected_reports
            details["actual_reports"] = reports

        if bool(output.get("passed", False)):
            score += weight
            details["passed_flag"] = True
        else:
            details["passed_flag"] = False

        final_score = round(score, 3)
        return SkillArenaCaseResult(case_id=case.case_id, score=final_score, passed=final_score >= 0.99, details=details)


    def _evaluate_response_planning(self, case, output):
        details: dict[str, Any] = {}
        if not isinstance(output, dict):
            return SkillArenaCaseResult(case_id=case.case_id, score=0.0, passed=False, details={"error": "non_mapping_output"})

        steps = list(output.get("steps", []))
        required_steps = list(case.expected.get("required_steps", []))
        forbidden_steps = list(case.expected.get("forbidden_steps", []))
        expected_policy = case.expected.get("policy")
        score = 0.0
        weight = 1.0 / 4.0

        required_ok = all(step in steps for step in required_steps)
        if required_ok:
            score += weight
            details["required_steps_ok"] = True
        else:
            details["required_steps_ok"] = False
            details["expected_required_steps"] = required_steps
            details["actual_steps"] = steps

        forbidden_ok = all(step not in steps for step in forbidden_steps)
        if forbidden_ok:
            score += weight
            details["forbidden_steps_ok"] = True
        else:
            details["forbidden_steps_ok"] = False
            details["expected_forbidden_steps"] = forbidden_steps
            details["actual_steps"] = steps

        if expected_policy is None or output.get("policy") == expected_policy:
            score += weight
            details["policy_match"] = True
        else:
            details["policy_match"] = False
            details["expected_policy"] = expected_policy
            details["actual_policy"] = output.get("policy")

        if bool(output.get("passed", False)):
            score += weight
            details["passed_flag"] = True
        else:
            details["passed_flag"] = False

        final_score = round(score, 3)
        return SkillArenaCaseResult(case_id=case.case_id, score=final_score, passed=final_score >= 0.99, details=details)


    def _evaluate_response_risk_model(self, case, output):
        details: dict[str, Any] = {}
        if not isinstance(output, dict):
            return SkillArenaCaseResult(case_id=case.case_id, score=0.0, passed=False, details={"error": "non_mapping_output"})

        expected_verify = case.expected.get("predicted_verify")
        required_steps = list(case.expected.get("required_steps", []))
        forbidden_steps = list(case.expected.get("forbidden_steps", []))
        steps = list(output.get("steps", []))
        score = 0.0
        weight = 1.0 / 4.0

        if expected_verify is None or bool(output.get("predicted_verify")) == bool(expected_verify):
            score += weight
            details["verify_decision_match"] = True
        else:
            details["verify_decision_match"] = False
            details["expected_verify"] = bool(expected_verify)
            details["actual_verify"] = bool(output.get("predicted_verify"))

        required_ok = all(step in steps for step in required_steps)
        if required_ok:
            score += weight
            details["required_steps_ok"] = True
        else:
            details["required_steps_ok"] = False
            details["expected_required_steps"] = required_steps
            details["actual_steps"] = steps

        forbidden_ok = all(step not in steps for step in forbidden_steps)
        if forbidden_ok:
            score += weight
            details["forbidden_steps_ok"] = True
        else:
            details["forbidden_steps_ok"] = False
            details["expected_forbidden_steps"] = forbidden_steps
            details["actual_steps"] = steps

        if bool(output.get("policy") == "adaptive_risk_model"):
            score += weight
            details["policy_match"] = True
        else:
            details["policy_match"] = False
            details["actual_policy"] = output.get("policy")

        final_score = round(score, 3)
        return SkillArenaCaseResult(case_id=case.case_id, score=final_score, passed=final_score >= 0.99, details=details)

    def _evaluate_memory_retrieval(self, case, output):
        details: dict[str, Any] = {}
        if not isinstance(output, dict):
            return SkillArenaCaseResult(case_id=case.case_id, score=0.0, passed=False, details={"error": "non_mapping_output"})

        expected_retrieved = case.expected.get("retrieved")
        expected_policy = case.expected.get("policy")
        score = 0.0
        weight = 1.0 / 3.0

        if output.get("retrieved") == expected_retrieved:
            score += weight
            details["retrieval_match"] = True
        else:
            details["retrieval_match"] = False
            details["expected_retrieved"] = expected_retrieved
            details["actual_retrieved"] = output.get("retrieved")

        if expected_policy is None or output.get("policy") == expected_policy:
            score += weight
            details["policy_match"] = True
        else:
            details["policy_match"] = False
            details["expected_policy"] = expected_policy
            details["actual_policy"] = output.get("policy")

        if bool(output.get("passed", False)):
            score += weight
            details["passed_flag"] = True
        else:
            details["passed_flag"] = False

        final_score = round(score, 3)
        return SkillArenaCaseResult(case_id=case.case_id, score=final_score, passed=final_score >= 0.99, details=details)

    def _evaluate_generic(self, case: SkillArenaCase, output: Any) -> SkillArenaCaseResult:
        details = {"output": output}
        if "equals" in case.expected:
            passed = output == case.expected["equals"]
        elif "contains" in case.expected:
            passed = str(case.expected["contains"]) in str(output)
        else:
            passed = bool(output)
        return SkillArenaCaseResult(case_id=case.case_id, score=1.0 if passed else 0.0, passed=passed, details=details)
