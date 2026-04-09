from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class OperatorCharter:
    """Operator-defined guardrails for autonomous AGI-relevant work."""

    name: str = "default_charter"
    description: str = "Autonomous progress is allowed only inside safe local boundaries."
    allowed_goal_tags: list[str] = field(
        default_factory=lambda: [
            "grounded_self_training",
            "grounded_navigation",
            "maintenance",
            "skill_audit",
            "transfer_curriculum",
            "navigation_route_explanation",
            "local_only",
            "planning",
            "memory",
            "response_planning_patch",
            "memory_retrieval_patch",
            "automl_model_upgrade",
            "response_risk_model",
            "future_integration",
            "deferred_execution",
        ]
    )
    blocked_goal_tags: list[str] = field(default_factory=lambda: ["networked", "cloud_only", "unsafe"])
    allow_grounded_navigation: bool = True
    allow_tool_generation: bool = True
    allow_cloud_llm: bool = False
    allow_quantum_solver: bool = False
    require_provider_qualification: bool = True
    provider_rollout_min_correctness: float = 0.95
    provider_rollout_min_safety: float = 1.0
    provider_rollout_max_avg_latency_ms: float = 250.0
    allowed_live_providers: dict[str, list[str]] = field(default_factory=dict)
    enable_cost_aware_fallback_routing: bool = True
    provider_routing_quality_weight: float = 0.8
    provider_routing_cost_weight: float = 0.2
    provider_routing_max_score_gap: float = 0.08
    enable_canary_rollout: bool = False
    canary_live_fraction: float = 0.34
    canary_window_size: int = 6
    canary_min_samples: int = 4
    canary_min_correctness_rate: float = 0.75
    canary_min_safety_rate: float = 1.0
    canary_max_latency_multiplier: float = 1.5
    canary_min_text_agreement: float = 0.35
    auto_promote_canary: bool = True
    auto_rollback_canary: bool = True
    enable_shadow_traffic: bool = True
    shadow_sample_rate: float = 1.0
    shadow_candidate_limit: int = 2
    shadow_consensus_min_support: int = 1
    shadow_consensus_pairwise_min_agreement: float = 0.45
    post_promotion_window_size: int = 4
    post_promotion_min_samples: int = 3
    post_promotion_min_correctness_rate: float = 0.75
    post_promotion_min_safety_rate: float = 1.0
    post_promotion_max_latency_multiplier: float = 1.75
    post_promotion_min_text_agreement: float = 0.35
    auto_demote_on_drift: bool = True
    rollback_cooldown_rollouts: int = 1
    anti_flap_window_rollouts: int = 4
    anti_flap_repeat_failures: int = 2
    anti_flap_freeze_rollouts: int = 2
    max_internal_goals_per_cycle: int = 1
    max_cycles_per_run: int = 12
    navigation_batch_size: int = 3
    navigation_max_difficulty: int = 2
    navigation_success_threshold: float = 0.67
    require_human_approval_for: list[str] = field(
        default_factory=lambda: ["cloud_llm", "quantum_solver", "network_install"]
    )

    def __post_init__(self) -> None:
        if self.max_internal_goals_per_cycle <= 0:
            raise ValueError("max_internal_goals_per_cycle must be positive")
        if self.max_cycles_per_run <= 0:
            raise ValueError("max_cycles_per_run must be positive")
        if self.navigation_batch_size <= 0:
            raise ValueError("navigation_batch_size must be positive")
        if self.navigation_max_difficulty <= 0:
            raise ValueError("navigation_max_difficulty must be positive")
        if not (0.0 < self.navigation_success_threshold <= 1.0):
            raise ValueError("navigation_success_threshold must be in (0, 1]")
        if not (0.0 <= self.provider_rollout_min_correctness <= 1.0):
            raise ValueError("provider_rollout_min_correctness must be in [0, 1]")
        if not (0.0 <= self.provider_rollout_min_safety <= 1.0):
            raise ValueError("provider_rollout_min_safety must be in [0, 1]")
        if self.provider_rollout_max_avg_latency_ms < 0.0:
            raise ValueError("provider_rollout_max_avg_latency_ms must be non-negative")
        if not (0.0 <= self.provider_routing_quality_weight <= 1.0):
            raise ValueError("provider_routing_quality_weight must be in [0, 1]")
        if not (0.0 <= self.provider_routing_cost_weight <= 1.0):
            raise ValueError("provider_routing_cost_weight must be in [0, 1]")
        if (self.provider_routing_quality_weight + self.provider_routing_cost_weight) <= 0.0:
            raise ValueError("provider routing weights must sum to a positive value")
        if self.provider_routing_max_score_gap < 0.0:
            raise ValueError("provider_routing_max_score_gap must be non-negative")
        if not (0.0 < self.canary_live_fraction <= 1.0):
            raise ValueError("canary_live_fraction must be in (0, 1]")
        if self.canary_window_size <= 0:
            raise ValueError("canary_window_size must be positive")
        if self.canary_min_samples <= 0:
            raise ValueError("canary_min_samples must be positive")
        if self.canary_min_samples > self.canary_window_size:
            raise ValueError("canary_min_samples must be <= canary_window_size")
        if not (0.0 <= self.canary_min_correctness_rate <= 1.0):
            raise ValueError("canary_min_correctness_rate must be in [0, 1]")
        if not (0.0 <= self.canary_min_safety_rate <= 1.0):
            raise ValueError("canary_min_safety_rate must be in [0, 1]")
        if self.canary_max_latency_multiplier <= 0.0:
            raise ValueError("canary_max_latency_multiplier must be positive")
        if not (0.0 <= self.canary_min_text_agreement <= 1.0):
            raise ValueError("canary_min_text_agreement must be in [0, 1]")
        if not (0.0 <= self.shadow_sample_rate <= 1.0):
            raise ValueError("shadow_sample_rate must be in [0, 1]")
        if self.shadow_candidate_limit <= 0:
            raise ValueError("shadow_candidate_limit must be positive")
        if self.shadow_consensus_min_support <= 0:
            raise ValueError("shadow_consensus_min_support must be positive")
        if self.shadow_consensus_min_support > self.shadow_candidate_limit:
            raise ValueError("shadow_consensus_min_support must be <= shadow_candidate_limit")
        if not (0.0 <= self.shadow_consensus_pairwise_min_agreement <= 1.0):
            raise ValueError("shadow_consensus_pairwise_min_agreement must be in [0, 1]")
        if self.post_promotion_window_size <= 0:
            raise ValueError("post_promotion_window_size must be positive")
        if self.post_promotion_min_samples <= 0:
            raise ValueError("post_promotion_min_samples must be positive")
        if self.post_promotion_min_samples > self.post_promotion_window_size:
            raise ValueError("post_promotion_min_samples must be <= post_promotion_window_size")
        if not (0.0 <= self.post_promotion_min_correctness_rate <= 1.0):
            raise ValueError("post_promotion_min_correctness_rate must be in [0, 1]")
        if not (0.0 <= self.post_promotion_min_safety_rate <= 1.0):
            raise ValueError("post_promotion_min_safety_rate must be in [0, 1]")
        if self.post_promotion_max_latency_multiplier <= 0.0:
            raise ValueError("post_promotion_max_latency_multiplier must be positive")
        if not (0.0 <= self.post_promotion_min_text_agreement <= 1.0):
            raise ValueError("post_promotion_min_text_agreement must be in [0, 1]")
        if self.rollback_cooldown_rollouts < 0:
            raise ValueError("rollback_cooldown_rollouts must be non-negative")
        if self.anti_flap_window_rollouts <= 0:
            raise ValueError("anti_flap_window_rollouts must be positive")
        if self.anti_flap_repeat_failures <= 0:
            raise ValueError("anti_flap_repeat_failures must be positive")
        if self.anti_flap_freeze_rollouts < 0:
            raise ValueError("anti_flap_freeze_rollouts must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def provider_allowed(self, *, adapter_name: str, provider: str) -> tuple[bool, str | None]:
        if not provider:
            return False, "provider_missing"
        allowlist = self.allowed_live_providers.get(adapter_name)
        if allowlist and provider not in allowlist:
            return False, f"provider_not_in_allowlist:{provider}"
        return True, None

    def rollout_report_allowed(self, report: Any) -> tuple[bool, list[str]]:
        reasons: list[str] = []
        provider = str(getattr(report, "provider", "") or "")
        adapter_name = str(getattr(report, "adapter_name", "") or "")
        provider_allowed, provider_reason = self.provider_allowed(adapter_name=adapter_name, provider=provider)
        if not provider_allowed and provider_reason is not None:
            reasons.append(provider_reason)
        if getattr(report, "correctness_rate", 0.0) < self.provider_rollout_min_correctness:
            reasons.append("correctness_threshold_not_met")
        if getattr(report, "safety_rate", 0.0) < self.provider_rollout_min_safety:
            reasons.append("safety_threshold_not_met")
        if getattr(report, "avg_latency_ms", 0.0) > self.provider_rollout_max_avg_latency_ms:
            reasons.append("latency_threshold_not_met")
        if self.require_provider_qualification and not bool(getattr(report, "eligible", False)):
            reasons.append("qualification_required")
        return (not reasons), reasons

    def goal_allowed(self, *, tags: list[str], required_capabilities: list[str]) -> tuple[bool, str | None]:
        tag_set = set(tags)
        if tag_set & set(self.blocked_goal_tags):
            blocked = sorted(tag_set & set(self.blocked_goal_tags))
            return False, f"blocked tags: {blocked}"
        if self.allowed_goal_tags:
            implicitly_allowed = {"patch", "strategy_reuse", "grounded_navigation_patch", "grounded_navigation_audit"}
            unknown = [tag for tag in tags if tag not in self.allowed_goal_tags and tag not in implicitly_allowed]
            if unknown:
                return False, f"tags not allowed by charter: {unknown}"
        caps = set(required_capabilities)
        if not self.allow_cloud_llm and "cloud_llm" in caps:
            return False, "cloud_llm is disabled by charter"
        if not self.allow_quantum_solver and "quantum_solver" in caps:
            return False, "quantum_solver is disabled by charter"
        if not self.allow_grounded_navigation and "grounded_navigation" in caps:
            return False, "grounded_navigation is disabled by charter"
        return True, None


DEFAULT_CHARTER = OperatorCharter()


def load_charter(path: str | Path | None) -> OperatorCharter:
    if path is None:
        return DEFAULT_CHARTER
    charter_path = Path(path)
    if not charter_path.exists():
        raise FileNotFoundError(f"Charter file not found: {charter_path}")
    data = json.loads(charter_path.read_text(encoding="utf-8"))
    return OperatorCharter(**data)


def save_charter(charter: OperatorCharter, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(charter.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return target
