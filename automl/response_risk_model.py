from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from itertools import product
from typing import Any
from uuid import uuid4

from motivation.goal_schema import Goal


RESPONSE_RISK_FAMILY = "response_risk_model"
RISK_FEATURE_NAMES = (
    "requires_verification",
    "insufficient_evidence",
    "high_risk_signal",
    "critical_tag",
)


def new_model_id(prefix: str = "model") -> str:
    return f"{prefix}_{uuid4().hex[:12]}"


@dataclass(slots=True)
class ResponseRiskTrainingExample:
    goal: dict[str, Any]
    label: bool
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ResponseRiskTrainingExample":
        return cls(**dict(data))


@dataclass(slots=True)
class ResponseRiskModel:
    model_id: str
    version: str
    threshold: float
    weights: dict[str, float]
    bias: float = 0.0
    family: str = RESPONSE_RISK_FAMILY
    metrics: dict[str, float] = field(default_factory=dict)
    training_summary: dict[str, Any] = field(default_factory=dict)
    status: str = "stable"
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def touch(self) -> None:
        self.updated_at = datetime.now(timezone.utc)

    def feature_vector(self, goal: Goal | dict[str, Any], world_state: dict[str, Any] | None = None) -> dict[str, float]:
        goal_obj = goal if isinstance(goal, Goal) else Goal.from_dict(goal)
        world_state = dict(world_state or {})
        evidence = dict(goal_obj.evidence)
        risk_signal = float(evidence.get("risk_signal", goal_obj.risk_estimate))
        feature_map = {
            "requires_verification": 1.0 if bool(evidence.get("requires_verification", False)) else 0.0,
            "insufficient_evidence": 1.0 if bool(evidence.get("insufficient_evidence", False) or world_state.get("insufficient_evidence", False)) else 0.0,
            "high_risk_signal": 1.0 if risk_signal >= 0.6 else 0.0,
            "critical_tag": 1.0 if ("critical_answer" in goal_obj.tags or "high_risk" in goal_obj.tags or "critical" in goal_obj.tags) else 0.0,
        }
        return feature_map

    def score_goal(self, goal: Goal | dict[str, Any], world_state: dict[str, Any] | None = None) -> float:
        features = self.feature_vector(goal, world_state)
        score = float(self.bias)
        for name in RISK_FEATURE_NAMES:
            score += float(self.weights.get(name, 0.0)) * float(features.get(name, 0.0))
        return round(score, 6)

    def probability(self, goal: Goal | dict[str, Any], world_state: dict[str, Any] | None = None) -> float:
        score = self.score_goal(goal, world_state)
        return max(0.0, min(1.0, score))

    def should_verify(self, goal: Goal | dict[str, Any], world_state: dict[str, Any] | None = None) -> bool:
        return self.probability(goal, world_state) >= float(self.threshold)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["created_at"] = self.created_at.isoformat()
        payload["updated_at"] = self.updated_at.isoformat()
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ResponseRiskModel":
        raw = dict(data)
        raw["created_at"] = datetime.fromisoformat(raw["created_at"])
        raw["updated_at"] = datetime.fromisoformat(raw["updated_at"])
        return cls(**raw)

    @classmethod
    def baseline(cls) -> "ResponseRiskModel":
        return cls(
            model_id=new_model_id("riskmodel"),
            version="heuristic_v1",
            threshold=0.5,
            weights={
                "requires_verification": 1.0,
                "insufficient_evidence": 0.0,
                "high_risk_signal": 0.0,
                "critical_tag": 0.0,
            },
            bias=0.0,
            metrics={"accuracy": 1.0},
            training_summary={"origin": "baseline"},
            status="stable",
        )


@dataclass(slots=True)
class ResponseRiskTrainingReport:
    model: ResponseRiskModel
    search_space: dict[str, list[float]]
    leaderboard: list[dict[str, Any]]
    best_config: dict[str, float]
    train_metrics: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model.to_dict(),
            "search_space": {key: list(value) for key, value in self.search_space.items()},
            "leaderboard": [dict(item) for item in self.leaderboard],
            "best_config": dict(self.best_config),
            "train_metrics": dict(self.train_metrics),
        }


def _metric_summary(predictions: list[bool], labels: list[bool]) -> dict[str, float]:
    if not labels:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0}
    tp = sum(1 for pred, label in zip(predictions, labels) if pred and label)
    tn = sum(1 for pred, label in zip(predictions, labels) if (not pred) and (not label))
    fp = sum(1 for pred, label in zip(predictions, labels) if pred and (not label))
    fn = sum(1 for pred, label in zip(predictions, labels) if (not pred) and label)
    accuracy = (tp + tn) / len(labels)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return {
        "accuracy": round(accuracy, 3),
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
    }


DEFAULT_RESPONSE_RISK_SEARCH_SPACE: dict[str, list[float]] = {
    "threshold": [0.5, 0.6],
    "bias": [0.0],
    "weight_requires_verification": [1.0],
    "weight_insufficient_evidence": [0.0, 0.5, 1.0],
    "weight_high_risk_signal": [0.0, 0.5, 1.0],
    "weight_critical_tag": [0.0, 0.25, 0.5],
}


def evaluate_response_risk_model(model: ResponseRiskModel, examples: list[ResponseRiskTrainingExample]) -> dict[str, float]:
    labels = [bool(example.label) for example in examples]
    predictions = [model.should_verify(example.goal) for example in examples]
    return _metric_summary(predictions, labels)


def _config_product(search_space: dict[str, list[float]]) -> list[dict[str, float]]:
    if not search_space:
        return [{}]
    keys = sorted(search_space)
    values = [list(search_space[key]) for key in keys]
    return [dict(zip(keys, combo)) for combo in product(*values)]


def train_response_risk_model(
    examples: list[ResponseRiskTrainingExample],
    *,
    search_space: dict[str, list[float]] | None = None,
    version_prefix: str = "heuristic",
) -> ResponseRiskTrainingReport:
    if not examples:
        raise ValueError("response-risk training requires at least one labeled example")

    effective_space = {key: list(values) for key, values in (search_space or DEFAULT_RESPONSE_RISK_SEARCH_SPACE).items()}
    leaderboard: list[dict[str, Any]] = []
    best_tuple: tuple[float, float, float, float] | None = None
    best_model: ResponseRiskModel | None = None
    best_config: dict[str, float] = {}

    for config in _config_product(effective_space):
        model = ResponseRiskModel(
            model_id=new_model_id("riskmodel"),
            version=f"{version_prefix}_candidate",
            threshold=float(config.get("threshold", 0.5)),
            bias=float(config.get("bias", 0.0)),
            weights={
                "requires_verification": float(config.get("weight_requires_verification", 1.0)),
                "insufficient_evidence": float(config.get("weight_insufficient_evidence", 0.0)),
                "high_risk_signal": float(config.get("weight_high_risk_signal", 0.0)),
                "critical_tag": float(config.get("weight_critical_tag", 0.0)),
            },
            status="candidate",
        )
        metrics = evaluate_response_risk_model(model, examples)
        row = {**{key: float(value) for key, value in config.items()}, **metrics}
        leaderboard.append(row)
        ranking = (float(metrics["accuracy"]), float(metrics["recall"]), float(metrics["precision"]), -float(metrics["fp"]))
        if best_tuple is None or ranking > best_tuple:
            best_tuple = ranking
            best_model = model
            best_model.metrics = {key: float(value) for key, value in metrics.items() if key in {"accuracy", "precision", "recall"}}
            best_config = {key: float(value) for key, value in config.items()}

    assert best_model is not None
    best_model.version = f"{version_prefix}_v{uuid4().hex[:4]}"
    best_model.training_summary = {"examples": len(examples), "best_config": dict(best_config)}
    ordered = sorted(leaderboard, key=lambda item: (item["accuracy"], item["recall"], item["precision"], -item["fp"]), reverse=True)
    return ResponseRiskTrainingReport(
        model=best_model,
        search_space=effective_space,
        leaderboard=ordered,
        best_config=best_config,
        train_metrics=dict(best_model.metrics),
    )
