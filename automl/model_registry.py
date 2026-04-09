from __future__ import annotations

from collections import defaultdict

from automl.model_schema import model_from_payload
from automl.response_risk_model import ResponseRiskModel
from memory.goal_ledger import GoalLedger


class ModelRegistry:
    """Stores internal models across stable, candidate, canary, and rollback stages."""

    def __init__(self, ledger: GoalLedger | None = None) -> None:
        self.ledger = ledger
        self._models_by_id: dict[str, ResponseRiskModel] = {}
        self._stable_by_family: dict[str, str] = {}
        self._candidate_by_family: dict[str, str] = {}
        self._canary_by_family: dict[str, str] = {}
        self._rollback_stack: dict[str, list[str]] = defaultdict(list)
        if self.ledger is not None:
            self._rehydrate()

    def _rehydrate(self) -> None:
        for payload in self.ledger.load_model_records():
            model = model_from_payload(payload)
            self._models_by_id[model.model_id] = model
            if model.status == "stable":
                self._stable_by_family[model.family] = model.model_id
            elif model.status == "candidate":
                self._candidate_by_family[model.family] = model.model_id
            elif model.status == "canary":
                self._canary_by_family[model.family] = model.model_id

    def _persist(self, model: ResponseRiskModel) -> ResponseRiskModel:
        model.touch()
        self._models_by_id[model.model_id] = model
        if self.ledger is not None:
            self.ledger.save_model_record(model.to_dict())
        return model

    def register_stable(self, model: ResponseRiskModel) -> ResponseRiskModel:
        previous_stable = self._stable_by_family.get(model.family)
        if previous_stable is not None and previous_stable != model.model_id:
            self._rollback_stack[model.family].append(previous_stable)
        model.status = "stable"
        self._stable_by_family[model.family] = model.model_id
        self._candidate_by_family.pop(model.family, None)
        self._canary_by_family.pop(model.family, None)
        return self._persist(model)

    def register_candidate(self, model: ResponseRiskModel) -> ResponseRiskModel:
        model.status = "candidate"
        self._candidate_by_family[model.family] = model.model_id
        return self._persist(model)

    def discard_candidate(self, family: str) -> ResponseRiskModel | None:
        model_id = self._candidate_by_family.pop(family, None)
        if model_id is None:
            return None
        model = self._models_by_id[model_id]
        model.status = "discarded"
        return self._persist(model)

    def promote_candidate_to_canary(self, family: str) -> ResponseRiskModel:
        model_id = self._candidate_by_family.pop(family)
        model = self._models_by_id[model_id]
        model.status = "canary"
        self._canary_by_family[family] = model_id
        return self._persist(model)

    def finalize_canary(self, family: str) -> ResponseRiskModel:
        model_id = self._canary_by_family.pop(family)
        model = self._models_by_id[model_id]
        return self.register_stable(model)

    def rollback_canary(self, family: str) -> ResponseRiskModel | None:
        canary_id = self._canary_by_family.pop(family, None)
        if canary_id is not None:
            canary = self._models_by_id[canary_id]
            canary.status = "rolled_back"
            self._persist(canary)
        stable_id = self._stable_by_family.get(family)
        return self._models_by_id.get(stable_id) if stable_id is not None else None

    def rollback_family(self, family: str) -> ResponseRiskModel | None:
        if family in self._canary_by_family:
            return self.rollback_canary(family)
        history = self._rollback_stack.get(family, [])
        if not history:
            stable_id = self._stable_by_family.get(family)
            return self._models_by_id.get(stable_id) if stable_id else None
        restored_id = history.pop()
        restored = self._models_by_id[restored_id]
        restored.status = "stable"
        self._stable_by_family[family] = restored_id
        return self._persist(restored)

    def has_family(self, family: str) -> bool:
        return family in self._stable_by_family

    def has_candidate(self, family: str) -> bool:
        return family in self._candidate_by_family

    def has_canary(self, family: str) -> bool:
        return family in self._canary_by_family

    def get_model(self, model_id: str) -> ResponseRiskModel:
        return self._models_by_id[model_id]

    def get_active_model(self, family: str) -> ResponseRiskModel:
        return self._models_by_id[self._stable_by_family[family]]

    def get_candidate_model(self, family: str) -> ResponseRiskModel:
        return self._models_by_id[self._candidate_by_family[family]]

    def get_canary_model(self, family: str) -> ResponseRiskModel:
        return self._models_by_id[self._canary_by_family[family]]

    def list_models(self, family: str | None = None) -> list[ResponseRiskModel]:
        models = list(self._models_by_id.values())
        if family is not None:
            models = [model for model in models if model.family == family]
        return sorted(models, key=lambda item: (item.created_at, item.model_id))

    def stable_model_names(self) -> list[str]:
        return sorted(self._stable_by_family.values())
