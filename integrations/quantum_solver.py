from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any
from uuid import uuid4


@dataclass(slots=True)
class DeferredQuantumTask:
    capability_id: str
    payload: dict[str, Any]
    task_id: str = field(default_factory=lambda: f'dq_{uuid4().hex[:12]}')
    backend: str = 'quantum_solver'

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ClassicalFallbackQuantumSolver:
    def solve(self, task: dict[str, Any]) -> dict[str, Any]:
        values = list(task.get('values', []))
        if not values:
            return {'status': 'deferred', 'reason': 'empty_values'}
        best_value = min(values)
        return {'status': 'ok', 'solver': 'classical_fallback', 'best_value': best_value, 'best_index': values.index(best_value)}


class QuantumSolver:
    def __init__(self, available: bool = False) -> None:
        self.available = available
        self.fallback = ClassicalFallbackQuantumSolver()

    def solve(self, task: dict[str, Any]) -> dict[str, Any] | DeferredQuantumTask:
        if not self.available:
            if task.get('allow_fallback', True):
                return self.fallback.solve(task)
            return DeferredQuantumTask(capability_id=str(task.get('capability_id', 'unknown')), payload=task)
        return {'status': 'ok', 'solver': 'quantum', 'result': task.get('values', [])}
