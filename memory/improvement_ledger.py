from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from memory.goal_ledger import GoalLedger


@dataclass(slots=True)
class CapabilityGap:
    capability_id: str
    gap_type: str
    severity: float
    confidence: float
    evidence_count: int
    transfer_score: float
    regression_delta: float
    last_updated_step: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ImprovementGoal:
    goal_id: str
    capability_id: str
    goal_type: str
    target_metric: str
    target_value: float
    priority: float
    budget_steps: int
    cooldown_cycles: int
    status: str = 'proposed'
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload['created_at'] = self.created_at.isoformat()
        return payload


@dataclass(slots=True)
class CurriculumItem:
    item_id: str
    goal_id: str
    capability_id: str
    benchmark_name: str
    scenario_id: str
    action_hint: str
    expected_signal: str
    max_attempts: int
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ImprovementLedger:
    def __init__(self, goal_ledger: GoalLedger) -> None:
        self.goal_ledger = goal_ledger
        self.root: Path = goal_ledger.root
        self.goal_dir = self.root / 'improvement_goals'
        self.curriculum_dir = self.root / 'improvement_curriculum'
        self.tool_proposal_dir = self.root / 'tool_proposals'
        self.external_task_dir = self.root / 'deferred_external_tasks'
        for directory in (self.goal_dir, self.curriculum_dir, self.tool_proposal_dir, self.external_task_dir):
            directory.mkdir(parents=True, exist_ok=True)

    def _write_json(self, path: Path, payload: dict[str, Any]) -> Path:
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
        return path

    def save_goal(self, goal: ImprovementGoal) -> Path:
        path = self.goal_dir / f"{goal.goal_id}.json"
        self._write_json(path, goal.to_dict())
        self.goal_ledger.append_event({'event_type': 'improvement_goal_stored', 'goal_id': goal.goal_id, 'capability_id': goal.capability_id, 'goal_type': goal.goal_type})
        return path

    def save_curriculum_items(self, items: list[CurriculumItem]) -> list[Path]:
        paths: list[Path] = []
        for item in items:
            path = self.curriculum_dir / f"{item.item_id}.json"
            self._write_json(path, item.to_dict())
            self.goal_ledger.append_event({'event_type': 'curriculum_item_stored', 'item_id': item.item_id, 'goal_id': item.goal_id, 'capability_id': item.capability_id})
            paths.append(path)
        return paths

    def save_tool_proposal(self, proposal: dict[str, Any]) -> Path:
        proposal_id = str(proposal.get('proposal_id', 'unknown'))
        path = self.tool_proposal_dir / f"{proposal_id}.json"
        self._write_json(path, proposal)
        self.goal_ledger.append_event({'event_type': 'tool_proposal_stored', 'proposal_id': proposal_id, 'capability_id': proposal.get('capability_id'), 'status': proposal.get('status')})
        return path

    def save_deferred_task(self, task: dict[str, Any]) -> Path:
        task_id = str(task.get('task_id', 'unknown'))
        path = self.external_task_dir / f"{task_id}.json"
        self._write_json(path, task)
        self.goal_ledger.append_event({'event_type': 'deferred_external_task_stored', 'task_id': task_id, 'backend': task.get('backend'), 'capability_id': task.get('capability_id')})
        return path

    def list_goals(self) -> list[dict[str, Any]]:
        return [json.loads(path.read_text(encoding='utf-8')) for path in sorted(self.goal_dir.glob('*.json'))]
