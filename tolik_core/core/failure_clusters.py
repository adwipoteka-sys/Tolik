from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List


@dataclass
class FailureCase:
    family: str
    layout: str
    issue: str
    done: bool
    actual_steps: int
    optimal_steps: int
    ratio: float
    reward_total: float
    answer: str


class FailureClusters:
    def __init__(self, storage_dir: str = "data/runtime") -> None:
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.storage_dir / "failure_clusters.json"
        self.cases: List[FailureCase] = []
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            self.cases = [FailureCase(**item) for item in raw]

    def _save(self) -> None:
        self.path.write_text(
            json.dumps([asdict(c) for c in self.cases], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def clear(self) -> None:
        self.cases = []
        self._save()

    def add_case(
        self,
        *,
        family: str,
        layout: str,
        issue: str,
        done: bool,
        actual_steps: int,
        optimal_steps: int,
        ratio: float,
        reward_total: float,
        answer: str,
    ) -> None:
        self.cases.append(
            FailureCase(
                family=family,
                layout=layout,
                issue=issue,
                done=done,
                actual_steps=actual_steps,
                optimal_steps=optimal_steps,
                ratio=ratio,
                reward_total=reward_total,
                answer=answer,
            )
        )
        self._save()

    def list_cases(self) -> List[Dict[str, object]]:
        return [asdict(c) for c in self.cases]

    def summary(self) -> List[Dict[str, object]]:
        agg: Dict[str, int] = {}
        for case in self.cases:
            agg[case.issue] = agg.get(case.issue, 0) + 1

        rows = [{"issue": issue, "count": count} for issue, count in agg.items()]
        rows.sort(key=lambda x: (-x["count"], x["issue"]))
        return rows

    def propose_goals(self, limit: int = 5) -> List[str]:
        templates = {
            "nav_world_model": "улучшить построение внутренней world model и стратегию исследования в частично наблюдаемой навигации",
            "nav_planning_efficiency": "улучшить эффективность планирования и сократить лишние шаги в частично наблюдаемой навигации",
            "keydoor_state_tracking": "улучшить отслеживание латентного состояния ключа и двери в композиционных задачах",
            "keydoor_compositional_efficiency": "улучшить композиционное планирование в задачах key-door и уменьшить избыточные шаги",
        }

        out: List[str] = []
        for row in self.summary()[:limit]:
            out.append(templates.get(row["issue"], f"исследовать и исправить класс провалов: {row['issue']}"))
        return out

    def summary_counts(self) -> Dict[str, int]:
        return {
            "cases": len(self.cases),
            "clusters": len(self.summary()),
        }
