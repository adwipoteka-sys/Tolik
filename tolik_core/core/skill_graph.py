from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List


@dataclass
class SkillNode:
    name: str
    description: str
    parents: List[str] = field(default_factory=list)
    runs: int = 0
    passes: int = 0
    last_score: float = 0.0


class SkillGraph:
    def __init__(self, storage_dir: str = "data/runtime") -> None:
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.storage_dir / "skill_graph.json"
        self.nodes: Dict[str, SkillNode] = {}
        self._load()
        self.seed_defaults()

    def _load(self) -> None:
        if self.path.exists():
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            self.nodes = {item["name"]: SkillNode(**item) for item in raw}

    def _save(self) -> None:
        payload = [asdict(node) for node in self.nodes.values()]
        payload.sort(key=lambda x: x["name"])
        self.path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def seed_defaults(self) -> None:
        if self.nodes:
            return

        defaults = [
            SkillNode("perception_local", "Локальное восприятие частично наблюдаемой среды."),
            SkillNode("memory_consolidation", "Консолидация опыта, карт мира и долгосрочной памяти."),
            SkillNode("partial_world_model", "Построение внутренней модели мира по локальным наблюдениям.", ["perception_local", "memory_consolidation"]),
            SkillNode("compositional_state_tracking", "Отслеживание объектов и латентного состояния: ключ, дверь, подцели.", ["perception_local", "memory_consolidation"]),
            SkillNode("hierarchical_planning", "Многошаговое и композиционное планирование.", ["partial_world_model", "compositional_state_tracking"]),
            SkillNode("transfer_generalization", "Перенос навыков на новые layout и задачи.", ["hierarchical_planning"]),
            SkillNode("motivation_internal", "Внутренняя мотивация, curriculum и самотренировка.", ["memory_consolidation"]),
            SkillNode("metacognitive_repair", "Метакогнитивное обнаружение провалов и самоисправление.", ["memory_consolidation"]),
            SkillNode("autonomous_development", "Автономное развитие через skill graph, curriculum и repair-cycles.", ["transfer_generalization", "motivation_internal", "metacognitive_repair"]),
        ]
        self.nodes = {node.name: node for node in defaults}
        self._save()

    def mastery(self, name: str) -> float:
        node = self.nodes[name]
        if node.runs == 0:
            return 0.0
        return node.passes / node.runs

    def _record(self, name: str, ok: bool, score: float) -> None:
        node = self.nodes[name]
        node.runs += 1
        node.passes += int(ok)
        node.last_score = score

    def sync_from_eval(self, eval_results: Dict[str, List[Dict[str, object]]], curriculum_summary: Dict[str, int], repair_count: int) -> None:
        nav_rows = eval_results.get("nav", [])
        kd_rows = eval_results.get("keydoor", [])

        for row in nav_rows:
            ok = bool(row["ok"])
            score = float(row["reward_total"])
            self._record("perception_local", ok, score)
            self._record("memory_consolidation", ok, score)
            self._record("partial_world_model", ok, score)
            self._record("hierarchical_planning", ok, score)

        for row in kd_rows:
            ok = bool(row["ok"])
            score = float(row["reward_total"])
            self._record("perception_local", ok, score)
            self._record("memory_consolidation", ok, score)
            self._record("compositional_state_tracking", ok, score)
            self._record("hierarchical_planning", ok, score)

        nav_ok = bool(nav_rows) and all(bool(r["ok"]) for r in nav_rows)
        kd_ok = bool(kd_rows) and all(bool(r["ok"]) for r in kd_rows)
        if nav_rows or kd_rows:
            denom = len(nav_rows) + len(kd_rows)
            avg_score = sum(float(r["reward_total"]) for r in nav_rows + kd_rows) / max(1, denom)
            self._record("transfer_generalization", nav_ok and kd_ok, avg_score)

        curr_runs = int(curriculum_summary.get("runs", 0))
        curr_passes = int(curriculum_summary.get("passes", 0))
        curr_ok = curr_runs > 0 and curr_passes == curr_runs
        curr_score = (curr_passes / curr_runs) if curr_runs else 0.0
        self._record("motivation_internal", curr_ok, curr_score)

        repair_ok = repair_count == 0 and curr_runs > 0
        repair_score = 1.0 if repair_ok else 0.0
        self._record("metacognitive_repair", repair_ok, repair_score)

        auto_ok = nav_ok and kd_ok and curr_ok and repair_ok
        auto_score = (
            self.mastery("transfer_generalization")
            + self.mastery("motivation_internal")
            + self.mastery("metacognitive_repair")
        ) / 3.0
        self._record("autonomous_development", auto_ok, auto_score)

        self._save()

    def list_nodes(self) -> List[Dict[str, object]]:
        rows = []
        for name, node in self.nodes.items():
            rows.append(
                {
                    "name": name,
                    "description": node.description,
                    "parents": node.parents,
                    "runs": node.runs,
                    "passes": node.passes,
                    "mastery": round(self.mastery(name), 3),
                    "last_score": round(node.last_score, 3),
                }
            )
        rows.sort(key=lambda x: (x["mastery"], x["name"]))
        return rows

    def gaps(self, limit: int = 5) -> List[Dict[str, object]]:
        rows = []
        for item in self.list_nodes():
            parent_mastery = 0.0
            if item["parents"]:
                parent_mastery = sum(self.mastery(p) for p in item["parents"]) / len(item["parents"])
            gap = 1.0 - (0.7 * item["mastery"] + 0.3 * parent_mastery)
            rows.append({**item, "gap": round(gap, 3)})
        rows.sort(key=lambda x: x["gap"], reverse=True)
        return rows[:limit]

    def roadmap(self, skill_name: str) -> List[str]:
        seen = set()
        order: List[str] = []

        def dfs(name: str) -> None:
            if name in seen or name not in self.nodes:
                return
            seen.add(name)
            for parent in self.nodes[name].parents:
                dfs(parent)
            order.append(name)

        dfs(skill_name)
        return order

    def propose_goals(self, limit: int = 5) -> List[str]:
        templates = {
            "perception_local": "улучшить локальное восприятие и извлечение объектов из частично наблюдаемой среды",
            "memory_consolidation": "улучшить консолидацию опыта, карт мира и долгосрочную память",
            "partial_world_model": "улучшить построение внутренней world model по локальным наблюдениям",
            "compositional_state_tracking": "улучшить отслеживание латентного состояния объектов и подцелей",
            "hierarchical_planning": "улучшить иерархическое планирование и декомпозицию задач",
            "transfer_generalization": "улучшить перенос навыков между различными средами и типами задач",
            "motivation_internal": "улучшить внутреннюю мотивацию, curriculum и распределение внимания между задачами",
            "metacognitive_repair": "улучшить метакогнитическое обнаружение провалов и автоматическое самоисправление",
            "autonomous_development": "улучшить автономное развитие через skill graph и capability cycles",
        }
        out: List[str] = []
        for item in self.gaps(limit):
            out.append(templates.get(item["name"], f"улучшить способность {item['name']}"))
        return out

    def summary(self) -> Dict[str, float]:
        if not self.nodes:
            return {"skills": 0, "avg_mastery": 0.0}
        avg_mastery = sum(self.mastery(name) for name in self.nodes) / len(self.nodes)
        return {"skills": len(self.nodes), "avg_mastery": round(avg_mastery, 3)}
