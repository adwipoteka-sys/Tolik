from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict

from agency.agency_module import AgencyModule
from agency.tools import LocalToolbox
from core.global_workspace import GlobalWorkspace
from language.language_module import LanguageModule
from memory.memory_module import MemoryModule
from metacognition.metacognition_module import MetacognitionModule
from motivation.motivation_module import MotivationModule
from perception.perception_module import PerceptionModule
from planning.planning_module import PlanningModule
from reasoning.reasoning_module import ReasoningModule


class TolikAGI:
    def __init__(self) -> None:
        module_root = Path(__file__).resolve().parent
        repo_root = module_root.parent.resolve()
        runtime_dir = module_root / "data" / "runtime"

        self.workspace = GlobalWorkspace()
        self.perception = PerceptionModule()
        self.memory = MemoryModule(storage_dir=str(runtime_dir))
        self.memory.seed_defaults()
        self.reasoning = ReasoningModule()
        self.motivation = MotivationModule()
        self.planning = PlanningModule()
        self.language = LanguageModule()
        self.toolbox = LocalToolbox(str(repo_root))
        self.agency = AgencyModule(language=self.language, toolbox=self.toolbox)
        self.metacognition = MetacognitionModule()

    @staticmethod
    def _to_dict(obj: Any) -> Dict[str, Any]:
        if is_dataclass(obj):
            return asdict(obj)
        if isinstance(obj, dict):
            return obj
        raise TypeError(f"Cannot convert object of type {type(obj)!r} to dict")

    def run_cycle(self, user_text: str) -> Dict[str, Any]:
        perception = self._to_dict(self.perception.process_input(user_text))
        self.workspace.publish("perception", perception, source="perception")
        self.memory.remember_event({"type": "perception", "data": perception})

        self.motivation.ingest_perception(perception)
        goal = self.motivation.next_goal() or "maintain_stability"
        self.workspace.publish("goal", goal, source="motivation")

        reasoning = self.reasoning.analyze(goal, perception, self.memory.recent_context())
        self.workspace.publish("reasoning", reasoning, source="reasoning")

        plan = self.planning.make_plan(goal, reasoning)
        self.workspace.publish("plan", plan, source="planning")

        action_result = self.agency.execute_plan(plan, self.memory, reasoning)
        self.workspace.publish("action_result", action_result, source="agency")
        self.memory.remember_event({"type": "action_result", "data": action_result})

        meta = self.metacognition.review(perception, reasoning, action_result)
        self.workspace.publish("metacognition", meta, source="metacognition")
        self.motivation.ingest_metacognition(meta.get("recommendations", []))

        self.memory.save()

        return {
            "goal": goal,
            "reasoning": reasoning,
            "plan": plan,
            "answer": action_result["answer"],
            "meta": meta,
        }


def main() -> None:
    agi = TolikAGI()
    print("Tolik bootstrap AGI ready. Enter text. Type 'exit' to quit.")

    while True:
        user_text = input("you> ").strip()
        if user_text.lower() in {"exit", "quit", "q"}:
            break
        result = agi.run_cycle(user_text)
        print("\n[GOAL]", result["goal"])
        print("[ANSWER]\n" + result["answer"])
        print("[META]", result["meta"])
        print()


if __name__ == "__main__":
    main()
