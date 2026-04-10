from __future__ import annotations

from typing import Any, Dict, List

from language.language_module import LanguageModule
from memory.memory_module import MemoryModule


class AgencyModule:
    """Executes plan steps using internal tools only."""

    def __init__(self, language: LanguageModule) -> None:
        self.language = language

    def execute_plan(
        self,
        plan: List[Dict[str, str]],
        memory: MemoryModule,
        reasoning: Dict[str, Any],
    ) -> Dict[str, Any]:
        memory_hits: List[str] = []
        notes: List[str] = []
        final_answer = ""

        for step in plan:
            action = step["action"]
            payload = step["input"]

            if action == "search_memory":
                architecture = memory.recall_fact("architecture_principle")
                modules = memory.recall_fact("project_modules")
                if architecture:
                    memory_hits.append(str(architecture))
                if modules:
                    memory_hits.append("Modules: " + ", ".join(modules))
            elif action == "decompose_task":
                notes.append(f"Task decomposition created for: {payload}")
            elif action == "reflect":
                notes.append(f"Reflection note: {payload}")
            elif action == "note_subgoal":
                notes.append(f"Subgoal noted: {payload}")
            elif action == "compose_answer":
                final_answer = self.language.compose_answer(
                    user_prompt=payload,
                    memory_hits=memory_hits + notes,
                    reasoning=reasoning,
                    plan=plan,
                )

        return {
            "status": "ok",
            "answer": final_answer or "План выполнен без текстового ответа.",
            "memory_hits": memory_hits,
            "notes": notes,
        }
