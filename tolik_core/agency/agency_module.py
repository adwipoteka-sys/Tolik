from __future__ import annotations

from typing import Any, Dict, List, Tuple

from agency.tools import LocalToolbox
from language.language_module import LanguageModule
from memory.memory_module import MemoryModule


class AgencyModule:
    """Executes plan steps using internal tools only."""

    def __init__(self, language: LanguageModule, toolbox: LocalToolbox) -> None:
        self.language = language
        self.toolbox = toolbox

    @staticmethod
    def _parse_fact_payload(payload: str) -> Tuple[str, str]:
        for sep in ("=", ":"):
            if sep in payload:
                key, value = payload.split(sep, 1)
                return key.strip(), value.strip()
        return payload.strip(), "true"

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
                results = memory.search_facts(payload, limit=5)
                if results:
                    memory_hits.extend(results)
                else:
                    notes.append(f"No memory hits for: {payload}")

            elif action == "store_fact":
                key, value = self._parse_fact_payload(payload)
                memory.store_fact(key, value)
                notes.append(f"Stored fact: {key} = {value}")

            elif action == "recall_fact":
                value = memory.recall_fact(payload)
                notes.append(f"Recall {payload}: {value}")

            elif action == "list_files":
                files = self.toolbox.list_files(payload or ".")
                notes.append("Files:\n" + "\n".join(files))

            elif action == "read_file":
                content = self.toolbox.read_file(payload)
                notes.append(f"File {payload}:\n{content}")

            elif action == "write_note":
                title = payload.splitlines()[0][:40] if payload.strip() else "agent_note"
                path = self.toolbox.write_note(title=title, text=payload)
                notes.append(f"Note written to: {path}")

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
