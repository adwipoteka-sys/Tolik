set -euo pipefail

PROJECT_DIR="tolik_core"
if [ -e "$PROJECT_DIR" ]; then
  PROJECT_DIR="tolik_core_$(date +%Y%m%d_%H%M%S)"
fi

mkdir -p "$PROJECT_DIR"/{core,perception,memory,reasoning,planning,language,motivation,agency,metacognition}

for d in core perception memory reasoning planning language motivation agency metacognition; do
  touch "$PROJECT_DIR/$d/__init__.py"
done

cat > "$PROJECT_DIR/core/global_workspace.py" <<'PY'
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import RLock
from typing import Any, Dict, List, Optional


@dataclass
class WorkspaceEvent:
    topic: str
    payload: Any
    source: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class GlobalWorkspace:
    """Central shared memory/bus for AGI modules."""

    def __init__(self) -> None:
        self._state: Dict[str, Any] = {}
        self._events: List[WorkspaceEvent] = []
        self._lock = RLock()

    def publish(self, topic: str, payload: Any, source: str = "system") -> None:
        with self._lock:
            self._state[topic] = deepcopy(payload)
            self._events.append(WorkspaceEvent(topic=topic, payload=deepcopy(payload), source=source))

    def read(self, topic: str, default: Optional[Any] = None) -> Any:
        with self._lock:
            return deepcopy(self._state.get(topic, default))

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return deepcopy(self._state)

    def recent_events(self, limit: int = 20) -> List[Dict[str, Any]]:
        with self._lock:
            tail = self._events[-limit:]
            return [
                {
                    "topic": event.topic,
                    "payload": deepcopy(event.payload),
                    "source": event.source,
                    "timestamp": event.timestamp,
                }
                for event in tail
            ]
PY

cat > "$PROJECT_DIR/perception/perception_module.py" <<'PY'
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class PerceptionResult:
    raw_text: str
    cleaned_text: str
    intent: str
    entities: Dict[str, str]
    suggested_goal: Optional[str]


class PerceptionModule:
    """Converts raw user input into a structured internal representation."""

    def process_input(self, user_text: str) -> PerceptionResult:
        cleaned = " ".join(user_text.strip().split())
        lowered = cleaned.lower()

        intent = "statement"
        suggested_goal: Optional[str] = None
        entities: Dict[str, str] = {}

        if not cleaned:
            intent = "empty"
            suggested_goal = "stabilize_context"
        elif lowered.startswith("goal:"):
            intent = "explicit_goal"
            suggested_goal = cleaned.split(":", 1)[1].strip() or "clarify_goal"
        elif "?" in cleaned:
            intent = "question"
            suggested_goal = f"answer_user: {cleaned}"
        elif any(word in lowered for word in ["сделай", "создай", "реализуй", "напиши"]):
            intent = "task_request"
            suggested_goal = f"execute_request: {cleaned}"
        else:
            suggested_goal = f"analyze_input: {cleaned}"

        if "agi" in lowered:
            entities["domain"] = "agi"
        if "толик" in lowered:
            entities["project"] = "толик"

        return PerceptionResult(
            raw_text=user_text,
            cleaned_text=cleaned,
            intent=intent,
            entities=entities,
            suggested_goal=suggested_goal,
        )
PY

cat > "$PROJECT_DIR/memory/memory_module.py" <<'PY'
from __future__ import annotations

from collections import deque
from typing import Any, Deque, Dict, List, Optional


class MemoryModule:
    """Short-term + long-term memory skeleton."""

    def __init__(self, short_term_limit: int = 25) -> None:
        self.short_term: Deque[Dict[str, Any]] = deque(maxlen=short_term_limit)
        self.long_term: Dict[str, Any] = {}

    def remember_event(self, event: Dict[str, Any]) -> None:
        self.short_term.append(event)

    def store_fact(self, key: str, value: Any) -> None:
        self.long_term[key] = value

    def recall_fact(self, key: str) -> Optional[Any]:
        return self.long_term.get(key)

    def recent_context(self, limit: int = 5) -> List[Dict[str, Any]]:
        return list(self.short_term)[-limit:]

    def seed_defaults(self) -> None:
        self.store_fact(
            "architecture_principle",
            "AGI loop: perception -> memory -> reasoning -> goal selection -> planning -> action -> feedback -> metacognition",
        )
        self.store_fact(
            "project_modules",
            [
                "global_workspace",
                "perception",
                "memory",
                "reasoning",
                "planning",
                "language",
                "motivation",
                "agency",
                "metacognition",
            ],
        )
PY

cat > "$PROJECT_DIR/reasoning/reasoning_module.py" <<'PY'
from __future__ import annotations

from typing import Any, Dict, List


class ReasoningModule:
    """Basic reasoning / consistency check layer."""

    def analyze(
        self,
        goal: str,
        perception: Dict[str, Any],
        recent_context: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        confidence = 0.55
        warnings: List[str] = []
        inferred_subgoals: List[str] = []

        if not goal:
            warnings.append("goal_missing")
            confidence = 0.2
        if perception.get("intent") == "empty":
            warnings.append("empty_input")
            confidence = min(confidence, 0.2)
        if "answer_user:" in goal:
            inferred_subgoals.extend(["retrieve_relevant_memory", "compose_response"])
            confidence = max(confidence, 0.65)
        if "execute_request:" in goal:
            inferred_subgoals.extend(["decompose_request", "draft_execution_plan"])
            confidence = max(confidence, 0.7)

        if recent_context:
            last = recent_context[-1]
            if last.get("type") == "failure":
                warnings.append("previous_cycle_failed")
                inferred_subgoals.append("adapt_strategy")
                confidence -= 0.1

        return {
            "goal": goal,
            "confidence": round(max(0.0, min(1.0, confidence)), 2),
            "warnings": warnings,
            "inferred_subgoals": inferred_subgoals,
        }
PY

cat > "$PROJECT_DIR/motivation/motivation_module.py" <<'PY'
from __future__ import annotations

from typing import Dict, List, Optional


class MotivationModule:
    """Manages external and internal goals."""

    def __init__(self) -> None:
        self.goal_queue: List[str] = []

    def add_goal(self, goal: Optional[str]) -> None:
        if goal and goal not in self.goal_queue:
            self.goal_queue.append(goal)

    def ingest_perception(self, perception: Dict[str, object]) -> None:
        self.add_goal(perception.get("suggested_goal"))

    def ingest_metacognition(self, recommendations: List[str]) -> None:
        for goal in recommendations:
            self.add_goal(goal)

    def next_goal(self) -> Optional[str]:
        if not self.goal_queue:
            return None
        return self.goal_queue.pop(0)
PY

cat > "$PROJECT_DIR/planning/planning_module.py" <<'PY'
from __future__ import annotations

from typing import Any, Dict, List


class PlanningModule:
    """Turns goals into executable steps."""

    def make_plan(self, goal: str, reasoning: Dict[str, Any]) -> List[Dict[str, str]]:
        steps: List[Dict[str, str]] = []
        inferred = reasoning.get("inferred_subgoals", [])

        if goal.startswith("answer_user:"):
            question = goal.split(":", 1)[1].strip()
            steps = [
                {"action": "search_memory", "input": question},
                {"action": "compose_answer", "input": question},
            ]
        elif goal.startswith("execute_request:"):
            task = goal.split(":", 1)[1].strip()
            steps = [
                {"action": "decompose_task", "input": task},
                {"action": "compose_answer", "input": f"Plan for task: {task}"},
            ]
        elif goal == "stabilize_context":
            steps = [{"action": "compose_answer", "input": "Пустой ввод. Нужна цель или запрос."}]
        else:
            steps = [
                {"action": "reflect", "input": goal},
                {"action": "compose_answer", "input": f"Internal analysis for: {goal}"},
            ]

        for subgoal in inferred:
            steps.append({"action": "note_subgoal", "input": subgoal})

        return steps
PY

cat > "$PROJECT_DIR/language/language_module.py" <<'PY'
from __future__ import annotations

from typing import Any, Dict, List


class LanguageModule:
    """Lightweight text generation layer for the bootstrap version."""

    def compose_answer(
        self,
        user_prompt: str,
        memory_hits: List[str],
        reasoning: Dict[str, Any],
        plan: List[Dict[str, str]],
    ) -> str:
        lines: List[str] = []
        lines.append(f"Цель: {user_prompt}")

        if memory_hits:
            lines.append("Память:")
            lines.extend(f"- {item}" for item in memory_hits)

        warnings = reasoning.get("warnings", [])
        if warnings:
            lines.append(f"Предупреждения: {', '.join(warnings)}")

        lines.append(f"Уверенность: {reasoning.get('confidence', 0.0)}")
        lines.append("План:")
        for idx, step in enumerate(plan, start=1):
            lines.append(f"{idx}. {step['action']} -> {step['input']}")

        return "\n".join(lines)
PY

cat > "$PROJECT_DIR/agency/agency_module.py" <<'PY'
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
PY

cat > "$PROJECT_DIR/metacognition/metacognition_module.py" <<'PY'
from __future__ import annotations

from typing import Any, Dict, List


class MetacognitionModule:
    """Monitors cycles, logs issues, proposes internal improvements."""

    def __init__(self) -> None:
        self.history: List[Dict[str, Any]] = []

    def review(
        self,
        perception: Dict[str, Any],
        reasoning: Dict[str, Any],
        action_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        recommendations: List[str] = []

        if reasoning.get("confidence", 0) < 0.4:
            recommendations.append("improve_reasoning_on_low_confidence_cases")
        if not action_result.get("answer"):
            recommendations.append("improve_response_generation")
        if perception.get("intent") == "empty":
            recommendations.append("clarify_user_goal")

        report = {
            "cycle_ok": action_result.get("status") == "ok",
            "recommendations": recommendations,
            "confidence": reasoning.get("confidence", 0.0),
        }
        self.history.append(report)
        return report
PY

cat > "$PROJECT_DIR/main.py" <<'PY'
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Dict

from agency.agency_module import AgencyModule
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
        self.workspace = GlobalWorkspace()
        self.perception = PerceptionModule()
        self.memory = MemoryModule()
        self.memory.seed_defaults()
        self.reasoning = ReasoningModule()
        self.motivation = MotivationModule()
        self.planning = PlanningModule()
        self.language = LanguageModule()
        self.agency = AgencyModule(language=self.language)
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
PY

cat > "$PROJECT_DIR/requirements.txt" <<'TXT'
# Minimal bootstrap version uses only Python standard library.
# Add optional integrations later: transformers, faiss-cpu, chromadb, networkx, qiskit, qiskit-ibm-runtime, openai, langchain.
TXT

cat > "$PROJECT_DIR/README.md" <<'TXT'
# Tolik Core Bootstrap

Minimal runnable AGI skeleton with:
- global workspace
- perception
- memory
- reasoning
- motivation
- planning
- agency
- metacognition
- language layer

Run:

python3 main.py
TXT

printf '\nCreated project: %s\n' "$PROJECT_DIR"
printf 'Run it with:\n  cd %s && python3 main.py\n' "$PROJECT_DIR"
