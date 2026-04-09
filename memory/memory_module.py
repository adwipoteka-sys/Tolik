from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from memory.goal_ledger import GoalLedger
from memory.strategy_memory import StrategyMemory, StrategyPattern
from metacognition.postmortem import PostmortemReport
from tooling.tool_spec import GeneratedTool


@dataclass
class MemoryModule:
    """Lightweight memory for the autonomous-goal and safe-tool demo."""

    goal_ledger: GoalLedger | None = None
    strategy_memory: StrategyMemory | None = None
    working_memory: dict[str, Any] = field(default_factory=dict)
    long_term: dict[str, Any] = field(default_factory=dict)
    experience_log: list[dict[str, Any]] = field(default_factory=list)
    retrieval_policy: str = "exact_only"

    def set_retrieval_policy(self, policy: str) -> None:
        if policy not in {"exact_only", "working_then_semantic_backoff"}:
            raise ValueError(f"Unsupported retrieval policy: {policy}")
        self.retrieval_policy = policy

    def store_fact(self, key: str, info: Any) -> None:
        self.long_term[key] = info

    def store_semantic_promotion(self, fact_key: str, payload: dict[str, Any]) -> None:
        self.long_term[fact_key] = dict(payload)
        self.working_memory["last_semantic_promotion"] = {
            "fact_key": fact_key,
            "summary": payload.get("summary"),
            "support_count": payload.get("support_count"),
        }

    def semantic_facts(self) -> dict[str, Any]:
        return {key: value for key, value in self.long_term.items() if key.startswith("semantic:")}

    def store_tool(self, tool: GeneratedTool) -> None:
        self.long_term[f"tool:{tool.name}"] = tool.to_dict()
        self.working_memory["last_registered_tool"] = {
            "name": tool.name,
            "capability": tool.capability,
        }

    def _normalize(self, text: str) -> str:
        return " ".join(text.lower().replace("_", " ").split())

    def _lookup_in_store(self, store: dict[str, Any], query: str) -> Any | None:
        if query in store:
            return store[query]
        normalized_query = self._normalize(query)
        for key, value in store.items():
            if self._normalize(str(key)) == normalized_query:
                return value
            if isinstance(value, dict):
                aliases = value.get("aliases", [])
                for alias in aliases:
                    if self._normalize(str(alias)) == normalized_query:
                        return value
        return None

    def query_semantic(self, query: str) -> Any | None:
        self.working_memory["last_query"] = query
        result: Any | None = None
        if self.retrieval_policy == "exact_only":
            result = self.long_term.get(query)
        elif self.retrieval_policy == "working_then_semantic_backoff":
            result = self._lookup_in_store(self.working_memory, query)
            if result is None:
                result = self._lookup_in_store(self.long_term, query)
        else:
            raise ValueError(f"Unsupported retrieval policy: {self.retrieval_policy}")
        self.working_memory["last_query_miss"] = result is None
        return result

    def retrieval_confidence(self, query: str) -> float:
        if self.query_semantic(query) is not None:
            return 0.9
        return 0.1

    def get_recent_events(self) -> dict[str, Any]:
        payload = dict(self.working_memory)
        payload["experience_count"] = len(self.experience_log)
        payload["semantic_fact_count"] = len(self.semantic_facts())
        payload["retrieval_policy"] = self.retrieval_policy
        if self.experience_log:
            payload["last_event"] = self.experience_log[-1]
        return payload

    def add_experience(self, event: dict[str, Any]) -> None:
        self.experience_log.append(dict(event))
        self.working_memory["last_event"] = dict(event)
        if "query_miss" in event:
            self.working_memory["last_query_miss"] = bool(event["query_miss"])

    def store_strategy_pattern(self, pattern: StrategyPattern) -> None:
        self.long_term[f"strategy:{pattern.strategy_id}"] = pattern.to_dict()
        self.working_memory["last_strategy_pattern"] = {
            "strategy_id": pattern.strategy_id,
            "signature": pattern.signature,
            "capability": pattern.capability,
        }

    def list_strategy_patterns(self) -> list[dict[str, Any]]:
        if self.strategy_memory is None:
            return []
        return [pattern.to_dict() for pattern in self.strategy_memory.list_patterns()]

    def query_goal_history(self, goal_id: str) -> list[dict[str, Any]]:
        if self.goal_ledger is None:
            return []
        return self.goal_ledger.query_goal_history(goal_id)

    def retrieve_similar_failures(self, tags: list[str]) -> list[dict[str, Any]]:
        if self.goal_ledger is None:
            return []
        return self.goal_ledger.retrieve_similar_failures(tags)

    def store_postmortem(self, report: PostmortemReport) -> None:
        if self.goal_ledger is not None:
            self.goal_ledger.save_postmortem(report)
        self.long_term[f"postmortem:{report.goal_id}"] = report.to_dict()
