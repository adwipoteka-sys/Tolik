from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from typing import Any

from tooling.tool_spec import GeneratedTool


class ToolRegistry:
    """Stores generated tools across candidate, canary, stable, and rollback stages."""

    def __init__(self) -> None:
        self._tools_by_name: dict[str, GeneratedTool] = {}
        self._callables_by_name: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {}
        self._stable_by_capability: dict[str, str] = {}
        self._candidate_by_capability: dict[str, str] = {}
        self._canary_by_capability: dict[str, str] = {}
        self._rollback_stack: dict[str, list[str]] = defaultdict(list)

    def register(self, tool: GeneratedTool, runtime_callable: Callable[[dict[str, Any]], dict[str, Any]]) -> None:
        """Compatibility helper: register and immediately expose as stable."""
        self.register_stable(tool, runtime_callable)

    def register_stable(
        self,
        tool: GeneratedTool,
        runtime_callable: Callable[[dict[str, Any]], dict[str, Any]],
    ) -> None:
        self._tools_by_name[tool.name] = tool
        self._callables_by_name[tool.name] = runtime_callable
        previous_stable = self._stable_by_capability.get(tool.capability)
        if previous_stable is not None and previous_stable != tool.name:
            self._rollback_stack[tool.capability].append(previous_stable)
        self._stable_by_capability[tool.capability] = tool.name

    def register_candidate(
        self,
        tool: GeneratedTool,
        runtime_callable: Callable[[dict[str, Any]], dict[str, Any]],
    ) -> None:
        self._tools_by_name[tool.name] = tool
        self._callables_by_name[tool.name] = runtime_callable
        self._candidate_by_capability[tool.capability] = tool.name

    def discard_candidate(self, capability: str) -> GeneratedTool | None:
        candidate_name = self._candidate_by_capability.pop(capability, None)
        if candidate_name is None:
            return None
        return self._tools_by_name[candidate_name]

    def promote_candidate(self, capability: str) -> GeneratedTool:
        """Compatibility shortcut: move candidate directly to stable."""
        candidate_name = self._candidate_by_capability.pop(capability)
        previous_stable = self._stable_by_capability.get(capability)
        if previous_stable is not None and previous_stable != candidate_name:
            self._rollback_stack[capability].append(previous_stable)
        self._stable_by_capability[capability] = candidate_name
        return self._tools_by_name[candidate_name]

    def promote_candidate_to_canary(self, capability: str) -> GeneratedTool:
        candidate_name = self._candidate_by_capability.pop(capability)
        self._canary_by_capability[capability] = candidate_name
        return self._tools_by_name[candidate_name]

    def finalize_canary(self, capability: str) -> GeneratedTool:
        canary_name = self._canary_by_capability.pop(capability)
        previous_stable = self._stable_by_capability.get(capability)
        if previous_stable is not None and previous_stable != canary_name:
            self._rollback_stack[capability].append(previous_stable)
        self._stable_by_capability[capability] = canary_name
        return self._tools_by_name[canary_name]

    def discard_canary(self, capability: str) -> GeneratedTool | None:
        canary_name = self._canary_by_capability.pop(capability, None)
        if canary_name is None:
            return None
        return self._tools_by_name[canary_name]

    def rollback_canary(self, capability: str) -> GeneratedTool | None:
        self._canary_by_capability.pop(capability, None)
        stable_name = self._stable_by_capability.get(capability)
        if stable_name is None:
            return None
        return self._tools_by_name[stable_name]

    def rollback_capability(self, capability: str) -> GeneratedTool | None:
        if capability in self._canary_by_capability:
            return self.rollback_canary(capability)
        history = self._rollback_stack.get(capability, [])
        if not history:
            stable_name = self._stable_by_capability.get(capability)
            return self._tools_by_name.get(stable_name) if stable_name else None
        restored_name = history.pop()
        self._stable_by_capability[capability] = restored_name
        return self._tools_by_name[restored_name]

    def has_capability(self, capability: str) -> bool:
        return capability in self._stable_by_capability

    def has_candidate(self, capability: str) -> bool:
        return capability in self._candidate_by_capability

    def has_canary(self, capability: str) -> bool:
        return capability in self._canary_by_capability

    def capabilities(self) -> set[str]:
        return set(self._stable_by_capability)

    def candidate_capabilities(self) -> set[str]:
        return set(self._candidate_by_capability)

    def canary_capabilities(self) -> set[str]:
        return set(self._canary_by_capability)

    def candidate_name(self, capability: str) -> str | None:
        return self._candidate_by_capability.get(capability)

    def canary_name(self, capability: str) -> str | None:
        return self._canary_by_capability.get(capability)

    def execute_by_capability(self, capability: str, payload: dict[str, Any]) -> dict[str, Any]:
        tool_name = self._stable_by_capability[capability]
        return self._callables_by_name[tool_name](payload)

    def execute_candidate(self, capability: str, payload: dict[str, Any]) -> dict[str, Any]:
        tool_name = self._candidate_by_capability[capability]
        return self._callables_by_name[tool_name](payload)

    def execute_canary(self, capability: str, payload: dict[str, Any]) -> dict[str, Any]:
        tool_name = self._canary_by_capability[capability]
        return self._callables_by_name[tool_name](payload)

    def get_tool(self, name: str) -> GeneratedTool:
        return self._tools_by_name[name]

    def get_active_tool(self, capability: str) -> GeneratedTool:
        return self._tools_by_name[self._stable_by_capability[capability]]

    def get_canary_tool(self, capability: str) -> GeneratedTool:
        return self._tools_by_name[self._canary_by_capability[capability]]

    def list_tools(self) -> list[str]:
        return sorted(self._tools_by_name)

    def stable_tool_names(self) -> list[str]:
        return sorted(self._stable_by_capability.values())

    def canary_tool_names(self) -> list[str]:
        return sorted(self._canary_by_capability.values())

    def active_tool_names(self) -> list[str]:
        return self.stable_tool_names()
