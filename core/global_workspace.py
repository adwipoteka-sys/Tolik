from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any


@dataclass
class GlobalWorkspace:
    """Shared state exchanged between cognitive modules."""

    state: dict[str, Any] = field(default_factory=dict)

    def update(self, data: dict[str, Any]) -> None:
        """Merge new data into the global state."""
        self.state.update(data)

    def get_state(self) -> dict[str, Any]:
        """Return a deep copy of the workspace state."""
        return deepcopy(self.state)

    def get(self, key: str, default: Any = None) -> Any:
        return self.state.get(key, default)
