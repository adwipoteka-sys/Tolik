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
