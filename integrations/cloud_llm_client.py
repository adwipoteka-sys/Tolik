from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any
from uuid import uuid4


@dataclass(slots=True)
class DeferredLLMTask:
    capability_id: str
    prompt: str
    context: dict[str, Any] | None = None
    task_id: str = field(default_factory=lambda: f'dllm_{uuid4().hex[:12]}')
    backend: str = 'cloud_llm'

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class LocalStubLLMClient:
    def complete(self, prompt: str, context: dict[str, Any] | None = None) -> str:
        if context and context.get('summary'):
            return str(context['summary'])
        return f'STUB:{prompt[:80]}'


class CloudLLMClient:
    def __init__(self, available: bool = False) -> None:
        self.available = available
        self.stub = LocalStubLLMClient()

    def complete(self, prompt: str, context: dict[str, Any] | None = None) -> str | DeferredLLMTask:
        if not self.available:
            if context and context.get('allow_stub', True):
                return self.stub.complete(prompt, context)
            return DeferredLLMTask(capability_id=str(context.get('capability_id', 'unknown')) if context else 'unknown', prompt=prompt, context=context)
        return f'LIVE:{prompt[:120]}'
