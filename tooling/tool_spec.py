from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ToolSpec:
    name: str
    capability: str
    description: str
    template_name: str
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ToolValidationReport:
    allowed: bool
    violations: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "violations": list(self.violations),
            "warnings": list(self.warnings),
        }


@dataclass(slots=True)
class GeneratedTool:
    spec: ToolSpec
    source_code: str
    validation: ToolValidationReport
    version: str = "v3.133"

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def capability(self) -> str:
        return self.spec.capability

    def to_dict(self) -> dict[str, Any]:
        return {
            "spec": {
                "name": self.spec.name,
                "capability": self.spec.capability,
                "description": self.spec.description,
                "template_name": self.spec.template_name,
                "parameters": dict(self.spec.parameters),
            },
            "source_code": self.source_code,
            "validation": self.validation.to_dict(),
            "version": self.version,
        }
