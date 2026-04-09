from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any
from uuid import uuid4

_FORBIDDEN = ('exec', 'eval', 'subprocess', 'os.system', 'socket', 'requests.', 'http://', 'https://')


@dataclass(slots=True)
class ToolProposal:
    capability_id: str
    problem_signature: str
    interface_spec: dict[str, Any]
    test_spec: list[dict[str, Any]]
    required_permissions: list[str] = field(default_factory=list)
    risk_level: str = 'medium'
    status: str = 'proposed'
    proposal_id: str = field(default_factory=lambda: f'toolprop_{uuid4().hex[:12]}')

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ShadowValidationResult:
    allowed: bool
    status: str
    reasons: list[str]


class ToolProposalGuard:
    def review(self, proposal: ToolProposal) -> ShadowValidationResult:
        blob = str(proposal.interface_spec) + '\n' + str(proposal.test_spec)
        reasons = [token for token in _FORBIDDEN if token in blob]
        if reasons:
            proposal.status = 'rejected'
            return ShadowValidationResult(False, 'rejected', reasons)
        proposal.status = 'approved_for_shadow'
        return ShadowValidationResult(True, 'approved_for_shadow', [])
