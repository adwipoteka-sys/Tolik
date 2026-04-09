from __future__ import annotations

from policy.tool_proposal_guard import ToolProposal, ToolProposalGuard


def test_tool_proposal_guard_rejects_unsafe_operations():
    guard = ToolProposalGuard()
    proposal = ToolProposal(
        capability_id="x",
        problem_signature="unsafe",
        interface_spec={"code": "import subprocess\nsubprocess.run(['rm','-rf','/'])"},
        test_spec=[],
    )
    review = guard.review(proposal)
    assert review.allowed is False
    assert proposal.status == "rejected"


def test_tool_proposal_guard_allows_shadow_safe_proposals():
    guard = ToolProposalGuard()
    proposal = ToolProposal(
        capability_id="x",
        problem_signature="safe",
        interface_spec={"code": "def helper(x):\n    return x + 1"},
        test_spec=[{"input": 1, "expected": 2}],
    )
    review = guard.review(proposal)
    assert review.allowed is True
    assert proposal.status == "approved_for_shadow"
