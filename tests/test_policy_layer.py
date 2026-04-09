from __future__ import annotations

from tooling.policy_layer import PolicyLayer


def test_policy_layer_rejects_imports_and_open() -> None:
    source = """
def run_tool(payload):
    import os
    return open('x.txt').read()
"""
    report = PolicyLayer().validate_source(source)
    assert report.allowed is False
    assert "imports_are_forbidden" in report.violations
    assert "forbidden_call:open" in report.violations
