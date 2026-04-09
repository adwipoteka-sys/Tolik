from __future__ import annotations

from tooling.tool_registry import ToolRegistry
from tooling.tool_spec import GeneratedTool, ToolSpec, ToolValidationReport


def _tool(name: str, capability: str, version: str) -> GeneratedTool:
    return GeneratedTool(
        spec=ToolSpec(
            name=name,
            capability=capability,
            description=name,
            template_name="text_summarizer",
            parameters={},
        ),
        source_code="def run_tool(payload):\n    return {'summary': 'ok', 'source_count': 0, 'sentences_used': 0}\n",
        validation=ToolValidationReport(allowed=True),
        version=version,
    )


def test_candidate_is_inactive_until_promoted() -> None:
    registry = ToolRegistry()
    tool = _tool("generated_text_summarizer_v1", "text_summarizer", "v3.119")
    registry.register_candidate(tool, lambda payload: {"summary": "ok", "source_count": 1, "sentences_used": 1})

    assert registry.has_candidate("text_summarizer") is True
    assert registry.has_capability("text_summarizer") is False

    promoted = registry.promote_candidate("text_summarizer")
    assert promoted.name == tool.name
    assert registry.has_capability("text_summarizer") is True


def test_canary_does_not_replace_stable_until_finalized() -> None:
    registry = ToolRegistry()
    stable = _tool("generated_text_summarizer_v1", "text_summarizer", "v3.119")
    canary = _tool("generated_text_summarizer_v2", "text_summarizer", "v3.119")

    registry.register(stable, lambda payload: {"summary": "stable", "source_count": 1, "sentences_used": 1})
    registry.register_candidate(canary, lambda payload: {"summary": "canary", "source_count": 1, "sentences_used": 1})
    promoted = registry.promote_candidate_to_canary("text_summarizer")

    assert promoted.name == canary.name
    assert registry.execute_by_capability("text_summarizer", {"texts": ["x"]})["summary"] == "stable"
    assert registry.execute_canary("text_summarizer", {"texts": ["x"]})["summary"] == "canary"

    finalized = registry.finalize_canary("text_summarizer")
    assert finalized.name == canary.name
    assert registry.execute_by_capability("text_summarizer", {"texts": ["x"]})["summary"] == "canary"


def test_registry_rolls_back_to_previous_active_tool() -> None:
    registry = ToolRegistry()
    first = _tool("generated_text_summarizer_v1", "text_summarizer", "v3.119")
    second = _tool("generated_text_summarizer_v2", "text_summarizer", "v3.119")

    registry.register(first, lambda payload: {"summary": "first", "source_count": 1, "sentences_used": 1})
    registry.register_candidate(second, lambda payload: {"summary": "second", "source_count": 1, "sentences_used": 1})
    registry.promote_candidate("text_summarizer")

    restored = registry.rollback_capability("text_summarizer")
    assert restored is not None
    assert restored.name == first.name
    assert registry.execute_by_capability("text_summarizer", {"texts": ["x"]})["summary"] == "first"
