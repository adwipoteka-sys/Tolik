from __future__ import annotations

from tooling.tool_evaluator import ToolEvaluator
from tooling.tool_templates import render_text_summarizer
from tooling.tool_spec import ToolSpec
from tooling.sandbox import RestrictedSandbox


def test_tool_evaluator_accepts_controlled_summarizer() -> None:
    spec = ToolSpec(
        name="generated_text_summarizer_v1",
        capability="text_summarizer",
        description="demo",
        template_name="text_summarizer",
        parameters={"max_sentences": 2},
    )
    source = render_text_summarizer(spec)
    runtime = RestrictedSandbox().load_callable(source)
    report = ToolEvaluator().evaluate(
        tool_name=spec.name,
        capability=spec.capability,
        runtime_callable=runtime,
    )
    assert report.passed is True
    assert report.mean_score >= report.threshold
    assert len(report.cases) == 2
