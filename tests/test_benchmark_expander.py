from __future__ import annotations

from benchmarks.benchmark_expander import BenchmarkExpander
from metacognition.failure_miner import FailureCase
from tooling.tool_evaluator import ToolEvaluator
from tooling.tool_spec import ToolSpec
from tooling.tool_templates import render_text_summarizer
from tooling.sandbox import RestrictedSandbox


def _failure_case() -> FailureCase:
    return FailureCase(
        case_id="failure_tool_1",
        goal_id="tool_goal_1",
        capability="text_summarizer",
        tool_name="generated_text_summarizer_v2",
        tool_version="v3.120",
        rollout_stage="canary",
        payload={
            "texts": [
                "Stable output should survive rollback.",
                "   ",
                "Blank inputs must be handled.",
            ],
            "max_sentences": 2,
        },
        input_shape={"texts_count": 3, "nonempty_texts_count": 2, "blank_texts_count": 1, "requested_max_sentences": 2},
        violation_types=["source_count_mismatch", "sentence_limit_exceeded"],
        expected={"source_count": 2, "max_sentences": 2, "required_summary_nonempty": True},
        actual={"source_count": 3, "sentences_used": 3},
        rollback_target="generated_text_summarizer_v1",
        signature="text_summarizer|source_count_mismatch+sentence_limit_exceeded|blank_inputs|limit=2",
        notes=["ignore_blank_inputs"],
    )


def test_benchmark_expander_turns_failure_into_regression_case() -> None:
    expander = BenchmarkExpander()
    case = expander.expand_from_failure(_failure_case())
    assert case.case_id.startswith("regression_text_summarizer_")
    assert case.expected["source_count"] == 2
    assert len(expander.get_cases("text_summarizer")) == 1


def test_expanded_regression_case_fails_buggy_tool_and_passes_patch() -> None:
    expander = BenchmarkExpander()
    expander.expand_from_failure(_failure_case())
    evaluator = ToolEvaluator(benchmark_expander=expander)

    buggy_spec = ToolSpec(
        name="generated_text_summarizer_v2",
        capability="text_summarizer",
        description="buggy",
        template_name="text_summarizer",
        parameters={"variant": "counts_raw_inputs"},
    )
    patched_spec = ToolSpec(
        name="generated_text_summarizer_v3",
        capability="text_summarizer",
        description="patched",
        template_name="text_summarizer",
        parameters={"variant": "blank_input_guard", "max_sentences": 3},
    )
    sandbox = RestrictedSandbox()
    buggy_runtime = sandbox.load_callable(render_text_summarizer(buggy_spec))
    patched_runtime = sandbox.load_callable(render_text_summarizer(patched_spec))

    buggy_report = evaluator.evaluate(tool_name=buggy_spec.name, capability="text_summarizer", runtime_callable=buggy_runtime)
    patched_report = evaluator.evaluate(tool_name=patched_spec.name, capability="text_summarizer", runtime_callable=patched_runtime)

    assert buggy_report.passed is False
    assert patched_report.passed is True
    regression_eval = next(case for case in patched_report.cases if case.case_id.startswith("regression_text_summarizer_"))
    assert regression_eval.passed is True
