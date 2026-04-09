from __future__ import annotations

from main import run_demo


def test_v3122_demo_learns_and_reuses_strategy(tmp_path, capsys) -> None:
    runtime_dir = tmp_path / "runtime"
    run_demo(cycles=6, runtime_dir=runtime_dir)
    captured = capsys.readouterr().out

    assert "Strategy patterns:" in captured
    assert "Reuse learned strategy for text_summarizer edge canary" in captured
    assert "blank_input_guard" in captured
    assert "generated_text_summarizer_v4" in captured
