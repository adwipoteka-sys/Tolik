from __future__ import annotations

from main import run_demo



def test_v3123_demo_consolidates_semantics_and_schedules_audit(tmp_path, capsys) -> None:
    runtime_dir = tmp_path / "runtime"
    run_demo(cycles=7, runtime_dir=runtime_dir)
    captured = capsys.readouterr().out

    assert "Semantic promotions:" in captured
    assert "text_summarizer__blank_input_guard" in captured
    assert "Skill arena runs:" in captured
    assert "Audit text_summarizer after semantic promotion" in captured
    assert "Semantic consolidation reached: True" in captured
