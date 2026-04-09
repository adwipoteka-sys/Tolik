from __future__ import annotations

from benchmarks.skill_arena import SkillArena, SkillArenaCase
from memory.goal_ledger import GoalLedger



def _execute(payload):
    texts = [text for text in payload["texts"] if text.strip()]
    max_sentences = int(payload["max_sentences"])
    sentences = texts[:max_sentences]
    return {
        "summary": " ".join(sentences),
        "source_count": len(texts),
        "sentences_used": len(sentences),
    }



def test_skill_arena_records_successful_run(tmp_path) -> None:
    ledger = GoalLedger(tmp_path / "ledger")
    arena = SkillArena(ledger=ledger)
    run = arena.run(
        capability="text_summarizer",
        label="smoke",
        execute=_execute,
        cases=[
            SkillArenaCase(
                case_id="case1",
                payload={"texts": ["A.", " ", "B."], "max_sentences": 2},
                expected={"source_count": 2, "max_sentences": 2, "required_summary_nonempty": True},
            )
        ],
    )

    assert run.passed is True
    assert run.mean_score == 1.0

    restored = SkillArena(ledger=ledger)
    latest = restored.latest_run("text_summarizer")
    assert latest is not None
    assert latest.run_id == run.run_id
