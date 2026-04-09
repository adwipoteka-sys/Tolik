from __future__ import annotations

import json
from pathlib import Path

from interfaces.adapter_schema import AdapterSafetyPolicy
from interfaces.cloud_llm import CloudLLMClient
from interfaces.quantum_solver import QuantumSolver
from main import build_system
from memory.goal_ledger import GoalLedger
from motivation.operator_charter import OperatorCharter
from experiments.experiment_sources import build_future_interface_candidates


def test_cloud_llm_live_replay_redacts_and_audits(tmp_path: Path) -> None:
    ledger = GoalLedger(tmp_path / "ledger")
    adapter = CloudLLMClient(mode="disabled", ledger=ledger)
    deferred = adapter.generate("Email alice@example.com. Alpha. Beta.", task="summarize")
    assert deferred["status"] == "deferred"

    captured: list[str] = []

    def _transport(prompt: str, **_: object) -> dict[str, object]:
        captured.append(prompt)
        return {"text": "Alpha. Beta.", "provider": "mock_live"}

    adapter.policy.allow_live_calls = True
    adapter.set_mode("live", provider="mock_live", live_transport=_transport)
    replayed = adapter.replay_deferred()
    assert len(replayed) == 1
    assert replayed[0]["status"] == "ok"
    assert captured and "[redacted-email]" in captured[0]
    assert "alice@example.com" not in captured[0]

    calls = ledger.load_interface_calls()
    assert len(calls) >= 2
    states = ledger.load_interface_states()
    assert any(state["adapter_name"] == "cloud_llm" and state["live_ready"] for state in states)


def test_quantum_solver_policy_validates_problem_size_and_value() -> None:
    solver = QuantumSolver(
        mode="live",
        provider="mock_live",
        policy=AdapterSafetyPolicy(allow_live_calls=True, max_problem_size=3, max_numeric_value=100),
        live_transport=lambda operation, payload: {"factors": [3, 7], "provider": "mock_live"}
        if operation == "factorize"
        else {"best_value": 1, "best_index": 1, "provider": "mock_live"},
    )

    too_large_value = solver.factorize(101)
    assert too_large_value["status"] == "rejected"
    assert too_large_value["reason"] == "value_too_large"

    too_large_problem = solver.solve_optimization({"values": [5, 3, 4, 1]})
    assert too_large_problem["status"] == "rejected"
    assert too_large_problem["reason"] == "problem_too_large"

    ok = solver.factorize(21)
    assert ok["status"] == "ok"
    assert ok["mode"] == "live"
    assert ok["factors"] == [3, 7]


def test_build_system_with_live_interface_config_unblocks_future_campaigns(tmp_path: Path) -> None:
    interfaces_config = tmp_path / "interfaces.json"
    interfaces_config.write_text(
        json.dumps(
            {
                "cloud_llm": {
                    "mode": "live",
                    "provider": "mock_live",
                    "policy": {"allow_live_calls": True},
                },
                "quantum_solver": {
                    "mode": "live",
                    "provider": "mock_live",
                    "policy": {"allow_live_calls": True},
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    system = build_system(tmp_path / "runtime", interfaces_config_path=interfaces_config)
    workspace = system["workspace"]
    board = system["experiment_board"]
    scheduler = system["experiment_scheduler"]
    quantum_solver = system["quantum_solver"]
    cloud_llm = system["cloud_llm"]

    state = workspace.get_state()
    assert state["future_interfaces"]["quantum_solver"] == "live"
    assert state["future_interfaces"]["cloud_llm"] == "live"
    assert state["interface_adapter_summary"]["quantum_solver"]["live_ready"] is True
    assert state["interface_adapter_summary"]["cloud_llm"]["live_ready"] is True

    charter = OperatorCharter(allow_cloud_llm=True, allow_quantum_solver=True)
    candidates = build_future_interface_candidates(scheduler=scheduler, quantum_solver=quantum_solver, cloud_llm=cloud_llm)
    quantum_candidate = next(candidate for candidate in candidates if "quantum_solver" in candidate.required_capabilities)
    campaign, ticket = board.stage_selected_execution(
        quantum_candidate,
        current_cycle=1,
        context=state,
        charter=charter,
    )
    assert campaign.status.value == "selected"
    assert ticket["materialize_now"] is True
