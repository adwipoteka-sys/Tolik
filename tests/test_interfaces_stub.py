from __future__ import annotations

from interfaces.cloud_llm import CloudLLMClient
from interfaces.quantum_solver import QuantumSolver


def test_quantum_solver_stub_and_simulated_modes() -> None:
    stub = QuantumSolver(mode="stub")
    assert stub.solve_optimization({"values": [7, 2, 5]})["best_value"] == 2

    disabled = QuantumSolver(mode="disabled")
    deferred = disabled.factorize(21)
    assert deferred["status"] == "deferred"
    assert disabled.deferred_queue


def test_cloud_llm_stub_and_disabled_modes() -> None:
    client = CloudLLMClient(mode="simulated")
    result = client.generate("Alpha. Beta. Gamma.", task="summarize")
    assert result["text"].count(".") <= 2

    disabled = CloudLLMClient(mode="disabled")
    deferred = disabled.generate("Hello")
    assert deferred["status"] == "deferred"
    assert disabled.deferred_requests
