from __future__ import annotations

from integrations.cloud_llm_client import CloudLLMClient, DeferredLLMTask
from integrations.quantum_solver import DeferredQuantumTask, QuantumSolver


def test_quantum_solver_falls_back_or_defers():
    solver = QuantumSolver(available=False)
    out = solver.solve({"values": [5.0, 1.0, 3.0], "allow_fallback": True})
    assert isinstance(out, dict)
    assert out["solver"] == "classical_fallback"
    deferred = solver.solve({"capability_id": "quantum_reasoning", "allow_fallback": False})
    assert isinstance(deferred, DeferredQuantumTask)


def test_cloud_llm_client_uses_stub_or_defers():
    client = CloudLLMClient(available=False)
    out = client.complete("hello", {"allow_stub": True})
    assert isinstance(out, str)
    deferred = client.complete("hello", {"capability_id": "qa", "allow_stub": False})
    assert isinstance(deferred, DeferredLLMTask)
