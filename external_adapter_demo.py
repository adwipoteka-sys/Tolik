from __future__ import annotations

import argparse
from pathlib import Path

from interfaces.cloud_llm import CloudLLMClient
from interfaces.quantum_solver import QuantumSolver
from memory.goal_ledger import GoalLedger


def main() -> None:
    parser = argparse.ArgumentParser(description="Demonstrate Tolik v3.136 safe external adapters with deferred replay and live/mock switch.")
    parser.add_argument("--runtime-dir", type=Path, default=Path("runtime_v3136_interfaces"), help="Directory used for adapter audit logs.")
    args = parser.parse_args()

    ledger = GoalLedger(args.runtime_dir / "ledger")
    quantum_solver = QuantumSolver(mode="disabled", ledger=ledger)
    cloud_llm = CloudLLMClient(mode="disabled", ledger=ledger)

    cloud_result = cloud_llm.generate("Reach me at alice@example.com. Alpha. Beta. Gamma.", task="summarize")
    quantum_result = quantum_solver.factorize(21)
    print("disabled cloud result:", cloud_result)
    print("disabled quantum result:", quantum_result)
    print("initial summaries:", {"cloud_llm": cloud_llm.summary(), "quantum_solver": quantum_solver.summary()})

    cloud_llm.policy.allow_live_calls = True
    cloud_llm.set_mode("live", provider="mock_live", live_transport=lambda prompt, **kwargs: {"text": "Alpha. Beta.", "provider": "demo_mock"})
    quantum_solver.policy.allow_live_calls = True
    quantum_solver.set_mode("live", provider="mock_live", live_transport=lambda operation, payload: {"factors": [3, 7], "provider": "demo_mock"} if operation == "factorize" else {"best_value": 2, "best_index": 1, "provider": "demo_mock"})

    replayed_cloud = cloud_llm.replay_deferred()
    replayed_quantum = quantum_solver.replay_deferred()
    print("replayed cloud:", replayed_cloud)
    print("replayed quantum:", replayed_quantum)
    print("final summaries:", {"cloud_llm": cloud_llm.summary(), "quantum_solver": quantum_solver.summary()})
    print("ledger interface calls:", len(ledger.load_interface_calls()))
    print("ledger interface states:", len(ledger.load_interface_states()))


if __name__ == "__main__":
    main()
