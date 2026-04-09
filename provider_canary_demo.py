from __future__ import annotations

import argparse
from pathlib import Path

from autonomous_agi import _apply_provider_rollout
from main import build_system
from motivation.operator_charter import load_charter


def main() -> None:
    parser = argparse.ArgumentParser(description="Demonstrate canary-percent rollout with continuous re-qualification.")
    parser.add_argument("--runtime-dir", type=Path, default=Path("runtime_v3139_canary_demo"))
    parser.add_argument("--charter", type=Path, default=Path("configs/operator_charter.canary_rollout.example.json"))
    parser.add_argument("--provider-catalog", type=Path, default=Path("configs/provider_catalog.canary_rollout.example.json"))
    args = parser.parse_args()

    system = build_system(args.runtime_dir)
    charter = load_charter(args.charter)
    summary = _apply_provider_rollout(system, charter=charter, provider_catalog_path=args.provider_catalog)

    cloud_llm = system["cloud_llm"]
    quantum_solver = system["quantum_solver"]
    ledger = system["ledger"]

    print("Canary rollout + continuous re-qualification demo")
    print(f"Runtime ledger: {args.runtime_dir / 'ledger'}")
    print(f"Initial rollout summary: {summary}")

    print("\n[cloud_llm] promoting safe candidate from canary to full live...")
    for index, prompt in enumerate([
        "Alpha. Beta. Gamma.",
        "Delta. Epsilon. Zeta.",
        "Eta. Theta. Iota.",
        "Kappa. Lambda. Mu.",
    ], start=1):
        result = cloud_llm.generate(prompt, task="summarize")
        print(f"  call {index}: {result}")

    print("\n[quantum_solver] rolling back regressing canary to fallback...")
    q1 = quantum_solver.factorize(21)
    q2 = quantum_solver.solve_optimization({"values": [5.0, 1.0, 3.0]})
    q3 = quantum_solver.factorize(21)
    q4 = quantum_solver.solve_optimization({"values": [5.0, 1.0, 3.0]})
    print(f"  factorize #1: {q1}")
    print(f"  optimize  #1: {q2}")
    print(f"  factorize #2: {q3}")
    print(f"  optimize  #2: {q4}")

    print("\nFinal adapter summaries")
    print(f"  cloud_llm: {cloud_llm.summary()}")
    print(f"  quantum_solver: {quantum_solver.summary()}")

    print("\nLedger stats")
    print(f"  canary samples: {len(ledger.load_interface_canary_samples())}")
    print(f"  requalification reports: {len(ledger.load_interface_requalification_reports())}")
    print(f"  canary decisions: {ledger.load_interface_canary_decisions()}")
    print(f"  drift reports: {len(ledger.load_interface_drift_reports())}")


if __name__ == "__main__":
    main()
