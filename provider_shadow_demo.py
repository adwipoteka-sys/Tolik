from __future__ import annotations

import argparse
from pathlib import Path

from autonomous_agi import _apply_provider_rollout
from main import build_system
from motivation.operator_charter import load_charter


def main() -> None:
    parser = argparse.ArgumentParser(description="Demonstrate shadow traffic, drift monitoring, and automatic fallback demotion.")
    parser.add_argument("--runtime-dir", type=Path, default=Path("runtime_v3138_shadow_drift_demo"))
    parser.add_argument("--charter", type=Path, default=Path("configs/operator_charter.shadow_drift.example.json"))
    parser.add_argument("--provider-catalog", type=Path, default=Path("configs/provider_catalog.shadow_drift.example.json"))
    args = parser.parse_args()

    system = build_system(args.runtime_dir)
    charter = load_charter(args.charter)
    summary = _apply_provider_rollout(system, charter=charter, provider_catalog_path=args.provider_catalog)

    cloud_llm = system["cloud_llm"]
    quantum_solver = system["quantum_solver"]
    ledger = system["ledger"]

    print("Shadow traffic + drift monitoring demo")
    print(f"Runtime ledger: {args.runtime_dir / 'ledger'}")
    print(f"Initial rollout summary: {summary}")

    print("\n[cloud_llm] triggering rollout-time drift...")
    for index, prompt in enumerate([
        "Alpha. Beta. Gamma.",
        "Delta. Epsilon. Zeta.",
        "Eta. Theta. Iota.",
    ], start=1):
        result = cloud_llm.generate(prompt, task="summarize")
        print(f"  call {index}: {result}")

    print("\n[quantum_solver] triggering rollout-time drift...")
    q1 = quantum_solver.factorize(21)
    q2 = quantum_solver.solve_optimization({"values": [5.0, 1.0, 3.0]})
    q3 = quantum_solver.factorize(21)
    print(f"  factorize #1: {q1}")
    print(f"  optimize #1: {q2}")
    print(f"  factorize #2: {q3}")

    print("\nFinal adapter summaries")
    print(f"  cloud_llm: {cloud_llm.summary()}")
    print(f"  quantum_solver: {quantum_solver.summary()}")

    print("\nLedger stats")
    print(f"  shadow runs: {len(ledger.load_interface_shadow_runs())}")
    print(f"  drift reports: {len(ledger.load_interface_drift_reports())}")
    print(f"  demotion decisions: {ledger.load_interface_demotion_decisions()}")


if __name__ == "__main__":
    main()
