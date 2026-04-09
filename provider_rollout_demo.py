from __future__ import annotations

import argparse
from pathlib import Path

from autonomous_agi import _apply_provider_rollout
from main import build_system
from motivation.operator_charter import load_charter


def main() -> None:
    parser = argparse.ArgumentParser(description="Demonstrate provider qualification and charter-aware live rollout gate.")
    parser.add_argument("--runtime-dir", type=Path, default=Path("runtime_v3138_provider_rollout_demo"))
    parser.add_argument("--charter", type=Path, default=Path("configs/operator_charter.live_interfaces.example.json"))
    parser.add_argument("--provider-catalog", type=Path, default=Path("configs/provider_catalog.mock_qualification.example.json"))
    args = parser.parse_args()

    system = build_system(args.runtime_dir)
    charter = load_charter(args.charter)
    cloud_llm = system["cloud_llm"]
    quantum_solver = system["quantum_solver"]

    cloud_llm.set_mode("disabled")
    quantum_solver.set_mode("disabled")

    deferred_cloud = cloud_llm.generate("Alpha. Beta. Gamma.", task="summarize")
    deferred_quantum = quantum_solver.factorize(21)

    summary = _apply_provider_rollout(system, charter=charter, provider_catalog_path=args.provider_catalog)
    cloud_replay = cloud_llm.replay_deferred()
    quantum_replay = quantum_solver.replay_deferred()

    print("Provider qualification + rollout demo")
    print(f"Runtime ledger: {args.runtime_dir / 'ledger'}")
    print(f"Deferred cloud before rollout: {deferred_cloud}")
    print(f"Deferred quantum before rollout: {deferred_quantum}")
    print(f"Rollout summary: {summary}")
    print(f"Cloud replay: {cloud_replay}")
    print(f"Quantum replay: {quantum_replay}")
    print(f"Final cloud summary: {cloud_llm.summary()}")
    print(f"Final quantum summary: {quantum_solver.summary()}")


if __name__ == "__main__":
    main()
