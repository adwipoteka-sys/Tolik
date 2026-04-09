from __future__ import annotations

import argparse
from pathlib import Path

from autonomous_agi import _apply_provider_rollout
from main import build_system
from motivation.operator_charter import load_charter


def main() -> None:
    parser = argparse.ArgumentParser(description="Demonstrate task-aware shadow consensus scoring for harder LLM tasks.")
    parser.add_argument("--runtime-dir", type=Path, default=Path("runtime_v3140_shadow_consensus_demo"))
    parser.add_argument("--charter", type=Path, default=Path("configs/operator_charter.shadow_consensus.example.json"))
    parser.add_argument("--provider-catalog", type=Path, default=Path("configs/provider_catalog.shadow_consensus.example.json"))
    args = parser.parse_args()

    system = build_system(args.runtime_dir)
    charter = load_charter(args.charter)
    summary = _apply_provider_rollout(system, charter=charter, provider_catalog_path=args.provider_catalog)

    cloud_llm = system["cloud_llm"]
    ledger = system["ledger"]

    print("Shadow consensus scoring demo")
    print(f"Runtime ledger: {args.runtime_dir / 'ledger'}")
    print(f"Initial rollout summary: {summary}")
    print(f"Initial provider: {cloud_llm.provider}")

    prompt = "Urgent fraud breach detected in production account."
    for index in range(1, 4):
        result = cloud_llm.generate(prompt, task="classify_risk")
        print(f"\ncall {index}: {result}")
        monitoring = result.get("post_promotion_monitoring", {})
        if monitoring:
            print("  shadow consensus:", monitoring.get("shadow_consensus"))
            print("  drift report:", monitoring.get("drift_report"))
            if monitoring.get("demotion") is not None:
                print("  demotion:", monitoring.get("demotion"))

    print("\nFinal provider:", cloud_llm.provider)
    print("Consensus records:", len(ledger.load_interface_shadow_consensus()))
    print("Demotion decisions:", ledger.load_interface_demotion_decisions())


if __name__ == "__main__":
    main()
