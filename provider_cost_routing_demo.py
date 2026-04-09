from __future__ import annotations

import json
from pathlib import Path

from autonomous_agi import _apply_provider_rollout
from main import build_system
from motivation.operator_charter import load_charter


def main() -> None:
    root = Path(__file__).resolve().parent
    runtime_dir = root / "runtime_v3141_cost_routing_demo"
    charter_path = root / "configs" / "operator_charter.canary_rollout.example.json"
    catalog_path = root / "configs" / "provider_catalog.canary_rollout.example.json"

    system = build_system(runtime_dir)
    charter = load_charter(charter_path)
    summary = _apply_provider_rollout(system, charter=charter, provider_catalog_path=catalog_path)

    print("Tolik v3.141 — cost-aware fallback routing demo")
    print("Cloud rollout decision:")
    print(json.dumps(summary["decisions"]["cloud_llm"], ensure_ascii=False, indent=2))
    print("Quantum rollout decision:")
    print(json.dumps(summary["decisions"]["quantum_solver"], ensure_ascii=False, indent=2))

    ledger = system["ledger"]
    routing_records = ledger.load_interface_provider_routing()
    print(f"Routing audit records: {len(routing_records)}")
    for record in routing_records:
        print(
            f"  {record['adapter_name']} | role={record['role']} | "
            f"selected={record['selected_provider']} | strategy={record['strategy']}"
        )


if __name__ == "__main__":
    main()
