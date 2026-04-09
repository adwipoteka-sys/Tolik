from __future__ import annotations

import json
from pathlib import Path

from autonomous_agi import _apply_provider_rollout
from main import build_system
from motivation.operator_charter import load_charter


def main() -> None:
    root = Path(__file__).resolve().parent
    runtime_dir = root / "runtime_v3142_anti_flap_demo"
    charter_path = root / "configs" / "operator_charter.canary_rollout.example.json"
    catalog_path = root / "configs" / "provider_catalog.shadow_drift.example.json"

    system = build_system(runtime_dir)
    charter = load_charter(charter_path)

    print("Tolik v3.142 — rollback cooldown + anti-flap demo")
    print(f"Runtime ledger: {runtime_dir / 'ledger'}")

    first = _apply_provider_rollout(system, charter=charter, provider_catalog_path=catalog_path)
    print("\nInitial rollout decision:")
    print(json.dumps(first["decisions"]["quantum_solver"], ensure_ascii=False, indent=2))

    solver = system["quantum_solver"]
    print("\nTriggering first canary rollback...")
    solver.factorize(21)
    solver.solve_optimization({"values": [5.0, 1.0, 3.0]})
    solver.factorize(21)
    first_failure = solver.solve_optimization({"values": [5.0, 1.0, 3.0]})
    print(json.dumps(first_failure.get("canary_rollout", {}), ensure_ascii=False, indent=2))

    second = _apply_provider_rollout(system, charter=charter, provider_catalog_path=catalog_path)
    print("\nImmediate re-rollout after rollback (cooldown should block the regressing provider):")
    print(json.dumps(second["decisions"]["quantum_solver"], ensure_ascii=False, indent=2))

    third = _apply_provider_rollout(system, charter=charter, provider_catalog_path=catalog_path)
    print("\nCanary returns after cooldown expires:")
    print(json.dumps(third["decisions"]["quantum_solver"], ensure_ascii=False, indent=2))

    print("\nTriggering second rollback to arm anti-flap freeze...")
    solver.factorize(21)
    solver.solve_optimization({"values": [5.0, 1.0, 3.0]})
    solver.factorize(21)
    second_failure = solver.solve_optimization({"values": [5.0, 1.0, 3.0]})
    print(json.dumps(second_failure.get("canary_rollout", {}), ensure_ascii=False, indent=2))

    fourth = _apply_provider_rollout(system, charter=charter, provider_catalog_path=catalog_path)
    print("\nRe-rollout while anti-flap freeze is active:")
    print(json.dumps(fourth["decisions"]["quantum_solver"], ensure_ascii=False, indent=2))

    protections = system["ledger"].load_interface_rollout_protections()
    print("\nProtection ledger records:")
    print(json.dumps(protections, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
