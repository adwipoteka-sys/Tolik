from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable
from interfaces.adapter_schema import InterfaceRuntimeSpec
from interfaces.cloud_llm import CloudLLMClient
from interfaces.provider_registry import build_cloud_transport, build_quantum_transport
from interfaces.quantum_solver import QuantumSolver


def load_interface_runtime(
    path: str | Path | None,
    *,
    ledger: Any | None = None,
) -> tuple[QuantumSolver, CloudLLMClient, dict[str, dict[str, Any]]]:
    if path is None:
        quantum_spec = InterfaceRuntimeSpec()
        cloud_spec = InterfaceRuntimeSpec()
    else:
        config_path = Path(path)
        data = json.loads(config_path.read_text(encoding="utf-8"))
        quantum_spec = InterfaceRuntimeSpec.from_dict(data.get("quantum_solver"))
        cloud_spec = InterfaceRuntimeSpec.from_dict(data.get("cloud_llm"))

    quantum_solver = QuantumSolver(
        mode=quantum_spec.mode,
        provider=quantum_spec.provider,
        policy=quantum_spec.policy,
        ledger=ledger,
        live_transport=build_quantum_transport(quantum_spec),
    )
    cloud_llm = CloudLLMClient(
        mode=cloud_spec.mode,
        provider=cloud_spec.provider,
        policy=cloud_spec.policy,
        ledger=ledger,
        live_transport=build_cloud_transport(cloud_spec),
    )
    summaries = {
        "quantum_solver": quantum_solver.summary(),
        "cloud_llm": cloud_llm.summary(),
    }
    return quantum_solver, cloud_llm, summaries
