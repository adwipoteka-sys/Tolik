from __future__ import annotations

from collections.abc import Callable
from typing import Any


class RestrictedSandbox:
    """Loads approved tool code into a minimal execution environment."""

    SAFE_BUILTINS = {
        "abs": abs,
        "all": all,
        "any": any,
        "bool": bool,
        "dict": dict,
        "enumerate": enumerate,
        "float": float,
        "int": int,
        "isinstance": isinstance,
        "len": len,
        "list": list,
        "max": max,
        "min": min,
        "range": range,
        "round": round,
        "set": set,
        "sorted": sorted,
        "str": str,
        "sum": sum,
        "tuple": tuple,
        "ValueError": ValueError,
        "TypeError": TypeError,
    }

    def load_callable(self, source_code: str) -> Callable[[dict[str, Any]], dict[str, Any]]:
        globals_dict = {"__builtins__": dict(self.SAFE_BUILTINS)}
        locals_dict: dict[str, Any] = {}
        exec(source_code, globals_dict, locals_dict)
        func = locals_dict.get("run_tool") or globals_dict.get("run_tool")
        if not callable(func):
            raise ValueError("Generated tool must define callable run_tool(payload).")
        return func
