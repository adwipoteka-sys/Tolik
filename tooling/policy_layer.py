from __future__ import annotations

import ast

from tooling.tool_spec import ToolValidationReport


class PolicyLayer:
    """Static guardrail for generated tools.

    The policy intentionally allows only small, pure-Python functions that do not touch
    the network, filesystem, subprocesses, dynamic evaluation, or reflection-heavy APIs.
    """

    FORBIDDEN_NAMES = {
        "eval",
        "exec",
        "compile",
        "open",
        "input",
        "__import__",
        "globals",
        "locals",
        "vars",
        "getattr",
        "setattr",
        "delattr",
        "breakpoint",
        "help",
        "dir",
    }
    FORBIDDEN_ATTRIBUTES = {
        "system",
        "popen",
        "remove",
        "unlink",
        "rmdir",
        "mkdir",
        "makedirs",
        "rename",
        "chmod",
        "chown",
        "connect",
        "send",
        "recv",
        "request",
        "urlopen",
        "loads",
        "dumps",
    }
    FORBIDDEN_MODULE_TOKENS = {
        "os",
        "sys",
        "subprocess",
        "socket",
        "pathlib",
        "requests",
        "urllib",
        "http",
        "shutil",
        "pickle",
        "marshal",
        "importlib",
        "ctypes",
        "multiprocessing",
        "threading",
        "asyncio",
    }

    def __init__(self, max_lines: int = 120, max_ast_nodes: int = 500) -> None:
        self.max_lines = max_lines
        self.max_ast_nodes = max_ast_nodes

    def validate_source(self, source_code: str) -> ToolValidationReport:
        violations: list[str] = []
        warnings: list[str] = []

        if len(source_code.splitlines()) > self.max_lines:
            violations.append(f"source_too_long>{self.max_lines}")

        try:
            tree = ast.parse(source_code)
        except SyntaxError as exc:
            return ToolValidationReport(False, [f"syntax_error:{exc.msg}"], [])

        node_count = sum(1 for _ in ast.walk(tree))
        if node_count > self.max_ast_nodes:
            violations.append(f"ast_too_large>{self.max_ast_nodes}")

        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
        if len(functions) != 1:
            violations.append("expected_exactly_one_function")
        elif functions[0].name != "run_tool":
            violations.append("entrypoint_must_be_run_tool")

        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                violations.append("imports_are_forbidden")
            elif isinstance(node, ast.Call):
                call_name = self._resolve_call_name(node.func)
                if call_name in self.FORBIDDEN_NAMES:
                    violations.append(f"forbidden_call:{call_name}")
            elif isinstance(node, ast.Attribute):
                if node.attr.startswith("__"):
                    violations.append("dunder_attribute_forbidden")
                if node.attr in self.FORBIDDEN_ATTRIBUTES:
                    violations.append(f"forbidden_attribute:{node.attr}")
            elif isinstance(node, ast.Name):
                if node.id.startswith("__"):
                    violations.append("dunder_name_forbidden")
                if node.id in self.FORBIDDEN_MODULE_TOKENS:
                    violations.append(f"forbidden_module_token:{node.id}")
            elif isinstance(node, ast.With):
                warnings.append("with_statement_present")
            elif isinstance(node, ast.Try):
                violations.append("try_statements_forbidden")
            elif isinstance(node, ast.ClassDef):
                violations.append("class_definitions_forbidden")

        return ToolValidationReport(not violations, sorted(set(violations)), sorted(set(warnings)))

    def _resolve_call_name(self, node: ast.AST) -> str | None:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return node.attr
        return None
