from __future__ import annotations

from pathlib import Path
from typing import List


class LocalToolbox:
    """Safe local tools restricted to the repo root."""

    def __init__(self, repo_root: str) -> None:
        self.repo_root = Path(repo_root).resolve()
        self.notes_dir = self.repo_root / "tolik_core" / "notes"
        self.notes_dir.mkdir(parents=True, exist_ok=True)

    def _resolve(self, rel_path: str) -> Path:
        target = (self.repo_root / rel_path).resolve()
        if not str(target).startswith(str(self.repo_root)):
            raise ValueError("Path escapes repo root")
        return target

    def list_files(self, rel_path: str = ".") -> List[str]:
        target = self._resolve(rel_path)
        if target.is_file():
            return [target.relative_to(self.repo_root).as_posix()]
        if not target.exists():
            return [f"missing: {rel_path}"]

        items = sorted(target.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        return [
            p.relative_to(self.repo_root).as_posix() + ("/" if p.is_dir() else "")
            for p in items[:50]
        ]

    def read_file(self, rel_path: str, max_chars: int = 4000) -> str:
        target = self._resolve(rel_path)
        if not target.exists():
            return f"missing: {rel_path}"
        if target.is_dir():
            return f"is_directory: {rel_path}"
        return target.read_text(encoding="utf-8", errors="replace")[:max_chars]

    def write_note(self, title: str, text: str) -> str:
        safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in title).strip("_") or "note"
        path = self.notes_dir / f"{safe}.md"
        path.write_text(text, encoding="utf-8")
        return path.relative_to(self.repo_root).as_posix()
