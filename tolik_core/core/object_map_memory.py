from __future__ import annotations

from typing import Dict, List, Optional, Tuple


class ObjectMapMemory:
    def __init__(self) -> None:
        self.layout = "unknown"
        self.rows = 0
        self.cols = 0
        self.cells: Dict[Tuple[int, int], str] = {}
        self.target: Optional[Tuple[int, int]] = None
        self.key_pos: Optional[Tuple[int, int]] = None
        self.door_pos: Optional[Tuple[int, int]] = None
        self.has_key: bool = False
        self.door_open: bool = False

    def reset(self, layout: str, rows: int, cols: int) -> None:
        self.layout = layout
        self.rows = rows
        self.cols = cols
        self.cells = {}
        self.target = None
        self.key_pos = None
        self.door_pos = None
        self.has_key = False
        self.door_open = False

    def load_from_fact(self, fact: dict) -> None:
        self.layout = fact.get("layout", self.layout)
        self.rows = fact.get("rows", self.rows)
        self.cols = fact.get("cols", self.cols)
        self.target = tuple(fact["target"]) if fact.get("target") else None
        self.key_pos = tuple(fact["key_pos"]) if fact.get("key_pos") else None
        self.door_pos = tuple(fact["door_pos"]) if fact.get("door_pos") else None
        self.has_key = bool(fact.get("has_key", False))
        self.door_open = bool(fact.get("door_open", False))

        cells: Dict[Tuple[int, int], str] = {}
        for item in fact.get("cells", []):
            cells[(item["i"], item["j"])] = item["v"]
        self.cells = cells

    def to_fact(self) -> dict:
        return {
            "layout": self.layout,
            "rows": self.rows,
            "cols": self.cols,
            "target": list(self.target) if self.target else None,
            "key_pos": list(self.key_pos) if self.key_pos else None,
            "door_pos": list(self.door_pos) if self.door_pos else None,
            "has_key": self.has_key,
            "door_open": self.door_open,
            "cells": [{"i": i, "j": j, "v": v} for (i, j), v in sorted(self.cells.items())],
        }

    def update_from_local(self, obs: dict) -> None:
        self.has_key = bool(obs.get("has_key", False))
        self.door_open = bool(obs.get("door_open", False))

        top_i, top_j = obs["top_left"]
        patch = obs["patch"]

        for di, row in enumerate(patch):
            for dj, ch in enumerate(row):
                i, j = top_i + di, top_j + dj
                if not (0 <= i < self.rows and 0 <= j < self.cols):
                    continue

                if ch == "A":
                    self.cells[(i, j)] = "."
                elif ch == "T":
                    self.cells[(i, j)] = "T"
                    self.target = (i, j)
                elif ch == "K":
                    self.cells[(i, j)] = "K"
                    self.key_pos = (i, j)
                elif ch == "D":
                    self.cells[(i, j)] = "." if self.door_open else "D"
                    self.door_pos = (i, j)
                elif ch in {".", "#"}:
                    self.cells[(i, j)] = ch

        if self.has_key and self.key_pos is not None:
            self.cells[self.key_pos] = "."
        if self.door_open and self.door_pos is not None:
            self.cells[self.door_pos] = "."

    def frontier_positions(self) -> List[Tuple[int, int]]:
        out: List[Tuple[int, int]] = []
        for (i, j), val in self.cells.items():
            if not self.is_walkable((i, j)):
                continue
            for ni, nj in ((i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)):
                if 0 <= ni < self.rows and 0 <= nj < self.cols and (ni, nj) not in self.cells:
                    out.append((i, j))
                    break
        return out

    def is_walkable(self, pos: Tuple[int, int]) -> bool:
        cell = self.cells.get(pos)
        if cell is None:
            return False
        if cell in {".", "K", "T"}:
            return True
        if cell == "D":
            return self.has_key or self.door_open
        return False

    def render_known(self, agent: Tuple[int, int]) -> str:
        canvas = [["?" for _ in range(self.cols)] for _ in range(self.rows)]

        for (i, j), v in self.cells.items():
            canvas[i][j] = v

        if self.target is not None:
            ti, tj = self.target
            canvas[ti][tj] = "T"

        if self.key_pos is not None and not self.has_key:
            ki, kj = self.key_pos
            canvas[ki][kj] = "K"

        if self.door_pos is not None and not self.door_open:
            di, dj = self.door_pos
            canvas[di][dj] = "D"

        ai, aj = agent
        canvas[ai][aj] = "A"

        return "\n".join("".join(row) for row in canvas)
