from __future__ import annotations

from typing import Dict, List, Optional, Tuple


class MapMemory:
    def __init__(self) -> None:
        self.layout = "unknown"
        self.rows = 0
        self.cols = 0
        self.cells: Dict[Tuple[int, int], str] = {}
        self.target: Optional[Tuple[int, int]] = None

    def reset(self, layout: str, rows: int, cols: int) -> None:
        self.layout = layout
        self.rows = rows
        self.cols = cols
        self.cells = {}
        self.target = None

    def load_from_fact(self, fact: dict) -> None:
        self.layout = fact.get("layout", self.layout)
        self.rows = fact.get("rows", self.rows)
        self.cols = fact.get("cols", self.cols)
        self.target = tuple(fact["target"]) if fact.get("target") else None

        cells = {}
        for item in fact.get("cells", []):
            cells[(item["i"], item["j"])] = item["v"]
        self.cells = cells

    def to_fact(self) -> dict:
        return {
            "layout": self.layout,
            "rows": self.rows,
            "cols": self.cols,
            "target": list(self.target) if self.target else None,
            "cells": [{"i": i, "j": j, "v": v} for (i, j), v in sorted(self.cells.items())],
        }

    def update_from_local(self, obs: dict) -> None:
        top_i, top_j = obs["top_left"]
        patch = obs["patch"]

        for di, row in enumerate(patch):
            for dj, ch in enumerate(row):
                i, j = top_i + di, top_j + dj
                if not (0 <= i < self.rows and 0 <= j < self.cols):
                    continue

                if ch == "A":
                    self.cells[(i, j)] = "."
                elif ch in {".", "#"}:
                    self.cells[(i, j)] = ch
                elif ch == "T":
                    self.cells[(i, j)] = "."
                    self.target = (i, j)

    def is_free_known(self, pos: Tuple[int, int]) -> bool:
        return self.cells.get(pos) == "."

    def neighbors(self, pos: Tuple[int, int]) -> List[Tuple[str, Tuple[int, int]]]:
        i, j = pos
        return [
            ("up", (i - 1, j)),
            ("down", (i + 1, j)),
            ("left", (i, j - 1)),
            ("right", (i, j + 1)),
        ]

    def frontier_positions(self) -> List[Tuple[int, int]]:
        out: List[Tuple[int, int]] = []
        for pos, val in self.cells.items():
            if val != ".":
                continue
            for _, nxt in self.neighbors(pos):
                if 0 <= nxt[0] < self.rows and 0 <= nxt[1] < self.cols and nxt not in self.cells:
                    out.append(pos)
                    break
        return out

    def render_known(self, agent: Tuple[int, int]) -> str:
        canvas = [["?" for _ in range(self.cols)] for _ in range(self.rows)]

        for (i, j), v in self.cells.items():
            canvas[i][j] = v

        if self.target is not None:
            ti, tj = self.target
            canvas[ti][tj] = "T"

        ai, aj = agent
        canvas[ai][aj] = "A"

        return "\n".join("".join(row) for row in canvas)
