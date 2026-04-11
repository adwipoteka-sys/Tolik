from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple


ACTIONS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}


DEFAULT_LAYOUTS = {
    "easy": [
        "#####",
        "#A..#",
        "#.#T#",
        "#...#",
        "#####",
    ],
    "detour": [
        "#######",
        "#A#...#",
        "#.#.#T#",
        "#...#.#",
        "#######",
    ],
    "mirror": [
        "#######",
        "#...#A#",
        "#.#.#.#",
        "#T....#",
        "#######",
    ],
    "corridor": [
        "########",
        "#A.....#",
        "###.####",
        "#....T.#",
        "########",
    ],
}


@dataclass
class StepResult:
    observation: Dict[str, object]
    reward: float
    done: bool
    info: Dict[str, object]


class GridWorld:
    def __init__(self) -> None:
        self.layouts = DEFAULT_LAYOUTS
        self.layout_name = "easy"
        self.grid: List[List[str]] = []
        self.agent: Tuple[int, int] = (0, 0)
        self.target: Tuple[int, int] = (0, 0)
        self.reset("easy")

    def reset(self, layout_name: str = "easy") -> Dict[str, object]:
        if layout_name not in self.layouts:
            raise ValueError(f"Unknown layout: {layout_name}")
        self.layout_name = layout_name
        raw = [list(row) for row in self.layouts[layout_name]]
        self.grid = raw

        for i, row in enumerate(self.grid):
            for j, cell in enumerate(row):
                if cell == "A":
                    self.agent = (i, j)
                    self.grid[i][j] = "."
                elif cell == "T":
                    self.target = (i, j)
                    self.grid[i][j] = "."

        return self.observe()

    def observe(self) -> Dict[str, object]:
        return {
            "layout": self.layout_name,
            "grid": ["".join(row) for row in self.grid],
            "agent": self.agent,
            "target": self.target,
            "distance_l1": abs(self.agent[0] - self.target[0]) + abs(self.agent[1] - self.target[1]),
        }

    def render(self) -> str:
        canvas = [row[:] for row in self.grid]
        ai, aj = self.agent
        ti, tj = self.target
        canvas[ai][aj] = "A"
        canvas[ti][tj] = "T"
        return "\n".join("".join(row) for row in canvas)

    def _is_free(self, i: int, j: int) -> bool:
        return 0 <= i < len(self.grid) and 0 <= j < len(self.grid[0]) and self.grid[i][j] != "#"

    def step(self, action: str) -> StepResult:
        if action not in ACTIONS:
            return StepResult(self.observe(), -0.2, False, {"error": f"unknown_action:{action}"})

        di, dj = ACTIONS[action]
        ni, nj = self.agent[0] + di, self.agent[1] + dj

        if self._is_free(ni, nj):
            self.agent = (ni, nj)
            moved = True
        else:
            moved = False

        done = self.agent == self.target
        reward = 1.0 if done else (-0.01 if moved else -0.05)

        return StepResult(
            observation=self.observe(),
            reward=reward,
            done=done,
            info={"moved": moved},
        )
