from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


ACTIONS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

DEFAULT_LAYOUTS = {
    "corridor_kd": [
        "#########",
        "#A..K.DT#",
        "#########",
    ],
    "mirror_kd": [
        "#########",
        "#TD.K..A#",
        "#########",
    ],
    "detour_kd": [
        "#########",
        "#A..#...#",
        "#.#.#.#T#",
        "#K....D.#",
        "#########",
    ],
    "room_kd": [
        "###########",
        "#A...#....#",
        "#.##.#.##T#",
        "#K...D....#",
        "###########",
    ],
}


@dataclass
class StepResult:
    observation: Dict[str, object]
    reward: float
    done: bool
    info: Dict[str, object]


class KeyDoorWorld:
    def __init__(self) -> None:
        self.layouts = DEFAULT_LAYOUTS
        self.layout_name = "corridor_kd"
        self.grid: List[List[str]] = []
        self.agent: Tuple[int, int] = (0, 0)
        self.target: Tuple[int, int] = (0, 0)
        self.key_pos: Optional[Tuple[int, int]] = None
        self.door_pos: Optional[Tuple[int, int]] = None
        self.has_key: bool = False
        self.door_open: bool = False
        self.reset("corridor_kd")

    def reset(self, layout_name: str = "corridor_kd") -> Dict[str, object]:
        if layout_name not in self.layouts:
            raise ValueError(f"Unknown layout: {layout_name}")

        self.layout_name = layout_name
        self.grid = [list(row) for row in self.layouts[layout_name]]
        self.key_pos = None
        self.door_pos = None
        self.has_key = False
        self.door_open = False

        for i, row in enumerate(self.grid):
            for j, cell in enumerate(row):
                if cell == "A":
                    self.agent = (i, j)
                    self.grid[i][j] = "."
                elif cell == "T":
                    self.target = (i, j)
                    self.grid[i][j] = "."
                elif cell == "K":
                    self.key_pos = (i, j)
                    self.grid[i][j] = "."
                elif cell == "D":
                    self.door_pos = (i, j)
                    self.grid[i][j] = "."

        return self.observe()

    def _cell_view(self, i: int, j: int) -> str:
        if (i, j) == self.agent:
            return "A"
        if (i, j) == self.target:
            return "T"
        if self.key_pos is not None and (i, j) == self.key_pos and not self.has_key:
            return "K"
        if self.door_pos is not None and (i, j) == self.door_pos and not self.door_open:
            return "D"
        return self.grid[i][j]

    def observe(self) -> Dict[str, object]:
        return {
            "layout": self.layout_name,
            "grid": ["".join(self._cell_view(i, j) for j in range(len(self.grid[0]))) for i in range(len(self.grid))],
            "agent": self.agent,
            "target": self.target,
            "has_key": self.has_key,
            "door_open": self.door_open,
        }

    def observe_local(self, radius: int = 1) -> Dict[str, object]:
        ai, aj = self.agent
        rows = len(self.grid)
        cols = len(self.grid[0])

        patch: List[str] = []
        for i in range(ai - radius, ai + radius + 1):
            chars: List[str] = []
            for j in range(aj - radius, aj + radius + 1):
                if not (0 <= i < rows and 0 <= j < cols):
                    chars.append("#")
                else:
                    chars.append(self._cell_view(i, j))
            patch.append("".join(chars))

        return {
            "layout": self.layout_name,
            "agent": self.agent,
            "patch": patch,
            "top_left": (ai - radius, aj - radius),
            "shape": (rows, cols),
            "has_key": self.has_key,
            "door_open": self.door_open,
        }

    def render(self) -> str:
        rows = len(self.grid)
        cols = len(self.grid[0])
        return "\n".join("".join(self._cell_view(i, j) for j in range(cols)) for i in range(rows))

    def _static_free(self, i: int, j: int) -> bool:
        return 0 <= i < len(self.grid) and 0 <= j < len(self.grid[0]) and self.grid[i][j] != "#"

    def step(self, action: str) -> StepResult:
        if action not in ACTIONS:
            return StepResult(self.observe(), -0.2, False, {"error": f"unknown_action:{action}"})

        di, dj = ACTIONS[action]
        ni, nj = self.agent[0] + di, self.agent[1] + dj

        if not self._static_free(ni, nj):
            return StepResult(self.observe(), -0.05, False, {"moved": False, "blocked": "wall"})

        blocked_by_door = self.door_pos is not None and (ni, nj) == self.door_pos and not self.door_open
        if blocked_by_door:
            if self.has_key:
                self.door_open = True
            else:
                return StepResult(self.observe(), -0.05, False, {"moved": False, "blocked": "door"})

        self.agent = (ni, nj)

        reward = -0.01
        done = False
        info: Dict[str, object] = {"moved": True}

        if self.key_pos is not None and self.agent == self.key_pos and not self.has_key:
            self.has_key = True
            reward += 0.2
            info["picked_key"] = True

        if self.agent == self.target:
            done = True
            reward = 1.0
            info["reached_target"] = True

        return StepResult(self.observe(), reward, done, info)
