from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional, Tuple


class KeyDoorPlanner:
    def _state_bfs(
        self,
        map_memory,
        start_pos: Tuple[int, int],
        targets: List[Tuple[int, int]],
    ) -> List[str]:
        if not targets:
            return []

        target_set = set(targets)
        start_state = (start_pos, map_memory.has_key, map_memory.door_open)

        q = deque([start_state])
        prev: Dict[Tuple[Tuple[int, int], bool, bool], Tuple[Tuple[Tuple[int, int], bool, bool], str]] = {}
        seen = {start_state}

        moves = [
            ("up", (-1, 0)),
            ("down", (1, 0)),
            ("left", (0, -1)),
            ("right", (0, 1)),
        ]

        found: Optional[Tuple[Tuple[int, int], bool, bool]] = None

        while q:
            pos, has_key, door_open = q.popleft()
            if pos in target_set:
                found = (pos, has_key, door_open)
                break

            for action, (di, dj) in moves:
                nxt = (pos[0] + di, pos[1] + dj)
                cell = map_memory.cells.get(nxt)

                if cell is None or cell == "#":
                    continue

                if cell == "D" and not (door_open or has_key):
                    continue

                new_has_key = has_key or (cell == "K")
                new_door_open = door_open or (cell == "D" and has_key)

                state = (nxt, new_has_key, new_door_open)
                if state in seen:
                    continue

                seen.add(state)
                prev[state] = ((pos, has_key, door_open), action)
                q.append(state)

        if found is None:
            return []

        actions: List[str] = []
        node = found
        while node != start_state:
            parent, act = prev[node]
            actions.append(act)
            node = parent
        actions.reverse()
        return actions

    def plan(self, map_memory, agent_pos: Tuple[int, int]) -> List[str]:
        if map_memory.target is not None:
            path = self._state_bfs(map_memory, agent_pos, [map_memory.target])
            if path:
                return path

        if not map_memory.has_key and map_memory.key_pos is not None:
            path = self._state_bfs(map_memory, agent_pos, [map_memory.key_pos])
            if path:
                return path

        if map_memory.has_key and map_memory.door_pos is not None and not map_memory.door_open:
            path = self._state_bfs(map_memory, agent_pos, [map_memory.door_pos])
            if path:
                return path

        frontier = map_memory.frontier_positions()
        if frontier:
            return self._state_bfs(map_memory, agent_pos, frontier)

        return []
