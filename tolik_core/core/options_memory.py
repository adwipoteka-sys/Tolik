from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class OptionRecord:
    id: str
    name: str
    family: str
    layout: str
    kind: str
    trigger: str
    actions: List[str]
    uses: int = 0
    successes: int = 0
    last_ratio: float = 0.0


class OptionsMemory:
    def __init__(self, storage_dir: str = "data/runtime") -> None:
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.storage_dir / "options_memory.json"
        self.options: List[OptionRecord] = []
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            self.options = [OptionRecord(**item) for item in raw]

    def _save(self) -> None:
        self.path.write_text(
            json.dumps([asdict(x) for x in self.options], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def add_or_replace(
        self,
        *,
        name: str,
        family: str,
        layout: str,
        kind: str,
        trigger: str,
        actions: List[str],
    ) -> Dict[str, object]:
        self.options = [
            x for x in self.options
            if not (x.name == name and x.family == family and x.layout == layout and x.kind == kind)
        ]
        rec = OptionRecord(
            id=str(uuid.uuid4())[:8],
            name=name,
            family=family,
            layout=layout,
            kind=kind,
            trigger=trigger,
            actions=actions,
        )
        self.options.append(rec)
        self._save()
        return asdict(rec)

    def match(self, family: str, layout: str, kind: Optional[str] = None) -> List[Dict[str, object]]:
        rows = [
            asdict(x) for x in self.options
            if x.family == family and x.layout == layout and (kind is None or x.kind == kind)
        ]
        rows.sort(key=lambda x: (-x["successes"], x["uses"], x["name"]))
        return rows

    def list_options(self) -> List[Dict[str, object]]:
        rows = [asdict(x) for x in self.options]
        rows.sort(key=lambda x: (x["family"], x["layout"], x["name"]))
        return rows

    def record_use(self, option_id: str, ok: bool, ratio: float) -> None:
        for x in self.options:
            if x.id == option_id:
                x.uses += 1
                x.successes += int(ok)
                x.last_ratio = ratio
                break
        self._save()

    def summary(self) -> Dict[str, int]:
        return {
            "options": len(self.options),
            "used": sum(x.uses > 0 for x in self.options),
        }
