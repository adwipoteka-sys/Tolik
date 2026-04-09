from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from memory.goal_ledger import GoalLedger
from memory.capability_portfolio import CapabilityPortfolio


_READY_STAGES = {"available", "stable", "transfer_validated"}
_STAGE_ORDER = {
    "latent": 0,
    "unknown": 1,
    "emerging": 2,
    "available": 3,
    "stable": 4,
    "transfer_validated": 5,
}


@dataclass(slots=True)
class CapabilityGraphNode:
    capability: str
    stage: str = "unknown"
    incoming: list[str] = field(default_factory=list)
    outgoing: list[str] = field(default_factory=list)
    suggested_targets: list[str] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["updated_at"] = self.updated_at.isoformat()
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CapabilityGraphNode":
        raw = dict(data)
        raw["updated_at"] = datetime.fromisoformat(raw["updated_at"])
        return cls(**raw)


@dataclass(frozen=True, slots=True)
class CapabilityTransferEdge:
    source_capability: str
    target_capability: str
    relation: str = "transfer_enables"
    support_capabilities: tuple[str, ...] = ()
    source_stage_required: str = "transfer_validated"
    description: str = ""
    strength: float = 0.5
    strategic_value: float = 0.5


class CapabilityGraph:
    """Persistent graph of capability dependencies and transfer opportunities."""

    DEFAULT_EDGES: tuple[CapabilityTransferEdge, ...] = (
        CapabilityTransferEdge(
            source_capability="grounded_navigation",
            target_capability="navigation_route_explanation",
            support_capabilities=("local_llm",),
            source_stage_required="transfer_validated",
            description="Explain grounded detours from validated graph-search traces.",
            strength=0.78,
            strategic_value=0.52,
        ),
        CapabilityTransferEdge(
            source_capability="grounded_navigation",
            target_capability="spatial_route_composition",
            source_stage_required="transfer_validated",
            description="Compose validated grounded-navigation primitives across waypoint chains.",
            strength=0.93,
            strategic_value=0.90,
        ),
        CapabilityTransferEdge(
            source_capability="spatial_route_composition",
            target_capability="route_mission_briefing",
            support_capabilities=("local_llm",),
            source_stage_required="transfer_validated",
            description="Turn composed waypoint routes into compact mission briefings.",
            strength=0.91,
            strategic_value=0.88,
        ),
    )



    def __init__(self, ledger: GoalLedger | None = None) -> None:
        self.ledger = ledger
        self._nodes: dict[str, CapabilityGraphNode] = {}
        self._edges = list(self.DEFAULT_EDGES)
        self._ensure_node("classical_planning", stage="available")
        self._ensure_node("local_llm", stage="available")
        for edge in self._edges:
            self._register_edge_nodes(edge)
        if self.ledger is not None:
            self._rehydrate()

    def _rehydrate(self) -> None:
        for payload in self.ledger.load_capability_graph_nodes():
            node = CapabilityGraphNode.from_dict(payload)
            current = self._nodes.get(node.capability)
            if current is None:
                self._nodes[node.capability] = node
                continue
            if _STAGE_ORDER.get(node.stage, 0) >= _STAGE_ORDER.get(current.stage, 0):
                current.stage = node.stage
            current.suggested_targets = sorted(set(current.suggested_targets) | set(node.suggested_targets))
            current.evidence.update(node.evidence)
            current.updated_at = node.updated_at

    def _persist(self, node: CapabilityGraphNode) -> CapabilityGraphNode:
        node.updated_at = datetime.now(timezone.utc)
        self._nodes[node.capability] = node
        if self.ledger is not None:
            self.ledger.save_capability_graph_node(node.to_dict())
        return node

    def _ensure_node(self, capability: str, *, stage: str | None = None) -> CapabilityGraphNode:
        node = self._nodes.get(capability)
        if node is None:
            node = CapabilityGraphNode(capability=capability, stage=stage or "unknown")
            self._nodes[capability] = node
        elif stage is not None and _STAGE_ORDER.get(stage, 0) >= _STAGE_ORDER.get(node.stage, 0):
            node.stage = stage
        return node

    def _register_edge_nodes(self, edge: CapabilityTransferEdge) -> None:
        src = self._ensure_node(edge.source_capability)
        tgt = self._ensure_node(edge.target_capability, stage="latent")
        if edge.target_capability not in src.outgoing:
            src.outgoing.append(edge.target_capability)
        if edge.source_capability not in tgt.incoming:
            tgt.incoming.append(edge.source_capability)
        for support in edge.support_capabilities:
            support_node = self._ensure_node(support, stage="available" if support in {"classical_planning", "local_llm"} else "unknown")
            if edge.target_capability not in support_node.outgoing:
                support_node.outgoing.append(edge.target_capability)
            if support not in tgt.incoming:
                tgt.incoming.append(support)

    def list_transfer_edges(
        self,
        *,
        source_capability: str | None = None,
        target_capability: str | None = None,
    ) -> list[CapabilityTransferEdge]:
        edges = list(self._edges)
        if source_capability is not None:
            edges = [edge for edge in edges if edge.source_capability == source_capability]
        if target_capability is not None:
            edges = [edge for edge in edges if edge.target_capability == target_capability]
        return sorted(edges, key=lambda item: (item.source_capability, item.target_capability))

    def add_transfer_edge(self, edge: CapabilityTransferEdge) -> None:
        if edge not in self._edges:
            self._edges.append(edge)
            self._register_edge_nodes(edge)

    def get(self, capability: str) -> CapabilityGraphNode | None:
        return self._nodes.get(capability)

    def list_nodes(self) -> list[CapabilityGraphNode]:
        return sorted(self._nodes.values(), key=lambda item: item.capability)

    def update_stage(self, capability: str, stage: str, *, evidence: dict[str, Any] | None = None) -> CapabilityGraphNode:
        node = self._ensure_node(capability)
        if _STAGE_ORDER.get(stage, 0) >= _STAGE_ORDER.get(node.stage, 0):
            node.stage = stage
        if evidence:
            node.evidence.update(evidence)
        return self._persist(node)

    def sync_from_portfolio(self, portfolio: CapabilityPortfolio) -> None:
        for state in portfolio.list_states():
            self.update_stage(
                state.capability,
                state.maturity_stage,
                evidence={
                    "latest_strategy": state.latest_strategy,
                    "latest_skill_score": state.latest_skill_score,
                    "latest_transfer_score": state.latest_transfer_score,
                },
            )

    def mark_transfer_goal_suggested(self, *, source_capability: str, target_capability: str) -> None:
        node = self._ensure_node(source_capability)
        if target_capability not in node.suggested_targets:
            node.suggested_targets.append(target_capability)
            self._persist(node)

    def is_ready(self, capability: str) -> bool:
        node = self._nodes.get(capability)
        return bool(node and node.stage in _READY_STAGES)

    def suggest_transfer_edges(self, *, existing_titles: set[str] | None = None) -> list[CapabilityTransferEdge]:
        existing_titles = {" ".join(title.lower().split()) for title in (existing_titles or set())}
        suggestions: list[CapabilityTransferEdge] = []
        for edge in self._edges:
            source = self._nodes.get(edge.source_capability)
            target = self._nodes.get(edge.target_capability)
            if source is None or target is None:
                continue
            if _STAGE_ORDER.get(source.stage, 0) < _STAGE_ORDER.get(edge.source_stage_required, 0):
                continue
            if target.stage in {"stable", "transfer_validated"}:
                continue
            if edge.target_capability in source.suggested_targets:
                continue
            if any(not self.is_ready(support) for support in edge.support_capabilities):
                continue
            normalized_title = " ".join(
                f"Bootstrap {edge.target_capability.replace('_', ' ')} from {edge.source_capability.replace('_', ' ')}".lower().split()
            )
            if normalized_title in existing_titles:
                continue
            suggestions.append(edge)
        return suggestions

    def summary(self) -> dict[str, dict[str, Any]]:
        return {
            node.capability: {
                "stage": node.stage,
                "incoming": list(sorted(node.incoming)),
                "outgoing": list(sorted(node.outgoing)),
                "suggested_targets": list(sorted(node.suggested_targets)),
            }
            for node in self.list_nodes()
        }
