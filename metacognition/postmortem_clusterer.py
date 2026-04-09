from __future__ import annotations

from dataclasses import asdict, dataclass, field

from metacognition.failure_miner import FailureCase


@dataclass(slots=True)
class FailureCluster:
    signature: str
    capability: str
    occurrence_count: int = 0
    violation_types: list[str] = field(default_factory=list)
    latest_goal_id: str | None = None
    latest_tool_name: str | None = None
    open: bool = True
    example_case_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class PostmortemClusterer:
    """Groups repeated canary failures into recurring remediation patterns."""

    def __init__(self) -> None:
        self._clusters: dict[str, FailureCluster] = {}

    def add_failure(self, failure: FailureCase) -> FailureCluster:
        cluster = self._clusters.get(failure.signature)
        if cluster is None:
            cluster = FailureCluster(
                signature=failure.signature,
                capability=failure.capability,
                violation_types=sorted(set(failure.violation_types)),
            )
            self._clusters[failure.signature] = cluster

        cluster.occurrence_count += 1
        cluster.latest_goal_id = failure.goal_id
        cluster.latest_tool_name = failure.tool_name
        cluster.open = True
        if failure.case_id not in cluster.example_case_ids:
            cluster.example_case_ids.append(failure.case_id)
        return cluster

    def mark_closed(self, signature: str) -> None:
        cluster = self._clusters.get(signature)
        if cluster is not None:
            cluster.open = False

    def get(self, signature: str) -> FailureCluster | None:
        return self._clusters.get(signature)

    def all_clusters(self) -> list[FailureCluster]:
        return list(self._clusters.values())
