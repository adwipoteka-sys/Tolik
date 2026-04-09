from __future__ import annotations

from agency.agency_module import AgencyModule
from environments.route_briefing import RouteBriefingLab


def test_route_mission_briefing_generates_deterministic_briefings():
    agency = AgencyModule()
    agency.set_navigation_strategy("graph_search")
    lab = RouteBriefingLab()

    payload = {
        "tasks": [
            lab.get_task("route_train_open_chain").to_dict(),
            lab.get_task("route_train_detour_chain").to_dict(),
        ]
    }
    result = agency.execute_capability("route_mission_briefing", payload)

    assert result["success"] is True
    output = result["output"]
    assert output["strategy"] == "graph_search"
    assert output["task_count"] == 2
    assert output["briefings"] == [
        lab.brief_task(lab.get_task("route_train_open_chain"), strategy="graph_search"),
        lab.brief_task(lab.get_task("route_train_detour_chain"), strategy="graph_search"),
    ]
