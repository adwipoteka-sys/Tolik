from __future__ import annotations

from agency.agency_module import AgencyModule
from environments.grounded_navigation import GroundedNavigationLab


def test_navigation_route_explanation_describes_detours_once_bootstrapped():
    agency = AgencyModule()
    agency.set_navigation_strategy("graph_search")
    agency.add_capability("navigation_route_explanation")
    lab = GroundedNavigationLab()

    result = agency.execute_capability(
        "navigation_route_explanation",
        {
            "tasks": [
                lab.get_task("nav_detour_wall").to_dict(),
                lab.get_task("nav_detour_channel").to_dict(),
            ],
            "success_threshold": 1.0,
            "detour_explanation_threshold": 1.0,
        },
    )

    assert result["success"] is True
    output = result["output"]
    assert output["success_rate"] == 1.0
    assert output["detour_explanation_rate"] == 1.0
    assert all(item["mentions_detour"] for item in output["task_results"])
    assert all("blocked" in item["explanation"] for item in output["task_results"])
