from __future__ import annotations

import copy
from typing import Any, Callable

from motivation.goal_schema import Goal


SelfModExecutor = Callable[[dict[str, Any], dict[str, Any], str], dict[str, Any]]


def execute_grounded_navigation(components: dict[str, Any], payload: dict[str, Any], rollout_stage: str) -> dict[str, Any]:
    agency = components["agency"]
    return agency.execute_capability("grounded_navigation", payload, rollout_stage=rollout_stage)


def execute_response_planning(components: dict[str, Any], payload: dict[str, Any], rollout_stage: str) -> dict[str, Any]:
    planning = components["planning"]
    raw_goal = payload.get("goal")
    goal = raw_goal if isinstance(raw_goal, Goal) else Goal.from_dict(raw_goal)
    world_state = dict(payload.get("world_state", {}))
    plan = planning.make_plan(goal, world_state)
    steps = [step.name for step in plan.steps]
    return {
        "steps": steps,
        "policy": planning.response_planning_policy,
        "requires_verification": bool(goal.evidence.get("requires_verification", False)),
        "rollout_stage": rollout_stage,
        "passed": "form_response" in steps and (
            not goal.evidence.get("requires_verification", False) or "verify_outcome" in steps
        ),
    }


def execute_memory_retrieval(components: dict[str, Any], payload: dict[str, Any], rollout_stage: str) -> dict[str, Any]:
    memory = components["memory"]
    original_long_term = copy.deepcopy(memory.long_term)
    original_working_memory = copy.deepcopy(memory.working_memory)
    try:
        memory.long_term = dict(payload.get("long_term_facts", {}))
        memory.working_memory = dict(payload.get("working_facts", {}))
        for key, info in payload.get("store_facts", {}).items():
            memory.store_fact(str(key), info)
        query = str(payload["query"])
        retrieved = memory.query_semantic(query)
        return {
            "query": query,
            "retrieved": retrieved,
            "policy": memory.retrieval_policy,
            "rollout_stage": rollout_stage,
            "passed": retrieved is not None,
        }
    finally:
        memory.long_term = original_long_term
        memory.working_memory = original_working_memory


def default_self_mod_executors() -> dict[str, SelfModExecutor]:
    return {
        "grounded_navigation": execute_grounded_navigation,
        "response_planning": execute_response_planning,
        "memory_retrieval": execute_memory_retrieval,
    }
