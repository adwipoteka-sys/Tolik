from __future__ import annotations

from metacognition.transfer_curriculum import TransferCurriculumTask, transfer_task_to_goal
from motivation.goal_schema import GoalBudget, SuccessCriterion


def test_transfer_task_to_goal_preserves_target_capability_and_tags():
    task = TransferCurriculumTask(
        source_capability="grounded_navigation",
        target_capability="navigation_route_explanation",
        title="Bootstrap navigation route explanation from grounded navigation",
        description="desc",
        budget=GoalBudget(max_steps=4, max_seconds=18.0),
        success_criteria=[SuccessCriterion(metric="success_rate", comparator=">=", target=1.0)],
        tags=["transfer_curriculum", "navigation_route_explanation", "local_only"],
        required_capabilities=["grounded_navigation", "local_llm"],
        evidence={"target_capability": "navigation_route_explanation", "source_capability": "grounded_navigation"},
    )

    goal = transfer_task_to_goal(task)
    assert goal.kind.value == "learning"
    assert goal.source.value == "curriculum"
    assert goal.evidence["target_capability"] == "navigation_route_explanation"
    assert "transfer_curriculum" in goal.tags
