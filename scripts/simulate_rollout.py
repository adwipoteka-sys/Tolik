import argparse
import json
from pathlib import Path


def load_scenario(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_decision(data: dict) -> dict:
    metrics = data.get("metrics", {})
    thresholds = data.get("thresholds", {})

    correctness = metrics.get("correctness", 0.0)
    safety = metrics.get("safety", 0.0)
    latency_ms = metrics.get("latency_ms", 0)

    correctness_min = thresholds.get("correctness_min", 0.88)
    safety_min = thresholds.get("safety_min", 0.95)
    latency_max_ms = thresholds.get("latency_max_ms", 1200)

    reasons = []

    if correctness < correctness_min:
        reasons.append(f"correctness_below_threshold:{correctness:.3f}")
    if safety < safety_min:
        reasons.append(f"safety_below_threshold:{safety:.3f}")
    if latency_ms > latency_max_ms:
        reasons.append(f"latency_above_threshold:{latency_ms}")

    recent_failures = data.get("recent_failures", 0)
    anti_flap_repeat_failures = data.get("anti_flap_repeat_failures", 0)
    anti_flap_window_rollouts = data.get("anti_flap_window_rollouts", 0)
    cooldown_rollouts = data.get("cooldown_rollouts", 0)

    anti_flap_active = False
    cooldown_active = False

    if recent_failures >= anti_flap_repeat_failures and anti_flap_repeat_failures > 0:
        anti_flap_active = True
        reasons.append(f"anti_flap_triggered:{recent_failures}")

    if reasons:
        if anti_flap_active:
            decision = "freeze_canary"
        else:
            decision = "rollback_with_cooldown"
            cooldown_active = cooldown_rollouts > 0
            if cooldown_active:
                reasons.append(f"cooldown_rollouts:{cooldown_rollouts}")
    else:
        decision = "promote_canary"

    return {
        "scenario": data.get("scenario"),
        "provider": data.get("provider"),
        "fallback_provider": data.get("fallback_provider"),
        "decision": decision,
        "metrics": metrics,
        "thresholds": thresholds,
        "anti_flap_window_rollouts": anti_flap_window_rollouts,
        "anti_flap_repeat_failures": anti_flap_repeat_failures,
        "anti_flap_active": anti_flap_active,
        "cooldown_rollouts": cooldown_rollouts,
        "cooldown_active": cooldown_active,
        "reasons": reasons,
        "expected_decision": data.get("expected_decision"),
        "matches_expected": decision == data.get("expected_decision"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Simulate Tolik rollout / rollback / anti-flap decisions"
    )
    parser.add_argument(
        "--scenario",
        required=True,
        help="Path to scenario JSON file",
    )
    args = parser.parse_args()

    scenario_path = Path(args.scenario)
    if not scenario_path.exists():
        raise FileNotFoundError(f"Scenario file not found: {scenario_path}")

    data = load_scenario(str(scenario_path))
    result = evaluate_decision(data)

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()