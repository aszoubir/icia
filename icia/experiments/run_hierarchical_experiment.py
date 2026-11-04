"""
Hierarchical Intention Experiment Runner
Quantifies performance of Hierarchical ICIA vs Autoregressive Inertia
on multi-room puzzle with stochastic roadblocking.
"""

import json
import numpy as np
from icia.tasks.hierarchical_foraging import HierarchicalForagingEnv
from icia.models.hierarchical_engine import HierarchicalIntentionController, SubGoal


def build_icia_controller() -> HierarchicalIntentionController:
    controller = HierarchicalIntentionController(
        macro_goal=np.array([10.0, 10.0]),
        impasse_threshold=0.8,
        operational_step_size=0.3
    )

    # Hierarchical plan with contingency attached to corridor sub-goal
    subgoals = [
        SubGoal(
            id="SG1_GET_KEY_A",
            description="Acquire Key A at (2.0, 2.0)",
            target_state=np.array([2.0, 2.0]),
            completion_radius=0.4
        ),
        SubGoal(
            id="SG2_ENTER_DOOR_1",
            description="Traverse Door 1 at (4.0, 2.0)",
            target_state=np.array([4.0, 2.0]),
            preconditions=["has_key_A"],
            completion_radius=0.4
        ),
        SubGoal(
            id="SG3_CORRIDOR_KEY_B",
            description="Traverse Corridor at (6.0, 4.0) to acquire Key B",
            target_state=np.array([6.0, 4.0]),
            preconditions=["door_1_unlocked"],
            completion_radius=0.4,
            metadata={
                "contingencies": [
                    {
                        "id": "SG3_ALT_VENT_ROUTE",
                        "description": "Pivot to Vent route at (5.0, 6.0)",
                        "target_state": [5.0, 6.0],
                        "completion_radius": 0.4
                    },
                    {
                        "id": "SG3_GET_KEY_B",
                        "description": "Acquire Key B at (8.0, 6.0)",
                        "target_state": [8.0, 6.0],
                        "completion_radius": 0.4
                    }
                ]
            }
        ),
        SubGoal(
            id="SG4_UNLOCK_VAULT",
            description="Unlock Terminal Vault at (10.0, 10.0)",
            target_state=np.array([10.0, 10.0]),
            preconditions=["has_key_B"],
            completion_radius=0.5
        )
    ]
    controller.decompose(subgoals)
    return controller


def run_autoregressive_baseline(env: HierarchicalForagingEnv, max_steps: int = 120) -> dict:
    """
    Simulates autoregressive sequence generation:
    When an unexpected obstacle blocks the path, the agent experiences autoregressive inertia
    (it repeats previously generated tokens/directions for multiple steps before prompt clearing).
    """
    state, facts = env.reset()
    inertia_counter = 0
    total_steps = 0
    stuck_cycles = 0

    # Fixed predetermined plan waypoints
    waypoints = [
        np.array([2.0, 2.0]),  # Key A
        np.array([4.0, 2.0]),  # Door 1
        np.array([6.0, 4.0]),  # Corridor (JAMMED!)
        np.array([8.0, 6.0]),  # Key B
        np.array([10.0, 10.0]) # Vault
    ]
    wp_idx = 0

    while total_steps < max_steps:
        total_steps += 1
        target = waypoints[wp_idx]

        diff = target - state
        dist = np.linalg.norm(diff)
        if dist < 0.4:
            wp_idx = min(wp_idx + 1, len(waypoints) - 1)
            target = waypoints[wp_idx]
            diff = target - state
            dist = np.linalg.norm(diff)

        action = (diff / (dist + 1e-6)) * 0.3
        next_state, disc, done, info = env.step(action)

        if disc > 0.5:
            # Hit roadblock! Due to autoregressive inertia, the agent keeps retrying the blocked path
            inertia_counter += 1
            stuck_cycles += 1
            if inertia_counter > 15:
                # Finally breaks out of inertia after 15 wasted tokens/steps
                wp_idx = min(wp_idx + 1, len(waypoints) - 1)
                inertia_counter = 0

        state = next_state
        if done and info["vault_unlocked"]:
            break

    return {
        "success": info["vault_unlocked"],
        "steps_taken": total_steps,
        "stuck_cycles": stuck_cycles,
        "final_facts": info["facts"]
    }


def run_icia_hierarchical(env: HierarchicalForagingEnv, max_steps: int = 120) -> dict:
    state, facts = env.reset()
    controller = build_icia_controller()
    total_steps = 0
    discrepancy = 0.0

    while total_steps < max_steps:
        total_steps += 1
        action, telemetry = controller.compute_action(state, facts, discrepancy=discrepancy)
        state, discrepancy, done, info = env.step(action)
        facts = info["facts"]

        if done and info["vault_unlocked"]:
            break

    return {
        "success": info["vault_unlocked"],
        "steps_taken": total_steps,
        "replan_count": controller.total_replan_count,
        "completed_subgoals": controller.completed_sub_goals,
        "final_facts": info["facts"]
    }


def run_benchmark():
    print("=" * 70)
    print("PHASE 8 BENCHMARK: HIERARCHICAL INTENTION DECOMPOSITION & IMPASSE RESOLUTION")
    print("=" * 70)

    n_trials = 20
    ar_successes = 0
    ar_steps = []
    ar_stuck = []

    icia_successes = 0
    icia_steps = []
    icia_replans = []

    for trial in range(n_trials):
        env_ar = HierarchicalForagingEnv(stochastic_jam=True)
        res_ar = run_autoregressive_baseline(env_ar)
        if res_ar["success"]:
            ar_successes += 1
        ar_steps.append(res_ar["steps_taken"])
        ar_stuck.append(res_ar["stuck_cycles"])

        env_icia = HierarchicalForagingEnv(stochastic_jam=True)
        res_icia = run_icia_hierarchical(env_icia)
        if res_icia["success"]:
            icia_successes += 1
        icia_steps.append(res_icia["steps_taken"])
        icia_replans.append(res_icia["replan_count"])

    summary = {
        "trials": n_trials,
        "autoregressive_baseline": {
            "success_rate": float(ar_successes / n_trials),
            "avg_steps": float(np.mean(ar_steps)),
            "avg_stuck_cycles": float(np.mean(ar_stuck))
        },
        "hierarchical_icia": {
            "success_rate": float(icia_successes / n_trials),
            "avg_steps": float(np.mean(icia_steps)),
            "avg_replans": float(np.mean(icia_replans))
        }
    }

    print("\nRESULTS SUMMARY:")
    print(f"Autoregressive Baseline: Success = {summary['autoregressive_baseline']['success_rate']*100:.1f}%, "
          f"Avg Steps = {summary['autoregressive_baseline']['avg_steps']:.1f}, "
          f"Avg Stuck Retries = {summary['autoregressive_baseline']['avg_stuck_cycles']:.1f}")
    print(f"Hierarchical ICIA:       Success = {summary['hierarchical_icia']['success_rate']*100:.1f}%, "
          f"Avg Steps = {summary['hierarchical_icia']['avg_steps']:.1f}, "
          f"Avg Replans = {summary['hierarchical_icia']['avg_replans']:.1f} (1-Step Pivots)")

    with open("icia_hierarchical_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary


if __name__ == "__main__":
    run_benchmark()
