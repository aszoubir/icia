"""
ICIA Embodied Control Master Experiment Runner: Continuous Physics & Impulse Recovery
Benchmarks:
1. Continuous ICIA Teleological Controller (Riemannian Gradient Flow + Reality Discrepancy Gate)
2. Open-Loop Sequence Controller (Standard sequence-to-action baseline)

Evaluates:
- Zero-shot continuous stabilization (0 trial-and-error training episodes)
- Recovery from unannounced external force impulse shocks (5N, 10N, 15N)
- Boundary barrier constraint compliance (|x| <= 2.4m)
- Control compute latency (microseconds per step)
"""

import time
import os
import json
import math
import random
from typing import Dict, Any, List, Tuple
import numpy as np

from icia.tasks.continuous_cartpole_env import ContinuousCartPoleEnv
from icia.models.continuous_icia_controller import ContinuousICIAController
from icia.models.decision_transformer_baseline import OpenLoopSequenceController


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)


def run_nominal_balancing(
    controller,
    env: ContinuousCartPoleEnv,
    num_episodes: int = 25,
    max_steps_per_episode: int = 500
) -> Dict[str, float]:
    """
    Evaluates nominal balancing stability without external shocks.
    """
    success_count = 0
    total_rewards = []
    angle_deviations = []
    position_drifts = []

    for ep in range(num_episodes):
        obs = env.reset()
        ep_reward = 0.0
        done = False
        step = 0

        while not done and step < max_steps_per_episode:
            action, _ = controller.compute_action(obs)
            next_obs, reward, done, info = env.step(action)
            ep_reward += reward
            angle_deviations.append(abs(obs[2]))
            position_drifts.append(abs(obs[0]))
            obs = next_obs
            step += 1

        if step >= max_steps_per_episode and not (info["rail_violation"] or info["angle_violation"]):
            success_count += 1
        total_rewards.append(ep_reward)

    return {
        "success_rate_pct": (success_count / num_episodes) * 100.0,
        "mean_reward": float(np.mean(total_rewards)),
        "mean_abs_angle_rad": float(np.mean(angle_deviations)),
        "mean_abs_position_m": float(np.mean(position_drifts)),
        "episodes_required_to_learn": 0 # Zero-shot!
    }


def run_impulse_shock_benchmark(
    controller,
    env: ContinuousCartPoleEnv,
    shock_magnitudes: List[float] = [5.0, 10.0, 15.0],
    num_trials_per_shock: int = 20,
    shock_step: int = 100,
    max_steps: int = 400
) -> Dict[float, Dict[str, Any]]:
    """
    Evaluates resilience to unannounced physical force impulses injected at step 100.
    """
    results = {}

    for shock_f in shock_magnitudes:
        successes = 0
        rail_violations = 0
        angle_violations = 0
        recovery_times = []
        peak_discrepancies = []

        for trial in range(num_trials_per_shock):
            obs = env.reset(initial_state=[0.0, 0.0, 0.0, 0.0])
            done = False
            step = 0
            recovered_step = None
            peak_disc = 0.0

            while not done and step < max_steps:
                if step == shock_step:
                    env.apply_impulse(shock_f)

                action, telemetry = controller.compute_action(obs)
                disc = telemetry.get("discrepancy_norm", 0.0)
                if disc > peak_disc:
                    peak_disc = disc

                next_obs, reward, done, info = env.step(action)

                # Check if recovered after shock (|theta| < 0.03 rad)
                if step > shock_step and recovered_step is None and abs(next_obs[2]) < 0.03 and abs(next_obs[3]) < 0.05:
                    recovered_step = step

                obs = next_obs
                step += 1

            if info.get("rail_violation", False):
                rail_violations += 1
            if info.get("angle_violation", False):
                angle_violations += 1

            if step >= max_steps:
                successes += 1
                rec_t = (recovered_step - shock_step) * env.tau if recovered_step else max_steps * env.tau
                recovery_times.append(rec_t)
            else:
                recovery_times.append(max_steps * env.tau)

            peak_discrepancies.append(peak_disc)

        success_pct = (successes / num_trials_per_shock) * 100.0
        mean_rec_time = float(np.mean(recovery_times))
        mean_peak_disc = float(np.mean(peak_discrepancies))

        results[shock_f] = {
            "shock_force_newtons": shock_f,
            "survival_rate_pct": success_pct,
            "rail_crashes": rail_violations,
            "angle_crashes": angle_violations,
            "mean_recovery_time_sec": mean_rec_time,
            "mean_peak_discrepancy": mean_peak_disc
        }

    return results


def benchmark_control_latency(controller, env: ContinuousCartPoleEnv, n_steps: int = 1000) -> float:
    obs = env.reset()
    start_t = time.perf_counter()
    for _ in range(n_steps):
        action, _ = controller.compute_action(obs)
        obs, _, _, _ = env.step(action)
    total_time = time.perf_counter() - start_t
    return (total_time / n_steps) * 1e6 # microseconds per step


def main():
    set_seed(42)
    print("=" * 75)
    print("  ICIA-CONTROL: CONTINUOUS EMBODIED MOTOR CONTROL & IMPULSE RECOVERY")
    print("=" * 75)

    env = ContinuousCartPoleEnv(tau=0.02, x_threshold=2.4, theta_threshold=math.pi / 3.0)
    icia_controller = ContinuousICIAController(dt=0.02, x_max=2.4)
    baseline_controller = OpenLoopSequenceController()

    # 1. Benchmark Zero-Shot Balancing
    print("\n--- 1. Evaluating Zero-Shot Balancing Stability (No Impulses) ---")
    icia_nominal = run_nominal_balancing(icia_controller, env, num_episodes=20)
    base_nominal = run_nominal_balancing(baseline_controller, env, num_episodes=20)

    print(f"ICIA-Control:     Survival Rate = {icia_nominal['success_rate_pct']:5.1f}% | Angle Error = {icia_nominal['mean_abs_angle_rad']:.4f} rad | Position Drift = {icia_nominal['mean_abs_position_m']:.4f} m (0 RL episodes)")
    print(f"Baseline Policy:  Survival Rate = {base_nominal['success_rate_pct']:5.1f}% | Angle Error = {base_nominal['mean_abs_angle_rad']:.4f} rad | Position Drift = {base_nominal['mean_abs_position_m']:.4f} m")

    # 2. Benchmark Impulse Shock Recovery
    print("\n--- 2. Evaluating Unannounced Physical Force Impulse Shocks ---")
    shocks = [5.0, 10.0, 15.0]
    icia_shocks = run_impulse_shock_benchmark(icia_controller, env, shock_magnitudes=shocks, num_trials_per_shock=20)
    base_shocks = run_impulse_shock_benchmark(baseline_controller, env, shock_magnitudes=shocks, num_trials_per_shock=20)

    for sf in shocks:
        ic = icia_shocks[sf]
        bc = base_shocks[sf]
        print(f"Shock Force: {sf:4.1f} N | "
              f"ICIA: {ic['survival_rate_pct']:5.1f}% Survival, {ic['rail_crashes']} Rail Crashes, Rec: {ic['mean_recovery_time_sec']:.2f} s | "
              f"Base: {bc['survival_rate_pct']:5.1f}% Survival, {bc['rail_crashes']} Rail Crashes, Rec: {bc['mean_recovery_time_sec']:.2f} s")

    # 3. Benchmark Computational Latency
    print("\n--- 3. Measuring Control Decision Latency ---")
    lat_icia = benchmark_control_latency(icia_controller, env)
    lat_base = benchmark_control_latency(baseline_controller, env)
    print(f"ICIA Control Decision Latency:      {lat_icia:6.2f} microseconds / step")
    print(f"Sequence Baseline Decision Latency:  {lat_base:6.2f} microseconds / step")

    # Serialize complete results
    final_output = {
        "metadata": {
            "experiment": "ICIA Embodied Continuous Motor Control",
            "date": "2026-11-04",
            "environment": "ContinuousCartPole-v1",
            "integration_timestep_tau": env.tau,
            "rail_threshold_m": env.x_threshold,
            "angle_threshold_rad": env.theta_threshold
        },
        "zero_shot_nominal_balancing": {
            "icia": icia_nominal,
            "baseline": base_nominal
        },
        "impulse_shock_benchmark": {
            "icia": icia_shocks,
            "baseline": base_shocks
        },
        "latency_microseconds": {
            "icia": lat_icia,
            "baseline": lat_base
        }
    }

    output_path = "icia_continuous_control_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2)

    print(f"\n[OK] Embodied control benchmark completed. Results saved to {output_path}")


if __name__ == "__main__":
    main()
