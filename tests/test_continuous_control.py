"""
Unit Tests: ICIA Embodied Continuous Motor Control & Non-Linear Physics
"""

import math
import numpy as np
import pytest
from icia.tasks.continuous_cartpole_env import ContinuousCartPoleEnv
from icia.models.continuous_icia_controller import ContinuousICIAController


def test_continuous_physics_energy_and_boundaries():
    """Verify non-linear ODE integration and rail boundary detection."""
    env = ContinuousCartPoleEnv(tau=0.02, x_threshold=2.4)
    # Start near rail boundary
    state = env.reset(initial_state=[2.39, 1.0, 0.0, 0.0])
    
    # Step forward with positive force should push cart across boundary
    next_state, _, done, info = env.step(10.0)
    
    assert next_state[0] > 2.4
    assert bool(done) is True
    assert bool(info["rail_violation"]) is True


def test_icia_continuous_controller_zero_shot_balancing():
    """Verify that continuous ICIA controller stabilizes inverted pendulum from perturbation."""
    env = ContinuousCartPoleEnv(tau=0.02, x_threshold=2.4)
    controller = ContinuousICIAController(dt=0.02, x_max=2.4)
    
    # Initial perturbation of 0.1 rad (~5.7 degrees)
    obs = env.reset(initial_state=[0.0, 0.0, 0.1, 0.0])
    
    for _ in range(200):
        action, _ = controller.compute_action(obs)
        obs, _, done, _ = env.step(action)
        assert not done
        
    # Pole angle should contract toward 0 rad (|theta| < 0.02)
    assert abs(obs[2]) < 0.02
    # Cart position should be bounded near center (|x| < 0.1)
    assert abs(obs[0]) < 0.1


def test_icia_impulse_discrepancy_detection():
    """Verify that external force impulse triggers reality discrepancy gate."""
    env = ContinuousCartPoleEnv(tau=0.02)
    controller = ContinuousICIAController(dt=0.02)
    
    obs = env.reset(initial_state=[0.0, 0.0, 0.0, 0.0])
    
    # Step 1: Nominal
    action, telemetry = controller.compute_action(obs)
    obs, _, _, _ = env.step(action)
    
    # Step 2: Inject unannounced +15N shock
    env.apply_impulse(15.0)
    action, telemetry = controller.compute_action(obs)
    obs, _, _, _ = env.step(action)
    
    # Step 3: Observe shock discrepancy
    action, telemetry = controller.compute_action(obs)
    assert telemetry["discrepancy_norm"] > 0.1
    assert telemetry["shock_detected"] is True
