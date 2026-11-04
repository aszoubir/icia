"""
Unit Tests: ICIA Neural Web Server & Endpoints
"""

import json
import pytest
from icia.web.server import physics_env, physics_controller


def test_physics_endpoint_logic():
    """Verify physics controller step logic called by web server."""
    obs = physics_env.reset(initial_state=[0.0, 0.0, 0.05, 0.0])
    action, telemetry = physics_controller.compute_action(obs)
    next_obs, reward, done, info = physics_env.step(action)
    
    assert len(next_obs) == 4
    assert isinstance(action, float)
    assert "discrepancy_norm" in telemetry
    assert "shock_detected" in telemetry
