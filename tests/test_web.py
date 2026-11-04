"""
Unit Tests: ICIA Neural Web Server & Endpoints
"""

import json
import numpy as np
import pytest
from icia.web.server import (
    physics_env,
    physics_controller,
    hierarchical_env,
    hierarchical_controller,
    build_icia_controller
)
from icia.primitives.collective_intention import AgentState, CollectiveIntentionField


def test_physics_endpoint_logic():
    """Verify physics controller step logic called by web server."""
    obs = physics_env.reset(initial_state=[0.0, 0.0, 0.05, 0.0])
    action, telemetry = physics_controller.compute_action(obs)
    next_obs, reward, done, info = physics_env.step(action)
    
    assert len(next_obs) == 4
    assert isinstance(action, float)
    assert "discrepancy_norm" in telemetry
    assert "shock_detected" in telemetry


def test_multiagent_endpoint_logic():
    """Verify collective intention field step logic called by web server."""
    agents = [
        AgentState(agent_id=0, initial_pos=np.array([0.0, 0.0]), goal_pos=np.array([5.0, 0.0])),
        AgentState(agent_id=1, initial_pos=np.array([5.0, 0.0]), goal_pos=np.array([0.0, 0.0]))
    ]
    field = CollectiveIntentionField(agents=agents, dt=0.05)
    res = field.step()
    assert len(field.agents) == 2
    assert field.agents[0].pos[0] > 0.0
    assert field.agents[1].pos[0] < 5.0
    assert "collisions" in res


def test_hierarchical_endpoint_logic():
    """Verify hierarchical deliberation step logic called by web server."""
    env = hierarchical_env
    controller = build_icia_controller()
    state, facts = env.reset()
    action, telemetry = controller.compute_action(state, facts, discrepancy=0.0)
    
    assert "active_subgoal" in telemetry
    assert telemetry["active_subgoal"] is not None
    assert len(action) == 2


def test_code_synthesis_endpoint_logic():
    """Verify code synthesis endpoint execution logic."""
    from icia.tasks.code_synthesis_task import get_binary_search_task
    from icia.experiments.run_code_synthesis_experiment import run_icia_generator
    task = get_binary_search_task()
    res = run_icia_generator(task)
    assert res["syntax_valid"] is True
    assert res["exec_success"] is True


def test_nl_program_endpoint_logic():
    """Verify NL programmer endpoint execution logic."""
    from icia.code.nl_programmer import NLProgrammerEngine
    engine = NLProgrammerEngine()
    res = engine.compile_and_synthesize("Filter even numbers and square them, sorting in descending order.")
    assert res["syntax_valid"] is True
    assert res["tests_passed"] is True
    assert "process_even_squares" in res["code"]
