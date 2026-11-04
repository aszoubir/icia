"""
Unit Tests: ICIA Engines (NumPy Reference & PyTorch Autonomous Engine)
"""

import numpy as np
import torch
import pytest
from icia.models.numpy_engine import NumPyDIVREModel
from icia.models.icia_engine import ICIAAutonomousEngine


def test_numpy_divre_forward_pass():
    """Verify that pure NumPy DIVRE model processes tokens and computes discrepancy."""
    model = NumPyDIVREModel(vocab_size=32, d_model=16, seed=42)
    tokens = [1, 5, 12, 18, 22]
    goal_vec = np.ones(16) * 0.5
    
    logits, errors = model.forward(tokens, goal_vec)
    
    assert logits.shape == (5, 32)
    assert len(errors) == 5
    assert all(e >= 0.0 for e in errors)


def test_autonomous_engine_closed_loop_step():
    """Verify that PyTorch autonomous engine executes closed-loop step and produces actions."""
    engine = ICIAAutonomousEngine(d_obs=16, d_state=32, d_action=4, d_intention=32)
    engine.eval()
    
    device = torch.device("cpu")
    t_0, v_0, m_k, m_v = engine.init_state(batch_size=2, device=device)
    
    x_obs = torch.randn(2, 16)
    goal = torch.randn(2, 32)
    
    with torch.no_grad():
        out = engine.step(x_obs, goal, t_0, v_0, m_k, m_v)
        
    assert out["logits"].shape == (2, 4)
    assert out["action"].shape == (2, 4)
    assert out["t_state"].shape == (2, 32)
    assert out["v_state"].shape == (2, 32)
    assert out["reality_error"].shape == (2, 32)
