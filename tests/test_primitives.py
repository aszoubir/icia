"""
Unit Tests: ICIA Mathematical Primitives & Dynamical Systems Invariants
"""

import numpy as np
import torch
import pytest
from icia.primitives.intention_state import IntentionState
from icia.primitives.adaptive_memory import IntentionGatedAssociativeMemory


def test_intention_state_mathematical_invariants():
    """Verify Riemannian potential and barrier calculation."""
    goal = np.array([5.0, 5.0])
    constraints = np.array([[1.0, 0.0, -10.0]]) # 1*x_0 + 0*x_1 - 10 <= 0 => x_0 <= 10
    priorities = np.array([1.0])
    belief = np.array([0.0, 0.0])
    uncertainty = np.array([0.1])
    
    intent = IntentionState(
        goal=goal,
        constraints=constraints,
        priorities=priorities,
        belief=belief,
        uncertainty=uncertainty
    )
    
    # At goal, distance is zero
    assert intent.compute_goal_distance(np.array([5.0, 5.0])) == pytest.approx(0.0)
    
    # Distance to (2.0, 1.0) should be 5.0 (3^2 + 4^2 = 25)
    assert intent.compute_goal_distance(np.array([2.0, 1.0])) == pytest.approx(5.0)

    # Safe state vs violation
    safe_state = np.array([8.0, 0.0])
    viol_state = np.array([12.0, 0.0])
    
    assert intent.compute_constraint_violation(safe_state) == pytest.approx(0.0)
    assert intent.compute_constraint_violation(viol_state) == pytest.approx(2.0)


def test_igam_surprise_gated_associative_memory():
    """Verify surprise-gated writes and intention-addressed recall."""
    d_key, d_val, num_slots = 16, 32, 8
    mem = IntentionGatedAssociativeMemory(num_slots=num_slots, d_key=d_key, d_val=d_val)
    device = torch.device("cpu")
    
    keys, values = mem.init_memory(batch_size=2, device=device)
    assert keys.shape == (2, num_slots, d_key)
    assert values.shape == (2, num_slots, d_val)

    # Read context indexed by intention
    intent = torch.randn(2, d_key)
    context = mem.read(intent, keys, values)
    assert context.shape == (2, d_val)

    # Perform surprise-gated write with discrepancy
    new_val = torch.randn(2, d_val)
    discrepancy = torch.ones(2, d_val) * 2.5 # High surprise
    up_keys, up_values = mem.write(new_val, discrepancy, keys, values, intent)
    
    assert up_keys.shape == keys.shape
    assert up_values.shape == values.shape
    # Values should be modified due to surprise update
    assert not torch.allclose(up_values, values)
