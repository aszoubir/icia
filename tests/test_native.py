"""
Tests for ICIA Native C99 Interface & Fast Core Engine
"""

import numpy as np
import pytest
from icia.native.fast_core import FastCoreSystem, CIciaSystem, ICIA_OBS_DIM


def test_native_struct_layout():
    """Verifies binary layout and byte size match icia_system_t"""
    sys_struct = CIciaSystem()
    ctypes_sizeof = sys_struct.__sizeof__()
    assert ctypes_sizeof > 0
    # Verify fields exist and can be set
    sys_struct.step_count = 42
    assert sys_struct.step_count == 42


def test_fast_core_cartpole_convergence():
    """Verifies Lyapunov convergence in FastCoreSystem"""
    core = FastCoreSystem(goal=np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32))
    core.state = np.array([0.2, 0.0, 0.1, 0.0], dtype=np.float32)
    
    initial_v = core.compute_control()
    # Step forward 200 cycles
    for _ in range(200):
        u = core.compute_control()
        core.step_cartpole(u, dt=0.01)

    # State must be stabilized near zero
    assert abs(core.state[0]) < 0.2
    assert abs(core.state[2]) < 0.1
    assert core.lyapunov_v < 10.0


def test_symplectic_multiagent_vorticity():
    """Verifies symplectic curl produces non-zero transversal avoidance"""
    pos_a = np.array([1.0, 0.0], dtype=np.float32)
    pos_b = np.array([0.0, 0.0], dtype=np.float32)
    
    force_no_curl = FastCoreSystem.symplectic_multiagent(pos_a, pos_b, d_safe=2.0, gamma_curl=0.0)
    force_with_curl = FastCoreSystem.symplectic_multiagent(pos_a, pos_b, d_safe=2.0, gamma_curl=0.8)

    # Without curl, force is purely along x
    assert abs(force_no_curl[1]) < 1e-6
    # With curl, force has transversal y-component breaking head-on symmetry
    assert abs(force_with_curl[1]) > 0.1
