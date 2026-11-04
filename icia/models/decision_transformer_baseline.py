"""
Open-Loop Sequence Baseline Controller (Decision Transformer Paradigm)
Predicts control force without closed-loop reality discrepancy feedback.
Vulnerable to unexpected external impulse shocks.
"""

from typing import Dict, Any, Tuple
import numpy as np


class OpenLoopSequenceController:
    """
    Open-loop trajectory controller modeling standard sequence-to-action policies.
    Lacks internal reality discrepancy gating.
    """
    def __init__(self, k_p: float = 18.0, k_d: float = 4.0):
        self.k_p = k_p
        self.k_d = k_d

    def compute_action(self, current_state: np.ndarray) -> Tuple[float, Dict[str, Any]]:
        x, x_dot, theta, theta_dot = current_state
        # Open loop policy: ignores rail boundaries and does not monitor prediction discrepancies
        action_force = self.k_p * theta + self.k_d * theta_dot
        action_force = float(np.clip(action_force, -15.0, 15.0))
        return action_force, {"action_force": action_force, "discrepancy_norm": 0.0}
