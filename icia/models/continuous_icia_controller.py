"""
ICIA Continuous Controller: Continuous Teleological Potential Flow with Reality Innovation Gate
Directly evaluates Euclidean teleological gradient flow (identity metric, hand-tuned gains):
  u_t = - K_goal * ∇Ψ_G(S) - mu * ∇B(x) + K_error * E_t
"""

import math
from typing import Dict, Any, Tuple, Optional
import numpy as np


class ContinuousICIAController:
    """
    Continuous Teleological Controller for Embodied Physical Systems.
    Combines:
    1. Teleological Potential Gradient: -∇Ψ_G(S)
    2. Boundary Barrier Repulsion: -∇B(x)
    3. Internal Forward Prediction Model: W_theta(S_{t-1}, u_{t-1})
    4. Closed-loop Reality Innovation Discrepancy Gate: E_t = S_t - \hat{S}_t
    """
    def __init__(
        self,
        q_theta: float = 35.0,
        q_omega: float = 8.5,
        q_x: float = 4.0,
        q_v: float = 3.5,
        x_max: float = 2.4,
        barrier_weight: float = 0.5,
        innovation_gain: float = 12.0,
        dt: float = 0.02
    ):
        self.q_theta = q_theta
        self.q_omega = q_omega
        self.q_x = q_x
        self.q_v = q_v
        self.x_max = x_max
        self.barrier_weight = barrier_weight
        self.innovation_gain = innovation_gain
        self.dt = dt

        # Forward transition state prediction: \hat{S}_t
        self.last_state: Optional[np.ndarray] = None
        self.last_action: float = 0.0
        self.predicted_state: Optional[np.ndarray] = None

    def internal_world_model_step(self, state: np.ndarray, action: float) -> np.ndarray:
        """
        Internal forward physics model: analytic cartpole dynamics (closed-form, not learned).
        Predicts expected next state under nominal dynamics.
        """
        x, x_dot, theta, theta_dot = state
        g, m, M, l = 9.81, 0.1, 1.0, 0.5
        total_mass = M + m
        pole_mass_length = m * l

        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        temp = (action + pole_mass_length * theta_dot * theta_dot * sintheta) / total_mass
        theta_acc = (g * sintheta - costheta * temp) / (
            l * (4.0 / 3.0 - m * costheta * costheta / total_mass)
        )
        x_acc = temp - pole_mass_length * theta_acc * costheta / total_mass

        x_dot_next = x_dot + self.dt * x_acc
        x_next = x + self.dt * x_dot_next
        theta_dot_next = theta_dot + self.dt * theta_acc
        theta_next = theta + self.dt * theta_dot_next

        return np.array([x_next, x_dot_next, theta_next, theta_dot_next])

    def compute_action(self, current_state: np.ndarray) -> Tuple[float, Dict[str, Any]]:
        """
        Computes continuous control force in real time via teleological gradient flow.
        """
        x, x_dot, theta, theta_dot = current_state

        # 1. Reality Discrepancy Gate: E_t = S_t - \hat{S}_t
        if self.predicted_state is not None:
            discrepancy = current_state - self.predicted_state
            discrepancy_norm = float(np.linalg.norm(discrepancy))
            shock_detected = discrepancy_norm > 0.05
        else:
            discrepancy = np.zeros(4)
            discrepancy_norm = 0.0
            shock_detected = False

        # 2. Teleological Potential Gradient Pull toward (theta=0, x=0)
        # Note: To accelerate theta upright, positive theta requires positive cart force
        teleological_pull = (
            self.q_theta * theta +
            self.q_omega * theta_dot +
            self.q_x * x +
            self.q_v * x_dot
        )

        # 3. Rail Boundary Barrier Repulsion: ∇B(x) = 2x / (x_max^2 - x^2)
        dist_to_rail = max(0.05, (self.x_max ** 2) - (x ** 2))
        barrier_repulsion = self.barrier_weight * (2.0 * x / dist_to_rail)

        # 4. Closed-Loop Innovation Compensation: K * E_t
        # Directly opposes unmodeled physical shocks (e.g. wind impulse on theta_dot or x_dot)
        innovation_force = -self.innovation_gain * (discrepancy[3] * 2.0 + discrepancy[1])

        # Synthesized continuous actuator force
        action_force = teleological_pull + barrier_repulsion + innovation_force
        action_force = float(np.clip(action_force, -20.0, 20.0))

        # 5. Predict next state in imagination using internal world model
        self.predicted_state = self.internal_world_model_step(current_state, action_force)
        self.last_state = current_state.copy()
        self.last_action = action_force

        telemetry = {
            "discrepancy_norm": discrepancy_norm,
            "shock_detected": shock_detected,
            "teleological_pull": teleological_pull,
            "barrier_repulsion": barrier_repulsion,
            "action_force": action_force
        }

        return action_force, telemetry
