"""
ICIA Continuous Physics Environment: Non-linear Continuous Inverted Pendulum
Simulates full non-linear continuous Newtonian dynamics with gravity, inertia,
rail boundary constraints, and unannounced external force impulses.
"""

import math
from typing import Tuple, Dict, Any, Optional
import numpy as np


class ContinuousCartPoleEnv:
    """
    Continuous Non-linear Cart-Pole Dynamical System.
    Governed by non-linear ODEs:
      ddot_theta = (g*sin(theta) + cos(theta)*((-F - m*l*theta_dot^2*sin(theta))/(M + m))) /
                   (l * (4/3 - (m*cos^2(theta))/(M + m)))
      ddot_x = (F + m*l*(theta_dot^2*sin(theta) - ddot_theta*cos(theta))) / (M + m)
    """
    def __init__(
        self,
        gravity: float = 9.81,
        cart_mass: float = 1.0,
        pole_mass: float = 0.1,
        pole_length: float = 0.5, # half-length (center of mass)
        tau: float = 0.02, # 50 Hz integration timestep
        x_threshold: float = 2.4, # rail boundary in meters
        theta_threshold: float = math.pi / 3.0 # 60 degrees failure angle
    ):
        self.gravity = gravity
        self.cart_mass = cart_mass
        self.pole_mass = pole_mass
        self.total_mass = cart_mass + pole_mass
        self.pole_length = pole_length
        self.pole_mass_length = pole_mass * pole_length
        self.tau = tau
        self.x_threshold = x_threshold
        self.theta_threshold = theta_threshold

        self.state: Optional[np.ndarray] = None
        self.steps = 0
        self.external_impulse = 0.0

    def reset(self, initial_state: Optional[np.ndarray] = None) -> np.ndarray:
        if initial_state is not None:
            self.state = np.array(initial_state, dtype=np.float64)
        else:
            # Small random perturbation around upright equilibrium
            self.state = np.random.uniform(low=-0.04, high=0.04, size=(4,))
        self.steps = 0
        self.external_impulse = 0.0
        return self.state.copy()

    def apply_impulse(self, force_magnitude: float):
        """Injects an unannounced external force impulse (e.g., wind gust or collision shock)."""
        self.external_impulse = force_magnitude

    def step(self, action_force: float) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Executes one physical integration step using semi-implicit Euler / RK2.
        action_force: Continuous force in Newtons.
        """
        x, x_dot, theta, theta_dot = self.state
        
        # Total applied force = actuator force + external physical impulse
        total_force = np.clip(action_force, -20.0, 20.0) + self.external_impulse
        self.external_impulse = 0.0 # Impulse is instantaneous

        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        temp = (total_force + self.pole_mass_length * theta_dot * theta_dot * sintheta) / self.total_mass
        theta_acc = (self.gravity * sintheta - costheta * temp) / (
            self.pole_length * (4.0 / 3.0 - self.pole_mass * costheta * costheta / self.total_mass)
        )
        x_acc = temp - self.pole_mass_length * theta_acc * costheta / self.total_mass

        # Numerical integration (Symplectic Euler)
        x_dot = x_dot + self.tau * x_acc
        x = x + self.tau * x_dot
        theta_dot = theta_dot + self.tau * theta_acc
        theta = theta + self.tau * theta_dot

        self.state = np.array([x, x_dot, theta, theta_dot], dtype=np.float64)
        self.steps += 1

        # Check termination / boundary failure
        rail_violation = abs(x) > self.x_threshold
        angle_violation = abs(theta) > self.theta_threshold
        done = rail_violation or angle_violation

        # Teleological potential reward: negative of quadratic deviation
        cost = (
            5.0 * (theta ** 2) +
            0.5 * (theta_dot ** 2) +
            1.0 * (x ** 2) +
            0.1 * (x_dot ** 2) +
            0.001 * (action_force ** 2)
        )
        reward = -cost

        info = {
            "rail_violation": rail_violation,
            "angle_violation": angle_violation,
            "total_force": total_force,
            "step": self.steps
        }

        return self.state.copy(), reward, done, info
