"""
Non-Convex Concave Labyrinth Benchmark Environment.
Designed to test local minimum traps where pure gradient flow fails
and test-time tree search (T-MCTS) is mathematically necessary.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, List, Dict, Any
import numpy as np


class ConcaveLabyrinthEnv:
    """
    2D continuous grid environment with a U-shaped concave trap.
    Grid size: 10 x 10.
    Start: [1.0, 5.0]
    Goal:  [8.0, 5.0]
    Concave Wall:
      Back wall: x = 5.0, y in [3.0, 7.0]
      North arm: y = 7.0, x in [3.0, 5.0]
      South arm: y = 3.0, x in [3.0, 5.0]
      Opening faces West (towards start position).
    """

    def __init__(self):
        self.start = np.array([1.0, 5.0], dtype=np.float32)
        self.goal = np.array([8.0, 5.0], dtype=np.float32)
        self.actions = [
            np.array([1.0, 0.0]),   # 0: East
            np.array([-1.0, 0.0]),  # 1: West
            np.array([0.0, 1.0]),   # 2: North
            np.array([0.0, -1.0])   # 3: South
        ]
        self.action_dim = 4

    def is_in_barrier(self, pos: np.ndarray) -> bool:
        """Returns True if position collides with the concave obstacle."""
        x, y = pos[0], pos[1]
        # Out of bounds [0, 10]
        if x < 0.0 or x > 10.0 or y < 0.0 or y > 10.0:
            return True

        # Back wall: x in [4.8, 5.2], y in [2.8, 7.2]
        if 4.8 <= x <= 5.2 and 2.8 <= y <= 7.2:
            return True

        # North arm: y in [6.8, 7.2], x in [2.8, 5.2]
        if 2.8 <= x <= 5.2 and 6.8 <= y <= 7.2:
            return True

        # South arm: y in [2.8, 3.2], x in [2.8, 5.2]
        if 2.8 <= x <= 5.2 and 2.8 <= y <= 3.2:
            return True

        return False

    def forward_step(self, state: np.ndarray, action: int) -> Tuple[np.ndarray, bool, float]:
        """(state, action) -> (next_state, is_goal_reached, step_cost)"""
        act_vec = self.actions[action]
        next_pos = state + act_vec

        if self.is_in_barrier(next_pos):
            # Collision: rebound to current position with heavy penalty
            return state.copy(), False, 10.0

        is_reached = float(np.linalg.norm(next_pos - self.goal)) < 0.75
        step_cost = 1.0
        return next_pos, is_reached, step_cost

    def lyapunov_potential(self, state: np.ndarray, goal: np.ndarray) -> float:
        """Euclidean distance squared to goal."""
        return float(np.sum((state - goal) ** 2))

    def run_pure_gradient_flow(self, max_steps: int = 50) -> Dict[str, Any]:
        """
        Simulates standard greedy Riemannian gradient descent without search.
        Stops when reaching local minimum (stuck in concave trap).
        """
        pos = self.start.copy()
        path = [pos.copy()]
        stuck_cycles = 0

        for step in range(max_steps):
            if np.linalg.norm(pos - self.goal) < 0.75:
                return {"success": True, "steps": step, "path": path, "stuck": False}

            # Greedy step towards -grad V
            best_action = None
            best_dist = float("inf")
            for a_idx, act in enumerate(self.actions):
                cand = pos + act
                if not self.is_in_barrier(cand):
                    dist = np.linalg.norm(cand - self.goal)
                    if dist < best_dist:
                        best_dist = dist
                        best_action = a_idx

            if best_action is None or best_dist >= np.linalg.norm(pos - self.goal):
                stuck_cycles += 1
                if stuck_cycles >= 3:
                    # Trapped in local minimum
                    return {"success": False, "steps": step, "path": path, "stuck": True, "trap_pos": pos.tolist()}
            else:
                stuck_cycles = 0
                pos = pos + self.actions[best_action]
                path.append(pos.copy())

        return {"success": False, "steps": max_steps, "path": path, "stuck": True, "trap_pos": pos.tolist()}
