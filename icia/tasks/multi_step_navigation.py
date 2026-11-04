"""
ICIA Benchmark: Multi-Step Stochastic Navigation with Dynamic Roadblocks
Simulates an environment requiring:
1. Long-horizon planning toward intended Goal G
2. Resistance to stochastic environmental slips (p_slip = 0.15)
3. Zero-shot recovery from unannounced dynamic roadblocks
"""

import random
from typing import Tuple, Dict, Any, List
import numpy as np
import torch


class StochasticGridNavigationEnv:
    """
    2D Grid Navigation Environment with stochastic slips and dynamic obstacles.
    """
    def __init__(self, grid_size: int = 8, p_slip: float = 0.15, max_steps: int = 25, seed: int = 42):
        self.grid_size = grid_size
        self.p_slip = p_slip
        self.max_steps = max_steps
        random.seed(seed)
        np.random.seed(seed)

        # Actions: 0: Up (-y), 1: Down (+y), 2: Left (-x), 3: Right (+x)
        self.action_deltas = [
            (0, -1),  # Up
            (0, 1),   # Down
            (-1, 0),  # Left
            (1, 0)    # Right
        ]
        self.reset()

    def reset(self, start_pos: Tuple[int, int] = (0, 0), goal_pos: Tuple[int, int] = (6, 6)) -> np.ndarray:
        self.pos = list(start_pos)
        self.goal = list(goal_pos)
        self.step_count = 0
        self.roadblocks = set()
        return self._get_obs()

    def _get_obs(self) -> np.ndarray:
        """One-hot spatial grid representation of size (grid_size * grid_size)"""
        obs = np.zeros(self.grid_size * self.grid_size, dtype=np.float32)
        idx = self.pos[1] * self.grid_size + self.pos[0]
        obs[idx] = 1.0
        return obs

    def get_goal_vector(self) -> np.ndarray:
        """One-hot spatial representation of goal"""
        vec = np.zeros(self.grid_size * self.grid_size, dtype=np.float32)
        idx = self.goal[1] * self.grid_size + self.goal[0]
        vec[idx] = 1.0
        return vec

    def inject_roadblock(self, pos: Tuple[int, int]):
        self.roadblocks.add(pos)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        self.step_count += 1

        # Apply stochastic slip with probability p_slip
        if random.random() < self.p_slip:
            # Random perpendicular slip
            action = random.choice([0, 1, 2, 3])

        dx, dy = self.action_deltas[action]
        new_x = max(0, min(self.grid_size - 1, self.pos[0] + dx))
        new_y = max(0, min(self.grid_size - 1, self.pos[1] + dy))

        # Check roadblock collision
        collision = (new_x, new_y) in self.roadblocks
        if not collision:
            self.pos = [new_x, new_y]

        dist = abs(self.pos[0] - self.goal[0]) + abs(self.pos[1] - self.goal[1])
        reached_goal = (dist == 0)
        done = reached_goal or (self.step_count >= self.max_steps)

        reward = 10.0 if reached_goal else (-0.1 - 0.05 * dist)
        if collision:
            reward -= 1.0

        info = {
            "pos": tuple(self.pos),
            "goal": tuple(self.goal),
            "distance_to_goal": dist,
            "reached_goal": reached_goal,
            "collision": collision
        }
        return self._get_obs(), reward, done, info
