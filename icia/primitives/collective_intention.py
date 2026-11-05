"""
ICIA Primitive: Collective Intention Field & Multi-Agent Coordination Algebra
Implements coupled teleological potential flow on the 2D Euclidean plane with inter-agent barrier repulsion.
"""

import math
from typing import Dict, Any, List, Tuple, Optional
import numpy as np


class AgentState:
    """Represents an individual agent within a collective intention field."""
    def __init__(
        self,
        agent_id: int,
        initial_pos: np.ndarray,
        goal_pos: np.ndarray,
        radius: float = 0.4,
        max_speed: float = 1.5,
        priority: float = 1.0
    ):
        self.agent_id = agent_id
        self.pos = np.array(initial_pos, dtype=np.float64)
        self.goal = np.array(goal_pos, dtype=np.float64)
        self.radius = radius
        self.max_speed = max_speed
        self.priority = priority
        self.velocity = np.zeros_like(self.pos)
        self.reached_goal = False

    def distance_to_goal(self) -> float:
        return float(np.linalg.norm(self.pos - self.goal))


class CollectiveIntentionField:
    """
    Manages N coupled agents on a continuous Riemannian manifold.
    Dynamics:
      dS^(i)/dt = - α_i ∇Ψ_goal(S^(i)) - ∑_j ∇B_ij(S^(i), S^(j)) - ∑_k ∇B_obstacle(S^(i), O_k)
    """
    def __init__(
        self,
        agents: List[AgentState],
        obstacles: Optional[List[Tuple[np.ndarray, float]]] = None, # (pos, radius)
        dt: float = 0.05,
        safety_margin: float = 0.2
    ):
        self.agents = agents
        self.obstacles = obstacles or []
        self.dt = dt
        self.safety_margin = safety_margin
        self.collisions_recorded = 0

    def compute_gradient_for_agent(self, agent_idx: int) -> np.ndarray:
        ai = self.agents[agent_idx]
        if ai.reached_goal:
            return np.zeros_like(ai.pos)

        # 1. Teleological Goal Potential Pull
        diff_g = ai.goal - ai.pos
        dist_g = np.linalg.norm(diff_g)
        if dist_g < 0.15:
            ai.reached_goal = True
            return np.zeros_like(ai.pos)

        grad_goal = (diff_g / (dist_g + 1e-6)) * min(ai.max_speed, dist_g * 1.5)

        # 2. Inter-Agent Collision Avoidance Barrier
        grad_avoidance = np.zeros_like(ai.pos)
        for j, aj in enumerate(self.agents):
            if j == agent_idx:
                continue
            diff_agents = ai.pos - aj.pos
            dist_agents = np.linalg.norm(diff_agents)
            min_dist = ai.radius + aj.radius + self.safety_margin

            if dist_agents < (ai.radius + aj.radius):
                self.collisions_recorded += 1

            if dist_agents < min_dist and not aj.reached_goal:
                overlap = min_dist - dist_agents
                # Radial log-barrier repulsive force
                radial = (diff_agents / (dist_agents + 1e-6)) * (overlap / (dist_agents - (ai.radius + aj.radius) + 1e-4))
                # Anti-symmetric rotation term (J = [[0, -1], [1, 0]]): torque that breaks
                # head-on deadlocks without adding to the radial potential energy.
                tangential = np.array([-diff_agents[1], diff_agents[0]]) / (dist_agents + 1e-6)
                p_ratio = aj.priority / (ai.priority + 1e-6)
                grad_avoidance += (radial + 0.8 * tangential) * (1.0 + p_ratio)

        # 3. Environmental Obstacle Barrier
        grad_obs = np.zeros_like(ai.pos)
        for obs_pos, obs_radius in self.obstacles:
            diff_obs = ai.pos - obs_pos
            dist_obs = np.linalg.norm(diff_obs)
            min_obs_dist = ai.radius + obs_radius + self.safety_margin
            if dist_obs < min_obs_dist:
                overlap = min_obs_dist - dist_obs
                repulsion = (diff_obs / (dist_obs + 1e-6)) * (overlap / (dist_obs - (ai.radius + obs_radius) + 1e-4))
                grad_obs += repulsion

        # Synthesize coupled velocity
        total_flow = grad_goal + grad_avoidance * 1.2 + grad_obs * 1.5
        speed = np.linalg.norm(total_flow)
        if speed > ai.max_speed:
            total_flow = (total_flow / speed) * ai.max_speed

        return total_flow

    def step(self) -> Dict[str, Any]:
        """Synchronously advances all agents in the collective intention field."""
        new_velocities = [self.compute_gradient_for_agent(i) for i in range(len(self.agents))]
        
        for i, ai in enumerate(self.agents):
            ai.velocity = new_velocities[i]
            ai.pos += ai.velocity * self.dt

        all_reached = all(a.reached_goal for a in self.agents)
        return {
            "all_reached": all_reached,
            "collisions": self.collisions_recorded,
            "positions": [a.pos.copy() for a in self.agents]
        }
