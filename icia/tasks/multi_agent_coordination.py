"""
ICIA Multi-Agent Coordination Task: Symmetric Bottleneck Corridor
Tests deadlock resolution and dynamic priority arbitration in narrow spatial bottlenecks.
"""

from typing import List, Tuple, Dict, Any
import numpy as np
from icia.primitives.collective_intention import AgentState, CollectiveIntentionField


class BottleneckCorridorEnv:
    """
    Symmetric 2-agent or 4-agent corridor bottleneck environment.
    Corridor walls restrict passage to a single-agent width at x in [-1.0, 1.0].
    """
    def __init__(self, num_agents: int = 2):
        self.num_agents = num_agents
        self.corridor_half_width = 0.55 # Narrow bottleneck
        self.obstacles: List[Tuple[np.ndarray, float]] = []

        # Create upper and lower obstacle walls defining the corridor
        for x_val in np.linspace(-2.0, 2.0, 9):
            self.obstacles.append((np.array([x_val, 1.2]), 0.5))
            self.obstacles.append((np.array([x_val, -1.2]), 0.5))

    def create_field(self) -> CollectiveIntentionField:
        agents = []
        if self.num_agents == 2:
            # Agent 0: Moves Left to Right (Start: -3.0, Goal: +3.0)
            agents.append(AgentState(agent_id=0, initial_pos=[-3.0, 0.0], goal_pos=[3.0, 0.0], priority=1.2))
            # Agent 1: Moves Right to Left (Start: +3.0, Goal: -3.0)
            agents.append(AgentState(agent_id=1, initial_pos=[3.0, 0.0], goal_pos=[-3.0, 0.0], priority=0.8))
        elif self.num_agents == 4:
            # 4-way crossway intersection
            agents.append(AgentState(agent_id=0, initial_pos=[-3.0, 0.0], goal_pos=[3.0, 0.0], priority=1.5))
            agents.append(AgentState(agent_id=1, initial_pos=[3.0, 0.0], goal_pos=[-3.0, 0.0], priority=1.0))
            agents.append(AgentState(agent_id=2, initial_pos=[0.0, -3.0], goal_pos=[0.0, 3.0], priority=0.8))
            agents.append(AgentState(agent_id=3, initial_pos=[0.0, 3.0], goal_pos=[0.0, -3.0], priority=0.5))

        return CollectiveIntentionField(agents=agents, obstacles=self.obstacles, dt=0.04)
