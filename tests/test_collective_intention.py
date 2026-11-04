"""
Unit Tests: ICIA Collective Intention Field & Multi-Agent Coordination
"""

import numpy as np
import pytest
from icia.primitives.collective_intention import AgentState, CollectiveIntentionField
from icia.tasks.multi_agent_coordination import BottleneckCorridorEnv


def test_collective_intention_two_agent_crossing():
    """Verify that 2 agents cross the corridor bottleneck with 100% success and 0 collisions."""
    env = BottleneckCorridorEnv(num_agents=2)
    field = env.create_field()
    
    for _ in range(250):
        res = field.step()
        if res["all_reached"]:
            break
            
    assert all(a.reached_goal for a in field.agents)
    assert field.collisions_recorded == 0


def test_collective_intention_four_agent_intersection():
    """Verify that 4 agents cross the intersection with 100% success and 0 collisions."""
    env = BottleneckCorridorEnv(num_agents=4)
    field = env.create_field()
    
    for _ in range(350):
        res = field.step()
        if res["all_reached"]:
            break
            
    assert all(a.reached_goal for a in field.agents)
    assert field.collisions_recorded == 0
