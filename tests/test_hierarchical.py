"""
Tests for ICIA Hierarchical Intention Decomposition Engine
"""

import numpy as np
import pytest
from icia.models.hierarchical_engine import HierarchicalIntentionController, SubGoal
from icia.tasks.hierarchical_foraging import HierarchicalForagingEnv
from icia.experiments.run_hierarchical_experiment import build_icia_controller, run_icia_hierarchical


def test_subgoal_satisfaction():
    sg = SubGoal(
        id="SG_TEST",
        description="Reach waypoint",
        target_state=np.array([1.0, 1.0]),
        preconditions=["unlocked"],
        completion_radius=0.5
    )
    
    # Precondition false
    assert not sg.is_satisfied(np.array([1.0, 1.0]), {"unlocked": False})
    
    # Distance too large
    assert not sg.is_satisfied(np.array([3.0, 3.0]), {"unlocked": True})
    
    # Both satisfied
    assert sg.is_satisfied(np.array([1.2, 1.1]), {"unlocked": True})


def test_hierarchical_controller_1step_pivot():
    controller = HierarchicalIntentionController(
        macro_goal=np.array([10.0, 10.0]),
        impasse_threshold=0.8
    )
    controller.decompose([
        SubGoal(
            id="SG_BLOCKED",
            description="Blocked corridor",
            target_state=np.array([5.0, 5.0]),
            metadata={
                "contingencies": [
                    {
                        "id": "SG_ALT",
                        "description": "Clear vent",
                        "target_state": [2.0, 8.0],
                        "completion_radius": 0.4
                    }
                ]
            }
        )
    ])
    
    assert controller.get_active_sub_goal().id == "SG_BLOCKED"
    
    # Deliver high discrepancy indicating impasse
    step, telemetry = controller.compute_action(
        current_state=np.array([4.8, 4.8]),
        world_facts={},
        discrepancy=1.5
    )
    
    assert telemetry["impasse"] is True
    assert telemetry["replan"] is True
    assert controller.total_replan_count == 1
    assert controller.get_active_sub_goal().id == "SG_ALT"


def test_hierarchical_foraging_full_task():
    env = HierarchicalForagingEnv(stochastic_jam=True)
    res = run_icia_hierarchical(env, max_steps=100)
    assert res["success"] is True
    assert res["replan_count"] == 1
    assert res["final_facts"]["vault_unlocked"] is True
    assert res["steps_taken"] < 70
