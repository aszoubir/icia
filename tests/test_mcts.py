"""
Unit Tests for Phase 14: Teleological Monte Carlo Tree Search (T-MCTS).
Verifies that T-MCTS escapes non-convex concave traps and prunes barrier branches.
"""

import numpy as np
from icia.tasks.maze_labyrinth_env import ConcaveLabyrinthEnv
from icia.models.teleological_mcts import TeleologicalMCTS


def test_greedy_gradient_flow_trapped_in_concave_minimum():
    env = ConcaveLabyrinthEnv()
    res = env.run_pure_gradient_flow(max_steps=30)
    assert not res["success"]
    assert res["stuck"]
    assert res["trap_pos"][0] == 4.0 and res["trap_pos"][1] == 5.0


def test_teleological_mcts_escapes_trap_and_reaches_goal():
    env = ConcaveLabyrinthEnv()
    mcts = TeleologicalMCTS(
        c_puct=2.0,
        beta_lyapunov=0.5,
        max_depth=35,
        discount=0.98
    )

    res = mcts.search(
        initial_state=env.start,
        goal_state=env.goal,
        num_simulations=500,
        action_dim=env.action_dim,
        forward_model=env.forward_step,
        barrier_fn=env.is_in_barrier,
        lyapunov_fn=env.lyapunov_potential
    )

    assert res["goal_reached"]
    assert len(res["trajectory"]) > 10
    assert res["barrier_prunes"] > 50
    # Final state must be near goal
    final_pos = res["trajectory"][-1]
    dist_to_goal = np.linalg.norm(final_pos - env.goal)
    assert dist_to_goal < 0.75
