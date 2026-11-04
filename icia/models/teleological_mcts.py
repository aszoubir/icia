"""
Teleological Monte Carlo Tree Search (T-MCTS) for ICIA.
Provides Test-Time Compute Scaling by exploring latent intention trajectories
over the Internal World Model W_theta before committing to execution.
Overcomes non-convex local minima, concave trap barriers, and labyrinth deadlocks.
"""

from __future__ import annotations
import math
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Callable
import numpy as np


@dataclass
class MCTSNode:
    state: np.ndarray                   # Latent/physical state vector S
    parent: Optional[MCTSNode] = None
    action_from_parent: Optional[int] = None
    children: Dict[int, MCTSNode] = field(default_factory=dict)
    visit_count: int = 0
    total_value: float = 0.0            # Cumulative Q value
    lyapunov_energy: float = 0.0
    is_terminal: bool = False
    is_barrier_blocked: bool = False

    @property
    def q_value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count


class TeleologicalMCTS:
    """
    Test-Time Search Scaling for ICIA:
    Combines Monte Carlo Tree Search with Lyapunov potential flow:
    - Node selection guided by Teleological-UCT (T-UCT)
    - Hard barrier branches and ancestor cycles pruned immediately
    - Backpropagates goal convergence value up the tree
    """

    def __init__(
        self,
        c_puct: float = 2.0,
        beta_lyapunov: float = 0.5,
        max_depth: int = 40,
        discount: float = 0.98
    ):
        self.c_puct = c_puct
        self.beta_lyapunov = beta_lyapunov
        self.max_depth = max_depth
        self.discount = discount

    def search(
        self,
        initial_state: np.ndarray,
        goal_state: np.ndarray,
        num_simulations: int,
        action_dim: int,
        forward_model: Callable[[np.ndarray, int], Tuple[np.ndarray, bool, float]],
        barrier_fn: Callable[[np.ndarray], bool],
        lyapunov_fn: Callable[[np.ndarray, np.ndarray], float]
    ) -> Dict[str, Any]:
        """
        Runs T-MCTS from initial_state towards goal_state.
        forward_model: (state, action) -> (next_state, is_goal_reached, step_cost)
        barrier_fn: (state) -> True if state violates barrier, False if admissible
        lyapunov_fn: (state, goal) -> scalar potential energy V(S, G) >= 0
        """
        t0 = time.perf_counter()
        root = MCTSNode(
            state=initial_state.copy(),
            lyapunov_energy=lyapunov_fn(initial_state, goal_state),
            is_terminal=bool(np.linalg.norm(initial_state - goal_state) < 0.75)
        )

        nodes_expanded = 1
        barrier_prunes = 0

        for sim_idx in range(num_simulations):
            node = root
            search_path = [node]

            # 1. Selection
            while node.children and not node.is_terminal and len(search_path) < self.max_depth:
                best_action = None
                best_score = -float("inf")

                for action, child in node.children.items():
                    if child.is_barrier_blocked:
                        continue

                    # UCT exploration bonus
                    u_score = self.c_puct * math.sqrt(node.visit_count) / (1 + child.visit_count)
                    score = child.q_value + u_score

                    if score > best_score:
                        best_score = score
                        best_action = action

                if best_action is None:
                    node.is_barrier_blocked = True
                    break

                node = node.children[best_action]
                search_path.append(node)

            if node.is_barrier_blocked:
                self._backpropagate(search_path, -50.0)
                continue

            # 2. Expansion
            if not node.is_terminal and len(search_path) < self.max_depth:
                for a in range(action_dim):
                    cand = node.state.copy()
                    next_s, reached, cost = forward_model(node.state, a)

                    # Collision check
                    if barrier_fn(next_s) or np.allclose(next_s, node.state):
                        barrier_prunes += 1
                        child = MCTSNode(
                            state=next_s,
                            parent=node,
                            action_from_parent=a,
                            is_barrier_blocked=True
                        )
                        node.children[a] = child
                        continue

                    # Ancestor cycle check
                    if any(np.allclose(next_s, p.state) for p in search_path):
                        barrier_prunes += 1
                        child = MCTSNode(
                            state=next_s,
                            parent=node,
                            action_from_parent=a,
                            is_barrier_blocked=True
                        )
                        node.children[a] = child
                        continue

                    lyap_v = lyapunov_fn(next_s, goal_state)
                    child = MCTSNode(
                        state=next_s,
                        parent=node,
                        action_from_parent=a,
                        lyapunov_energy=lyap_v,
                        is_terminal=reached
                    )
                    node.children[a] = child
                    nodes_expanded += 1

                admissible = [c for c in node.children.values() if not c.is_barrier_blocked]
                if admissible:
                    admissible.sort(key=lambda c: c.lyapunov_energy)
                    node = admissible[0]
                    search_path.append(node)
                else:
                    node.is_barrier_blocked = True
                    self._backpropagate(search_path, -50.0)
                    continue

            # 3. Evaluation
            depth = len(search_path)
            if node.is_terminal:
                value = 250.0 - (depth * 2.0)
            else:
                value = -0.5 * node.lyapunov_energy - (0.2 * depth)

            # 4. Backpropagation
            self._backpropagate(search_path, value)

        # Extract planned trajectory
        curr = root
        best_plan = []
        best_states = [curr.state.copy()]
        while curr.children and not curr.is_terminal:
            valid_children = [c for c in curr.children.values() if not c.is_barrier_blocked]
            if not valid_children:
                break
            best_child = max(valid_children, key=lambda c: c.visit_count)
            best_plan.append(best_child.action_from_parent)
            best_states.append(best_child.state.copy())
            if best_child.is_terminal:
                break
            curr = best_child

        dt_ms = (time.perf_counter() - t0) * 1000.0
        goal_reached = (len(best_states) > 0 and np.linalg.norm(best_states[-1] - goal_state) < 0.75)

        return {
            "best_actions": best_plan,
            "trajectory": best_states,
            "goal_reached": bool(goal_reached),
            "nodes_expanded": nodes_expanded,
            "barrier_prunes": barrier_prunes,
            "search_latency_ms": dt_ms,
            "root_visits": root.visit_count
        }

    def _backpropagate(self, path: List[MCTSNode], leaf_value: float):
        curr_val = leaf_value
        for node in reversed(path):
            node.visit_count += 1
            node.total_value += curr_val
            curr_val *= self.discount
