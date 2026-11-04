"""
ICIA Primitive: Intention State Formalization
Mathematical representation of I_t = (G_t, C_t, P_t, B_t, U_t, H_t, R_t, V_t)
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@dataclass
class IntentionState:
    """
    Formal 8-tuple Intention representation on a continuous-symbolic manifold:
    I_t = (G_t, C_t, P_t, B_t, U_t, H_t, R_t, V_t)
    """
    # G_t: Goal Manifold representation (dim: d_g)
    goal: np.ndarray
    # C_t: Constraint boundaries (k constraints, dim: k x d_c)
    constraints: np.ndarray
    # P_t: Priority simplex vector (dim: m)
    priorities: np.ndarray
    # B_t: Epistemic Belief state (dim: d_b)
    belief: np.ndarray
    # U_t: Uncertainty covariance/entropy tensor (dim: d_b x d_b or scalar)
    uncertainty: np.ndarray
    # H_t: Temporal horizon (scalar in R^+)
    horizon: float = 100.0
    # R_t: Computational Resource budget (FLOPs, Memory, Max Steps)
    resource_budget: Dict[str, float] = field(default_factory=lambda: {"max_steps": 50, "flops_cap": 1e6})
    # V_t: Verification threshold (scalar in [0, 1])
    verification_threshold: float = 0.95
    # Teleological Error History (Reality Discrepancy metric)
    teleological_error: float = 0.0

    def compute_goal_distance(self, state: np.ndarray) -> float:
        """Computes geodesic distance on the belief manifold: ||Proj_G(S) - G||_2"""
        dim = min(len(state), len(self.goal))
        return float(np.linalg.norm(state[:dim] - self.goal[:dim]))

    def compute_constraint_violation(self, state: np.ndarray) -> float:
        """
        Evaluates boundary conditions: sum_i max(0, c_i(S))
        Constraints are defined as hyperplanes: w_i . S + b_i <= 0
        """
        if self.constraints.size == 0:
            return 0.0
        # If constraints are provided as (k, d+1) matrix [W | b]
        k, d_plus_1 = self.constraints.shape
        d = d_plus_1 - 1
        W = self.constraints[:, :d]
        b = self.constraints[:, d]
        s_proj = state[:d]
        violations = np.dot(W, s_proj) + b
        return float(np.sum(np.maximum(0.0, violations)))

    def verify_success(self, state: np.ndarray) -> bool:
        """Verifies if state fulfills the goal within verification threshold without constraint violation"""
        dist = self.compute_goal_distance(state)
        viol = self.compute_constraint_violation(state)
        fidelity = 1.0 / (1.0 + dist)
        return (fidelity >= self.verification_threshold) and (viol == 0.0)

    def to_vector(self) -> np.ndarray:
        """Flattens primary intention components into a unified conditioning vector"""
        parts = [
            self.goal.flatten(),
            self.priorities.flatten(),
            self.belief.flatten(),
            np.array([self.horizon, self.teleological_error], dtype=np.float32)
        ]
        return np.concatenate(parts)


if HAS_TORCH:
    class NeuralIntentionEncoder(nn.Module):
        """
        Differentiable neural projection of structured Intention tuple into latent space.
        """
        def __init__(self, goal_dim: int, belief_dim: int, latent_dim: int):
            super().__init__()
            self.goal_proj = nn.Linear(goal_dim, latent_dim)
            self.belief_proj = nn.Linear(belief_dim, latent_dim)
            self.gate = nn.Linear(latent_dim * 2 + 2, latent_dim)
            self.layer_norm = nn.LayerNorm(latent_dim)

        def forward(self, goal_t: torch.Tensor, belief_t: torch.Tensor, horizon_t: torch.Tensor, error_t: torch.Tensor) -> torch.Tensor:
            g_emb = F.relu(self.goal_proj(goal_t))
            b_emb = F.relu(self.belief_proj(belief_t))
            meta = torch.cat([horizon_t, error_t], dim=-1)
            raw = torch.cat([g_emb, b_emb, meta], dim=-1)
            intention_latent = self.layer_norm(F.gelu(self.gate(raw)))
            return intention_latent
