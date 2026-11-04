"""
ICIA Module: Differentiable Latent World Model
Implements forward state simulation, sensory prediction,
and parallel counterfactual action evaluation (imagination rollouts).
"""

from typing import Tuple, Dict, Any, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class LatentWorldModel(nn.Module):
    """
    Differentiable Forward World Model:
    (S_t, A_t, I_t) -> (\hat{S}_{t+1}, \hat{X}_{t+1})
    """
    def __init__(self, d_state: int = 64, d_action: int = 4, d_intention: int = 64, d_obs: int = 64):
        super().__init__()
        self.d_state = d_state
        self.d_action = d_action
        self.d_intention = d_intention
        self.d_obs = d_obs

        # Action embedding
        self.action_emb = nn.Linear(d_action, d_state)
        self.intention_proj = nn.Linear(d_intention, d_state)

        # Transition dynamics network (MLP with residual connection)
        self.transition_net = nn.Sequential(
            nn.Linear(d_state * 3, d_state * 2),
            nn.GELU(),
            nn.Linear(d_state * 2, d_state)
        )
        self.ln_s = nn.LayerNorm(d_state)

        # Observation decoder: \hat{S}_{t+1} -> \hat{X}_{t+1}
        self.obs_decoder = nn.Sequential(
            nn.Linear(d_state, d_obs),
            nn.Tanh()
        )

        # Discrepancy predictor: estimates expected sensory surprise
        self.discrepancy_estimator = nn.Linear(d_state, 1)

    def forward(
        self,
        s_t: torch.Tensor,
        a_t: torch.Tensor,
        intention_t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        s_t: (B, d_state)
        a_t: (B, d_action) one-hot action vector or continuous embedding
        intention_t: (B, d_intention)
        Returns:
          s_hat_next: (B, d_state)
          x_hat_next: (B, d_obs)
          expected_discrepancy: (B, 1)
        """
        a_emb = F.gelu(self.action_emb(a_t))
        i_emb = F.gelu(self.intention_proj(intention_t))

        in_feat = torch.cat([s_t, a_emb, i_emb], dim=-1)
        delta_s = self.transition_net(in_feat)
        s_hat_next = self.ln_s(s_t + delta_s)

        x_hat_next = self.obs_decoder(s_hat_next)
        expected_disc = F.softplus(self.discrepancy_estimator(s_hat_next))

        return s_hat_next, x_hat_next, expected_disc

    def counterfactual_search(
        self,
        s_t: torch.Tensor,
        candidate_actions: torch.Tensor,
        goal_vec: torch.Tensor,
        constraint_boundaries: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parallel counterfactual rollout over N candidate actions in imagination:
        s_t: (B, d_state)
        candidate_actions: (num_actions, d_action)
        goal_vec: (B, d_state)
        Returns:
          best_action: (B, d_action)
          scores: (B, num_actions)
        """
        B = s_t.shape[0]
        N_act = candidate_actions.shape[0]

        # Expand batch for parallel candidate evaluation: (B * N_act, ...)
        s_exp = s_t.unsqueeze(1).expand(-1, N_act, -1).reshape(B * N_act, self.d_state)
        a_exp = candidate_actions.unsqueeze(0).expand(B, -1, -1).reshape(B * N_act, self.d_action)
        g_exp = goal_vec.unsqueeze(1).expand(-1, N_act, -1).reshape(B * N_act, self.d_state)

        # Forward simulate all candidates simultaneously
        s_hat, _, _ = self.forward(s_exp, a_exp, g_exp)

        # Compute goal distance: - ||s_hat - g||^2
        distances = torch.sum((s_hat - g_exp) ** 2, dim=-1)  # (B * N_act,)
        scores = -distances.view(B, N_act)

        # Constraint penalty if applicable
        if constraint_boundaries is not None:
            # penalize candidates that approach barrier
            barrier_penalty = torch.relu(torch.bmm(s_hat.unsqueeze(1), constraint_boundaries.transpose(1, 2)))
            scores = scores - 2.0 * barrier_penalty.squeeze(1).view(B, N_act)

        best_act_idx = torch.argmax(scores, dim=-1)  # (B,)
        best_actions = candidate_actions[best_act_idx]  # (B, d_action)

        return best_actions, scores
