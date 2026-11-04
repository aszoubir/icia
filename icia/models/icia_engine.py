"""
ICIA Autonomous Engine (Stage 9: Complete Integrated Prototype)
Unified Architecture combining:
1. DIVRE-v2 Dual-Stream Recurrent Core
2. Active Intention Attractor Anchoring (AIAA)
3. Intention-Gated Associative Memory (IGAM)
4. Differentiable Latent World Model (W_theta)
5. Closed-Loop Reality Innovation Correction
"""

from typing import Tuple, Dict, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from icia.models.candidate_c_divre_v2 import DIVREv2Cell
from icia.primitives.adaptive_memory import IntentionGatedAssociativeMemory
from icia.models.world_model import LatentWorldModel


class ICIAAutonomousEngine(nn.Module):
    """
    Stage 9 Integrated Prototype of the Intention-Centred Intelligence Architecture.
    """
    def __init__(
        self,
        d_obs: int = 64,
        d_state: int = 64,
        d_action: int = 4,
        d_intention: int = 64,
        num_mem_slots: int = 16
    ):
        super().__init__()
        self.d_obs = d_obs
        self.d_state = d_state
        self.d_action = d_action
        self.d_intention = d_intention

        # Observation encoder: X_t -> d_state
        self.obs_encoder = nn.Sequential(
            nn.Linear(d_obs, d_state),
            nn.LayerNorm(d_state),
            nn.GELU()
        )

        # Intention projection
        self.intention_proj = nn.Linear(d_intention, d_state)
        self.memory_key_proj = nn.Linear(d_state, 32)

        # Memory and Recurrent Cell
        self.memory = IntentionGatedAssociativeMemory(num_slots=num_mem_slots, d_key=32, d_val=d_state)
        self.cell = DIVREv2Cell(d_model=d_state, d_intention=d_state, d_memory_key=32)

        # Internal World Model (Forward Simulator)
        self.world_model = LatentWorldModel(
            d_state=d_state, d_action=d_action, d_intention=d_state, d_obs=d_obs
        )

        # Kalman-style Innovation Gain for reality correction
        self.kalman_gain = nn.Sequential(
            nn.Linear(d_state + d_obs, d_state),
            nn.Sigmoid()
        )

        # Action emission policy head
        self.policy_head = nn.Sequential(
            nn.Linear(d_state * 2, d_state),
            nn.GELU(),
            nn.Linear(d_state, d_action)
        )

    def init_state(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        t_state = torch.zeros(batch_size, self.d_state, device=device)
        v_state = torch.zeros(batch_size, self.d_state, device=device)
        mem_keys, mem_vals = self.memory.init_memory(batch_size, device)
        return t_state, v_state, mem_keys, mem_vals

    def step(
        self,
        x_obs: torch.Tensor,
        goal_vec: torch.Tensor,
        t_prev: torch.Tensor,
        v_prev: torch.Tensor,
        mem_keys: torch.Tensor,
        mem_vals: torch.Tensor,
        candidate_actions: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """
        Executes a single closed-loop teleological step:
        1. Encodes observation X_t
        2. Retrieves goal-congruent memory
        3. Computes discrepancy E_t and updates teleological states (T_t, V_t)
        4. Updates episodic memory (surprise-gated)
        5. Simulates candidate actions in the internal World Model
        6. Selects optimal action A*
        """
        B = x_obs.shape[0]
        x_emb = self.obs_encoder(x_obs)
        intention = self.intention_proj(goal_vec)
        mem_key = self.memory_key_proj(intention)

        # 1. Memory read
        mem_context = self.memory.read(mem_key, mem_keys, mem_vals)

        # 2. Recurrent update with AIAA
        t_state, v_state, error_t = self.cell(x_emb, t_prev, v_prev, intention, mem_context)

        # 3. Memory write (surprise-gated)
        mem_keys, mem_vals = self.memory.write(
            t_state * v_state, error_t, mem_keys, mem_vals, mem_key
        )

        # 4. Verified state
        verified_state = torch.cat([t_state * v_state, mem_context], dim=-1)

        # 5. Action selection via counterfactual search or policy head
        if candidate_actions is not None:
            # Parallel imagination rollout in latent world model
            action_choice, action_scores = self.world_model.counterfactual_search(
                s_t=t_state * v_state,
                candidate_actions=candidate_actions,
                goal_vec=intention
            )
            logits = action_scores
        else:
            logits = self.policy_head(verified_state)
            action_choice = F.one_hot(torch.argmax(logits, dim=-1), num_classes=self.d_action).float()

        # 6. Predict next state in imagination
        s_hat_next, x_hat_next, _ = self.world_model(t_state * v_state, action_choice, intention)

        return {
            "action": action_choice,
            "logits": logits,
            "t_state": t_state,
            "v_state": v_state,
            "mem_keys": mem_keys,
            "mem_vals": mem_vals,
            "reality_error": error_t,
            "predicted_next_obs": x_hat_next,
            "predicted_next_state": s_hat_next
        }
