"""
ICIA Model Candidate C: Dual-Stream Intention-Verification Recurrent Engine (DIVRE)
Natively intention-centred, closed-loop predictive active-inference architecture.
Features:
- Top-Down Teleological Stream T_t
- Bottom-Up Sensory Verification Stream V_t
- Continuous Reality Discrepancy Tensor E_t = X_t - Pred(T_{t-1})
- O(1) inference memory; linear O(T) sequence computation.
"""

from typing import Tuple, Dict, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class DIVRECell(nn.Module):
    """
    A single recurrent step of the Dual-Stream Intention-Verification Recurrent Engine.
    """
    def __init__(self, d_model: int, d_intention: int):
        super().__init__()
        self.d_model = d_model
        self.d_intention = d_intention

        # Predictive sensory decoder: T_{t-1} -> Expected sensory observation \hat{X}_t
        self.sensory_predictor = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh()
        )

        # Bottom-up verification update: takes (V_{t-1}, X_t, Error E_t)
        self.w_v = nn.Linear(d_model * 3, d_model)
        self.ln_v = nn.LayerNorm(d_model)

        # Top-down teleological update: takes (T_{t-1}, Intention I_t, Verified V_t)
        self.w_t = nn.Linear(d_model * 2 + d_intention, d_model)
        self.ln_t = nn.LayerNorm(d_model)

        # Symplectic Discrepancy Gate: modulates top-down state revision based on prediction error
        self.discrepancy_gate = nn.Linear(d_model, d_model)

        # Constraint barrier projector (penalizes trajectories near prohibited boundaries)
        self.constraint_barrier = nn.Linear(d_model, d_model)

    def forward(
        self,
        x_t: torch.Tensor,
        t_prev: torch.Tensor,
        v_prev: torch.Tensor,
        intention_t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        x_t: (B, d_model) incoming sensory token representation
        t_prev: (B, d_model) prior top-down teleological state
        v_prev: (B, d_model) prior bottom-up verification state
        intention_t: (B, d_intention) active intention state vector
        """
        # 1. Predictive top-down expectation
        x_hat = self.sensory_predictor(t_prev)

        # 2. Reality Discrepancy computation (Sensory Surprise)
        error_t = x_t - x_hat

        # 3. Bottom-up verification stream update
        v_in = torch.cat([v_prev, x_t, error_t], dim=-1)
        v_t = self.ln_v(F.gelu(self.w_v(v_in)))

        # 4. Teleological error gate: how strongly reality discrepancy impacts the plan
        gate = torch.sigmoid(self.discrepancy_gate(error_t))

        # 5. Top-down teleological stream update
        t_in = torch.cat([t_prev, intention_t, v_t], dim=-1)
        t_candidate = torch.tanh(self.w_t(t_in))
        
        # Constraint barrier penalty
        c_barrier = torch.tanh(self.constraint_barrier(t_prev))

        # Symplectic coupling: T_t is pulled toward candidate plan, gated by reality discrepancy
        t_t = self.ln_t(t_prev + gate * t_candidate - 0.05 * c_barrier)

        return t_t, v_t, error_t


class DIVREModel(nn.Module):
    """
    Complete ICIA DIVRE Architecture.
    Processes token sequences through closed-loop teleological recurrence.
    """
    def __init__(self, vocab_size: int, d_model: int = 64, d_intention: int = 64):
        super().__init__()
        self.d_model = d_model
        self.d_intention = d_intention

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.intention_proj = nn.Linear(d_model, d_intention)

        self.cell = DIVRECell(d_model, d_intention)

        # Decision head: conditions on the verified state product (T_t \odot V_t)
        self.action_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, vocab_size, bias=False)
        )

    def forward(
        self,
        idx: torch.Tensor,
        goal_vec: torch.Tensor,
        return_discrepancy: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        idx: (B, T) sequence tokens
        goal_vec: (B, d_model) intention vector
        """
        B, T = idx.shape
        x_seq = self.token_emb(idx)  # (B, T, d_model)
        intention = self.intention_proj(goal_vec)  # (B, d_intention)

        # Initialize internal dual states
        t_state = torch.zeros(B, self.d_model, device=idx.device)
        v_state = torch.zeros(B, self.d_model, device=idx.device)

        logits_list = []
        errors_list = []

        for step in range(T):
            x_t = x_seq[:, step, :]
            t_state, v_state, error_t = self.cell(x_t, t_state, v_state, intention)

            # Verified state synthesis: Symplectic product
            verified_state = t_state * v_state
            step_logits = self.action_head(verified_state)
            logits_list.append(step_logits.unsqueeze(1))
            errors_list.append(error_t.unsqueeze(1))

        all_logits = torch.cat(logits_list, dim=1)  # (B, T, vocab_size)
        all_errors = torch.cat(errors_list, dim=1)  # (B, T, d_model)

        if return_discrepancy:
            return all_logits, all_errors
        return all_logits, verified_state
