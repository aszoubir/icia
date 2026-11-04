"""
ICIA Model Candidate C v2: DIVRE-v2 with Attractor Anchoring & Adaptive Memory
Integrated Prototype advancing Stages 4, 5, and 8 of the Master Directive.
Features:
- Active Intention Attractor Anchoring (AIAA) for invariant long-horizon preservation.
- Intention-Gated Associative Memory (IGAM) with surprise-gated writes.
- Instantaneous dynamic constraint re-routing.
"""

from typing import Tuple, Dict, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from icia.primitives.adaptive_memory import IntentionGatedAssociativeMemory


class DIVREv2Cell(nn.Module):
    def __init__(self, d_model: int, d_intention: int, d_memory_key: int = 32):
        super().__init__()
        self.d_model = d_model
        self.d_intention = d_intention

        # Predictive sensory decoder: T_{t-1} -> \hat{X}_t
        self.sensory_predictor = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh()
        )

        # Bottom-up verification update: takes [V_{t-1}, X_t, Error, Memory]
        self.w_v = nn.Linear(d_model * 4, d_model)
        self.ln_v = nn.LayerNorm(d_model)

        # Top-down teleological update: takes [T_{t-1}, Intention, V_t, Memory]
        self.w_t = nn.Linear(d_model * 3 + d_intention, d_model)
        self.ln_t = nn.LayerNorm(d_model)

        # Symplectic Discrepancy Gate
        self.discrepancy_gate = nn.Linear(d_model, d_model)

        # Active Intention Attractor Anchor: generates alpha_t in (0, 1)
        self.anchor_gate = nn.Linear(d_model + d_intention, d_model)
        self.anchor_proj = nn.Linear(d_intention, d_model)

        # Constraint impasse projector
        self.constraint_proj = nn.Linear(d_model, d_model)

    def forward(
        self,
        x_t: torch.Tensor,
        t_prev: torch.Tensor,
        v_prev: torch.Tensor,
        intention_t: torch.Tensor,
        mem_context: torch.Tensor,
        is_constraint_flip: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        x_t: (B, d_model)
        t_prev: (B, d_model)
        v_prev: (B, d_model)
        intention_t: (B, d_intention)
        mem_context: (B, d_model)
        """
        # 1. Predictive top-down expectation
        x_hat = self.sensory_predictor(t_prev)

        # 2. Reality Discrepancy computation
        error_t = x_t - x_hat

        # 3. Bottom-up verification stream update
        v_in = torch.cat([v_prev, x_t, error_t, mem_context], dim=-1)
        v_t = self.ln_v(F.gelu(self.w_v(v_in)))

        # 4. Teleological error gate
        gate = torch.sigmoid(self.discrepancy_gate(error_t))

        # 5. Top-down recurrent candidate
        t_in = torch.cat([t_prev, intention_t, v_t, mem_context], dim=-1)
        t_candidate = torch.tanh(self.w_t(t_in))
        t_recurrent = self.ln_t(t_prev + gate * t_candidate)

        # 6. ACTIVE INTENTION ATTRACTOR ANCHORING (AIAA)
        # Prevents state drift across unbounded sequences!
        anchor_val = self.anchor_proj(intention_t)
        alpha_t = torch.sigmoid(self.anchor_gate(torch.cat([error_t, intention_t], dim=-1)))
        # Anchored combination: restorative pull toward active intention
        t_t = (1.0 - 0.6 * alpha_t) * t_recurrent + (0.6 * alpha_t) * anchor_val

        # 7. Dynamic constraint impasse projection
        if is_constraint_flip is not None:
            c_mask = is_constraint_flip.unsqueeze(-1).float()
            # Instantaneously shift state along the constraint null space
            t_t = t_t + c_mask * torch.tanh(self.constraint_proj(t_t))

        return t_t, v_t, error_t


class DIVREv2Model(nn.Module):
    """
    ICIA DIVRE-v2: Integrated Prototype with AIAA and IGAM.
    """
    def __init__(self, vocab_size: int, d_model: int = 64, d_intention: int = 64, num_mem_slots: int = 16):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.d_intention = d_intention

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.intention_proj = nn.Linear(d_model, d_intention)
        self.memory_key_proj = nn.Linear(d_intention, 32)

        self.memory = IntentionGatedAssociativeMemory(num_slots=num_mem_slots, d_key=32, d_val=d_model)
        self.cell = DIVREv2Cell(d_model, d_intention, d_memory_key=32)

        self.action_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, vocab_size, bias=False)
        )

    def forward(
        self,
        idx: torch.Tensor,
        goal_vec: torch.Tensor,
        constraint_mask: Optional[torch.Tensor] = None,
        return_discrepancy: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T = idx.shape
        x_seq = self.token_emb(idx)
        intention = self.intention_proj(goal_vec)  # (B, d_intention)
        mem_key = self.memory_key_proj(intention)  # (B, 32)

        # Initialize internal states
        t_state = torch.zeros(B, self.d_model, device=idx.device)
        v_state = torch.zeros(B, self.d_model, device=idx.device)
        mem_keys, mem_vals = self.memory.init_memory(B, idx.device)

        logits_list = []
        errors_list = []

        for step in range(T):
            x_t = x_seq[:, step, :]
            
            # Read from adaptive memory
            mem_context = self.memory.read(mem_key, mem_keys, mem_vals)

            # Check constraint flip
            flip_t = constraint_mask[:, step] if constraint_mask is not None else None

            # Recurrent step with AIAA
            t_state, v_state, error_t = self.cell(
                x_t, t_state, v_state, intention, mem_context, is_constraint_flip=flip_t
            )

            # Surprise-gated memory update
            mem_keys, mem_vals = self.memory.write(
                t_state * v_state, error_t, mem_keys, mem_vals, mem_key
            )

            # Verified state synthesis
            verified_state = torch.cat([t_state * v_state, mem_context], dim=-1)
            step_logits = self.action_head(verified_state)
            logits_list.append(step_logits.unsqueeze(1))
            errors_list.append(error_t.unsqueeze(1))

        all_logits = torch.cat(logits_list, dim=1)
        all_errors = torch.cat(errors_list, dim=1)

        if return_discrepancy:
            return all_logits, all_errors
        return all_logits, verified_state
