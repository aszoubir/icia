"""
ICIA Primitive: Intention-Gated Associative Memory (IGAM)
Persistent, bounded episodic memory with surprise-gated writes
and teleologically-indexed reads.
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class IntentionGatedAssociativeMemory(nn.Module):
    """
    Differentiable associative memory with fixed slot capacity K.
    - Write gate: modulated by reality discrepancy ||E_t|| (surprise)
    - Read gate: indexed by active intention I_t
    """
    def __init__(self, num_slots: int = 32, d_key: int = 32, d_val: int = 64):
        super().__init__()
        self.num_slots = num_slots
        self.d_key = d_key
        self.d_val = d_val

        # Initial memory parameters
        self.init_keys = nn.Parameter(torch.randn(num_slots, d_key) * 0.02)
        self.init_values = nn.Parameter(torch.zeros(num_slots, d_val))

        # Projections
        self.key_proj = nn.Linear(d_val, d_key)
        self.write_gate = nn.Linear(d_val, 1)

    def init_memory(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        keys = self.init_keys.unsqueeze(0).expand(batch_size, -1, -1)
        values = self.init_values.unsqueeze(0).expand(batch_size, -1, -1)
        return keys, values

    def read(self, intention: torch.Tensor, keys: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        """
        intention: (B, d_key)
        keys: (B, K, d_key)
        values: (B, K, d_val)
        Returns: retrieved memory context (B, d_val)
        """
        # Query-key dot product: (B, 1, d_key) x (B, d_key, K) -> (B, 1, K)
        scores = torch.bmm(intention.unsqueeze(1), keys.transpose(1, 2)) / (self.d_key ** 0.5)
        attn = F.softmax(scores, dim=-1)
        # Context: (B, 1, K) x (B, K, d_val) -> (B, d_val)
        context = torch.bmm(attn, values).squeeze(1)
        return context

    def write(
        self,
        new_val: torch.Tensor,
        discrepancy: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        intention: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Surprise-gated episodic write.
        Only writes if discrepancy ||E_t|| is significant and congruent with intention.
        """
        B = new_val.shape[0]
        # Writing weight based on error magnitude
        err_mag = torch.norm(discrepancy, dim=-1, keepdim=True)
        w_gate = torch.sigmoid(self.write_gate(new_val) + 0.5 * err_mag)  # (B, 1)

        # Slot assignment based on intention congruency
        scores = torch.bmm(intention.unsqueeze(1), keys.transpose(1, 2)) / (self.d_key ** 0.5)
        slot_weights = F.softmax(scores, dim=-1).transpose(1, 2)  # (B, K, 1)

        # Memory update
        new_k = self.key_proj(new_val).unsqueeze(1)  # (B, 1, d_key)
        new_v = new_val.unsqueeze(1)  # (B, 1, d_val)

        updated_keys = keys + (w_gate.unsqueeze(-1) * slot_weights) * (new_k - keys)
        updated_values = values + (w_gate.unsqueeze(-1) * slot_weights) * (new_v - values)

        return updated_keys, updated_values
