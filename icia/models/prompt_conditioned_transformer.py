"""
ICIA Model Baseline: Intention-Conditioned Transformer (ICT)
Transformer augmented with explicit goal-prefix tokens and goal cross-attention.
Represents prompt-engineering / prompt-tuning / prefix-conditioned LLM baselines.
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from .standard_transformer import TransformerBlock


class GoalCrossAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, goal_emb: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        _, G_len, _ = goal_emb.shape

        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(goal_emb).view(B, G_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(goal_emb).view(B, G_len, self.n_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)


class IntentionConditionedTransformer(nn.Module):
    """
    Intention-Conditioned Transformer baseline.
    Conditioned on an explicit intention representation prepended and cross-attended.
    """
    def __init__(self, vocab_size: int, d_model: int = 64, n_heads: int = 4, n_layers: int = 2, d_ff: int = 128, max_len: int = 1024):
        super().__init__()
        self.d_model = d_model
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.goal_proj = nn.Linear(d_model, d_model)

        self.causal_blocks = nn.ModuleList([TransformerBlock(d_model, n_heads, d_ff) for _ in range(n_layers)])
        self.goal_cross_attns = nn.ModuleList([GoalCrossAttention(d_model, n_heads) for _ in range(n_layers)])
        self.cross_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])

        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx: torch.Tensor, goal_vec: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        idx: (B, T) sequence tokens
        goal_vec: (B, d_model) or (B, G_len, d_model) intention vector
        """
        B, T = idx.shape
        pos = torch.arange(0, T, device=idx.device).unsqueeze(0)
        x = self.token_emb(idx) + self.pos_emb(pos)

        if goal_vec.dim() == 2:
            goal_emb = self.goal_proj(goal_vec).unsqueeze(1)  # (B, 1, d_model)
        else:
            goal_emb = self.goal_proj(goal_vec)

        for block, cross_attn, cross_norm in zip(self.causal_blocks, self.goal_cross_attns, self.cross_norms):
            x, _ = block(x)
            # Add intention cross-attention
            x = x + cross_attn(cross_norm(x), goal_emb)

        x = self.ln_f(x)
        logits = self.head(x)
        return logits, x
