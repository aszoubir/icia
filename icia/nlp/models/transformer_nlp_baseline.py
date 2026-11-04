"""
ICIA NLP: Parameter-Matched Standard Transformer Baseline
Causal Byte-Level Transformer for direct generation and Chain-of-Thought (CoT) reasoning.
"""

import math
from typing import Dict, Any, Tuple, Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, nhead: int = 4):
        super().__init__()
        assert d_model % nhead == 0
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.scale = 1.0 / math.sqrt(self.head_dim)

    def forward(
        self,
        x: torch.Tensor,
        causal: bool = True,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, L, D = x.size()
        qkv = self.qkv_proj(x) # (B, L, 3D)
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        
        # Reshape to (B, nhead, L, head_dim)
        q = q.view(B, L, self.nhead, self.head_dim).transpose(1, 2)
        k = k.view(B, L, self.nhead, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.nhead, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale # (B, nhead, L, L)
        
        if causal:
            causal_mask = torch.triu(torch.ones(L, L, device=x.device, dtype=torch.bool), diagonal=1)
            scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), -1e9)
            
        if mask is not None:
            # mask: (B, L)
            key_mask = (mask == 0).unsqueeze(1).unsqueeze(2) # (B, 1, 1, L)
            scores = scores.masked_fill(key_mask, -1e9)
            
        attn_weights = F.softmax(scores, dim=-1)
        out = torch.matmul(attn_weights, v) # (B, nhead, L, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        out = self.out_proj(out)
        
        return out, attn_weights


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, nhead: int = 4, d_ff: int = 256):
        super().__init__()
        self.attn = CausalSelfAttention(d_model, nhead=nhead)
        self.norm1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        causal: bool = True,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        attn_out, attn_weights = self.attn(self.norm1(x), causal=causal, mask=mask)
        x = x + attn_out
        mlp_out = self.mlp(self.norm2(x))
        x = x + mlp_out
        return x, attn_weights


class TransformerNLPBaseline(nn.Module):
    """
    Parameter-matched Causal Byte Transformer Baseline.
    Used for both Direct answering and Chain-of-Thought (CoT) sequence generation.
    """
    def __init__(
        self,
        vocab_size: int = 262,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 3,
        d_ff: int = 128,
        max_seq_len: int = 512
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers

        self.token_embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)

        self.layers = nn.ModuleList([
            TransformerBlock(d_model, nhead=nhead, d_ff=d_ff)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

        # Numeric regression head from sequence pooling (for direct evaluation)
        self.numeric_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1)
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        causal: bool = True,
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        B, L = input_ids.size()
        pos = torch.arange(0, L, device=input_ids.device).unsqueeze(0).expand(B, L)
        
        x = self.token_embed(input_ids) + self.pos_embed(pos)
        
        attn_maps = []
        for layer in self.layers:
            x, attn = layer(x, causal=causal, mask=mask)
            attn_maps.append(attn)
            
        x_norm = self.norm(x)
        logits = self.lm_head(x_norm)
        
        # Numeric prediction from last token position
        last_h = x_norm[:, -1, :]
        pred_val = self.numeric_head(last_h).squeeze(-1)

        return {
            "logits": logits,
            "pred_val": pred_val,
            "hidden_states": x_norm,
            "attn_maps": attn_maps
        }

    def generate(
        self,
        prompt_tokens: torch.Tensor,
        max_new_tokens: int = 80,
        stop_token: int = 2 # EOS_TOKEN
    ) -> Tuple[torch.Tensor, int]:
        """
        Autoregressively generates continuation tokens (scratchpad or direct answer).
        Returns: (generated_sequence, count_of_generated_tokens)
        """
        self.eval()
        with torch.no_grad():
            curr_tokens = prompt_tokens.clone()
            tokens_generated = 0
            
            for _ in range(max_new_tokens):
                out = self.forward(curr_tokens, causal=True)
                next_token_logits = out["logits"][:, -1, :]
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                curr_tokens = torch.cat([curr_tokens, next_token], dim=1)
                tokens_generated += 1
                
                # If all sequences in batch produced stop_token, break
                if (next_token == stop_token).all():
                    break
                    
            return curr_tokens, tokens_generated
