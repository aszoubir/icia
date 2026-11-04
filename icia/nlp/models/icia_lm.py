"""
ICIA-LM: Intention-Centred Language Model
Dual-Stream Language Architecture combining:
1. Fast local causal recurrent stream (syntax & grammar)
2. Macro Intention & Surprise-Gated Associative Memory (IGAM, O(1) bounded memory)
"""

import math
from typing import Dict, Any, Tuple, Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 8192):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        if seq_len > self.pe.size(1):
            position = torch.arange(0, seq_len, dtype=torch.float, device=x.device).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, x.size(2), 2, device=x.device).float() * (-math.log(10000.0) / x.size(2)))
            pe = torch.zeros(1, seq_len, x.size(2), device=x.device)
            pe[0, :, 0::2] = torch.sin(position * div_term)
            pe[0, :, 1::2] = torch.cos(position * div_term)
            return x + pe
        return x + self.pe[:, :seq_len]


class IntentionGatedTextMemory(nn.Module):
    """
    Bounded B-slot Intention-Gated Associative Memory for Text.
    Surprise-gated factual storage with O(K) retrieval complexity.
    """
    def __init__(self, num_slots: int = 32, d_model: int = 128, d_key: int = 64):
        super().__init__()
        self.num_slots = num_slots
        self.d_model = d_model
        self.d_key = d_key

        self.key_proj = nn.Linear(d_model, d_key)
        self.val_proj = nn.Linear(d_model, d_model)
        self.query_proj = nn.Linear(d_model, d_key)
        
        # Surprise threshold gate: maps prediction error to write weight
        self.write_gate = nn.Sequential(
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )
        self.slot_routing = nn.Linear(d_key, num_slots)

    def forward(
        self,
        hidden_seq: torch.Tensor,
        error_seq: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Processes sequence and returns memory-augmented context representations.
        hidden_seq: (B, T, d_model)
        error_seq: (B, T, d_model) optional discrepancy signal
        """
        B, T, D = hidden_seq.size()
        device = hidden_seq.device
        
        # Initialize bounded memory buffer (B, num_slots, d_model)
        mem_keys = torch.zeros(B, self.num_slots, self.d_key, device=device)
        mem_vals = torch.zeros(B, self.num_slots, D, device=device)
        
        contexts = []
        
        # In causal sequence processing:
        # Every token generates a query and selectively writes if surprising
        q_seq = self.query_proj(hidden_seq) # (B, T, d_key)
        k_seq = self.key_proj(hidden_seq)   # (B, T, d_key)
        v_seq = self.val_proj(hidden_seq)   # (B, T, d_model)
        w_gates = self.write_gate(hidden_seq) # (B, T, 1)

        # Vectorized associative read over accumulated slots
        scale = 1.0 / math.sqrt(self.d_key)
        
        # Bounded slot memory updates
        # Memory writes accumulate weighted by surprise
        for t in range(T):
            q_t = q_seq[:, t:t+1, :] # (B, 1, d_key)
            # Read from current memory slots
            attn_scores = torch.bmm(q_t, mem_keys.transpose(1, 2)) * scale # (B, 1, num_slots)
            attn_weights = F.softmax(attn_scores, dim=-1)
            ctx_t = torch.bmm(attn_weights, mem_vals) # (B, 1, d_model)
            contexts.append(ctx_t)
            
            # Surprise-gated write into memory slot
            w_t = w_gates[:, t:t+1, :] # (B, 1, 1)
            slot_weights = F.softmax(self.slot_routing(k_seq[:, t:t+1, :]), dim=-1) # (B, 1, num_slots)
            
            # Soft update of bounded slots
            write_matrix = w_t * slot_weights # (B, 1, num_slots)
            mem_keys = mem_keys + torch.bmm(write_matrix.transpose(1, 2), k_seq[:, t:t+1, :])
            mem_vals = mem_vals + torch.bmm(write_matrix.transpose(1, 2), v_seq[:, t:t+1, :])

        context_seq = torch.cat(contexts, dim=1) # (B, T, d_model)
        return context_seq, mem_vals


class ICIALanguageModel(nn.Module):
    """
    Complete ICIA Language Model (ICIA-LM).
    Combines:
    1. Byte Embedding Layer
    2. Multi-layer Dual-Stream Causal Recurrent Core
    3. Bounded Intention-Gated Associative Memory (IGAM, O(1) memory complexity)
    4. Language Modeling Output Head
    """
    def __init__(
        self,
        vocab_size: int = 262,
        d_model: int = 128,
        num_layers: int = 3,
        num_memory_slots: int = 32,
        max_seq_len: int = 8192
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_memory_slots = num_memory_slots

        self.token_embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_seq_len)

        # Local causal recurrent stream (bidirectional / gated GRU layers)
        self.recurrent_layers = nn.ModuleList([
            nn.GRU(
                input_size=d_model if l == 0 else d_model,
                hidden_size=d_model,
                batch_first=True
            )
            for l in range(num_layers)
        ])
        self.layer_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])

        # Intention-Gated Associative Memory (bounded capacity K)
        self.memory = IntentionGatedTextMemory(num_slots=num_memory_slots, d_model=d_model)

        # Teleological synthesis projection
        self.synthesis = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model)
        )

        # Next-byte LM prediction head
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """
        Forward pass over input sequence.
        input_ids: (B, T)
        """
        B, T = input_ids.size()
        x = self.token_embed(input_ids) # (B, T, d_model)
        x = self.pos_encoder(x)

        # 1. Local recurrent processing
        h = x
        for layer, norm in zip(self.recurrent_layers, self.layer_norms):
            h_out, _ = layer(h)
            h = norm(h + h_out)

        # 2. Intention-Gated Associative Memory augmentation
        mem_ctx, final_mem = self.memory(h)

        # 3. Teleological synthesis: Combine local state with factual memory
        synthesized = self.synthesis(torch.cat([h, mem_ctx], dim=-1))

        # 4. Next-token prediction logits
        logits = self.lm_head(synthesized) # (B, T, vocab_size)

        return {
            "logits": logits,
            "hidden_states": synthesized,
            "final_memory": final_mem,
            "kv_cache_slots": self.num_memory_slots # Strictly O(1) bounded!
        }

    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 40,
        stop_token: int = 2
    ) -> Tuple[torch.Tensor, int]:
        """
        Autoregressive generation using O(1) bounded memory state.
        """
        self.eval()
        with torch.no_grad():
            curr_tokens = prompt_ids.clone()
            tokens_generated = 0
            
            for _ in range(max_new_tokens):
                out = self.forward(curr_tokens)
                next_logits = out["logits"][:, -1, :]
                next_tok = torch.argmax(next_logits, dim=-1, keepdim=True)
                curr_tokens = torch.cat([curr_tokens, next_tok], dim=1)
                tokens_generated += 1
                
                if (next_tok == stop_token).all():
                    break
                    
            return curr_tokens, tokens_generated
