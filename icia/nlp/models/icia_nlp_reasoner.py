"""
ICIA NLP: Latent Teleological Reasoner (ICIANLPReasoner)
Performs multi-step symbolic deduction in latent cognitive space without emitting external scratchpad tokens.
"""

import math
from typing import Dict, Any, Tuple, Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, seq_len, d_model)
        return x + self.pe[:, :x.size(1)]


class IntentionGatedPremiseMemory(nn.Module):
    """
    Intention-Gated Associative Memory for NLP.
    Queries tokenized premises conditioned on the current internal intention state S_k.
    """
    def __init__(self, d_latent: int, d_model: int):
        super().__init__()
        self.q_proj = nn.Linear(d_latent, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_latent)
        self.scale = 1.0 / math.sqrt(d_model)

    def forward(
        self,
        query_state: torch.Tensor,       # (batch, d_latent)
        premise_keys: torch.Tensor,      # (batch, seq_len, d_model)
        premise_vals: torch.Tensor,      # (batch, seq_len, d_model)
        mask: Optional[torch.Tensor] = None # (batch, seq_len)
    ) -> torch.Tensor:
        # q: (batch, 1, d_model)
        q = self.q_proj(query_state).unsqueeze(1)
        # k: (batch, seq_len, d_model)
        k = self.k_proj(premise_keys)
        # scores: (batch, 1, seq_len)
        scores = torch.bmm(q, k.transpose(1, 2)) * self.scale
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1) == 0, -1e9)
        attn = F.softmax(scores, dim=-1)
        # attended: (batch, 1, d_model)
        attended = torch.bmm(attn, premise_vals).squeeze(1)
        # project to d_latent: (batch, d_latent)
        return self.v_proj(attended)


class LatentDeliberationCore(nn.Module):
    """
    Internal latent thinking core.
    Executes K latent steps of teleological recurrence without emitting scratchpad tokens.
    """
    def __init__(self, d_latent: int, d_model: int):
        super().__init__()
        self.d_latent = d_latent
        self.memory = IntentionGatedPremiseMemory(d_latent, d_model)
        
        # Teleological gradient drive
        self.goal_drive = nn.Sequential(
            nn.Linear(d_latent, d_latent),
            nn.Tanh()
        )
        
        # State update network
        self.state_update = nn.Sequential(
            nn.Linear(d_latent * 3, d_latent * 2),
            nn.GELU(),
            nn.Linear(d_latent * 2, d_latent)
        )
        self.norm = nn.LayerNorm(d_latent)

        # Verification head: outputs verification confidence nu_k in [0, 1]
        self.verifier = nn.Sequential(
            nn.Linear(d_latent, d_latent // 2),
            nn.GELU(),
            nn.Linear(d_latent // 2, 1),
            nn.Sigmoid()
        )

    def forward_step(
        self,
        s_prev: torch.Tensor,
        goal: torch.Tensor,
        premise_keys: torch.Tensor,
        premise_vals: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Single step of latent deliberation.
        Returns: (s_next, verification_score)
        """
        # 1. Teleological gradient pull towards goal
        goal_diff = goal - s_prev
        drive = self.goal_drive(goal_diff)
        
        # 2. Intention-gated retrieval from premise memory
        retrieved = self.memory(s_prev, premise_keys, premise_vals, mask)
        
        # 3. Latent state update
        combined = torch.cat([s_prev, drive, retrieved], dim=-1)
        delta = self.state_update(combined)
        s_next = self.norm(s_prev + delta)
        
        # 4. Continuous verification
        nu = self.verifier(s_next).squeeze(-1)
        return s_next, nu


class ICIANLPReasoner(nn.Module):
    """
    Complete ICIA Natural Language and Symbolic Reasoner.
    Combines:
    1. Byte Embedding & Positional Encoding
    2. Bidirectional Intention Extractor
    3. Latent Deliberation Core (K internal steps, 0 scratchpad tokens)
    4. Direct Answer Decoder
    """
    def __init__(
        self,
        vocab_size: int = 262,
        d_model: int = 64,
        d_latent: int = 128,
        k_think_steps: int = 4,
        max_seq_len: int = 512
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.d_latent = d_latent
        self.k_think_steps = k_think_steps

        # 1. Embeddings
        self.token_embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_seq_len)

        # 2. Intention Extractor (Bidirectional GRU)
        self.context_encoder = nn.GRU(
            input_size=d_model,
            hidden_size=d_model // 2,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )

        # Projections to latent space
        self.goal_proj = nn.Linear(d_model, d_latent)
        self.belief_proj = nn.Linear(d_model, d_latent)
        self.init_state_proj = nn.Linear(d_latent * 2, d_latent)

        # 3. Latent Deliberation Core
        self.think_core = LatentDeliberationCore(d_latent, d_model)

        # 4. Output Decoders
        # Direct numeric value regression/prediction head
        self.numeric_head = nn.Sequential(
            nn.Linear(d_latent, d_latent // 2),
            nn.GELU(),
            nn.Linear(d_latent // 2, 1)
        )
        
        # Direct Byte Token Emission Decoder (for generating answer string)
        self.ans_token_embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.ans_decoder_cell = nn.GRUCell(d_model, d_latent)
        self.ans_out_head = nn.Linear(d_latent, vocab_size)

    def extract_intention(
        self,
        prompt_tokens: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extracts goal G, premise representations B, and initial state S_0.
        """
        x_emb = self.token_embed(prompt_tokens) # (B, L, d_model)
        x_emb = self.pos_encoder(x_emb)
        
        # Context encoding
        h_seq, _ = self.context_encoder(x_emb) # (B, L, d_model)
        
        # Find query token position (token 4 = QUERY_TOKEN) or use last token
        is_query = (prompt_tokens == 4).float()
        has_query = (is_query.sum(dim=-1, keepdim=True) > 0).float()
        
        # Masked average for query representation if present, else last valid token
        if mask is not None:
            query_weights = is_query * mask
        else:
            query_weights = is_query
            
        sum_weights = query_weights.sum(dim=-1, keepdim=True) + 1e-8
        g_pooled = torch.bmm(query_weights.unsqueeze(1) / sum_weights.unsqueeze(1), h_seq).squeeze(1)
        
        # For samples without explicit query token, fallback to last token
        last_tok = h_seq[:, -1, :]
        g_rep = has_query * g_pooled + (1.0 - has_query) * last_tok
        
        # Belief representation: average over all premise tokens
        if mask is not None:
            b_weights = mask / (mask.sum(dim=-1, keepdim=True) + 1e-8)
            b_rep = torch.bmm(b_weights.unsqueeze(1), h_seq).squeeze(1)
        else:
            b_rep = h_seq.mean(dim=1)

        goal = self.goal_proj(g_rep)
        belief = self.belief_proj(b_rep)
        s_0 = torch.tanh(self.init_state_proj(torch.cat([goal, belief], dim=-1)))

        return s_0, goal, h_seq, h_seq

    def deliberate(
        self,
        s_0: torch.Tensor,
        goal: torch.Tensor,
        premise_keys: torch.Tensor,
        premise_vals: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        k_steps: Optional[int] = None,
        halt_threshold: float = 0.95
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        """
        Executes internal latent thinking.
        Returns: (final_state, state_trajectory, verification_trajectory)
        """
        k_max = k_steps or self.k_think_steps
        s_curr = s_0
        states = [s_curr]
        verifications = []

        for step in range(k_max):
            s_next, nu = self.think_core.forward_step(
                s_curr, goal, premise_keys, premise_vals, mask
            )
            states.append(s_next)
            verifications.append(nu)
            s_curr = s_next
            
            # Early stopping check (if dynamic inference requested and all batch verified)
            if not self.training and (nu >= halt_threshold).all():
                break

        return s_curr, states, verifications

    def forward(
        self,
        prompt_tokens: torch.Tensor,
        target_answer_tokens: Optional[torch.Tensor] = None,
        prompt_mask: Optional[torch.Tensor] = None,
        k_steps: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Full forward pass:
        1. Extract intention
        2. Deliberate internally (K steps, 0 scratchpad tokens)
        3. Predict numeric answer and decode answer token logits
        """
        s_0, goal, p_keys, p_vals = self.extract_intention(prompt_tokens, prompt_mask)
        s_final, states, verifications = self.deliberate(s_0, goal, p_keys, p_vals, prompt_mask, k_steps)

        # Numeric prediction
        pred_val = self.numeric_head(s_final).squeeze(-1)

        # Answer token decoding logits
        token_logits = None
        if target_answer_tokens is not None:
            # Teacher-forced byte decoding
            # target_answer_tokens: (B, A_len)
            batch_size, a_len = target_answer_tokens.size()
            logits_list = []
            dec_state = s_final
            
            # Start with BOS or 0
            curr_in = torch.ones((batch_size,), dtype=torch.long, device=prompt_tokens.device) # BOS=1
            
            for t in range(a_len):
                emb_in = self.ans_token_embed(curr_in)
                dec_state = self.ans_decoder_cell(emb_in, dec_state)
                logits = self.ans_out_head(dec_state)
                logits_list.append(logits.unsqueeze(1))
                curr_in = target_answer_tokens[:, t]
                
            token_logits = torch.cat(logits_list, dim=1) # (B, A_len, vocab_size)

        return {
            "s_final": s_final,
            "pred_val": pred_val,
            "token_logits": token_logits,
            "states": states,
            "verifications": verifications,
            "scratchpad_tokens_emitted": 0 # Mathematically ZERO external tokens!
        }

    def generate_answer(
        self,
        prompt_tokens: torch.Tensor,
        prompt_mask: Optional[torch.Tensor] = None,
        max_ans_len: int = 10,
        k_steps: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Autoregressive direct emission of final answer string without intermediate scratchpad.
        Returns: (predicted_tokens, predicted_values)
        """
        self.eval()
        with torch.no_grad():
            s_0, goal, p_keys, p_vals = self.extract_intention(prompt_tokens, prompt_mask)
            s_final, _, _ = self.deliberate(s_0, goal, p_keys, p_vals, prompt_mask, k_steps)
            pred_val = self.numeric_head(s_final).squeeze(-1)
            
            batch_size = prompt_tokens.size(0)
            dec_state = s_final
            curr_in = torch.ones((batch_size,), dtype=torch.long, device=prompt_tokens.device) # BOS=1
            
            generated = []
            for _ in range(max_ans_len):
                emb_in = self.ans_token_embed(curr_in)
                dec_state = self.ans_decoder_cell(emb_in, dec_state)
                logits = self.ans_out_head(dec_state)
                next_tok = torch.argmax(logits, dim=-1)
                generated.append(next_tok.unsqueeze(1))
                curr_in = next_tok
                
            gen_tokens = torch.cat(generated, dim=1)
            return gen_tokens, pred_val
