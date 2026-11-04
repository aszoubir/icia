"""
ICIA Pure NumPy Engine
Zero-dependency, mathematically transparent implementation of:
1. Standard Causal Self-Attention (SAT)
2. Intention-Conditioned Self-Attention (ICT)
3. ICIA Dual-Stream Intention-Verification Recurrent Engine (DIVRE)
Demonstrates foundational mathematical viability on raw linear algebra.
"""

import numpy as np
from typing import Tuple, Dict, Any, List


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


def gelu(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * np.power(x, 3))))


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -15.0, 15.0)))


class NumPyStandardTransformer:
    """Standard Causal Self-Attention sequence model implemented in pure NumPy"""
    def __init__(self, vocab_size: int, d_model: int = 32, max_len: int = 512, seed: int = 42):
        np.random.seed(seed)
        self.vocab_size = vocab_size
        self.d_model = d_model
        
        # Weights
        self.W_emb = np.random.randn(vocab_size, d_model) * 0.02
        self.W_pos = np.random.randn(max_len, d_model) * 0.02
        self.W_q = np.random.randn(d_model, d_model) * 0.02
        self.W_k = np.random.randn(d_model, d_model) * 0.02
        self.W_v = np.random.randn(d_model, d_model) * 0.02
        self.W_out = np.random.randn(d_model, d_model) * 0.02
        self.W_head = np.random.randn(d_model, vocab_size) * 0.02

    def forward(self, token_seq: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        T = len(token_seq)
        x = self.W_emb[token_seq] + self.W_pos[:T]  # (T, d_model)

        Q = np.dot(x, self.W_q)
        K = np.dot(x, self.W_k)
        V = np.dot(x, self.W_v)

        # Scaled dot-product with causal mask
        scores = np.dot(Q, K.T) / np.sqrt(self.d_model)
        mask = np.triu(np.full((T, T), -1e9), k=1)
        scores += mask
        attn_weights = softmax(scores, axis=-1)

        context = np.dot(attn_weights, V)
        h = np.dot(context, self.W_out) + x
        logits = np.dot(h, self.W_head)
        return logits, attn_weights


class NumPyIntentionConditionedTransformer:
    """Intention-augmented Transformer in pure NumPy"""
    def __init__(self, vocab_size: int, d_model: int = 32, max_len: int = 128, seed: int = 42):
        np.random.seed(seed)
        self.vocab_size = vocab_size
        self.d_model = d_model
        
        self.W_emb = np.random.randn(vocab_size, d_model) * 0.02
        self.W_pos = np.random.randn(max_len, d_model) * 0.02
        self.W_q = np.random.randn(d_model, d_model) * 0.02
        self.W_k = np.random.randn(d_model, d_model) * 0.02
        self.W_v = np.random.randn(d_model, d_model) * 0.02
        self.W_cross = np.random.randn(d_model, d_model) * 0.02
        self.W_head = np.random.randn(d_model, vocab_size) * 0.02

    def forward(self, token_seq: List[int], goal_vec: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        T = len(token_seq)
        x = self.W_emb[token_seq] + self.W_pos[:T]

        # Standard self-attention
        Q = np.dot(x, self.W_q)
        K = np.dot(x, self.W_k)
        V = np.dot(x, self.W_v)
        scores = np.dot(Q, K.T) / np.sqrt(self.d_model) + np.triu(np.full((T, T), -1e9), k=1)
        attn_weights = softmax(scores, axis=-1)
        h = np.dot(attn_weights, V)

        # Goal modulation
        goal_proj = np.dot(goal_vec, self.W_cross)
        h = h + goal_proj
        logits = np.dot(h, self.W_head)
        return logits, attn_weights


class NumPyDIVREModel:
    """
    ICIA Candidate C: Dual-Stream Intention-Verification Recurrent Engine
    Pure NumPy implementation of closed-loop teleological intelligence.
    """
    def __init__(self, vocab_size: int, d_model: int = 32, seed: int = 42):
        np.random.seed(seed)
        self.vocab_size = vocab_size
        self.d_model = d_model

        self.W_emb = np.random.randn(vocab_size, d_model) * 0.02
        # Sensory predictor: T_{t-1} -> \hat{X}_t
        self.W_pred = np.random.randn(d_model, d_model) * 0.02
        # Bottom-up verification: [V_{t-1}, X_t, Error] -> V_t
        self.W_v = np.random.randn(d_model * 3, d_model) * 0.02
        # Top-down teleological update: [T_{t-1}, Intention, V_t] -> T_t
        self.W_t = np.random.randn(d_model * 3, d_model) * 0.02
        # Discrepancy gate
        self.W_gate = np.random.randn(d_model, d_model) * 0.02
        # Action head
        self.W_action = np.random.randn(d_model, vocab_size) * 0.02

    def forward(self, token_seq: List[int], goal_vec: np.ndarray) -> Tuple[np.ndarray, List[float]]:
        T = len(token_seq)
        t_state = np.zeros(self.d_model)
        v_state = np.zeros(self.d_model)

        logits_list = []
        errors_norm = []

        for step in range(T):
            x_t = self.W_emb[token_seq[step]]

            # 1. Top-down sensory prediction
            x_hat = np.tanh(np.dot(t_state, self.W_pred))

            # 2. Reality Discrepancy computation
            error_t = x_t - x_hat
            errors_norm.append(float(np.linalg.norm(error_t)))

            # 3. Bottom-up verification update
            v_in = np.concatenate([v_state, x_t, error_t])
            v_state = gelu(np.dot(v_in, self.W_v))

            # 4. Teleological error gate
            gate = sigmoid(np.dot(error_t, self.W_gate))

            # 5. Top-down teleological state update
            t_in = np.concatenate([t_state, goal_vec, v_state])
            t_candidate = np.tanh(np.dot(t_in, self.W_t))
            t_state = t_state + gate * t_candidate

            # 6. Verified state synthesis
            verified_state = t_state * v_state
            step_logits = np.dot(verified_state, self.W_action)
            logits_list.append(step_logits)

        return np.array(logits_list), errors_norm
