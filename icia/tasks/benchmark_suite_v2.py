"""
ICIA Benchmark Suite v2: High-Performance Vectorized Benchmark Suite
Eliminates tensor conversion overhead; supports long-horizon evaluations up to L=1024.
"""

import random
from typing import Tuple, Dict, Any, List
import numpy as np
import torch


class VectorizedBenchmarkGenerator:
    """
    High-speed, zero-copy synthetic task generator.
    """
    def __init__(self, vocab_size: int = 128, d_model: int = 64, seed: int = 42):
        self.vocab_size = vocab_size
        self.d_model = d_model
        random.seed(seed)
        np.random.seed(seed)

        self.GOAL_START = 1
        self.DISTRACTOR_TOKEN = 2
        self.CONSTRAINT_TOKEN = 3
        self.EXECUTE_TOKEN = 4

    def generate_dpgp_batch(self, batch_size: int, distractor_len: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Task 1: Distractor Perturbation & Goal Preservation (Vectorized)
        Returns:
          x_tensor: (B, distractor_len + 2)
          g_tensor: (B, d_model)
          y_tensor: (B,)
        """
        total_len = distractor_len + 2
        seqs = np.empty((batch_size, total_len), dtype=np.int64)
        goals = np.zeros((batch_size, self.d_model), dtype=np.float32)
        targets = np.random.randint(10, self.vocab_size, size=batch_size, dtype=np.int64)

        # Distractor tokens
        distractors = np.random.randint(10, self.vocab_size, size=(batch_size, distractor_len), dtype=np.int64)

        seqs[:, 0] = targets
        seqs[:, 1:-1] = distractors
        seqs[:, -1] = self.EXECUTE_TOKEN

        # Goal vectors: one-hot projection
        col_indices = targets % self.d_model
        goals[np.arange(batch_size), col_indices] = 1.0

        return (
            torch.from_numpy(seqs),
            torch.from_numpy(goals),
            torch.from_numpy(targets)
        )

    def generate_dcpr_batch(self, batch_size: int, seq_len: int = 24) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """
        Task 2: Dynamic Constraint Pivot & Runtime Recovery (Vectorized)
        Returns:
          x_tensor: (B, seq_len)
          g_tensor: (B, d_model)
          y_tensor: (B, seq_len)
          c_mask: (B, seq_len) boolean mask where constraint flip occurs
          t_flip: int
        """
        t_flip = seq_len // 2
        goal_a = np.random.randint(10, self.vocab_size // 2, size=batch_size, dtype=np.int64)
        goal_b = np.random.randint(self.vocab_size // 2 + 1, self.vocab_size, size=batch_size, dtype=np.int64)

        seqs = np.empty((batch_size, seq_len), dtype=np.int64)
        target_seqs = np.empty((batch_size, seq_len), dtype=np.int64)
        c_mask = np.zeros((batch_size, seq_len), dtype=bool)
        goals = np.zeros((batch_size, self.d_model), dtype=np.float32)

        # Pre-flip: tracking Goal A
        for t in range(t_flip):
            seqs[:, t] = goal_a
            target_seqs[:, t] = goal_a

        # Step t_flip: Constraint obstacle injected!
        seqs[:, t_flip] = self.CONSTRAINT_TOKEN
        target_seqs[:, t_flip] = goal_b
        c_mask[:, t_flip] = True

        # Post-flip: Re-routing to Goal B
        for t in range(t_flip + 1, seq_len):
            seqs[:, t] = goal_b
            target_seqs[:, t] = goal_b

        # Goal conditioning vector
        col_indices = goal_a % self.d_model
        goals[np.arange(batch_size), col_indices] = 1.0

        return (
            torch.from_numpy(seqs),
            torch.from_numpy(goals),
            torch.from_numpy(target_seqs),
            torch.from_numpy(c_mask),
            t_flip
        )
