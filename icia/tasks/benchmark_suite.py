"""
ICIA Benchmark Suite: Empirical Validation Tasks
1. Distractor Perturbation & Goal Preservation (DPGP)
2. Dynamic Constraint Pivot & Runtime Recovery (DCPR)
3. Teleological Discrepancy & Verification Efficiency (TDVE)
"""

import random
from typing import Tuple, Dict, Any, List
import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class SyntheticBenchmarkGenerator:
    """
    Generates synthetic task batches testing goal preservation, distractor resistance,
    and dynamic constraint satisfaction.
    """
    def __init__(self, vocab_size: int = 128, d_model: int = 64, seed: int = 42):
        self.vocab_size = vocab_size
        self.d_model = d_model
        random.seed(seed)
        np.random.seed(seed)

        # Reserved tokens
        self.GOAL_START = 1
        self.DISTRACTOR_TOKEN = 2
        self.CONSTRAINT_TOKEN = 3
        self.EXECUTE_TOKEN = 4

    def generate_dpgp_batch(self, batch_size: int, distractor_len: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Task 1: Distractor Perturbation & Goal Preservation
        Format: [GOAL_A] [DISTRACTOR_1 ... DISTRACTOR_L] [EXECUTE] -> Target Action = Action_A
        """
        seqs = []
        targets = []
        goals = []

        for _ in range(batch_size):
            # Target goal ID between 10 and vocab_size - 1
            goal_id = random.randint(10, self.vocab_size - 1)
            # Create intention vector representation
            goal_vec = np.zeros(self.d_model, dtype=np.float32)
            goal_vec[goal_id % self.d_model] = 1.0

            # Distractor tokens: random tokens urging alternate actions
            distractors = [random.randint(10, self.vocab_size - 1) for _ in range(distractor_len)]

            # Sequence: [GOAL_ID] + distractors + [EXECUTE]
            seq = [goal_id] + distractors + [self.EXECUTE_TOKEN]
            seqs.append(seq)
            targets.append(goal_id)  # Correct final action is always the original intended goal!
            goals.append(goal_vec)

        if HAS_TORCH:
            return (
                torch.tensor(seqs, dtype=torch.long),
                torch.tensor(goals, dtype=torch.float32),
                torch.tensor(targets, dtype=torch.long)
            )
        return np.array(seqs), np.array(goals), np.array(targets)

    def generate_dcpr_batch(self, batch_size: int, seq_len: int = 20) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """
        Task 2: Dynamic Constraint Pivot & Runtime Recovery
        Agent is moving toward Goal_A. At step t_flip, Constraint_B activates, forcing pivot to Goal_B.
        """
        seqs = []
        targets = []
        goals = []
        t_flip = seq_len // 2

        for _ in range(batch_size):
            goal_a = random.randint(10, self.vocab_size // 2)
            goal_b = random.randint(self.vocab_size // 2 + 1, self.vocab_size - 1)

            # Combined intention vector
            goal_vec = np.zeros(self.d_model, dtype=np.float32)
            goal_vec[goal_a % self.d_model] = 1.0

            seq = []
            target_seq = []
            for t in range(seq_len):
                if t < t_flip:
                    seq.append(goal_a)
                    target_seq.append(goal_a)
                elif t == t_flip:
                    seq.append(self.CONSTRAINT_TOKEN)  # Environment triggers constraint flip!
                    target_seq.append(goal_b)
                else:
                    seq.append(goal_b)
                    target_seq.append(goal_b)

            seqs.append(seq)
            targets.append(target_seq)
            goals.append(goal_vec)

        if HAS_TORCH:
            return (
                torch.tensor(seqs, dtype=torch.long),
                torch.tensor(goals, dtype=torch.float32),
                torch.tensor(targets, dtype=torch.long),
                t_flip
            )
        return np.array(seqs), np.array(goals), np.array(targets), t_flip
