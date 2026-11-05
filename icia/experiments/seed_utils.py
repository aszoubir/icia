"""
Shared seeding for ICIA benchmark experiments.

Run with `ICIA_SEED=<int>` in the environment to make every stochastic
benchmark (torch init, numpy, python random, env slip noise) reproducible.
Default seed is 42, matching the historical benchmark artifacts.
"""

import os
import random

import numpy as np


def apply_experiment_seed(default: int = 42) -> int:
    seed = int(os.environ.get("ICIA_SEED", default))
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass
    return seed
