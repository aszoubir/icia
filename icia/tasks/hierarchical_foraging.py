"""
Hierarchical Multi-Room Key-Door Foraging Environment
Benchmarks hierarchical intention decomposition vs. autoregressive inertia.
"""

from typing import Dict, Tuple, List, Optional
import numpy as np


class HierarchicalForagingEnv:
    """
    Multi-stage continuous-symbolic puzzle environment:
    - Stage 1: Obtain Key A at (2.0, 2.0)
    - Stage 2: Unlock Door 1 at (4.0, 2.0)
    - Stage 3: Obtain Key B (Primary route at (6.0, 4.0) is dynamically blocked with roadblock;
      Alternate Vent route at (5.0, 6.0) is clear)
    - Stage 4: Unlock Terminal Vault at (10.0, 10.0)
    """
    def __init__(self, stochastic_jam: bool = True):
        self.stochastic_jam = stochastic_jam
        self.reset()

    def reset(self) -> Tuple[np.ndarray, Dict[str, bool]]:
        self.agent_pos = np.array([0.0, 0.0], dtype=np.float32)
        self.step_count = 0
        self.facts = {
            "has_key_A": False,
            "door_1_unlocked": False,
            "has_key_B": False,
            "vault_unlocked": False
        }
        self.jammed_corridor_active = self.stochastic_jam
        self.jammed_pos = np.array([6.0, 4.0], dtype=np.float32)
        return self.agent_pos.copy(), self.facts.copy()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, any]]:
        """
        Executes a continuous 2D motion step.
        Returns: next_state, discrepancy_error, done, info
        """
        self.step_count += 1
        new_pos = self.agent_pos + action
        discrepancy = 0.0

        # Collision with jammed corridor if attempting to pass (6.0, 4.0)
        if self.jammed_corridor_active:
            dist_to_jam = np.linalg.norm(new_pos - self.jammed_pos)
            if dist_to_jam < 0.6:
                # Roadblock collision! Agent is bounced back, high reality discrepancy
                discrepancy = 1.5
                new_pos = self.agent_pos  # Cannot proceed through blocked corridor

        # Check Door 1 constraint
        if np.linalg.norm(new_pos - np.array([4.0, 2.0])) < 0.5 and not self.facts["has_key_A"]:
            # Locked door!
            discrepancy = 1.0
            new_pos = self.agent_pos

        self.agent_pos = new_pos

        # Key A pickup
        if np.linalg.norm(self.agent_pos - np.array([2.0, 2.0])) < 0.5:
            self.facts["has_key_A"] = True

        # Door 1 unlocked
        if self.facts["has_key_A"] and np.linalg.norm(self.agent_pos - np.array([4.0, 2.0])) < 0.5:
            self.facts["door_1_unlocked"] = True

        # Key B pickup via primary route (if unjammed) or alternate vent route at (8.0, 6.0)
        if np.linalg.norm(self.agent_pos - np.array([8.0, 6.0])) < 0.5:
            self.facts["has_key_B"] = True

        # Vault unlock
        if self.facts["has_key_B"] and np.linalg.norm(self.agent_pos - np.array([10.0, 10.0])) < 0.6:
            self.facts["vault_unlocked"] = True

        done = self.facts["vault_unlocked"] or (self.step_count >= 120)

        info = {
            "step": self.step_count,
            "facts": self.facts.copy(),
            "vault_unlocked": self.facts["vault_unlocked"],
            "discrepancy": discrepancy
        }

        return self.agent_pos.copy(), discrepancy, done, info
