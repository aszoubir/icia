"""
ICIA Hierarchical Intention Decomposition Engine
Implements Multi-Scale Teleological Hierarchy:
Level 2: Macro-Intention (Global mission, safety invariants, horizon)
Level 1: Tactical Sub-Goal Manifold (Dynamic decomposition, sequencing, impasse re-routing)
Level 0: Operational Actuation (Micro-step gradient flow / action emission)
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Callable
import numpy as np

from icia.primitives.intention_state import IntentionState


@dataclass
class SubGoal:
    """Represents a discrete milestone in the teleological hierarchy."""
    id: str
    description: str
    target_state: np.ndarray
    preconditions: List[str] = field(default_factory=list)
    completion_radius: float = 0.5
    is_completed: bool = False
    is_invalidated: bool = False
    metadata: Dict[str, any] = field(default_factory=dict)

    def is_satisfied(self, current_state: np.ndarray, world_facts: Dict[str, bool]) -> bool:
        """Evaluates whether this sub-goal is achieved."""
        # Check world state facts (e.g. 'has_key_A': True)
        for pre in self.preconditions:
            if not world_facts.get(pre, False):
                return False
        # Check geometric or state distance
        dim = min(len(current_state), len(self.target_state))
        dist = np.linalg.norm(current_state[:dim] - self.target_state[:dim])
        return dist <= self.completion_radius


class HierarchicalIntentionController:
    """
    3-Tier Hierarchical Intention Engine.
    Coordinates between invariant macro-mission and reactive operational steps.
    """
    def __init__(
        self,
        macro_goal: np.ndarray,
        global_constraints: Optional[np.ndarray] = None,
        impasse_threshold: float = 0.8,
        operational_step_size: float = 0.2
    ):
        self.macro_goal = np.asarray(macro_goal, dtype=np.float32)
        self.global_constraints = (
            np.asarray(global_constraints, dtype=np.float32)
            if global_constraints is not None
            else np.empty((0, len(macro_goal) + 1), dtype=np.float32)
        )
        self.impasse_threshold = impasse_threshold
        self.operational_step_size = operational_step_size

        # Teleological hierarchy components
        self.sub_goals: List[SubGoal] = []
        self.active_sub_goal_idx: Optional[int] = None
        self.completed_sub_goals: List[str] = []
        self.discovered_obstacles: List[np.ndarray] = []
        
        # Telemetry
        self.total_replan_count: int = 0
        self.impasse_detected: bool = False
        self.macro_completed: bool = False

    def decompose(self, sub_goals: List[SubGoal]) -> None:
        """Sets or replaces the active tactical plan."""
        self.sub_goals = sub_goals
        self.active_sub_goal_idx = 0 if sub_goals else None

    def get_active_sub_goal(self) -> Optional[SubGoal]:
        """Returns the current active milestone."""
        if self.active_sub_goal_idx is not None and self.active_sub_goal_idx < len(self.sub_goals):
            return self.sub_goals[self.active_sub_goal_idx]
        return None

    def compute_action(
        self,
        current_state: np.ndarray,
        world_facts: Dict[str, bool],
        discrepancy: float = 0.0
    ) -> Tuple[np.ndarray, Dict[str, any]]:
        """
        Executes one hierarchical deliberation cycle:
        1. Reality Verification & Impasse Detection.
        2. Sub-Goal Progression & Retraction.
        3. Operational Direction Calculation.
        """
        telemetry = {
            "impasse": False,
            "replan": False,
            "subgoal_completed": None,
            "active_subgoal": None
        }

        # 1. Impasse Detection & Instantaneous Pivot
        if discrepancy > self.impasse_threshold:
            self.impasse_detected = True
            telemetry["impasse"] = True
            self._handle_impasse(current_state, world_facts)
            telemetry["replan"] = True

        # 2. Check and advance active sub-goal
        active_sg = self.get_active_sub_goal()
        if active_sg is not None:
            if active_sg.is_satisfied(current_state, world_facts):
                active_sg.is_completed = True
                self.completed_sub_goals.append(active_sg.id)
                telemetry["subgoal_completed"] = active_sg.id
                
                # Advance to next uncompleted sub-goal
                self._advance_to_next_valid_subgoal(world_facts)
                active_sg = self.get_active_sub_goal()

        # 3. Check macro completion
        macro_dist = np.linalg.norm(current_state[:len(self.macro_goal)] - self.macro_goal)
        if (active_sg is None and macro_dist <= 0.5) or macro_dist <= 0.2:
            self.macro_completed = True

        # 4. Level 0: Operational Gradient Flow towards active target
        target = active_sg.target_state if active_sg is not None else self.macro_goal
        telemetry["active_subgoal"] = active_sg.id if active_sg else "MACRO_TARGET"

        direction = target[:len(current_state)] - current_state
        dist = np.linalg.norm(direction)
        if dist > 1e-6:
            unit_dir = direction / dist
            # Repel from discovered obstacle barriers
            repulsion = np.zeros_like(current_state)
            for obs in self.discovered_obstacles:
                obs_diff = current_state - obs
                obs_d = np.linalg.norm(obs_diff)
                if 0.0 < obs_d < 1.2:
                    repulsion += (obs_diff / (obs_d ** 2)) * 0.3
            step = (unit_dir + repulsion) * self.operational_step_size
        else:
            step = np.zeros_like(current_state)

        return step, telemetry

    def _advance_to_next_valid_subgoal(self, world_facts: Dict[str, bool]) -> None:
        """Finds next valid sub-goal whose preconditions are satisfied."""
        if self.active_sub_goal_idx is None:
            return
        
        next_idx = self.active_sub_goal_idx + 1
        while next_idx < len(self.sub_goals):
            candidate = self.sub_goals[next_idx]
            if not candidate.is_completed and not candidate.is_invalidated:
                preconditions_met = all(world_facts.get(p, False) for p in candidate.preconditions)
                if preconditions_met or not candidate.preconditions:
                    self.active_sub_goal_idx = next_idx
                    return
            next_idx += 1
        
        # All sub-goals satisfied or exhausted
        self.active_sub_goal_idx = len(self.sub_goals)

    def _handle_impasse(self, current_state: np.ndarray, world_facts: Dict[str, bool]) -> None:
        """
        Instantaneous 1-step re-routing:
        Invalidates current blocked sub-goal, records obstacle, and pivots to alternate contingency branch.
        """
        self.total_replan_count += 1
        active_sg = self.get_active_sub_goal()
        if active_sg is not None:
            active_sg.is_invalidated = True
            # Record current blocked point
            self.discovered_obstacles.append(active_sg.target_state.copy())
            
            # Check if alternate contingency branch exists
            contingencies = active_sg.metadata.get("contingencies", [])
            if not contingencies and "contingency" in active_sg.metadata:
                contingencies = [active_sg.metadata["contingency"]]

            if contingencies:
                for offset, c_dict in enumerate(contingencies):
                    alt_subgoal = SubGoal(
                        id=c_dict["id"],
                        description=c_dict["description"],
                        target_state=np.array(c_dict["target_state"], dtype=np.float32),
                        preconditions=c_dict.get("preconditions", []),
                        completion_radius=c_dict.get("completion_radius", 0.4),
                        metadata=c_dict.get("metadata", {})
                    )
                    self.sub_goals.insert(self.active_sub_goal_idx + 1 + offset, alt_subgoal)
                self.active_sub_goal_idx += 1
            else:
                self._advance_to_next_valid_subgoal(world_facts)
