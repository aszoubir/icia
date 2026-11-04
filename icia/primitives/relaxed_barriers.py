"""
Soft-Slack Augmented Lagrangian Barrier Relaxation for ICIA.
Solves the Infeasible Optimization Problem when barrier constraints conflict,
enabling graceful Pareto-optimal constraint relaxation while preserving hard safety invariants.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional
import numpy as np


@dataclass
class BarrierConstraint:
    """
    Represents an invariant constraint c_j(S) <= 0.
    is_hard: True for strict non-negotiable invariants (syntax, safety, physics boundaries).
             False for soft preferences that can be relaxed under infeasibility.
    weight: Penalty weight w_j for relaxing this constraint. Hard constraints have infinite weight.
    """
    name: str
    is_hard: bool = True
    weight: float = 1000.0  # High weight for soft constraints, hard constraints are non-relaxable
    multiplier: float = 0.0  # Dual multiplier lambda_j
    slack: float = 0.0       # Non-negative slack variable xi_j >= 0

    def evaluate(self, state: np.ndarray) -> float:
        """Computes c_j(S). Subclasses or custom callables define this."""
        return 0.0


@dataclass
class RelaxationDiagnosis:
    feasible: bool
    relaxed_constraints: List[str]
    total_slack: float
    hard_invariants_preserved: bool
    explanation: str


class RelaxedBarrierOptimizer:
    """
    Augmented Lagrangian barrier solver with slack variables:
    min_S  Psi_G(S) + sum_j [ lambda_j (c_j(S) - xi_j) + 0.5 * rho * max(0, c_j(S) - xi_j)^2 + w_j * xi_j ]
    """

    def __init__(self, rho: float = 10.0, max_slack: float = 100.0):
        self.rho = rho
        self.max_slack = max_slack
        self.constraints: List[BarrierConstraint] = []

    def add_constraint(self, name: str, is_hard: bool = True, weight: float = 100.0) -> BarrierConstraint:
        c = BarrierConstraint(name=name, is_hard=is_hard, weight=float("inf") if is_hard else weight)
        self.constraints.append(c)
        return c

    def solve_relaxation(self, violations: Dict[str, float]) -> RelaxationDiagnosis:
        """
        Given measured constraint violations c_j(S) > 0:
        1. Checks if any hard invariants are violated.
        2. If hard invariants are violated, they CANNOT be relaxed -> infeasible.
        3. For soft constraints, allocates minimal slack xi_j = c_j(S) to restore feasibility.
        4. Computes total relaxation cost and returns full diagnosis.
        """
        hard_violations = []
        soft_violations = []

        for name, val in violations.items():
            if val > 0.0:
                # Find matching constraint
                c = next((item for item in self.constraints if item.name == name), None)
                if c is None or c.is_hard:
                    hard_violations.append((name, val))
                else:
                    soft_violations.append((c, val))

        if hard_violations:
            names = [h[0] for h in hard_violations]
            return RelaxationDiagnosis(
                feasible=False,
                relaxed_constraints=[],
                total_slack=0.0,
                hard_invariants_preserved=False,
                explanation=f"CRITICAL: Non-negotiable hard invariants violated: {names}. Cannot relax safety barriers."
            )

        if not soft_violations:
            return RelaxationDiagnosis(
                feasible=True,
                relaxed_constraints=[],
                total_slack=0.0,
                hard_invariants_preserved=True,
                explanation="All constraints strictly satisfied. No relaxation required."
            )

        # Allocate slack to soft constraints in order of penalty weight (cheapest first)
        soft_violations.sort(key=lambda x: x[0].weight)
        relaxed_names = []
        total_slack = 0.0

        for c, val in soft_violations:
            slack_needed = min(val, self.max_slack)
            c.slack = slack_needed
            c.multiplier += self.rho * (val - slack_needed)
            total_slack += slack_needed
            relaxed_names.append(f"{c.name} (+{slack_needed:.2f} slack)")

        return RelaxationDiagnosis(
            feasible=True,
            relaxed_constraints=relaxed_names,
            total_slack=total_slack,
            hard_invariants_preserved=True,
            explanation=f"Feasibility restored by soft relaxation of: {', '.join(relaxed_names)}. Hard safety invariants preserved."
        )
