"""
Unit Tests for Phase 13: Hybrid Foundation Governor & Soft-Slack Relaxation.
"""

from icia.primitives.relaxed_barriers import RelaxedBarrierOptimizer
from icia.models.hybrid_bridge import ICIAHybridGovernor
from icia.tasks.mbpp_constrained_benchmark import CONSTRAINED_BENCHMARK_SUITE


def test_relaxed_barrier_optimizer_soft_and_hard():
    opt = RelaxedBarrierOptimizer()
    opt.add_constraint("hard_safety", is_hard=True)
    opt.add_constraint("soft_latency", is_hard=False, weight=10.0)

    # 1. No violations
    diag1 = opt.solve_relaxation({"hard_safety": 0.0, "soft_latency": 0.0})
    assert diag1.feasible
    assert diag1.total_slack == 0.0
    assert diag1.hard_invariants_preserved

    # 2. Soft violation -> should relax with slack
    diag2 = opt.solve_relaxation({"hard_safety": 0.0, "soft_latency": 5.0})
    assert diag2.feasible
    assert diag2.total_slack == 5.0
    assert diag2.hard_invariants_preserved
    assert len(diag2.relaxed_constraints) == 1

    # 3. Hard violation -> cannot relax
    diag3 = opt.solve_relaxation({"hard_safety": 1.0, "soft_latency": 5.0})
    assert not diag3.feasible
    assert not diag3.hard_invariants_preserved


def test_hybrid_governor_interception_and_repair():
    governor = ICIAHybridGovernor()

    for ch in CONSTRAINED_BENCHMARK_SUITE:
        res = governor.govern_and_execute(
            prompt=ch.prompt,
            foundation_code=ch.llm_baseline_code,
            target_fn_name=ch.target_fn_name,
            test_cases=ch.test_cases,
            explicit_prohibitions=ch.prohibited_patterns
        )

        assert res.foundation_had_violations, f"Task {ch.task_id} did not detect LLM violations"
        assert len(res.interventions_applied) > 0
        assert res.tests_passed, f"Task {ch.task_id} failed after governance"
        assert res.syntax_valid
        assert res.relaxation.hard_invariants_preserved
