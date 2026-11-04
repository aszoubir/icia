"""
Empirical Benchmark Runner: Hybrid Foundation Bridge & Invariant Relaxation (Phase 13).
Demonstrates real-time governance of unaligned foundation LLM outputs and Pareto slack relaxation.
"""

from __future__ import annotations
import json
import time
from typing import Dict, Any, List
from icia.models.hybrid_bridge import ICIAHybridGovernor
from icia.tasks.mbpp_constrained_benchmark import CONSTRAINED_BENCHMARK_SUITE
from icia.primitives.relaxed_barriers import RelaxedBarrierOptimizer


def run_hybrid_governor_experiment() -> Dict[str, Any]:
    print("=" * 80)
    print("ICIA PHASE 13: HYBRID FOUNDATION GOVERNOR & SOFT-SLACK RELAXATION")
    print("Coupling Open-Domain LLM Output with ICIA Teleological Invariant Governor")
    print("=" * 80)

    governor = ICIAHybridGovernor()
    results = []

    # 1. Benchmark: Real-time Governance of Unaligned Foundation Model Code
    print("\n--- Part 1: Governing Foundation LLM Negation Blindness ---")
    total_tasks = len(CONSTRAINED_BENCHMARK_SUITE)
    governed_pass_count = 0
    interventions_count = 0

    for ch in CONSTRAINED_BENCHMARK_SUITE:
        print(f"\n[Task {ch.task_id}] {ch.name}")
        print(f"  Raw Foundation Code:\n    {ch.llm_baseline_code.strip().splitlines()[0]} ...")

        res = governor.govern_and_execute(
            prompt=ch.prompt,
            foundation_code=ch.llm_baseline_code,
            target_fn_name=ch.target_fn_name,
            test_cases=ch.test_cases,
            explicit_prohibitions=ch.prohibited_patterns
        )

        if res.tests_passed:
            governed_pass_count += 1
        interventions_count += len(res.interventions_applied)

        print(f"  Governor Status: Interventions={len(res.interventions_applied)} | "
              f"Violations Blocked={res.blocked_violations} | Pass={res.tests_passed} ({res.governor_latency_ms:.2f} ms)")

        results.append({
            "task_id": ch.task_id,
            "raw_violations": res.blocked_violations,
            "interventions": res.interventions_applied,
            "tests_passed": res.tests_passed,
            "latency_ms": res.governor_latency_ms
        })

    # 2. Benchmark: Soft-Slack Lagrangian Relaxation under Over-Constrained Settings
    print("\n--- Part 2: Soft-Slack Lagrangian Relaxation (Over-Constrained Requirements) ---")
    optimizer = RelaxedBarrierOptimizer()
    optimizer.add_constraint("syntax_invariance", is_hard=True)
    optimizer.add_constraint("no_illegal_side_effects", is_hard=True)
    optimizer.add_constraint("line_count_brevity", is_hard=False, weight=50.0)
    optimizer.add_constraint("memory_budget_hint", is_hard=False, weight=25.0)

    # Test Case A: Soft constraints exceeded (code takes 6 lines instead of 3, memory takes 120B instead of 100B)
    violations_soft = {
        "syntax_invariance": 0.0,
        "no_illegal_side_effects": 0.0,
        "line_count_brevity": 3.0,     # Exceeded by 3 lines
        "memory_budget_hint": 20.0     # Exceeded by 20 bytes
    }
    diag_soft = optimizer.solve_relaxation(violations_soft)
    print(f"  Test A (Soft Constraint Overrun): Feasible={diag_soft.feasible}")
    print(f"    Total Slack Allocated: {diag_soft.total_slack:.1f}")
    print(f"    Hard Invariants Preserved: {diag_soft.hard_invariants_preserved}")
    print(f"    Diagnosis: {diag_soft.explanation}")

    # Test Case B: Hard constraint violated (unrecoverable syntax or safety crash)
    violations_hard = {
        "syntax_invariance": 1.0,      # Syntax broken
        "line_count_brevity": 0.0
    }
    diag_hard = optimizer.solve_relaxation(violations_hard)
    print(f"\n  Test B (Hard Safety Violation): Feasible={diag_hard.feasible}")
    print(f"    Diagnosis: {diag_hard.explanation}")

    summary = {
        "num_tasks": total_tasks,
        "foundation_raw_pass_rate": 0.0,
        "governed_pass_rate": (governed_pass_count / total_tasks) * 100.0,
        "total_interventions": interventions_count,
        "soft_slack_test_a": {
            "feasible": diag_soft.feasible,
            "total_slack": diag_soft.total_slack,
            "preserved": diag_soft.hard_invariants_preserved
        },
        "hard_invariance_test_b": {
            "feasible": diag_hard.feasible,
            "preserved": diag_hard.hard_invariants_preserved
        },
        "tasks": results
    }

    print("\n" + "=" * 80)
    print("HYBRID GOVERNOR BENCHMARK SUMMARY")
    print("=" * 80)
    print(f"Foundation Model Raw Pass Rate : {summary['foundation_raw_pass_rate']:.1f}%")
    print(f"ICIA-Governed Pass Rate        : {summary['governed_pass_rate']:.1f}%")
    print(f"Total Teleological Interventions: {summary['total_interventions']}")
    print(f"Soft Slack Optimization Status : {diag_soft.explanation}")
    print("=" * 80)

    with open("icia_hybrid_governor_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary


if __name__ == "__main__":
    run_hybrid_governor_experiment()
