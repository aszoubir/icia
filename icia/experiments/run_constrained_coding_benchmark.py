"""
Empirical Benchmark Runner: Constrained Coding & Negation Blindness Resolution (Phase 12).
Evaluates ICIA Repulsive Barrier Compilation vs. Autoregressive LLMs on HumanEval/MBPP-Hard Subset.
"""

from __future__ import annotations
import ast
import json
import time
from typing import Dict, Any, List
from icia.tasks.mbpp_constrained_benchmark import CONSTRAINED_BENCHMARK_SUITE, ConstrainedChallenge, check_ast_prohibitions
from icia.code.nl_programmer import NLProgrammerEngine


def evaluate_candidate_code(code: str, challenge: ConstrainedChallenge) -> Dict[str, Any]:
    """Tests syntax, checks prohibition violations via AST invariants, and executes unit tests."""
    is_valid_syntax = False
    syntax_err = None
    try:
        ast.parse(code)
        is_valid_syntax = True
    except SyntaxError as e:
        syntax_err = str(e)

    # Check prohibitions (negation violations) via AST invariants
    violations = check_ast_prohibitions(code, challenge.target_fn_name, challenge.prohibited_patterns)

    # Functional test execution
    tests_passed = False
    exec_err = None
    test_results = []

    if is_valid_syntax and not violations:
        scope = {}
        try:
            exec(code, scope)
            target_fn = scope.get(challenge.target_fn_name)
            if target_fn is None:
                # Fallback: search for any callable
                for k, v in scope.items():
                    if callable(v) and not k.startswith("__"):
                        target_fn = v
                        break

            if target_fn is not None:
                all_ok = True
                for tc in challenge.test_cases:
                    args = tc["args"]
                    expected = tc["expected"]
                    actual = target_fn(*args)
                    ok = (actual == expected)
                    if not ok:
                        all_ok = False
                    test_results.append({
                        "args": args,
                        "expected": expected,
                        "actual": actual,
                        "passed": ok
                    })
                tests_passed = all_ok and len(test_results) > 0
        except Exception as e:
            exec_err = str(e)

    return {
        "syntax_valid": is_valid_syntax,
        "syntax_error": syntax_err,
        "violations": violations,
        "adhered_to_constraints": len(violations) == 0,
        "tests_passed": tests_passed,
        "exec_error": exec_err,
        "test_count": len(test_results)
    }


def run_constrained_benchmark() -> Dict[str, Any]:
    print("=" * 80)
    print("ICIA PHASE 12: CONSTRAINED CODING & NEGATION BLINDNESS BENCHMARK")
    print("Dataset: MBPP-Hard Constrained Subset (5 Algorithmic Tasks with Negative Barriers)")
    print("=" * 80)

    engine = NLProgrammerEngine()
    results = []

    llm_syntax_count = 0
    llm_adherence_count = 0
    llm_pass_count = 0

    icia_syntax_count = 0
    icia_adherence_count = 0
    icia_pass_count = 0

    for ch in CONSTRAINED_BENCHMARK_SUITE:
        print(f"\n[Task {ch.task_id}] {ch.name}")
        print(f"  Prompt: {ch.prompt}")
        print(f"  Prohibitions: {ch.prohibited_patterns}")

        # 1. Evaluate LLM Baseline
        llm_eval = evaluate_candidate_code(ch.llm_baseline_code, ch)
        if llm_eval["syntax_valid"]:
            llm_syntax_count += 1
        if llm_eval["adhered_to_constraints"]:
            llm_adherence_count += 1
        if llm_eval["tests_passed"]:
            llm_pass_count += 1

        print(f"  Autoregressive LLM: Syntax={llm_eval['syntax_valid']}, "
              f"Violations={llm_eval['violations']}, Pass={llm_eval['tests_passed']}")

        # 2. Evaluate ICIA Intention Compiler
        t0 = time.perf_counter()
        icia_out = engine.compile_and_synthesize(ch.prompt)
        dt_ms = (time.perf_counter() - t0) * 1000.0

        icia_eval = evaluate_candidate_code(icia_out["code"], ch)
        if icia_eval["syntax_valid"]:
            icia_syntax_count += 1
        if icia_eval["adhered_to_constraints"]:
            icia_adherence_count += 1
        if icia_eval["tests_passed"]:
            icia_pass_count += 1

        print(f"  ICIA Architecture : Syntax={icia_eval['syntax_valid']}, "
              f"Violations={icia_eval['violations']}, Pass={icia_eval['tests_passed']} ({dt_ms:.2f} ms)")

        results.append({
            "task_id": ch.task_id,
            "name": ch.name,
            "llm": llm_eval,
            "icia": icia_eval,
            "icia_code": icia_out["code"],
            "latency_ms": dt_ms
        })

    n = len(CONSTRAINED_BENCHMARK_SUITE)
    summary = {
        "num_tasks": n,
        "llm_syntax_rate": (llm_syntax_count / n) * 100.0,
        "llm_constraint_adherence_rate": (llm_adherence_count / n) * 100.0,
        "llm_functional_pass_rate": (llm_pass_count / n) * 100.0,
        "icia_syntax_rate": (icia_syntax_count / n) * 100.0,
        "icia_constraint_adherence_rate": (icia_adherence_count / n) * 100.0,
        "icia_functional_pass_rate": (icia_pass_count / n) * 100.0,
        "tasks": results
    }

    print("\n" + "=" * 80)
    print("CONSTRAINED CODING BENCHMARK SCORECARD")
    print("=" * 80)
    print(f"Syntax Validity Rate        : LLM = {summary['llm_syntax_rate']:.1f}% | ICIA = {summary['icia_syntax_rate']:.1f}%")
    print(f"Constraint Adherence (No Negation Blindness): LLM = {summary['llm_constraint_adherence_rate']:.1f}% | ICIA = {summary['icia_constraint_adherence_rate']:.1f}%")
    print(f"Functional Test Pass Rate   : LLM = {summary['llm_functional_pass_rate']:.1f}% | ICIA = {summary['icia_functional_pass_rate']:.1f}%")
    print("=" * 80)

    with open("icia_constrained_coding_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary


if __name__ == "__main__":
    run_constrained_benchmark()
