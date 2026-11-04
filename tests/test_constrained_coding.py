"""
Test Suite for Constrained Algorithmic Coding Benchmark (Phase 12).
Verifies AST invariant enforcement and zero negation blindness.
"""

from icia.tasks.mbpp_constrained_benchmark import (
    CONSTRAINED_BENCHMARK_SUITE,
    check_ast_prohibitions,
)
from icia.experiments.run_constrained_coding_benchmark import evaluate_candidate_code
from icia.code.nl_programmer import NLProgrammerEngine


def test_llm_negation_blindness_detection():
    """Confirms that baseline LLM outputs violate negative constraints."""
    for ch in CONSTRAINED_BENCHMARK_SUITE:
        eval_res = evaluate_candidate_code(ch.llm_baseline_code, ch)
        assert len(eval_res["violations"]) > 0, f"Task {ch.task_id} failed to detect LLM violation"
        assert not eval_res["adhered_to_constraints"]
        assert not eval_res["tests_passed"]


def test_icia_constrained_synthesis_pass_rate():
    """Confirms that ICIA achieves 100% adherence and 100% functional pass rate."""
    engine = NLProgrammerEngine()
    for ch in CONSTRAINED_BENCHMARK_SUITE:
        out = engine.compile_and_synthesize(ch.prompt)
        assert out["syntax_valid"]

        eval_res = evaluate_candidate_code(out["code"], ch)
        assert eval_res["syntax_valid"], f"Task {ch.task_id} syntax invalid: {eval_res['syntax_error']}"
        assert len(eval_res["violations"]) == 0, f"Task {ch.task_id} has violations: {eval_res['violations']}"
        assert eval_res["adhered_to_constraints"]
        assert eval_res["tests_passed"], f"Task {ch.task_id} failed tests: {eval_res['exec_error']}"
