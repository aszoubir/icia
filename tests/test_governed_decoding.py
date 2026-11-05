"""
Tests for Phase 16: real-decoder governed benchmark pipeline.

Covers corpus validity (reference solutions must satisfy their own barriers),
pipeline schema, and seeded determinism at a tiny training budget.
"""

import os

import pytest

from icia.tasks.code_corpus import CHALLENGE_SOLUTIONS, AUX_FUNCTIONS
from icia.tasks.mbpp_constrained_benchmark import (
    CONSTRAINED_BENCHMARK_SUITE,
    check_ast_prohibitions,
)
from icia.experiments.run_governed_decoding_benchmark import (
    run_governed_decoding_benchmark,
    score_candidate,
    _run_tests,
)


@pytest.mark.parametrize("challenge", CONSTRAINED_BENCHMARK_SUITE, ids=[c.task_id for c in CONSTRAINED_BENCHMARK_SUITE])
def test_reference_solution_passes_own_barriers(challenge):
    code = CHALLENGE_SOLUTIONS[challenge.task_id]
    violations = check_ast_prohibitions(code, challenge.target_fn_name, challenge.prohibited_patterns)
    assert violations == [], f"{challenge.task_id} reference violates: {violations}"
    ok, err = _run_tests(code, challenge.target_fn_name, challenge.test_cases, timeout=3.0)
    assert ok, f"{challenge.task_id} reference fails tests: {err}"


def test_aux_corpus_is_valid_python():
    import ast
    for prompt, code in AUX_FUNCTIONS:
        ast.parse(code)


def test_pipeline_schema_and_determinism():
    os.environ["ICIA_SEED"] = "123"
    try:
        a = run_governed_decoding_benchmark(steps=8, num_samples=2, max_new_bytes=64, export=False)
        b = run_governed_decoding_benchmark(steps=8, num_samples=2, max_new_bytes=64, export=False)
    finally:
        del os.environ["ICIA_SEED"]

    for key in ("raw_syntax_pct", "raw_adherence_pct", "raw_pass_pct",
                "governed_adherence_pct", "governed_pass_pct",
                "mean_governance_attempts", "fallback_rate_pct"):
        assert key in a["aggregate"]

    # Same seed, same training, same sampling stream -> identical aggregates.
    assert a["aggregate"] == b["aggregate"]
    assert set(a["per_task"]) == {c.task_id for c in CONSTRAINED_BENCHMARK_SUITE}


def test_score_candidate_blocks_prohibited_import():
    challenge = next(c for c in CONSTRAINED_BENCHMARK_SUITE if c.task_id == "MBPP-C04")
    bad = "from collections import Counter\n\ndef top_k_words(words, k):\n    return []\n"
    sc = score_candidate(bad, challenge)
    assert sc["syntax_ok"] is True
    assert sc["adherent"] is False
    assert sc["tests_pass"] is False
    assert sc["error"] == "blocked_by_barrier"
