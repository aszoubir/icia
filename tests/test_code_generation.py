"""
Unit Tests for ICIA Intention-Guided Code Synthesis
"""

import ast
import pytest
from icia.code.ast_engine import (
    CodeIntentionSpec,
    SyntaxConstraintTracker,
    IntentionGuidedCodeGenerator
)
from icia.tasks.code_synthesis_task import (
    get_binary_search_task,
    get_matrix_transpose_task,
    get_parentheses_validator_task
)
from icia.experiments.run_code_synthesis_experiment import (
    run_icia_generator,
    run_autoregressive_baseline
)


def test_syntax_tracker_unmatched_delimiter():
    spec = CodeIntentionSpec(
        function_name="dummy",
        args=["x"],
        return_type="int",
        docstring="dummy doc"
    )
    tracker = SyntaxConstraintTracker(spec)
    
    # Valid candidate
    adm, pen, reas = tracker.check_token_admissibility("val = (x + 1) * 2")
    assert adm is True
    assert pen == 0.0

    # Unclosed delimiter candidate
    adm, pen, reas = tracker.check_token_admissibility("val = (x + 1 * 2")
    assert adm is False
    assert pen > 0.0


def test_syntax_tracker_undefined_symbol():
    spec = CodeIntentionSpec(
        function_name="dummy",
        args=["x"],
        return_type="int",
        docstring="dummy doc"
    )
    tracker = SyntaxConstraintTracker(spec)

    # Reference undefined symbol 'ghost_var'
    adm, pen, reas = tracker.check_token_admissibility("y = ghost_var + 1")
    assert adm is False
    assert "Undefined symbol" in reas


def test_icia_code_generator_binary_search():
    task = get_binary_search_task()
    res = run_icia_generator(task)
    assert res["syntax_valid"] is True
    assert res["exec_success"] is True
    assert res["latent_pivots"] >= 1
    assert res["has_returned"] is True


def test_icia_code_generator_matrix_transpose():
    task = get_matrix_transpose_task()
    res = run_icia_generator(task)
    assert res["syntax_valid"] is True
    assert res["exec_success"] is True


def test_icia_code_generator_parentheses():
    task = get_parentheses_validator_task()
    res = run_icia_generator(task)
    assert res["syntax_valid"] is True
    assert res["exec_success"] is True
