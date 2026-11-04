"""
Unit Tests for ICIA Natural Language Programming Subsystem
"""

import pytest
from icia.code.nl_programmer import NLIntentionParser, NLProgrammerEngine


def test_nl_parser_contract_extraction():
    parser = NLIntentionParser()
    prompt = "Filter even numbers and square them, sorting in descending order without numpy."
    contract = parser.parse(prompt)

    assert contract.function_name == "process_even_squares"
    assert "numbers" in contract.args
    assert any("not numbers" in g for g in contract.input_guards)
    assert any("numpy" in p for p in contract.prohibitions)


def test_nl_programmer_even_squares():
    engine = NLProgrammerEngine()
    prompt = "Given a list of integers, filter odd numbers, square the remaining even numbers, sort descending."
    res = engine.compile_and_synthesize(prompt)

    assert res["syntax_valid"] is True
    assert res["tests_passed"] is True
    assert len(res["prohibition_violations"]) == 0


def test_nl_programmer_palindrome():
    engine = NLProgrammerEngine()
    prompt = "Write a function is_palindrome ignoring casing and non-alphanumeric characters."
    res = engine.compile_and_synthesize(prompt)

    assert res["syntax_valid"] is True
    assert res["tests_passed"] is True


def test_nl_programmer_moving_average():
    engine = NLProgrammerEngine()
    prompt = "Compute the moving average of window size k over a list of numbers."
    res = engine.compile_and_synthesize(prompt)

    assert res["syntax_valid"] is True
    assert res["tests_passed"] is True


def test_nl_programmer_prohibition_barrier():
    engine = NLProgrammerEngine()
    prompt = "Find frequent words without using collections or external libraries."
    res = engine.compile_and_synthesize(prompt)

    assert res["syntax_valid"] is True
    assert res["tests_passed"] is True
    assert "import collections" not in res["code"]
