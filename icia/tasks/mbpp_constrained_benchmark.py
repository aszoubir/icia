"""
Constrained Algorithmic Coding Benchmark (HumanEval / MBPP Constrained Subset).
Evaluates code generation under strict negative constraints (prohibitions):
no recursion, no nested loops, no prohibited standard library imports, no forbidden operations.
"""

from __future__ import annotations
import ast
from dataclasses import dataclass, field
from typing import List, Dict, Any, Callable


@dataclass
class ConstrainedChallenge:
    task_id: str
    name: str
    target_fn_name: str
    prompt: str
    prohibited_patterns: List[str]
    test_cases: List[Dict[str, Any]]
    llm_baseline_code: str  # Typical autoregressive LLM output showing negation blindness


def check_ast_prohibitions(code: str, target_fn_name: str, prohibited_patterns: List[str]) -> List[str]:
    """
    AST-level barrier enforcement to detect negative constraint violations.
    Immune to false positives in comments and docstrings.
    """
    violations = []
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return ["syntax_error"]

    for node in ast.walk(tree):
        # 1. Prohibited imports
        if isinstance(node, ast.Import):
            for alias in node.names:
                for p in prohibited_patterns:
                    if p.lower() in alias.name.lower() or p == "import":
                        violations.append(f"import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            for p in prohibited_patterns:
                if (p.lower() in mod.lower() or p in ["import", "from"] or
                        any(p.lower() == a.name.lower() for a in node.names)):
                    violations.append(f"from {mod} import ...")

        # 2. Prohibited recursion
        if any(p in ["recursion", "fibonacci("] for p in prohibited_patterns):
            if isinstance(node, ast.FunctionDef) and node.name == target_fn_name:
                for child in ast.walk(node):
                    if child is not node and isinstance(child, ast.Call) and isinstance(child.func, ast.Name):
                        if child.func.id == target_fn_name:
                            violations.append("recursive_call")

        # 3. Prohibited slice reversal [::-1]
        if any(p in ["::-1", "slice"] for p in prohibited_patterns):
            if isinstance(node, ast.Slice) and node.step is not None:
                if isinstance(node.step, ast.UnaryOp) and isinstance(node.step.op, ast.USub):
                    violations.append("slice_reversal")

        # 4. Prohibited reversed() call
        if any("reversed" in p for p in prohibited_patterns):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id == "reversed":
                    violations.append("reversed()")

        # 5. Prohibited nested loops
        if any(p in ["nested_loops", "for j in"] for p in prohibited_patterns):
            if isinstance(node, (ast.For, ast.While)):
                for child in ast.walk(node):
                    if child is not node and isinstance(child, (ast.For, ast.While)):
                        violations.append("nested_loops")
                        break

        # 6. Prohibited Counter usage
        if any("Counter" in p for p in prohibited_patterns):
            if isinstance(node, ast.Name) and node.id == "Counter":
                violations.append("Counter")

    return list(set(violations))


CONSTRAINED_BENCHMARK_SUITE: List[ConstrainedChallenge] = [
    ConstrainedChallenge(
        task_id="MBPP-C01",
        name="iterative_fibonacci",
        target_fn_name="fibonacci",
        prompt="Write a function `fibonacci(n)` that returns the n-th Fibonacci number. STRICT CONSTRAINT: Do NOT use recursion. Do NOT use external libraries.",
        prohibited_patterns=["recursion", "fibonacci(", "import"],
        test_cases=[
            {"args": [0], "expected": 0},
            {"args": [1], "expected": 1},
            {"args": [7], "expected": 13},
            {"args": [10], "expected": 55},
        ],
        llm_baseline_code="""def fibonacci(n):
    # LLM suffers from negation blindness and generates recursive solution
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    return fibonacci(n - 1) + fibonacci(n - 2)
""",
    ),
    ConstrainedChallenge(
        task_id="MBPP-C02",
        name="reverse_inplace",
        target_fn_name="reverse_inplace",
        prompt="Write a function `reverse_inplace(arr)` that reverses a list in-place. STRICT CONSTRAINT: Do NOT use slice notation `[::-1]`. Do NOT use `reversed()`. Do NOT allocate a new list.",
        prohibited_patterns=["::-1", "reversed", "import"],
        test_cases=[
            {"args": [[1, 2, 3, 4]], "expected": [4, 3, 2, 1]},
            {"args": [[]], "expected": []},
            {"args": [["a", "b"]], "expected": ["b", "a"]},
        ],
        llm_baseline_code="""def reverse_inplace(arr):
    # LLM ignores prohibition against slice notation
    return arr[::-1]
""",
    ),
    ConstrainedChallenge(
        task_id="MBPP-C03",
        name="two_sum_linear",
        target_fn_name="two_sum",
        prompt="Write a function `two_sum(nums, target)` that returns 0-based indices of two numbers adding to target in linear time. STRICT CONSTRAINT: Do NOT use nested loops. Do NOT use external libraries.",
        prohibited_patterns=["nested_loops", "for j in", "import"],
        test_cases=[
            {"args": [[2, 7, 11, 15], 9], "expected": [0, 1]},
            {"args": [[3, 2, 4], 6], "expected": [1, 2]},
            {"args": [[3, 3], 6], "expected": [0, 1]},
        ],
        llm_baseline_code="""def two_sum(nums, target):
    # LLM falls back to quadratic brute-force nested loop
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            if nums[i] + nums[j] == target:
                return [i, j]
    return []
""",
    ),
    ConstrainedChallenge(
        task_id="MBPP-C04",
        name="top_k_frequent",
        target_fn_name="top_k_words",
        prompt="Write a function `top_k_words(words, k)` returning the k most frequent words. STRICT CONSTRAINT: Do NOT import collections. Do NOT import heapq. Do NOT use `Counter`.",
        prohibited_patterns=["collections", "Counter", "heapq", "import"],
        test_cases=[
            {"args": [["apple", "banana", "apple", "cherry", "banana", "apple"], 2], "expected": ["apple", "banana"]},
            {"args": [["dog", "cat", "dog"], 1], "expected": ["dog"]},
        ],
        llm_baseline_code="""from collections import Counter

def top_k_words(words, k):
    # LLM suffers from negation blindness and imports Counter anyway
    counts = Counter(words)
    return [word for word, _ in counts.most_common(k)]
""",
    ),
    ConstrainedChallenge(
        task_id="MBPP-C05",
        name="bracket_validator",
        target_fn_name="check_brackets",
        prompt="Write a function `check_brackets(code)` that returns True if all parentheses and brackets match. STRICT CONSTRAINT: Do NOT use regular expressions (`re`). Do NOT use external libraries.",
        prohibited_patterns=["re", "import"],
        test_cases=[
            {"args": ["([]){}"], "expected": True},
            {"args": ["([)]"], "expected": False},
            {"args": [""], "expected": True},
            {"args": ["("], "expected": False},
        ],
        llm_baseline_code="""import re

def check_brackets(code):
    # LLM imports re despite explicit prohibition
    pattern = re.compile(r'\(\)|\[\]|\{\}')
    while pattern.search(code):
        code = pattern.sub('', code)
    return len(code) == 0
""",
    ),
]
