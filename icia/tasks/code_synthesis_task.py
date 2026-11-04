"""
ICIA Code Synthesis Task Suite
Defines programming challenge benchmarks with distractor/error-injected candidate steps.
"""

from typing import List, Dict, Any
from icia.code.ast_engine import CodeIntentionSpec


def get_binary_search_task() -> Dict[str, Any]:
    spec = CodeIntentionSpec(
        function_name="binary_search",
        args=["arr", "target"],
        return_type="int",
        docstring="Performs binary search on a sorted array returning index or -1.",
        target_complexity="O(log N)"
    )

    plan = [
        {
            "description": "Initialize pointers",
            "candidates": [
                "left, right = 0, len(arr) - 1"
            ],
            "indent": 1
        },
        {
            "description": "Loop condition",
            "candidates": [
                # Inadmissible: references undefined 'invalid_ptr'
                "while left <= invalid_ptr:",
                "while left <= right:"
            ],
            "indent": 1
        },
        {
            "description": "Compute midpoint",
            "candidates": [
                # Inadmissible: mismatched delimiter
                "mid = (left + right // 2",
                "mid = (left + right) // 2"
            ],
            "indent": 2
        },
        {
            "description": "Target match check",
            "candidates": [
                "if arr[mid] == target:\n    return mid"
            ],
            "indent": 2
        },
        {
            "description": "Branching updates",
            "candidates": [
                "elif arr[mid] < target:\n    left = mid + 1\nelse:\n    right = mid - 1"
            ],
            "indent": 2
        },
        {
            "description": "Return not found",
            "candidates": [
                "return -1"
            ],
            "indent": 1
        }
    ]

    return {"spec": spec, "plan": plan}


def get_matrix_transpose_task() -> Dict[str, Any]:
    spec = CodeIntentionSpec(
        function_name="matrix_transpose",
        args=["matrix"],
        return_type="List[List[int]]",
        docstring="Transposes a 2D matrix returning rows as columns.",
        target_complexity="O(R * C)"
    )

    plan = [
        {
            "description": "Check empty matrix",
            "candidates": [
                "if not matrix or not matrix[0]:\n    return []"
            ],
            "indent": 1
        },
        {
            "description": "Get dimensions",
            "candidates": [
                # Inadmissible: references undefined 'mat'
                "rows, cols = len(mat), len(mat[0])",
                "rows, cols = len(matrix), len(matrix[0])"
            ],
            "indent": 1
        },
        {
            "description": "List comprehension transpose",
            "candidates": [
                # Inadmissible: unclosed bracket
                "return [[matrix[r][c] for r in range(rows)] for c in range(cols)",
                "return [[matrix[r][c] for r in range(rows)] for c in range(cols)]"
            ],
            "indent": 1
        }
    ]

    return {"spec": spec, "plan": plan}


def get_parentheses_validator_task() -> Dict[str, Any]:
    spec = CodeIntentionSpec(
        function_name="is_valid_brackets",
        args=["s"],
        return_type="bool",
        docstring="Determines if input string delimiters are validly nested.",
        target_complexity="O(N)"
    )

    plan = [
        {
            "description": "Init stack and mapping",
            "candidates": [
                "stack = []\nmapping = {')': '(', '}': '{', ']': '['}"
            ],
            "indent": 1
        },
        {
            "description": "Iterate characters",
            "candidates": [
                # Inadmissible: references undefined 'string_input'
                "for char in string_input:",
                "for char in s:"
            ],
            "indent": 1
        },
        {
            "description": "Delim logic",
            "candidates": [
                "if char in mapping:\n    top = stack.pop() if stack else '#'\n    if mapping[char] != top:\n        return False\nelse:\n    stack.append(char)"
            ],
            "indent": 2
        },
        {
            "description": "Final check",
            "candidates": [
                "return not stack"
            ],
            "indent": 1
        }
    ]

    return {"spec": spec, "plan": plan}
