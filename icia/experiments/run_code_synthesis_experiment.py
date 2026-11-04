"""
Phase 9 Benchmark: Intention-Guided Code Synthesis vs Autoregressive Baseline
Measures syntax validity, scope resolution, unit test pass rate, and 1-step error pivots.
"""

import ast
import json
from typing import Dict, Any, List
from icia.code.ast_engine import IntentionGuidedCodeGenerator, CodeIntentionSpec
from icia.tasks.code_synthesis_task import (
    get_binary_search_task,
    get_matrix_transpose_task,
    get_parentheses_validator_task
)


def run_autoregressive_baseline(task: Dict[str, Any]) -> Dict[str, Any]:
    """
    Simulates unconstrained autoregressive sequence emission:
    Always samples the first statistical candidate without AST or scope verification.
    """
    spec: CodeIntentionSpec = task["spec"]
    lines = [
        f"def {spec.function_name}({', '.join(spec.args)}):",
        f'    """{spec.docstring}"""'
    ]

    for step in task["plan"]:
        # Unconstrained autoregressive sampling (greedily picks candidate 0)
        cand = step["candidates"][0]
        step_indent = step.get("indent", 1)
        base_indent = "    " * step_indent
        for subline in cand.split("\n"):
            stripped = subline.strip()
            if not stripped:
                continue
            cand_extra = len(subline) - len(subline.lstrip(" "))
            extra_levels = cand_extra // 4
            if stripped.startswith("elif ") or stripped.startswith("else:"):
                full_indent = base_indent
            else:
                full_indent = base_indent + ("    " * extra_levels)
            lines.append(full_indent + stripped)

    full_code = "\n".join(lines)
    
    # Check syntax validity
    is_valid = False
    syntax_err = None
    try:
        ast.parse(full_code)
        is_valid = True
    except SyntaxError as e:
        syntax_err = str(e)

    # Check execution and functional tests
    exec_success = False
    exec_err = None
    if is_valid:
        scope = {}
        try:
            exec(full_code, scope)
            fn = scope[task["spec"].function_name]
            if task["spec"].function_name == "binary_search":
                assert fn([1, 3, 5, 7, 9], 7) == 3
            elif task["spec"].function_name == "matrix_transpose":
                assert fn([[1, 2], [3, 4]]) == [[1, 3], [2, 4]]
            elif task["spec"].function_name == "is_valid_brackets":
                assert fn("({[]})") is True
            exec_success = True
        except Exception as e:
            exec_err = str(e)

    return {
        "code": full_code,
        "syntax_valid": is_valid,
        "syntax_error": syntax_err,
        "exec_success": exec_success,
        "exec_error": exec_err
    }


def run_icia_generator(task: Dict[str, Any]) -> Dict[str, Any]:
    """
    Runs ICIA intention-guided code generator with live AST & scope barrier projection.
    """
    generator = IntentionGuidedCodeGenerator(task["spec"])
    res = generator.generate_program(task["plan"])

    # Test runtime execution against sample unit tests
    code = res["code"]
    exec_success = False
    exec_err = None
    test_results = {}

    if res["is_valid"]:
        scope = {}
        try:
            exec(code, scope)
            fn = scope[task["spec"].function_name]

            # Run task-specific functional unit test
            if task["spec"].function_name == "binary_search":
                assert fn([1, 3, 5, 7, 9], 7) == 3
                assert fn([1, 3, 5, 7, 9], 2) == -1
                test_results["passed"] = True
            elif task["spec"].function_name == "matrix_transpose":
                assert fn([[1, 2], [3, 4]]) == [[1, 3], [2, 4]]
                test_results["passed"] = True
            elif task["spec"].function_name == "is_valid_brackets":
                assert fn("({[]})") is True
                assert fn("([)]") is False
                test_results["passed"] = True
            
            exec_success = True
        except Exception as e:
            exec_err = str(e)

    return {
        "code": code,
        "syntax_valid": res["is_valid"],
        "syntax_error": res["error"],
        "exec_success": exec_success,
        "exec_error": exec_err,
        "latent_pivots": res["latent_pivots"],
        "symbol_table_size": len(res["symbol_table"]),
        "has_returned": res["has_returned"]
    }


def run_benchmark():
    print("=" * 70)
    print("PHASE 9 BENCHMARK: INTENTION-GUIDED CODE SYNTHESIS VS AUTOREGRESSIVE BASELINE")
    print("=" * 70)

    tasks = [
        ("Binary Search", get_binary_search_task()),
        ("Matrix Transpose", get_matrix_transpose_task()),
        ("Delimiter Validator", get_parentheses_validator_task())
    ]

    ar_syntax_valid = 0
    ar_exec_valid = 0

    icia_syntax_valid = 0
    icia_exec_valid = 0
    total_pivots = 0

    results = []

    for name, task in tasks:
        ar_res = run_autoregressive_baseline(task)
        icia_res = run_icia_generator(task)

        if ar_res["syntax_valid"]: ar_syntax_valid += 1
        if ar_res["exec_success"]: ar_exec_valid += 1

        if icia_res["syntax_valid"]: icia_syntax_valid += 1
        if icia_res["exec_success"]: icia_exec_valid += 1
        total_pivots += icia_res["latent_pivots"]

        print(f"\nTask: {name}")
        print(f"  Autoregressive Baseline: Syntax Valid: {ar_res['syntax_valid']}, "
              f"Exec Success: {ar_res['exec_success']} | Error: {ar_res['syntax_error'] or ar_res['exec_error']}")
        print(f"  ICIA Intention Synthesis: Syntax Valid: {icia_res['syntax_valid']}, "
              f"Exec Success: {icia_res['exec_success']} | Latent Pivots: {icia_res['latent_pivots']}")

        results.append({
            "task": name,
            "autoregressive": ar_res,
            "icia": icia_res
        })

    n = len(tasks)
    summary = {
        "total_tasks": n,
        "autoregressive_baseline": {
            "syntax_validity_rate": float(ar_syntax_valid / n),
            "functional_execution_rate": float(ar_exec_valid / n)
        },
        "icia_code_synthesis": {
            "syntax_validity_rate": float(icia_syntax_valid / n),
            "functional_execution_rate": float(icia_exec_valid / n),
            "total_latent_pivots": total_pivots
        },
        "tasks": results
    }

    print("\n" + "=" * 70)
    print("OVERALL SUMMARY:")
    print(f"Autoregressive Syntax Validity:     {summary['autoregressive_baseline']['syntax_validity_rate']*100:.1f}%")
    print(f"Autoregressive Functional Success:  {summary['autoregressive_baseline']['functional_execution_rate']*100:.1f}%")
    print(f"ICIA Syntax Validity Rate:          {summary['icia_code_synthesis']['syntax_validity_rate']*100:.1f}%")
    print(f"ICIA Functional Success Rate:       {summary['icia_code_synthesis']['functional_execution_rate']*100:.1f}%")
    print(f"ICIA Total Latent Invariant Pivots: {summary['icia_code_synthesis']['total_latent_pivots']} (Bugs blocked at barrier)")
    print("=" * 70)

    with open("icia_code_synthesis_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary


if __name__ == "__main__":
    run_benchmark()
