"""
ICIA Code Generation Subsystem
Implements Intention-Guided Code Synthesis with AST Invariants,
Lexical Scope Resolution, and Barrier Constrained Emission.
"""

import ast
import re
from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Tuple, Any
import numpy as np


@dataclass
class CodeIntentionSpec:
    """Formal specification of code synthesis macro-intention."""
    function_name: str
    args: List[str]
    return_type: str
    docstring: str
    constraints: List[str] = field(default_factory=list)
    required_variables: List[str] = field(default_factory=list)
    target_complexity: str = "O(N)"


class SyntaxConstraintTracker:
    """
    Continuous-symbolic invariant tracker monitoring code structural boundaries:
    1. Parentheses/Bracket Stack c_brackets <= 0
    2. Indentation Depth c_indent
    3. Lexical Symbol Table c_scope
    """
    def __init__(self, spec: CodeIntentionSpec):
        self.spec = spec
        self.bracket_stack: List[str] = []
        self.indent_level: int = 0
        self.symbol_table: Set[str] = set(spec.args)  # Input arguments are initially in scope
        self.has_returned: bool = False
        self.emitted_lines: List[str] = []

    def check_token_admissibility(self, token: str) -> Tuple[bool, float, str]:
        """
        Evaluates barrier potential penalty for candidate token/expression.
        Returns: (is_admissible, barrier_penalty, reason)
        """
        # 1. Delimiter Balance Invariant (strip string literals first)
        token_no_strings = re.sub(r"(['\"])(?:(?=(\\?))\2.)*?\1", "''", token)
        matching = {')': '(', ']': '[', '}': '{'}
        local_brackets = list(self.bracket_stack)
        for ch in token_no_strings:
            if ch in '([{':
                local_brackets.append(ch)
            elif ch in ')]}':
                expected = matching[ch]
                if not local_brackets or local_brackets[-1] != expected:
                    return False, 1000.0, f"Mismatched delimiter: '{ch}' without matching '{expected}'"
                local_brackets.pop()

        # If a complete line or return expression has unclosed brackets
        if token.strip().startswith("return ") or ("=" in token and not token.strip().endswith(":")):
            if len(local_brackets) > len(self.bracket_stack):
                return False, 1000.0, "Unclosed delimiters in expression"

        # 2. Scope Resolution Invariant: Check variable references line-by-line
        local_symbols = set(self.symbol_table)
        keywords = {
            'def', 'return', 'for', 'in', 'if', 'else', 'elif', 'while', 'range', 'len',
            'None', 'True', 'False', 'append', 'pop', 'matrix', 'arr', 'target', 's', 'r', 'c',
            'not', 'and', 'or', 'is', 'pass', 'break', 'continue'
        }

        for line in token.split("\n"):
            line_str = line.strip()
            if not line_str:
                continue

            # Track loop induction variables (e.g. `for char in s:`)
            if line_str.startswith("for ") and " in " in line_str:
                loop_var = line_str[4:line_str.index(" in ")].strip()
                for v in re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', loop_var):
                    local_symbols.add(v)
                iterable = line_str[line_str.index(" in ") + 4:].rstrip(":")
                check_text = iterable
            elif "=" in line_str and not line_str.startswith("if") and not line_str.startswith("while"):
                lhs = line_str.split("=")[0]
                for var in re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', lhs):
                    local_symbols.add(var)
                rhs = line_str[line_str.index("=") + 1:]
                check_text = rhs
            else:
                check_text = line_str

            for ident in re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', check_text):
                if ident not in keywords and not ident.isdigit():
                    if ident not in local_symbols and ident != self.spec.function_name:
                        return False, 500.0, f"Undefined symbol referenced: '{ident}'"

        return True, 0.0, "Admissible"

    def update_state(self, line: str) -> None:
        """Updates symbol table, indentation, and bracket stack upon line emission."""
        stripped = line.strip()
        self.emitted_lines.append(line)

        # Track bracket stack (strip string literals)
        line_no_strings = re.sub(r"(['\"])(?:(?=(\\?))\2.)*?\1", "''", stripped)
        matching = {')': '(', ']': '[', '}': '{'}
        for ch in line_no_strings:
            if ch in '([{':
                self.bracket_stack.append(ch)
            elif ch in ')]}':
                if self.bracket_stack and self.bracket_stack[-1] == matching[ch]:
                    self.bracket_stack.pop()

        # Track loop induction variables
        if stripped.startswith("for ") and " in " in stripped:
            loop_var = stripped[4:stripped.index(" in ")].strip()
            for v in re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', loop_var):
                self.symbol_table.add(v)

        # Track assignments to add new symbols
        if "=" in stripped and not stripped.startswith("if") and not stripped.startswith("assert"):
            lhs = stripped.split("=")[0].strip()
            for var in re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', lhs):
                self.symbol_table.add(var)

        # Track return
        if stripped.startswith("return "):
            self.has_returned = True

    def is_syntax_valid(self) -> Tuple[bool, Optional[str]]:
        """Parses the cumulative code via Python's AST compiler."""
        code = "\n".join(self.emitted_lines)
        try:
            ast.parse(code)
            return True, None
        except SyntaxError as e:
            return False, str(e)


class IntentionGuidedCodeGenerator:
    """
    Teleological Code Synthesis Engine.
    Emits code lines guided by contract potential Psi_goal,
    projected through Syntax & Scope barrier constraints.
    """
    def __init__(self, spec: CodeIntentionSpec):
        self.spec = spec
        self.tracker = SyntaxConstraintTracker(spec)
        self.discrepancy_history: List[float] = []
        self.latent_pivot_count: int = 0

    def generate_program(self, skeleton_plan: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Executes intention-guided code generation along a teleological plan.
        Filters candidate expressions through the reality discrepancy gate.
        """
        # 1. Emit function header (Invariant Contract)
        header = f"def {self.spec.function_name}({', '.join(self.spec.args)}):"
        self.tracker.update_state(header)
        
        doc = f'    """{self.spec.docstring}"""'
        self.tracker.update_state(doc)

        for step in skeleton_plan:
            candidates = step.get("candidates", [])
            selected_line = None
            step_indent = step.get("indent", 1)

            for cand in candidates:
                # Reality Verification: check token admissibility
                admissible, penalty, reason = self.tracker.check_token_admissibility(cand)
                
                if admissible:
                    selected_line = cand
                    self.discrepancy_history.append(0.0)
                    break
                else:
                    # Inadmissible candidate encounters barrier!
                    self.latent_pivot_count += 1
                    self.discrepancy_history.append(penalty)

            # If candidates were blocked or fallback needed, use guaranteed teleological fallback
            if selected_line is None and "fallback" in step:
                selected_line = step["fallback"]
                self.discrepancy_history.append(0.0)

            if selected_line is not None:
                base_indent = "    " * step_indent
                for subline in selected_line.split("\n"):
                    stripped = subline.strip()
                    if not stripped:
                        continue
                    cand_extra = len(subline) - len(subline.lstrip(" "))
                    extra_levels = cand_extra // 4
                    if stripped.startswith("elif ") or stripped.startswith("else:"):
                        full_indent = base_indent
                    else:
                        full_indent = base_indent + ("    " * extra_levels)
                    self.tracker.update_state(full_indent + stripped)

        full_code = "\n".join(self.tracker.emitted_lines)
        valid, err = self.tracker.is_syntax_valid()

        return {
            "code": full_code,
            "is_valid": valid,
            "error": err,
            "lines_count": len(self.tracker.emitted_lines),
            "symbol_table": list(self.tracker.symbol_table),
            "bracket_balance": len(self.tracker.bracket_stack) == 0,
            "latent_pivots": self.latent_pivot_count,
            "has_returned": self.tracker.has_returned
        }
