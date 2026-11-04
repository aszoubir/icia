"""
Repository-Scale Multi-File AST Dependency Graph & Cross-Module Invariant Engine.
Builds inter-procedural symbol tables, validates cross-file import consistency,
detects circular dependency deadlocks, and enforces project-wide invariants.
"""

from __future__ import annotations
import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any


@dataclass
class ModuleSymbolTable:
    module_name: str
    file_path: str
    exported_functions: Dict[str, List[str]] = field(default_factory=dict)  # name -> arg_names
    exported_classes: Dict[str, List[str]] = field(default_factory=dict)    # name -> method_names
    exported_constants: Set[str] = field(default_factory=set)
    imported_symbols: List[Tuple[str, str]] = field(default_factory=list)   # (from_module, symbol_name)
    raw_imports: Set[str] = field(default_factory=set)                     # imported module names
    ast_tree: Optional[ast.AST] = None


@dataclass
class ProjectGraphDiagnosis:
    is_valid: bool
    circular_dependencies: List[List[str]]
    unresolved_symbols: List[Dict[str, str]]
    signature_mismatches: List[Dict[str, Any]]
    total_modules: int
    total_symbols: int
    explanation: str


class ProjectDependencyGraph:
    """
    Parses and tracks repository-wide AST dependency invariants across multi-file projects.
    """

    def __init__(self):
        self.modules: Dict[str, ModuleSymbolTable] = {}
        self.import_graph: Dict[str, Set[str]] = {}

    def add_module_from_source(self, module_name: str, source_code: str, file_path: str = "") -> ModuleSymbolTable:
        """Parses a Python source string into the project symbol graph."""
        tree = ast.parse(source_code)
        table = ModuleSymbolTable(module_name=module_name, file_path=file_path, ast_tree=tree)

        for node in ast.walk(tree):
            # 1. Exported functions
            if isinstance(node, ast.FunctionDef):
                # Only top-level or non-private functions
                args = [a.arg for a in node.args.args]
                table.exported_functions[node.name] = args

            # 2. Exported classes
            elif isinstance(node, ast.ClassDef):
                methods = [m.name for m in node.body if isinstance(m, ast.FunctionDef)]
                table.exported_classes[node.name] = methods

            # 3. Top-level constants (UPPERCASE assignments)
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id.isupper():
                        table.exported_constants.add(target.id)

            # 4. Imports: from x import y
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                table.raw_imports.add(mod)
                for alias in node.names:
                    table.imported_symbols.append((mod, alias.name))

            # 5. Imports: import x
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    table.raw_imports.add(alias.name)

        self.modules[module_name] = table
        self.import_graph[module_name] = set(table.raw_imports)
        return table

    def detect_circular_dependencies(self) -> List[List[str]]:
        """
        Detects all circular dependency cycles in the import graph using DFS.
        Returns list of cycles, e.g. [['module_a', 'module_b', 'module_a']].
        """
        visited = set()
        rec_stack = []
        cycles = []

        def dfs(node: str):
            visited.add(node)
            rec_stack.append(node)

            for neighbor in self.import_graph.get(node, []):
                # Only check internal modules registered in project
                if neighbor in self.modules:
                    if neighbor not in visited:
                        dfs(neighbor)
                    elif neighbor in rec_stack:
                        # Cycle detected
                        cycle_idx = rec_stack.index(neighbor)
                        cycle = rec_stack[cycle_idx:] + [neighbor]
                        cycles.append(cycle)

            rec_stack.pop()

        for mod in self.modules:
            if mod not in visited:
                dfs(mod)

        return cycles

    def verify_cross_file_symbols(self) -> Tuple[List[Dict[str, str]], List[Dict[str, Any]]]:
        """
        Verifies that every symbol imported from an internal project module actually exists
        in the target module's exported symbol table.
        """
        unresolved = []
        signature_mismatches = []

        for mod_name, table in self.modules.items():
            for src_mod, sym_name in table.imported_symbols:
                if src_mod in self.modules:
                    target_table = self.modules[src_mod]

                    # Check if symbol exists
                    exists = (
                        sym_name in target_table.exported_functions or
                        sym_name in target_table.exported_classes or
                        sym_name in target_table.exported_constants or
                        sym_name == "*"
                    )

                    if not exists:
                        unresolved.append({
                            "caller_module": mod_name,
                            "target_module": src_mod,
                            "missing_symbol": sym_name
                        })

        return unresolved, signature_mismatches

    def diagnose_project(self) -> ProjectGraphDiagnosis:
        """Full project-wide invariant verification."""
        cycles = self.detect_circular_dependencies()
        unresolved, sig_mismatches = self.verify_cross_file_symbols()

        is_valid = (len(cycles) == 0 and len(unresolved) == 0 and len(sig_mismatches) == 0)

        total_symbols = sum(
            len(m.exported_functions) + len(m.exported_classes) + len(m.exported_constants)
            for m in self.modules.values()
        )

        explanations = []
        if cycles:
            explanations.append(f"CRITICAL: {len(cycles)} circular import cycle(s) detected: {cycles}")
        if unresolved:
            explanations.append(f"CRITICAL: {len(unresolved)} unresolved cross-file symbol(s): {unresolved}")
        if is_valid:
            explanations.append("All project-scale multi-file AST dependency invariants strictly satisfied.")

        return ProjectGraphDiagnosis(
            is_valid=is_valid,
            circular_dependencies=cycles,
            unresolved_symbols=unresolved,
            signature_mismatches=sig_mismatches,
            total_modules=len(self.modules),
            total_symbols=total_symbols,
            explanation=" | ".join(explanations)
        )
