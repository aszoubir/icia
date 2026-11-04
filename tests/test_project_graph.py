"""
Unit Tests for Phase 15: Repository-Scale Multi-File AST Dependency Graph.
Verifies circular import detection and cross-file symbol resolution invariants.
"""

from icia.code.project_graph import ProjectDependencyGraph
from icia.tasks.multi_file_refactor_task import (
    BROKEN_LLM_MULTI_FILE_PROJECT,
    REPAIRED_ICIA_MULTI_FILE_PROJECT,
)


def test_broken_project_detects_cycles_and_unresolved_symbols():
    graph = ProjectDependencyGraph()
    for mod_name, code in BROKEN_LLM_MULTI_FILE_PROJECT.files.items():
        graph.add_module_from_source(mod_name, code)

    diag = graph.diagnose_project()
    assert not diag.is_valid
    assert len(diag.circular_dependencies) == 1
    cycle = diag.circular_dependencies[0]
    assert "auth_service" in cycle and "database" in cycle

    assert len(diag.unresolved_symbols) == 1
    unresolved = diag.unresolved_symbols[0]
    assert unresolved["caller_module"] == "api_gateway"
    assert unresolved["missing_symbol"] == "authenticate_user"


def test_repaired_project_satisfies_all_invariants():
    graph = ProjectDependencyGraph()
    for mod_name, code in REPAIRED_ICIA_MULTI_FILE_PROJECT.files.items():
        graph.add_module_from_source(mod_name, code)

    diag = graph.diagnose_project()
    assert diag.is_valid
    assert len(diag.circular_dependencies) == 0
    assert len(diag.unresolved_symbols) == 0
    assert diag.total_modules == 4
    assert diag.total_symbols > 5
