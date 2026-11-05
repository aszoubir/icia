"""
Compile-and-run verification for the single-header C99 embedded engine.

Skips automatically when no C compiler is available on PATH.
"""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
HEADER = PROJECT_ROOT / "include" / "icia_core.h"

COMPILER = next(
    (c for c in ("gcc", "cc", "clang", "x86_64-w64-mingw32-gcc") if shutil.which(c)),
    None,
)

SMOKE_SOURCE = r"""
#define ICIA_CORE_IMPLEMENTATION
#include "{header}"
#include <stdio.h>

int main(void) {{
    icia_system_t sys;
    float goal[ICIA_OBS_DIM] = {{0.0f, 0.0f, 0.0f, 0.0f}};
    icia_init(&sys, goal);

    /* 1. Nominal equilibrium: zero state, zero goal must emit zero force. */
    float u_eq = icia_compute_control(&sys);

    /* 2. Teleological pull: tilted pole must produce non-zero corrective force. */
    sys.state[2] = 0.2f;
    sys.state[3] = 0.01f;
    float u_tilt = icia_compute_control(&sys);

    /* 3. Barrier push: at a sub-saturation tilt, a violated constraint must
          reduce the control signal relative to the same unconstrained state. */
    sys.state[2] = 0.3f;
    float u_hi = icia_compute_control(&sys);
    sys.num_constraints = 1;
    sys.constraint_weights[0][2] = 1.0f; /* w . S = theta */
    sys.constraint_bias[0] = 0.0f;
    float u_barrier = icia_compute_control(&sys);

    /* 4. Anti-symmetric multi-agent curl force. */
    float pa[2] = {{1.0f, 0.0f}}, pb[2] = {{0.0f, 0.0f}}, fa[2], fb[2];
    icia_symplectic_multiagent(pa, pb, fa, 0.6f, 0.8f);
    icia_symplectic_multiagent(pb, pa, fb, 0.6f, 0.8f);

    printf("%f %f %f %f %f %f %f %f\n",
           (double)u_eq, (double)u_tilt, (double)u_hi, (double)u_barrier,
           (double)fa[0], (double)fa[1], (double)fb[0], (double)fb[1]);
    return 0;
}}
"""


@pytest.mark.skipif(COMPILER is None, reason="No C compiler on PATH")
def test_c99_header_compiles_and_runs(tmp_path):
    assert HEADER.exists()
    src = tmp_path / "smoke.c"
    src.write_text(SMOKE_SOURCE.format(header=HEADER.as_posix()), encoding="utf-8")

    exe = tmp_path / ("smoke.exe" if sys.platform == "win32" else "smoke")
    compiled = subprocess.run(
        [COMPILER, "-std=c99", "-O2", str(src), "-o", str(exe)],
        capture_output=True, text=True,
    )
    assert compiled.returncode == 0, f"Compilation failed:\n{compiled.stderr}"

    run = subprocess.run([str(exe)], capture_output=True, text=True, timeout=30)
    assert run.returncode == 0, f"Binary crashed:\n{run.stderr}"
    u_eq, u_tilt, u_hi, u_barrier, fa0, fa1, fb0, fb1 = map(float, run.stdout.split())

    assert abs(u_eq) < 1e-5
    assert abs(u_tilt) > 1.0
    assert u_barrier < u_hi, "violated barrier must push control away from constraint"
    assert abs(fa0 + fb0) < 1e-4 and abs(fa1 + fb1) < 1e-4, "pairwise forces must be anti-symmetric"


@pytest.mark.skipif(COMPILER is None, reason="No C compiler on PATH")
def test_c99_header_compiles_cleanly_with_warnings(tmp_path):
    src = tmp_path / "warn.c"
    src.write_text(
        f'#define ICIA_CORE_IMPLEMENTATION\n#include "{HEADER.as_posix()}"\nint main(void) {{ return 0; }}\n',
        encoding="utf-8",
    )
    res = subprocess.run(
        [COMPILER, "-std=c99", "-Wall", "-Wextra", "-c", str(src), "-o", str(tmp_path / "warn.o")],
        capture_output=True, text=True,
    )
    assert res.returncode == 0, f"Compile errors:\n{res.stderr}"
    assert "error" not in res.stderr.lower()
