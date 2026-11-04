"""
Test Suite for Phase 11: WebAssembly & Hardware RTL Acceleration.
Verifies WebAssembly binary compilation, V8 execution, and cycle-accurate Q16.16 RTL simulation.
"""

import os
import subprocess
import pytest
from icia.native.wasm_builder import build_icia_wasm_core
from icia.hardware.sim_hardware_core import (
    ICIARiemannianHardwareSimulator,
    float_to_q16,
    q16_to_float,
)


def test_wasm_binary_builder_and_exports(tmp_path):
    wasm_bytes = build_icia_wasm_core()
    assert len(wasm_bytes) > 200
    assert wasm_bytes[:4] == b"\x00asm"
    assert wasm_bytes[4:8] == b"\x01\x00\x00\x00"

    out_file = tmp_path / "test_core.wasm"
    out_file.write_bytes(wasm_bytes)
    assert out_file.exists()


def test_wasm_execution_in_node_v8(tmp_path):
    wasm_bytes = build_icia_wasm_core()
    wasm_path = tmp_path / "core.wasm"
    wasm_path.write_bytes(wasm_bytes)

    # JavaScript script to verify wasm functions
    js_code = f"""
    const fs = require('fs');
    const buf = fs.readFileSync({repr(str(wasm_path))});
    WebAssembly.instantiate(buf).then(res => {{
        const exp = res.instance.exports;
        // Test 1: compute_control
        const u = exp.compute_control(0.1, 0.05, 0.2, 0.01, 80.0, 15.0, 10.0, 2.0, 30.0);
        if (Math.abs(u - 10.77) > 0.01) process.exit(1);

        // Test 2: compute_control saturation
        const u_sat = exp.compute_control(1.0, 5.0, 0.0, 0.0, 80.0, 15.0, 0.0, 0.0, 30.0);
        if (Math.abs(u_sat - 30.0) > 1e-4) process.exit(2);

        // Test 3: compute_lyapunov
        const v = exp.compute_lyapunov(0.1, 0.05, 0.2, 0.01, 80.0, 15.0, 10.0, 2.0);
        if (Math.abs(v - 0.61885) > 0.001) process.exit(3);

        // Test 4: symplectic forces
        const fx = exp.symplectic_force_x(1.0, 0.0, 2.0, 0.5);
        const fy = exp.symplectic_force_y(1.0, 0.0, 2.0, 0.5);
        if (Math.abs(fx - 1.0) > 1e-3 || Math.abs(fy - 0.5) > 1e-3) process.exit(4);

        console.log("OK");
    }}).catch(err => {{
        console.error(err);
        process.exit(5);
    }});
    """
    js_path = tmp_path / "test.js"
    js_path.write_text(js_code)

    res = subprocess.run(["node", str(js_path)], capture_output=True, text=True)
    assert res.returncode == 0, f"Node failed: {res.stderr}"
    assert "OK" in res.stdout


def test_hardware_q16_conversion_precision():
    val = 10.77
    q16 = float_to_q16(val)
    reconstructed = q16_to_float(q16)
    assert abs(val - reconstructed) < 0.0001


def test_hardware_pipeline_cycle_accuracy():
    sim = ICIARiemannianHardwareSimulator()

    # Cycle 1: input valid, pipeline filling
    valid_out, u, v, sat = sim.clock_step(True, 0.1, 0.05, 0.2, 0.01)
    assert not valid_out
    assert u == 0.0

    # Cycle 2: pipeline drained to Stage 2, output valid
    valid_out, u, v, sat = sim.clock_step(True, 0.1, 0.05, 0.2, 0.01)
    assert valid_out
    assert abs(u - 10.77) < 0.01
    assert not sat

    # Cycle 3: Saturation barrier test
    # Force theta=2.0 -> raw force ~ 160.0 -> clamp to 30.0
    valid_out, u, v, sat = sim.clock_step(True, 2.0, 0.0, 0.0, 0.0)
    # The output in cycle 3 is from the previous input (0.1)
    # Cycle 4 will be the saturated output
    valid_out, u, v, sat = sim.clock_step(True, 2.0, 0.0, 0.0, 0.0)
    assert valid_out
    assert abs(u - 30.0) < 0.001
    assert sat is True
