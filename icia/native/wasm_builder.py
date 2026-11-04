"""
Zero-dependency WebAssembly binary encoder and compiler for ICIA embedded execution.
Produces standalone, valid Wasm 1.0 modules executable in any browser V8/SpiderMonkey engine,
Node.js, or embedded Wasm runtime (Wasm3 / Wasmtime).
"""

from __future__ import annotations
import struct
from typing import List, Tuple


def encode_u32(val: int) -> bytes:
    res = bytearray()
    while True:
        b = val & 0x7F
        val >>= 7
        if val != 0:
            b |= 0x80
        res.append(b)
        if val == 0:
            break
    return bytes(res)


def encode_vector(items: List[bytes]) -> bytes:
    return encode_u32(len(items)) + b"".join(items)


def encode_section(sec_id: int, payload: bytes) -> bytes:
    return bytes([sec_id]) + encode_u32(len(payload)) + payload


def f32_const(val: float) -> bytes:
    return bytes([0x43]) + struct.pack("<f", val)


# Wasm basic types
TYPE_I32 = 0x7F
TYPE_I64 = 0x7E
TYPE_F32 = 0x7D
TYPE_F64 = 0x7C
TYPE_FUNC = 0x60
OP_F32_EQ = 0x5B
OP_F32_NE = 0x5C
OP_F32_LT = 0x5D
OP_F32_GT = 0x5E
OP_F32_LE = 0x5F
OP_F32_GE = 0x60


class WasmBuilder:
    """Helper to assemble WebAssembly 1.0 binary modules."""

    def __init__(self):
        self.types: List[bytes] = []
        self.type_map: dict = {}
        self.functions: List[int] = []
        self.exports: List[Tuple[str, int, int]] = []  # (name, kind, index)
        self.code_bodies: List[bytes] = []

    def add_type(self, params: List[int], results: List[int]) -> int:
        key = (tuple(params), tuple(results))
        if key in self.type_map:
            return self.type_map[key]

        sig = (
            bytes([TYPE_FUNC])
            + encode_vector([bytes([p]) for p in params])
            + encode_vector([bytes([r]) for r in results])
        )
        idx = len(self.types)
        self.types.append(sig)
        self.type_map[key] = idx
        return idx

    def add_function(self, type_idx: int, local_types: List[int], instructions: bytes, export_name: str | None = None) -> int:
        func_idx = len(self.functions)
        self.functions.append(type_idx)

        # Locals declaration
        # Group identical types
        local_entries = []
        if local_types:
            for l_type in local_types:
                local_entries.append(encode_u32(1) + bytes([l_type]))
        locals_vec = encode_vector(local_entries)

        body = locals_vec + instructions + bytes([0x0B])  # end opcode
        self.code_bodies.append(encode_u32(len(body)) + body)

        if export_name:
            self.exports.append((export_name, 0x00, func_idx))

        return func_idx

    def build(self) -> bytes:
        header = b"\x00asm\x01\x00\x00\x00"

        # Type section (1)
        type_sec = encode_section(1, encode_vector(self.types))

        # Function section (3)
        func_sec = encode_section(3, encode_vector([encode_u32(idx) for idx in self.functions]))

        # Export section (7)
        export_entries = []
        for name, kind, idx in self.exports:
            nb = name.encode("utf-8")
            export_entries.append(encode_u32(len(nb)) + nb + bytes([kind]) + encode_u32(idx))
        export_sec = encode_section(7, encode_vector(export_entries))

        # Code section (10)
        code_sec = encode_section(10, encode_vector(self.code_bodies))

        return header + type_sec + func_sec + export_sec + code_sec


def build_icia_wasm_core() -> bytes:
    """
    Constructs the canonical ICIA Embedded WebAssembly binary module.
    Exports:
    1. compute_control(theta, theta_dot, x, x_dot, k_th, k_th_d, k_x, k_x_d, sat): returns f32
    2. compute_lyapunov(theta, theta_dot, x, x_dot, k_th, k_th_d, k_x, k_x_d): returns f32
    3. symplectic_force_x(dx, dy, d_safe, gamma_curl): returns f32
    4. symplectic_force_y(dx, dy, d_safe, gamma_curl): returns f32
    """
    builder = WasmBuilder()

    # 1. compute_control: 9 params f32 -> returns f32
    # locals: local 9 (raw_u), local 10 (u_clamped)
    sig_control = builder.add_type([TYPE_F32] * 9, [TYPE_F32])
    # Instructions:
    # raw_u = (theta * k_th) + (theta_dot * k_th_d) + (x * k_x) + (x_dot * k_x_d)
    # Saturation:
    # if raw_u > sat -> sat
    # else if raw_u < -sat -> -sat
    # else -> raw_u
    code_ctrl = bytearray()
    # local.get 0 (theta), local.get 4 (k_th), f32.mul
    code_ctrl += bytes([0x20, 0x00, 0x20, 0x04, 0x94])
    # local.get 1 (theta_dot), local.get 5 (k_th_d), f32.mul, f32.add
    code_ctrl += bytes([0x20, 0x01, 0x20, 0x05, 0x94, 0x92])
    # local.get 2 (x), local.get 6 (k_x), f32.mul, f32.add
    code_ctrl += bytes([0x20, 0x02, 0x20, 0x06, 0x94, 0x92])
    # local.get 3 (x_dot), local.get 7 (k_x_d), f32.mul, f32.add
    code_ctrl += bytes([0x20, 0x03, 0x20, 0x07, 0x94, 0x92])
    # local.set 9 (raw_u)
    code_ctrl += bytes([0x21, 0x09])

    # Check raw_u > sat:
    code_ctrl += bytes([0x20, 0x09, 0x20, 0x08, 0x5E])  # local.get 9, local.get 8, f32.gt
    code_ctrl += bytes([0x04, TYPE_F32])  # if returning f32
    code_ctrl += bytes([0x20, 0x08])  # local.get 8 (sat)
    code_ctrl += bytes([0x05])  # else
    # Check raw_u < -sat:
    code_ctrl += bytes([0x20, 0x09, 0x43]) + struct.pack("<f", 0.0) + bytes([0x20, 0x08, 0x93, OP_F32_LT])  # raw_u, 0.0 - sat, f32.lt
    code_ctrl += bytes([0x04, TYPE_F32])
    code_ctrl += bytes([0x43]) + struct.pack("<f", 0.0) + bytes([0x20, 0x08, 0x93])  # 0.0 - sat
    code_ctrl += bytes([0x05])
    code_ctrl += bytes([0x20, 0x09])  # raw_u
    code_ctrl += bytes([0x0B])  # end inner if
    code_ctrl += bytes([0x0B])  # end outer if

    builder.add_function(sig_control, [TYPE_F32, TYPE_F32], bytes(code_ctrl), "compute_control")

    # 2. compute_lyapunov: 8 params f32 -> returns f32
    # V = 0.5 * (k_th*th^2 + k_th_d*th_d^2 + k_x*x^2 + k_x_d*x_d^2)
    sig_lyap = builder.add_type([TYPE_F32] * 8, [TYPE_F32])
    code_lyap = bytearray()
    # 0.5 const
    code_lyap += f32_const(0.5)
    # theta * theta * k_th
    code_lyap += bytes([0x20, 0x00, 0x20, 0x00, 0x94, 0x20, 0x04, 0x94])
    # theta_dot * theta_dot * k_th_d +
    code_lyap += bytes([0x20, 0x01, 0x20, 0x01, 0x94, 0x20, 0x05, 0x94, 0x92])
    # x * x * k_x +
    code_lyap += bytes([0x20, 0x02, 0x20, 0x02, 0x94, 0x20, 0x06, 0x94, 0x92])
    # x_dot * x_dot * k_x_d +
    code_lyap += bytes([0x20, 0x03, 0x20, 0x03, 0x94, 0x20, 0x07, 0x94, 0x92])
    # multiply by 0.5
    code_lyap += bytes([0x94])

    builder.add_function(sig_lyap, [], bytes(code_lyap), "compute_lyapunov")

    # 3. symplectic_force_x(dx, dy, d_safe, gamma_curl): 4 params f32 -> returns f32
    # dist = sqrt(dx*dx + dy*dy)
    # if dist < 1e-4 or dist >= d_safe: return 0.0
    # inv_d3 = 1.0 / (dist * dist * dist)
    # rep_x = dx * inv_d3
    # rep_y = dy * inv_d3
    # curl_x = -rep_y
    # return rep_x + gamma_curl * curl_x
    sig_sym = builder.add_type([TYPE_F32] * 4, [TYPE_F32])
    # locals: local 4 (dist), local 5 (inv_d3)
    code_sym_x = bytearray()
    # dist = sqrt(dx*dx + dy*dy)
    code_sym_x += bytes([0x20, 0x00, 0x20, 0x00, 0x94, 0x20, 0x01, 0x20, 0x01, 0x94, 0x92, 0x91])
    code_sym_x += bytes([0x21, 0x04])  # local.set 4 (dist)

    # if dist >= d_safe or dist < 1e-4: return 0.0
    code_sym_x += bytes([0x20, 0x04, 0x20, 0x02, OP_F32_LT])  # dist < d_safe
    code_sym_x += bytes([0x04, TYPE_F32])  # if inside safety radius
    # check dist > 1e-4
    code_sym_x += bytes([0x20, 0x04]) + f32_const(1e-4) + bytes([OP_F32_GT])  # dist > 1e-4
    code_sym_x += bytes([0x04, TYPE_F32])
    # compute inv_d3 = 1.0 / (dist * dist * dist)
    code_sym_x += f32_const(1.0) + bytes([0x20, 0x04, 0x20, 0x04, 0x94, 0x20, 0x04, 0x94, 0x95])
    code_sym_x += bytes([0x21, 0x05])  # local.set 5 (inv_d3)
    # rep_x = dx * inv_d3
    # curl_x = -(dy * inv_d3)
    # res = rep_x + gamma * curl_x = dx*inv_d3 - gamma*(dy*inv_d3)
    code_sym_x += bytes([0x20, 0x00, 0x20, 0x05, 0x94])  # dx * inv_d3
    code_sym_x += bytes([0x20, 0x03, 0x20, 0x01, 0x20, 0x05, 0x94, 0x94])  # gamma * (dy * inv_d3)
    code_sym_x += bytes([0x93])  # sub -> dx*inv_d3 - gamma*dy*inv_d3
    code_sym_x += bytes([0x05])  # else
    code_sym_x += f32_const(0.0)
    code_sym_x += bytes([0x0B])  # end inner if
    code_sym_x += bytes([0x05])  # else
    code_sym_x += f32_const(0.0)
    code_sym_x += bytes([0x0B])  # end outer if

    builder.add_function(sig_sym, [TYPE_F32, TYPE_F32], bytes(code_sym_x), "symplectic_force_x")

    # 4. symplectic_force_y(dx, dy, d_safe, gamma_curl): 4 params f32 -> returns f32
    # res = rep_y + gamma * curl_y = dy*inv_d3 + gamma*(dx*inv_d3)
    code_sym_y = bytearray()
    code_sym_y += bytes([0x20, 0x00, 0x20, 0x00, 0x94, 0x20, 0x01, 0x20, 0x01, 0x94, 0x92, 0x91])
    code_sym_y += bytes([0x21, 0x04])  # local.set 4 (dist)

    code_sym_y += bytes([0x20, 0x04, 0x20, 0x02, OP_F32_LT])  # dist < d_safe
    code_sym_y += bytes([0x04, TYPE_F32])
    code_sym_y += bytes([0x20, 0x04]) + f32_const(1e-4) + bytes([OP_F32_GT])
    code_sym_y += bytes([0x04, TYPE_F32])
    code_sym_y += f32_const(1.0) + bytes([0x20, 0x04, 0x20, 0x04, 0x94, 0x20, 0x04, 0x94, 0x95])
    code_sym_y += bytes([0x21, 0x05])
    # dy*inv_d3 + gamma*(dx*inv_d3)
    code_sym_y += bytes([0x20, 0x01, 0x20, 0x05, 0x94])  # dy * inv_d3
    code_sym_y += bytes([0x20, 0x03, 0x20, 0x00, 0x20, 0x05, 0x94, 0x94])  # gamma * (dx * inv_d3)
    code_sym_y += bytes([0x92])  # add -> dy*inv_d3 + gamma*dx*inv_d3
    code_sym_y += bytes([0x05])
    code_sym_y += f32_const(0.0)
    code_sym_y += bytes([0x0B])
    code_sym_y += bytes([0x05])
    code_sym_y += f32_const(0.0)
    code_sym_y += bytes([0x0B])

    builder.add_function(sig_sym, [TYPE_F32, TYPE_F32], bytes(code_sym_y), "symplectic_force_y")

    return builder.build()


if __name__ == "__main__":
    wasm = build_icia_wasm_core()
    with open("icia/web/icia_core.wasm", "wb") as f:
        f.write(wasm)
    print(f"Successfully generated ICIA WebAssembly core: {len(wasm)} bytes -> icia/web/icia_core.wasm")
