"""
Cycle-Accurate Bit-Level Python Simulator for icia_riemannian_core.v.
Emulates the exact 32-bit signed Q16.16 hardware pipeline arithmetic.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple


def float_to_q16(val: float) -> int:
    """Converts a float to a 32-bit signed integer in Q16.16 format."""
    clamped = max(-32768.0, min(32767.999, val))
    raw = int(round(clamped * 65536.0))
    # Keep within 32-bit signed
    if raw >= 0x80000000:
        raw -= 0x100000000
    return raw


def q16_to_float(raw: int) -> float:
    """Converts a 32-bit signed Q16.16 integer back to float."""
    # Ensure signed 32-bit interpretation
    raw = (raw + 0x80000000) % 0x100000000 - 0x80000000
    return float(raw) / 65536.0


@dataclass
class HardwarePipelineRegisters:
    # Stage 1 registers
    valid_stage1: bool = False
    diff_th: int = 0
    diff_th_d: int = 0
    diff_x: int = 0
    diff_x_d: int = 0
    prod_th: int = 0
    prod_th_d: int = 0
    prod_x: int = 0
    prod_x_d: int = 0
    sat_limit: int = 0

    # Stage 2 registers (Outputs)
    valid_out: bool = False
    control_u: int = 0
    lyapunov_v: int = 0
    barrier_saturated: bool = False


class ICIARiemannianHardwareSimulator:
    """
    Simulates the exact RTL execution of hardware/icia_riemannian_core.v
    with 2-clock pipeline latency and 1-output-per-cycle throughput.
    """

    def __init__(self):
        self.regs = HardwarePipelineRegisters()

    def reset(self):
        self.regs = HardwarePipelineRegisters()

    def clock_step(
        self,
        valid_in: bool,
        theta: float,
        theta_dot: float,
        x: float,
        x_dot: float,
        goal_theta: float = 0.0,
        goal_theta_dot: float = 0.0,
        goal_x: float = 0.0,
        goal_x_dot: float = 0.0,
        k_th: float = 80.0,
        k_th_d: float = 15.0,
        k_x: float = 10.0,
        k_x_d: float = 2.0,
        sat_limit: float = 30.0,
    ) -> Tuple[bool, float, float, bool]:
        """
        Executes one rising clock edge of the pipeline.
        Returns: (valid_out, control_u_float, lyapunov_energy_float, barrier_saturated)
        """
        # Read prior stage registers for Stage 2 computation
        s1_valid = self.regs.valid_stage1
        p_th = self.regs.prod_th
        p_th_d = self.regs.prod_th_d
        p_x = self.regs.prod_x
        p_x_d = self.regs.prod_x_d
        sat_lim = self.regs.sat_limit

        # Stage 2 execution:
        # Arithmetic right shift by 16 bits to normalize Q16.16 * Q16.16
        term_th = (p_th >> 16)
        term_th_d = (p_th_d >> 16)
        term_x = (p_x >> 16)
        term_x_d = (p_x_d >> 16)

        raw_force = term_th + term_th_d + term_x + term_x_d
        abs_th = abs(term_th)
        abs_x = abs(term_x)
        energy = abs_th + abs_x

        if raw_force > sat_lim:
            out_u = sat_lim
            out_sat = True
        elif raw_force < -sat_lim:
            out_u = -sat_lim
            out_sat = True
        else:
            out_u = raw_force
            out_sat = False

        # Stage 1 execution (from inputs):
        q_th = float_to_q16(theta)
        q_th_d = float_to_q16(theta_dot)
        q_x = float_to_q16(x)
        q_x_d = float_to_q16(x_dot)

        q_g_th = float_to_q16(goal_theta)
        q_g_th_d = float_to_q16(goal_theta_dot)
        q_g_x = float_to_q16(goal_x)
        q_g_x_d = float_to_q16(goal_x_dot)

        q_k_th = float_to_q16(k_th)
        q_k_th_d = float_to_q16(k_th_d)
        q_k_x = float_to_q16(k_x)
        q_k_x_d = float_to_q16(k_x_d)
        q_sat = float_to_q16(sat_limit)

        d_th = q_th - q_g_th
        d_th_d = q_th_d - q_g_th_d
        d_x = q_x - q_g_x
        d_x_d = q_x_d - q_g_x_d

        # Update registers synchronously
        self.regs.valid_stage1 = valid_in
        self.regs.diff_th = d_th
        self.regs.diff_th_d = d_th_d
        self.regs.diff_x = d_x
        self.regs.diff_x_d = d_x_d
        self.regs.prod_th = d_th * q_k_th
        self.regs.prod_th_d = d_th_d * q_k_th_d
        self.regs.prod_x = d_x * q_k_x
        self.regs.prod_x_d = d_x_d * q_k_x_d
        self.regs.sat_limit = q_sat

        self.regs.valid_out = s1_valid
        self.regs.control_u = out_u
        self.regs.lyapunov_v = energy
        self.regs.barrier_saturated = out_sat

        return (
            self.regs.valid_out,
            q16_to_float(self.regs.control_u),
            q16_to_float(self.regs.lyapunov_v),
            self.regs.barrier_saturated,
        )
