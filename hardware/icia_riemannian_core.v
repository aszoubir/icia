// ==============================================================================
// ICIA Hardware Core: Synthesizable Q16.16 Fixed-Point Riemannian Potential Engine
// Architecture: Intention-Centred Intelligence Architecture (ICIA)
// Standard: IEEE 1364-2001 Verilog
// Target: Bare-metal FPGA (Xilinx Artix-7/UltraScale+, Intel Cyclone/Stratix, Lattice)
// Characteristics: 
//   - Fully deterministic 2-stage pipelined execution (1 output per clock)
//   - 0 heap allocations, purely register-transfer level
//   - Fixed-point Q16.16 arithmetic (1 sign, 15 int, 16 fractional bits)
//   - Teleological gradient flow u = K_g * (S - G) with dynamic saturation barrier
// ==============================================================================

`timescale 1ns / 1ps

module icia_riemannian_core #(
    parameter DATA_WIDTH = 32,
    parameter FRAC_BITS  = 16
)(
    input  wire                   clk,
    input  wire                   rst_n,
    input  wire                   valid_in,

    // Continuous 4D State Vector S_t in Q16.16
    input  wire signed [DATA_WIDTH-1:0] state_theta,
    input  wire signed [DATA_WIDTH-1:0] state_theta_dot,
    input  wire signed [DATA_WIDTH-1:0] state_x,
    input  wire signed [DATA_WIDTH-1:0] state_x_dot,

    // Target Intention Vector G_t in Q16.16
    input  wire signed [DATA_WIDTH-1:0] goal_theta,
    input  wire signed [DATA_WIDTH-1:0] goal_theta_dot,
    input  wire signed [DATA_WIDTH-1:0] goal_x,
    input  wire signed [DATA_WIDTH-1:0] goal_x_dot,

    // Riemannian Metric Gains K_t in Q16.16
    input  wire signed [DATA_WIDTH-1:0] gain_k_th,
    input  wire signed [DATA_WIDTH-1:0] gain_k_th_d,
    input  wire signed [DATA_WIDTH-1:0] gain_k_x,
    input  wire signed [DATA_WIDTH-1:0] gain_k_x_d,

    // Actuator Saturation Barrier Limit (e.g., 30.0 in Q16.16)
    input  wire signed [DATA_WIDTH-1:0] sat_limit,

    // Outputs
    output reg                    valid_out,
    output reg  signed [DATA_WIDTH-1:0] control_u,
    output reg  signed [DATA_WIDTH-1:0] lyapunov_v,
    output reg                    barrier_saturated
);

    // =========================================================================
    // Pipeline Stage 1: Discrepancy & Partial Products
    // =========================================================================
    reg signed [DATA_WIDTH-1:0] diff_th_r;
    reg signed [DATA_WIDTH-1:0] diff_th_d_r;
    reg signed [DATA_WIDTH-1:0] diff_x_r;
    reg signed [DATA_WIDTH-1:0] diff_x_d_r;

    reg signed [2*DATA_WIDTH-1:0] prod_th_r;
    reg signed [2*DATA_WIDTH-1:0] prod_th_d_r;
    reg signed [2*DATA_WIDTH-1:0] prod_x_r;
    reg signed [2*DATA_WIDTH-1:0] prod_x_d_r;

    reg signed [DATA_WIDTH-1:0] sat_limit_r;
    reg                         valid_stage1;

    wire signed [DATA_WIDTH-1:0] diff_th   = state_theta     - goal_theta;
    wire signed [DATA_WIDTH-1:0] diff_th_d = state_theta_dot - goal_theta_dot;
    wire signed [DATA_WIDTH-1:0] diff_x    = state_x         - goal_x;
    wire signed [DATA_WIDTH-1:0] diff_x_d  = state_x_dot     - goal_x_dot;

    always @(posedge clk) begin
        if (!rst_n) begin
            valid_stage1  <= 1'b0;
            diff_th_r     <= 0;
            diff_th_d_r   <= 0;
            diff_x_r      <= 0;
            diff_x_d_r    <= 0;
            prod_th_r     <= 0;
            prod_th_d_r   <= 0;
            prod_x_r      <= 0;
            prod_x_d_r    <= 0;
            sat_limit_r   <= 0;
        end else begin
            valid_stage1  <= valid_in;
            diff_th_r     <= diff_th;
            diff_th_d_r   <= diff_th_d;
            diff_x_r      <= diff_x;
            diff_x_d_r    <= diff_x_d;
            sat_limit_r   <= sat_limit;

            // Q16.16 Multiplications (32-bit * 32-bit = 64-bit full precision)
            prod_th_r     <= diff_th   * gain_k_th;
            prod_th_d_r   <= diff_th_d * gain_k_th_d;
            prod_x_r      <= diff_x    * gain_k_x;
            prod_x_d_r    <= diff_x_d  * gain_k_x_d;
        end
    end

    // =========================================================================
    // Pipeline Stage 2: Accumulation, Barrier Saturation Clamping, & Lyapunov
    // =========================================================================
    // Shift right by FRAC_BITS (16) to normalize Q16.16 * Q16.16 -> Q16.16
    wire signed [DATA_WIDTH-1:0] term_th   = prod_th_r[DATA_WIDTH+FRAC_BITS-1 : FRAC_BITS];
    wire signed [DATA_WIDTH-1:0] term_th_d = prod_th_d_r[DATA_WIDTH+FRAC_BITS-1 : FRAC_BITS];
    wire signed [DATA_WIDTH-1:0] term_x    = prod_x_r[DATA_WIDTH+FRAC_BITS-1 : FRAC_BITS];
    wire signed [DATA_WIDTH-1:0] term_x_d  = prod_x_d_r[DATA_WIDTH+FRAC_BITS-1 : FRAC_BITS];

    wire signed [DATA_WIDTH-1:0] raw_force = term_th + term_th_d + term_x + term_x_d;

    // Approximate Lyapunov scalar: V = (diff_th^2 * K_th + diff_x^2 * K_x) >> 1
    // For single-cycle stage 2 telemetry, compute sum of absolute gradient contributions
    wire signed [DATA_WIDTH-1:0] abs_th = (term_th < 0) ? -term_th : term_th;
    wire signed [DATA_WIDTH-1:0] abs_x  = (term_x  < 0) ? -term_x  : term_x;
    wire signed [DATA_WIDTH-1:0] energy = abs_th + abs_x;

    always @(posedge clk) begin
        if (!rst_n) begin
            valid_out         <= 1'b0;
            control_u         <= 0;
            lyapunov_v        <= 0;
            barrier_saturated <= 1'b0;
        end else begin
            valid_out  <= valid_stage1;
            lyapunov_v <= energy;

            // Teleological Barrier Saturation Logic:
            if (raw_force > sat_limit_r) begin
                control_u         <= sat_limit_r;
                barrier_saturated <= 1'b1;
            end else if (raw_force < -sat_limit_r) begin
                control_u         <= -sat_limit_r;
                barrier_saturated <= 1'b1;
            end else begin
                control_u         <= raw_force;
                barrier_saturated <= 1'b0;
            end
        end
    end

endmodule
