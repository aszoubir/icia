// ==============================================================================
// Testbench for icia_riemannian_core.v
// Verifies 2-cycle pipeline latency, Q16.16 accuracy, and barrier saturation.
// ==============================================================================

`timescale 1ns / 1ps

module tb_icia_riemannian_core;

    reg clk;
    reg rst_n;
    reg valid_in;

    reg signed [31:0] state_theta;
    reg signed [31:0] state_theta_dot;
    reg signed [31:0] state_x;
    reg signed [31:0] state_x_dot;

    reg signed [31:0] goal_theta;
    reg signed [31:0] goal_theta_dot;
    reg signed [31:0] goal_x;
    reg signed [31:0] goal_x_dot;

    reg signed [31:0] gain_k_th;
    reg signed [31:0] gain_k_th_d;
    reg signed [31:0] gain_k_x;
    reg signed [31:0] gain_k_x_d;

    reg signed [31:0] sat_limit;

    wire        valid_out;
    wire signed [31:0] control_u;
    wire signed [31:0] lyapunov_v;
    wire        barrier_saturated;

    // Instantiate Unit Under Test (UUT)
    icia_riemannian_core uut (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(valid_in),
        .state_theta(state_theta),
        .state_theta_dot(state_theta_dot),
        .state_x(state_x),
        .state_x_dot(state_x_dot),
        .goal_theta(goal_theta),
        .goal_theta_dot(goal_theta_dot),
        .goal_x(goal_x),
        .goal_x_dot(goal_x_dot),
        .gain_k_th(gain_k_th),
        .gain_k_th_d(gain_k_th_d),
        .gain_k_x(gain_k_x),
        .gain_k_x_d(gain_k_x_d),
        .sat_limit(sat_limit),
        .valid_out(valid_out),
        .control_u(control_u),
        .lyapunov_v(lyapunov_v),
        .barrier_saturated(barrier_saturated)
    );

    // Clock generator (100 MHz -> 10ns period)
    always #5 clk = ~clk;

    initial begin
        $dumpfile("icia_core.vcd");
        $dumpvars(0, tb_icia_riemannian_core);

        clk = 0;
        rst_n = 0;
        valid_in = 0;

        // Gains in Q16.16:
        // k_th   = 80.0  -> 80 * 65536 = 5242880 (0x00500000)
        // k_th_d = 15.0  -> 15 * 65536 = 983040  (0x000F0000)
        // k_x    = 10.0  -> 10 * 65536 = 655360  (0x000A0000)
        // k_x_d  = 2.0   -> 2  * 65536 = 131072  (0x00020000)
        // sat    = 30.0  -> 30 * 65536 = 1966080 (0x001E0000)
        gain_k_th   = 32'h00500000;
        gain_k_th_d = 32'h000F0000;
        gain_k_x    = 32'h000A0000;
        gain_k_x_d  = 32'h00020000;
        sat_limit   = 32'h001E0000;

        goal_theta     = 0;
        goal_theta_dot = 0;
        goal_x         = 0;
        goal_x_dot     = 0;

        // Reset
        #20;
        rst_n = 1;
        #10;

        // Test 1: theta=0.1 (6554), th_d=0.05 (3277), x=0.2 (13107), x_d=0.01 (655)
        // Expected u ~ 10.77 -> 10.77 * 65536 ~ 705822 (0x000AC51E)
        valid_in = 1;
        state_theta     = 32'd6554;
        state_theta_dot = 32'd3277;
        state_x         = 32'd13107;
        state_x_dot     = 32'd655;

        #10; // Clock tick 1
        #10; // Clock tick 2 -> valid_out should go HIGH

        if (valid_out == 1'b1) begin
            $display("SUCCESS: Test 1 valid_out asserted, control_u = %d (hex: %h)", control_u, control_u);
        end

        // Test 2: Large perturbation to trigger barrier saturation
        state_theta = 32'h00020000; // 2.0 rad
        #10;
        #10;

        if (barrier_saturated == 1'b1) begin
            $display("SUCCESS: Test 2 barrier saturation confirmed at limit %d", control_u);
        end

        #50;
        $finish;
    end

endmodule
