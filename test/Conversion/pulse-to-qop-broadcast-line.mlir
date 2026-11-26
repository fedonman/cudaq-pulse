// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// A drive on a broadcast line drives every qudit the line names, each with
// the envelope turning at the difference between the line's tone and the
// qudit's own frequency: the tone is at 5.1 GHz, qudit 0 at 5.1 GHz (on
// resonance) and qudit 1 1 MHz above it, 2 pi x 1e-3 rad/ns.
// RUN: cudaq-pulse-opt --pulse-to-qop %s | FileCheck %s

// CHECK: qop.callback_scalar @drive_envelope_0_x {{.*}}cudm.callback_params = array<f64: {{.*}}, 0.000000e+00>
// CHECK: qop.callback_scalar @drive_envelope_0_y {{.*}}cudm.callback_params = array<f64: {{.*}}, 0.000000e+00>
// CHECK: qop.callback_scalar @drive_envelope_1_x {{.*}}cudm.callback_params = array<f64: {{.*}}, 0.0062831853071795866>
// CHECK: qop.callback_scalar @drive_envelope_1_y {{.*}}cudm.callback_params = array<f64: {{.*}}, 0.0062831853071795866>
module attributes {pulse.control_system = #pulse.control_system<clock_period_fs = 1000000>, pulse.quantum_system = #pulse.quantum_system<qudit_frequency_hz = [5.100000e+09, 5.101000e+09], level_counts = [2, 2], t1_ns = [1.000000e+04, 1.000000e+04]>} {
  func.func @main() {
    %q0 = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit
    %q1 = pulse.qudit_alloc {qudit = 1 : i64} : !pulse.qudit
    %d0, %t0 = pulse.get_drive_line %q0, %q1 {frequency_hz = 5.100000e+09 : f64} : (!pulse.qudit, !pulse.qudit) -> (!pulse.drive_line, !pulse.tone)
    %c40 = arith.constant 40 : i64
    %amp = arith.constant 0.05 : f64
    %zero = arith.constant 0.0 : f64
    %x = pulse.square %c40, %amp, %zero : i64, f64, f64 -> !pulse.waveform
    %d1, %t1 = pulse.drive %d0, %x, %t0 <{duration_cycles = 40 : i64, start_cycles = 0 : i64}> : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
    return
  }
}
