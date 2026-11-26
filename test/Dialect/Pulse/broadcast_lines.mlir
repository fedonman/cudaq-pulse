// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// A broadcast drive line reaches several qudits with one tone. A program is
// traced with the line acquired from its lowest qudit, and the pass gives
// the acquisition the line's other qudits and its tone's frequency, from the
// lines the module lists, and drops the list. The line of a qudit on no
// listed line, and a readout line, stay as they are. What the pass writes
// parses and prints the same.
// RUN: cudaq-pulse-opt --split-input-file --pulse-broadcast-lines %s | FileCheck %s
// RUN: cudaq-pulse-opt --split-input-file --pulse-broadcast-lines %s | cudaq-pulse-opt --split-input-file | FileCheck %s

// CHECK-LABEL: module @listed
// CHECK-NOT: pulse.broadcast_lines
// CHECK: %[[Q0:.*]] = pulse.qudit_alloc {qudit = 0 : i64}
// CHECK: %[[Q2:.*]] = pulse.qudit_alloc {qudit = 2 : i64}
// CHECK: %[[Q1:.*]] = pulse.qudit_alloc {qudit = 1 : i64}
// CHECK: pulse.get_drive_line %[[Q0]], %[[Q1]], %[[Q2]] {frequency_hz = 5.100000e+09 : f64} : (!pulse.qudit, !pulse.qudit, !pulse.qudit) -> (!pulse.drive_line, !pulse.tone)
// CHECK: pulse.get_readout_line %[[Q0]] {frequency_hz = 7.000000e+09 : f64} : (!pulse.qudit) -> (!pulse.readout_line, !pulse.tone)
// CHECK: %[[Q5:.*]] = pulse.qudit_alloc {qudit = 5 : i64}
// CHECK: pulse.get_drive_line %[[Q5]] {frequency_hz = 5.500000e+09 : f64} : (!pulse.qudit) -> (!pulse.drive_line, !pulse.tone)
module @listed attributes {pulse.broadcast_lines = [{qudits = [0, 1, 2], frequency_hz = 5.1e9 : f64}, {qudits = [3, 4], frequency_hz = 5.3e9 : f64}]} {
  func.func @main() {
    %q0 = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit
    %q2 = pulse.qudit_alloc {qudit = 2 : i64} : !pulse.qudit
    %d0, %t0 = pulse.get_drive_line %q0 {frequency_hz = 5.000000e+09 : f64} : (!pulse.qudit) -> (!pulse.drive_line, !pulse.tone)
    %r0, %rt0 = pulse.get_readout_line %q0 {frequency_hz = 7.000000e+09 : f64} : (!pulse.qudit) -> (!pulse.readout_line, !pulse.tone)
    %q5 = pulse.qudit_alloc {qudit = 5 : i64} : !pulse.qudit
    %d5, %t5 = pulse.get_drive_line %q5 {frequency_hz = 5.500000e+09 : f64} : (!pulse.qudit) -> (!pulse.drive_line, !pulse.tone)
    %c40 = arith.constant 40 : i64
    %amp = arith.constant 0.05 : f64
    %zero = arith.constant 0.0 : f64
    %x = pulse.square %c40, %amp, %zero : i64, f64, f64 -> !pulse.waveform
    %d1, %t1 = pulse.drive %d0, %x, %t0 : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
    %d6, %t6 = pulse.drive %d5, %x, %t5 : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
    %r1, %rt1, %m = pulse.readout %r0, %x, %rt0, iq : !pulse.readout_line, !pulse.waveform, !pulse.tone -> !pulse.readout_line, !pulse.tone, !pulse.measurement
    return
  }
}

// -----

// A module that lists no broadcast line is left as it is.
// CHECK-LABEL: module @unlisted
// CHECK: pulse.get_drive_line %{{.*}} {frequency_hz = 5.000000e+09 : f64} : (!pulse.qudit) -> (!pulse.drive_line, !pulse.tone)
module @unlisted {
  func.func @main() {
    %q0 = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit
    %d0, %t0 = pulse.get_drive_line %q0 {frequency_hz = 5.000000e+09 : f64} : (!pulse.qudit) -> (!pulse.drive_line, !pulse.tone)
    %c40 = arith.constant 40 : i64
    %amp = arith.constant 0.05 : f64
    %zero = arith.constant 0.0 : f64
    %x = pulse.square %c40, %amp, %zero : i64, f64, f64 -> !pulse.waveform
    %d1, %t1 = pulse.drive %d0, %x, %t0 : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
    return
  }
}
