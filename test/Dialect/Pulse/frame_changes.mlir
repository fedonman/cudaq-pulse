// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// RUN: cudaq-pulse-opt --pulse-frame-changes --split-input-file %s | FileCheck %s

// A change is stamped where its tone's previous drive ends, with no duration;
// one no later drive takes is kept and marked unconsumed.

// CHECK-LABEL: func.func @stamps
func.func @stamps() {
  %q = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit
  %d, %t = pulse.get_drive_line %q : (!pulse.qudit) -> (!pulse.drive_line, !pulse.tone)
  %f = arith.constant 1.000000e+06 : f64
  %ph = arith.constant 2.500000e-01 : f64
  %duration = arith.constant 16 : i64
  %amplitude = arith.constant 1.000000e-01 : f64
  %wf = pulse.gaussian %duration, %amplitude, %amplitude : i64, f64, f64 -> !pulse.waveform
  // CHECK: pulse.shift_frequency
  // CHECK-SAME: duration_cycles = 0 : i64, start_cycles = 0 : i64
  %t1 = pulse.shift_frequency %t, %f : !pulse.tone, f64 -> !pulse.tone
  %d1, %t2 = pulse.drive %d, %wf, %t1 <{duration_cycles = 16 : i64, start_cycles = 0 : i64}> : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
  // CHECK: pulse.set_phase
  // CHECK-SAME: duration_cycles = 0 : i64, start_cycles = 16 : i64
  %t3 = pulse.set_phase %t2, %ph : !pulse.tone, f64 -> !pulse.tone
  %d2, %t4 = pulse.drive %d1, %wf, %t3 <{duration_cycles = 16 : i64, start_cycles = 24 : i64}> : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
  // CHECK: pulse.shift_phase
  // CHECK-SAME: <{duration_cycles = 0 : i64, start_cycles = 40 : i64}> {pulse.unconsumed}
  %t5 = pulse.shift_phase %t4, %ph : !pulse.tone, f64 -> !pulse.tone
  return
}

// -----

// Frames that persist leave the change nothing takes unmarked.

// CHECK-LABEL: module attributes {pulse.control_system = #pulse.control_system<clock_period_fs = 1000000, controllers = [#pulse.controller<persistent_frames>]>}
// CHECK-LABEL: func.func @persistent
module attributes {pulse.control_system = #pulse.control_system<clock_period_fs = 1000000, controllers = [#pulse.controller<persistent_frames>]>} {
  func.func @persistent() {
    %q = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit
    %d, %t = pulse.get_drive_line %q : (!pulse.qudit) -> (!pulse.drive_line, !pulse.tone)
    %ph = arith.constant 2.500000e-01 : f64
    %duration = arith.constant 16 : i64
    %amplitude = arith.constant 1.000000e-01 : f64
    %wf = pulse.gaussian %duration, %amplitude, %amplitude : i64, f64, f64 -> !pulse.waveform
    %d1, %t1 = pulse.drive %d, %wf, %t <{duration_cycles = 16 : i64, start_cycles = 8 : i64}> : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
    // CHECK: pulse.shift_phase
    // CHECK-SAME: duration_cycles = 0 : i64, start_cycles = 24 : i64
    // CHECK-NOT: pulse.unconsumed
    %t2 = pulse.shift_phase %t1, %ph : !pulse.tone, f64 -> !pulse.tone
    return
  }
}

// -----

// The change a loop carries out of its last iteration to nothing is listed
// on the loop.

// CHECK-LABEL: func.func @carried
func.func @carried() {
  %q = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit
  %d, %t = pulse.get_drive_line %q : (!pulse.qudit) -> (!pulse.drive_line, !pulse.tone)
  %ph = arith.constant 2.500000e-01 : f64
  %duration = arith.constant 16 : i64
  %amplitude = arith.constant 1.000000e-01 : f64
  %wf = pulse.gaussian %duration, %amplitude, %amplitude : i64, f64, f64 -> !pulse.waveform
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %r:2 = scf.for %i = %c0 to %c4 step %c1 iter_args(%l = %d, %tone = %t) -> (!pulse.drive_line, !pulse.tone) {
    %l1, %t1 = pulse.drive %l, %wf, %tone <{duration_cycles = 16 : i64, period_cycles = array<i64: 16>, start_cycles = 0 : i64}> : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
    // CHECK: pulse.shift_phase
    // CHECK-SAME: <{duration_cycles = 0 : i64, period_cycles = array<i64: 16>, start_cycles = 16 : i64}> {pulse.carried = 0 : i64}
    %t2 = pulse.shift_phase %t1, %ph : !pulse.tone, f64 -> !pulse.tone
    scf.yield %l1, %t2 : !pulse.drive_line, !pulse.tone
  }
  // CHECK: {pulse.carried_in = array<i64: 0>, pulse.exit_updates = array<i64: 0>}
  return
}

// -----

// A change made before a branch that only one arm plays with is listed on
// the other arm.

// CHECK-LABEL: func.func @one_arm
func.func @one_arm(%m: i1) {
  %q = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit
  %d, %t = pulse.get_drive_line %q : (!pulse.qudit) -> (!pulse.drive_line, !pulse.tone)
  %ph = arith.constant 2.500000e-01 : f64
  %duration = arith.constant 16 : i64
  %amplitude = arith.constant 1.000000e-01 : f64
  %wf = pulse.gaussian %duration, %amplitude, %amplitude : i64, f64, f64 -> !pulse.waveform
  // CHECK: pulse.shift_phase
  // CHECK-SAME: pulse.carried = 0 : i64
  %t1 = pulse.shift_phase %t, %ph : !pulse.tone, f64 -> !pulse.tone
  %r:2 = scf.if %m -> (!pulse.drive_line, !pulse.tone) {
    %l1, %t2 = pulse.drive %d, %wf, %t1 <{duration_cycles = 16 : i64, start_cycles = 0 : i64}> : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
    scf.yield %l1, %t2 : !pulse.drive_line, !pulse.tone
  } else {
    scf.yield %d, %t1 : !pulse.drive_line, !pulse.tone
  }
  // CHECK: } {pulse.else_updates = array<i64: 0>}
  return
}

// -----

// In a loop whose iterations do not play its tone, a change is at the start
// of each iteration, though the tone is free only later.

// CHECK-LABEL: func.func @idle
func.func @idle() {
  %q = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit
  %r = pulse.qudit_alloc {qudit = 1 : i64} : !pulse.qudit
  %d, %t = pulse.get_drive_line %q : (!pulse.qudit) -> (!pulse.drive_line, !pulse.tone)
  %e, %u = pulse.get_drive_line %r : (!pulse.qudit) -> (!pulse.drive_line, !pulse.tone)
  %ph = arith.constant 2.500000e-01 : f64
  %duration = arith.constant 16 : i64
  %amplitude = arith.constant 1.000000e-01 : f64
  %wf = pulse.gaussian %duration, %amplitude, %amplitude : i64, f64, f64 -> !pulse.waveform
  %d1, %t1 = pulse.drive %d, %wf, %t <{duration_cycles = 16 : i64, start_cycles = 48 : i64}> : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %res:3 = scf.for %i = %c0 to %c4 step %c1 iter_args(%tone = %t1, %l = %e, %v = %u) -> (!pulse.tone, !pulse.drive_line, !pulse.tone) {
    %l1, %v1 = pulse.drive %l, %wf, %v <{duration_cycles = 16 : i64, period_cycles = array<i64: 16>, start_cycles = 0 : i64}> : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
    // CHECK: pulse.shift_phase
    // CHECK-SAME: <{duration_cycles = 0 : i64, period_cycles = array<i64: 16>, start_cycles = 0 : i64}> {pulse.unconsumed}
    %t2 = pulse.shift_phase %tone, %ph : !pulse.tone, f64 -> !pulse.tone
    scf.yield %t2, %l1, %v1 : !pulse.tone, !pulse.drive_line, !pulse.tone
  }
  return
}

// -----

// A change made after a wait on its tone's line is at the wait's end: a
// frequency shifted before the wait and back after it runs shifted through
// the wait.

// CHECK-LABEL: func.func @across_wait
func.func @across_wait() {
  %q = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit
  %d, %t = pulse.get_drive_line %q : (!pulse.qudit) -> (!pulse.drive_line, !pulse.tone)
  %up = arith.constant 1.000000e+06 : f64
  %down = arith.constant -1.000000e+06 : f64
  %duration = arith.constant 4 : i64
  %amplitude = arith.constant 1.000000e-01 : f64
  %wf = pulse.gaussian %duration, %amplitude, %amplitude : i64, f64, f64 -> !pulse.waveform
  %c500 = arith.constant 500 : i64
  %idle = pulse.duration_from_int %c500 : (i64) -> !pulse.duration
  %d1, %t1 = pulse.drive %d, %wf, %t <{duration_cycles = 4 : i64, start_cycles = 0 : i64}> : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
  // CHECK: pulse.shift_frequency
  // CHECK-SAME: <{duration_cycles = 0 : i64, start_cycles = 4 : i64}>
  %t2 = pulse.shift_frequency %t1, %up : !pulse.tone, f64 -> !pulse.tone
  %d2 = pulse.wait %d1, %idle <{duration_cycles = 500 : i64, start_cycles = 4 : i64}> : (!pulse.drive_line, !pulse.duration) -> !pulse.drive_line
  // CHECK: pulse.shift_frequency
  // CHECK-SAME: <{duration_cycles = 0 : i64, start_cycles = 504 : i64}>
  %t3 = pulse.shift_frequency %t2, %down : !pulse.tone, f64 -> !pulse.tone
  %d3, %t4 = pulse.drive %d2, %wf, %t3 <{duration_cycles = 4 : i64, start_cycles = 504 : i64}> : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
  return
}

// -----

// After a sync, where the latest of its lines is free; after a loop of
// waits, at the end of the last; on a tone of its own, after the waits on
// the line it last played on.

// CHECK-LABEL: func.func @after_sync_and_loop
func.func @after_sync_and_loop() {
  %q = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit
  %d, %t = pulse.get_drive_line %q : (!pulse.qudit) -> (!pulse.drive_line, !pulse.tone)
  %r, %rt = pulse.get_readout_line %q : (!pulse.qudit) -> (!pulse.readout_line, !pulse.tone)
  %f = arith.constant 1.000000e+06 : f64
  %g = arith.constant 6.000000e+09 : f64
  %zero = arith.constant 0.000000e+00 : f64
  %duration = arith.constant 16 : i64
  %amplitude = arith.constant 1.000000e-01 : f64
  %wf = pulse.gaussian %duration, %amplitude, %amplitude : i64, f64, f64 -> !pulse.waveform
  %ro_duration = arith.constant 100 : i64
  %ro = pulse.gaussian %ro_duration, %amplitude, %amplitude : i64, f64, f64 -> !pulse.waveform
  %c10 = arith.constant 10 : i64
  %idle = pulse.duration_from_int %c10 : (i64) -> !pulse.duration
  %d1, %t1 = pulse.drive %d, %wf, %t <{duration_cycles = 16 : i64, start_cycles = 0 : i64}> : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
  %r1, %rt1, %m = pulse.readout %r, %ro, %rt, iq <{duration_cycles = 100 : i64, start_cycles = 0 : i64}> : !pulse.readout_line, !pulse.waveform, !pulse.tone -> !pulse.readout_line, !pulse.tone, !pulse.measurement
  %s:2 = pulse.sync %d1, %r1 : !pulse.drive_line, !pulse.readout_line -> !pulse.drive_line, !pulse.readout_line
  // CHECK: pulse.shift_frequency
  // CHECK-SAME: <{duration_cycles = 0 : i64, start_cycles = 100 : i64}>
  %t2 = pulse.shift_frequency %t1, %f : !pulse.tone, f64 -> !pulse.tone
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %l = scf.for %i = %c0 to %c4 step %c1 iter_args(%li = %s#0) -> (!pulse.drive_line) {
    %li1 = pulse.wait %li, %idle <{duration_cycles = 10 : i64, period_cycles = array<i64: 10>, start_cycles = 100 : i64}> : (!pulse.drive_line, !pulse.duration) -> !pulse.drive_line
    scf.yield %li1 : !pulse.drive_line
  }
  // CHECK: pulse.shift_frequency
  // CHECK-SAME: <{duration_cycles = 0 : i64, start_cycles = 140 : i64}>
  %t3 = pulse.shift_frequency %t2, %f : !pulse.tone, f64 -> !pulse.tone
  %d2, %t4 = pulse.drive %l, %wf, %t3 <{duration_cycles = 16 : i64, start_cycles = 140 : i64}> : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
  %u = pulse.tone %g, %zero : f64, f64 -> !pulse.tone
  %d3, %u1 = pulse.drive %d2, %wf, %u <{duration_cycles = 16 : i64, start_cycles = 156 : i64}> : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
  %d4 = pulse.wait %d3, %idle <{duration_cycles = 10 : i64, start_cycles = 172 : i64}> : (!pulse.drive_line, !pulse.duration) -> !pulse.drive_line
  // CHECK: pulse.shift_frequency
  // CHECK-SAME: <{duration_cycles = 0 : i64, start_cycles = 182 : i64}>
  %u2 = pulse.shift_frequency %u1, %f : !pulse.tone, f64 -> !pulse.tone
  %d5, %u3 = pulse.drive %d4, %wf, %u2 <{duration_cycles = 16 : i64, start_cycles = 182 : i64}> : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
  return
}

// -----

// A change in an arm is made once the measurement the branch tests is in,
// as if its line were synced with the readout's.

// CHECK-LABEL: func.func @in_an_arm
func.func @in_an_arm() {
  %q = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit
  %d, %t = pulse.get_drive_line %q : (!pulse.qudit) -> (!pulse.drive_line, !pulse.tone)
  %r, %rt = pulse.get_readout_line %q : (!pulse.qudit) -> (!pulse.readout_line, !pulse.tone)
  %f = arith.constant 2.000000e+06 : f64
  %duration = arith.constant 16 : i64
  %amplitude = arith.constant 1.000000e-01 : f64
  %wf = pulse.gaussian %duration, %amplitude, %amplitude : i64, f64, f64 -> !pulse.waveform
  %ro_duration = arith.constant 100 : i64
  %ro = pulse.gaussian %ro_duration, %amplitude, %amplitude : i64, f64, f64 -> !pulse.waveform
  %d1, %t1 = pulse.drive %d, %wf, %t <{duration_cycles = 16 : i64, start_cycles = 0 : i64}> : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
  %r1, %rt1, %m = pulse.readout %r, %ro, %rt, iq <{duration_cycles = 100 : i64, start_cycles = 16 : i64}> : !pulse.readout_line, !pulse.waveform, !pulse.tone -> !pulse.readout_line, !pulse.tone, !pulse.measurement
  %c = pulse.is_set %m : !pulse.measurement -> i1
  %t3 = scf.if %c -> (!pulse.tone) {
    // CHECK: pulse.shift_frequency
    // CHECK-SAME: <{duration_cycles = 0 : i64, start_cycles = 116 : i64}>
    %t2 = pulse.shift_frequency %t1, %f : !pulse.tone, f64 -> !pulse.tone
    scf.yield %t2 : !pulse.tone
  } else {
    scf.yield %t1 : !pulse.tone
  }
  %d2, %t4 = pulse.drive %d1, %wf, %t3 <{duration_cycles = 16 : i64, start_cycles = 116 : i64}> : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
  return
}

// -----

// So is a change whose value a measurement chooses.

// CHECK-LABEL: func.func @by_value
func.func @by_value() {
  %q = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit
  %d, %t = pulse.get_drive_line %q : (!pulse.qudit) -> (!pulse.drive_line, !pulse.tone)
  %r, %rt = pulse.get_readout_line %q : (!pulse.qudit) -> (!pulse.readout_line, !pulse.tone)
  %up = arith.constant 2.000000e+06 : f64
  %down = arith.constant -1.000000e+06 : f64
  %duration = arith.constant 16 : i64
  %amplitude = arith.constant 1.000000e-01 : f64
  %wf = pulse.gaussian %duration, %amplitude, %amplitude : i64, f64, f64 -> !pulse.waveform
  %ro_duration = arith.constant 100 : i64
  %ro = pulse.gaussian %ro_duration, %amplitude, %amplitude : i64, f64, f64 -> !pulse.waveform
  %d1, %t1 = pulse.drive %d, %wf, %t <{duration_cycles = 16 : i64, start_cycles = 0 : i64}> : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
  %r1, %rt1, %m = pulse.readout %r, %ro, %rt, iq <{duration_cycles = 100 : i64, start_cycles = 16 : i64}> : !pulse.readout_line, !pulse.waveform, !pulse.tone -> !pulse.readout_line, !pulse.tone, !pulse.measurement
  %c = pulse.is_set %m : !pulse.measurement -> i1
  %f = arith.select %c, %up, %down : f64
  // CHECK: pulse.shift_frequency
  // CHECK-SAME: <{duration_cycles = 0 : i64, start_cycles = 116 : i64}>
  %t2 = pulse.shift_frequency %t1, %f : !pulse.tone, f64 -> !pulse.tone
  %d2, %t3 = pulse.drive %d1, %wf, %t2 <{duration_cycles = 16 : i64, start_cycles = 116 : i64}> : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
  return
}

// -----

// A change that a drive on another line plays with before the wait on its
// tone's line ends stays where the tone is free.

// CHECK-LABEL: func.func @played_on_another_line
func.func @played_on_another_line() {
  %q = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit
  %p = pulse.qudit_alloc {qudit = 1 : i64} : !pulse.qudit
  %d, %t = pulse.get_drive_line %q : (!pulse.qudit) -> (!pulse.drive_line, !pulse.tone)
  %e, %u = pulse.get_drive_line %p : (!pulse.qudit) -> (!pulse.drive_line, !pulse.tone)
  %f = arith.constant 1.000000e+06 : f64
  %duration = arith.constant 16 : i64
  %amplitude = arith.constant 1.000000e-01 : f64
  %wf = pulse.gaussian %duration, %amplitude, %amplitude : i64, f64, f64 -> !pulse.waveform
  %c200 = arith.constant 200 : i64
  %idle = pulse.duration_from_int %c200 : (i64) -> !pulse.duration
  %d1, %t1 = pulse.drive %d, %wf, %t <{duration_cycles = 16 : i64, start_cycles = 0 : i64}> : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
  %d2 = pulse.wait %d1, %idle <{duration_cycles = 200 : i64, start_cycles = 16 : i64}> : (!pulse.drive_line, !pulse.duration) -> !pulse.drive_line
  // CHECK: pulse.shift_frequency
  // CHECK-SAME: <{duration_cycles = 0 : i64, start_cycles = 16 : i64}>
  %t2 = pulse.shift_frequency %t1, %f : !pulse.tone, f64 -> !pulse.tone
  %e1, %t3 = pulse.drive %e, %wf, %t2 <{duration_cycles = 16 : i64, start_cycles = 16 : i64}> : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
  return
}
