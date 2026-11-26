// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// RUN: cudaq-pulse-opt %s --split-input-file --pulse-verify | FileCheck %s --check-prefix=KEEP
// RUN: cudaq-pulse-opt %s --split-input-file --pulse-verify --pulse-expand-calls --pulse-verify | FileCheck %s

// Calls placed by hand, as a compiler that schedules its own calls writes
// them: callees on lines, calls at stated starts, a call in an arm of a
// branch on a measurement a call returns, and calls in a loop with a
// period. pulse-verify accepts them as they are, and writing them out gives
// the flat program, which verifies too.

!d = !pulse.drive_line
!r = !pulse.readout_line
!t = !pulse.tone
!w = !pulse.waveform
!m = !pulse.measurement

func.func private @x90(%d: !d, %t: !t) -> (!d, !t)
    attributes {pulse.length_cycles = 40 : i64, pulse.schedule_state = #pulse.schedule_state<timed>} {
  %c40 = arith.constant 40 : i64
  %a = arith.constant 0.1 : f64
  %z = arith.constant 0.0 : f64
  %x = pulse.square %c40, %a, %z : i64, f64, f64 -> !w
  %d1, %t1 = pulse.drive %d, %x, %t <{duration_cycles = 40 : i64, start_cycles = 0 : i64}> : !d, !w, !t -> !d, !t
  return %d1, %t1 : !d, !t
}

func.func private @measure(%r: !r, %t: !t) -> (!r, !t, !m)
    attributes {pulse.length_cycles = 400 : i64, pulse.schedule_state = #pulse.schedule_state<timed>} {
  %c400 = arith.constant 400 : i64
  %a = arith.constant 0.05 : f64
  %z = arith.constant 0.0 : f64
  %x = pulse.square %c400, %a, %z : i64, f64, f64 -> !w
  %r1, %t1, %m = pulse.readout %r, %x, %t, iq <{duration_cycles = 400 : i64, start_cycles = 0 : i64}> : !r, !w, !t -> !r, !t, !m
  return %r1, %t1, %m : !r, !t, !m
}

// KEEP-LABEL: func.func @shot
// KEEP-COUNT-4: pulse.call

// CHECK-NOT: func.func private @x90
// CHECK-NOT: func.func private @measure
// CHECK-LABEL: func.func @shot
// CHECK-NOT: pulse.call
// CHECK: pulse.drive {{.*}}<{duration_cycles = 40 : i64, start_cycles = 0 : i64}>
// CHECK: pulse.readout {{.*}}<{duration_cycles = 400 : i64, start_cycles = 40 : i64}>
// CHECK: scf.if
// CHECK: pulse.drive {{.*}}<{duration_cycles = 40 : i64, start_cycles = 440 : i64}>
// CHECK: scf.for
// CHECK: pulse.drive {{.*}}<{duration_cycles = 40 : i64, period_cycles = array<i64: 50>, start_cycles = 480 : i64}>
func.func @shot() attributes {pulse.schedule_state = #pulse.schedule_state<timed>} {
  %q = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit
  %d0, %t0 = pulse.get_drive_line %q : (!pulse.qudit) -> (!pulse.drive_line, !pulse.tone)
  %r0, %u0 = pulse.get_readout_line %q : (!pulse.qudit) -> (!pulse.readout_line, !pulse.tone)
  %a:2 = pulse.call @x90(%d0, %t0) <{duration_cycles = 40 : i64, start_cycles = 0 : i64}> : (!d, !t) -> (!d, !t)
  %b:3 = pulse.call @measure(%r0, %u0) <{duration_cycles = 400 : i64, start_cycles = 40 : i64}> : (!r, !t) -> (!r, !t, !m)
  %bit = pulse.is_set %b#2 : !m -> i1
  %c:2 = scf.if %bit -> (!d, !t) {
    %e:2 = pulse.call @x90(%a#0, %a#1) <{duration_cycles = 40 : i64, start_cycles = 440 : i64}> : (!d, !t) -> (!d, !t)
    scf.yield %e#0, %e#1 : !d, !t
  } else {
    scf.yield %a#0, %a#1 : !d, !t
  }
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %f:2 = scf.for %k = %c0 to %c4 step %c1 iter_args(%d = %c#0, %t = %c#1) -> (!d, !t) {
    %g:2 = pulse.call @x90(%d, %t) <{duration_cycles = 40 : i64, period_cycles = array<i64: 50>, start_cycles = 480 : i64}> : (!d, !t) -> (!d, !t)
    scf.yield %g#0, %g#1 : !d, !t
  }
  return
}

// -----

!d = !pulse.drive_line
!t = !pulse.tone
!w = !pulse.waveform

// A call in a callee, and a frame change in a callee, are written out at
// the outer call's start; a line the callee does not play on keeps its own
// period in the loop.
func.func private @inner(%d: !d, %t: !t) -> (!d, !t)
    attributes {pulse.length_cycles = 40 : i64, pulse.schedule_state = #pulse.schedule_state<timed>} {
  %c40 = arith.constant 40 : i64
  %a = arith.constant 0.1 : f64
  %z = arith.constant 0.0 : f64
  %x = pulse.square %c40, %a, %z : i64, f64, f64 -> !w
  %p = arith.constant 0.5 : f64
  %t0 = pulse.shift_phase %t, %p <{duration_cycles = 0 : i64, start_cycles = 0 : i64}> : !pulse.tone, f64 -> !pulse.tone
  %d1, %t1 = pulse.drive %d, %x, %t0 <{duration_cycles = 40 : i64, start_cycles = 0 : i64}> : !d, !w, !t -> !d, !t
  return %d1, %t1 : !d, !t
}

func.func private @outer(%d: !d, %t: !t, %e: !d, %u: !t) -> (!d, !t, !d, !t)
    attributes {pulse.length_cycles = 50 : i64, pulse.schedule_state = #pulse.schedule_state<timed>} {
  %a:2 = pulse.call @inner(%d, %t) <{duration_cycles = 40 : i64, start_cycles = 10 : i64}> : (!d, !t) -> (!d, !t)
  return %a#0, %a#1, %e, %u : !d, !t, !d, !t
}

// KEEP-LABEL: func.func @nested
// CHECK-LABEL: func.func @nested
// CHECK: scf.for
// CHECK: pulse.shift_phase {{.*}}<{duration_cycles = 0 : i64, period_cycles = array<i64: 100>, start_cycles = 1010 : i64}>
// CHECK: pulse.drive {{.*}}<{duration_cycles = 40 : i64, period_cycles = array<i64: 100>, start_cycles = 1010 : i64}>
// CHECK: pulse.drive {{.*}}<{duration_cycles = 40 : i64, period_cycles = array<i64: 50>, start_cycles = 0 : i64}>
func.func @nested() attributes {pulse.schedule_state = #pulse.schedule_state<timed>} {
  %q = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit
  %r = pulse.qudit_alloc {qudit = 1 : i64} : !pulse.qudit
  %d0, %t0 = pulse.get_drive_line %q : (!pulse.qudit) -> (!pulse.drive_line, !pulse.tone)
  %e0, %u0 = pulse.get_drive_line %r : (!pulse.qudit) -> (!pulse.drive_line, !pulse.tone)
  %c40 = arith.constant 40 : i64
  %a = arith.constant 0.1 : f64
  %z = arith.constant 0.0 : f64
  %x = pulse.square %c40, %a, %z : i64, f64, f64 -> !w
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %f:4 = scf.for %k = %c0 to %c4 step %c1 iter_args(%d = %d0, %t = %t0, %e = %e0, %u = %u0) -> (!d, !t, !d, !t) {
    %g:4 = pulse.call @outer(%d, %t, %e, %u) <{duration_cycles = 50 : i64, period_cycles = array<i64: 100>, start_cycles = 1000 : i64}> : (!d, !t, !d, !t) -> (!d, !t, !d, !t)
    %h, %v = pulse.drive %g#2, %x, %g#3 <{duration_cycles = 40 : i64, period_cycles = array<i64: 50>, start_cycles = 0 : i64}> : !d, !w, !t -> !d, !t
    scf.yield %g#0, %g#1, %h, %v : !d, !t, !d, !t
  }
  return
}

// -----

!d = !pulse.drive_line
!r = !pulse.readout_line
!t = !pulse.tone
!w = !pulse.waveform
!m = !pulse.measurement

// A public callee that takes a readout line before a drive line and hands
// the drive line back first: each line type keeps its order, and the tones
// keep theirs. The call lasts
// longer than the callee, so each line it plays on is held to the call's
// end, from its own last end.
func.func @measure_and_drive(%r: !r, %u: !t, %d: !d, %t: !t) -> (!d, !t, !r, !t, !m)
    attributes {pulse.length_cycles = 400 : i64, pulse.schedule_state = #pulse.schedule_state<timed>} {
  %c400 = arith.constant 400 : i64
  %c40 = arith.constant 40 : i64
  %a = arith.constant 0.05 : f64
  %z = arith.constant 0.0 : f64
  %y = pulse.square %c400, %a, %z : i64, f64, f64 -> !w
  %x = pulse.square %c40, %a, %z : i64, f64, f64 -> !w
  %r1, %u1, %m = pulse.readout %r, %y, %u, iq <{duration_cycles = 400 : i64, start_cycles = 0 : i64}> : !r, !w, !t -> !r, !t, !m
  %d1, %t1 = pulse.drive %d, %x, %t <{duration_cycles = 40 : i64, start_cycles = 0 : i64}> : !d, !w, !t -> !d, !t
  return %d1, %u1, %r1, %t1, %m : !d, !t, !r, !t, !m
}

// KEEP-LABEL: func.func @held
// CHECK-NOT: func.func @measure_and_drive
// CHECK-LABEL: func.func @held
// CHECK: pulse.readout {{.*}}<{duration_cycles = 400 : i64, start_cycles = 0 : i64}>
// CHECK: pulse.drive {{.*}}<{duration_cycles = 40 : i64, start_cycles = 0 : i64}>
// CHECK-DAG: pulse.wait %{{.*}} <{duration_cycles = 460 : i64, start_cycles = 40 : i64}> : (!pulse.drive_line
// CHECK-DAG: pulse.wait %{{.*}} <{duration_cycles = 100 : i64, start_cycles = 400 : i64}> : (!pulse.readout_line
func.func @held() attributes {pulse.schedule_state = #pulse.schedule_state<timed>} {
  %q = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit
  %r0, %u0 = pulse.get_readout_line %q : (!pulse.qudit) -> (!pulse.readout_line, !pulse.tone)
  %d0, %t0 = pulse.get_drive_line %q : (!pulse.qudit) -> (!pulse.drive_line, !pulse.tone)
  %a:5 = pulse.call @measure_and_drive(%r0, %u0, %d0, %t0) <{duration_cycles = 500 : i64, start_cycles = 0 : i64}> : (!r, !t, !d, !t) -> (!d, !t, !r, !t, !m)
  return
}

// -----

!d = !pulse.drive_line
!t = !pulse.tone
!w = !pulse.waveform

// Numbers are the values at the call: the amplitude (f64) and the length
// (i64) of the written-out drive are the caller's.
func.func private @scaled(%d: !d, %t: !t, %amp: f64, %len: i64) -> (!d, !t)
    attributes {pulse.length_cycles = 24 : i64, pulse.schedule_state = #pulse.schedule_state<timed>} {
  %z = arith.constant 0.0 : f64
  %x = pulse.square %len, %amp, %z : i64, f64, f64 -> !w
  %d1, %t1 = pulse.drive %d, %x, %t <{duration_cycles = 24 : i64, start_cycles = 0 : i64}> : !d, !w, !t -> !d, !t
  return %d1, %t1 : !d, !t
}

// KEEP-LABEL: func.func @numbers
// KEEP: pulse.call @scaled
// CHECK-NOT: func.func private @scaled
// CHECK-LABEL: func.func @numbers
// CHECK: %[[AMP:.*]] = arith.constant 2.500000e-01 : f64
// CHECK: %[[LEN:.*]] = arith.constant 24 : i64
// CHECK: pulse.square %[[LEN]], %[[AMP]]
// CHECK: pulse.drive {{.*}}<{duration_cycles = 24 : i64, start_cycles = 8 : i64}>
func.func @numbers() attributes {pulse.schedule_state = #pulse.schedule_state<timed>} {
  %q = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit
  %d0, %t0 = pulse.get_drive_line %q : (!pulse.qudit) -> (!pulse.drive_line, !pulse.tone)
  %amp = arith.constant 0.25 : f64
  %len = arith.constant 24 : i64
  %a:2 = pulse.call @scaled(%d0, %t0, %amp, %len) <{duration_cycles = 24 : i64, start_cycles = 8 : i64}> : (!d, !t, f64, i64) -> (!d, !t)
  return
}
