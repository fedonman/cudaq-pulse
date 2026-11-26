// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// RUN: cudaq-pulse-opt %s | cudaq-pulse-opt | FileCheck %s
// RUN: cudaq-pulse-opt %s --mlir-print-op-generic | cudaq-pulse-opt | FileCheck %s

!d = !pulse.drive_line
!r = !pulse.readout_line
!t = !pulse.tone
!w = !pulse.waveform
!m = !pulse.measurement

func.func private @x90(%d: !d, %t: !t, %amp: f64) -> (!d, !t)
    attributes {pulse.length_cycles = 40 : i64} {
  %c40 = arith.constant 40 : i64
  %z = arith.constant 0.0 : f64
  %x = pulse.square %c40, %amp, %z : i64, f64, f64 -> !w
  %d1, %t1 = pulse.drive %d, %x, %t <{duration_cycles = 40 : i64, start_cycles = 0 : i64}> : !d, !w, !t -> !d, !t
  return %d1, %t1 : !d, !t
}

func.func private @measure(%r: !r, %t: !t) -> (!r, !t, !m)
    attributes {pulse.length_cycles = 400 : i64} {
  %c400 = arith.constant 400 : i64
  %a = arith.constant 0.05 : f64
  %z = arith.constant 0.0 : f64
  %x = pulse.square %c400, %a, %z : i64, f64, f64 -> !w
  %r1, %t1, %m = pulse.readout %r, %x, %t, iq <{duration_cycles = 400 : i64, start_cycles = 0 : i64}> : !r, !w, !t -> !r, !t, !m
  return %r1, %t1, %m : !r, !t, !m
}

// A call not yet placed, a placed one, one that lasts longer than its callee,
// and one in a loop with a period.
// CHECK-LABEL: func.func @calls
// CHECK: %{{.*}}:2 = pulse.call @x90(%{{.*}}, %{{.*}}, %{{.*}}) : (!pulse.drive_line, !pulse.tone, f64) -> (!pulse.drive_line, !pulse.tone)
// CHECK: pulse.call @x90(%{{.*}}, %{{.*}}, %{{.*}}) <{duration_cycles = 40 : i64, start_cycles = 0 : i64}> : (!pulse.drive_line, !pulse.tone, f64) -> (!pulse.drive_line, !pulse.tone)
// CHECK: pulse.call @measure(%{{.*}}, %{{.*}}) <{duration_cycles = 450 : i64, start_cycles = 40 : i64}> : (!pulse.readout_line, !pulse.tone) -> (!pulse.readout_line, !pulse.tone, !pulse.measurement)
// CHECK: pulse.call @x90(%{{.*}}, %{{.*}}, %{{.*}}) <{duration_cycles = 40 : i64, period_cycles = array<i64: 561>, start_cycles = 490 : i64}>
func.func @calls(%d0: !d, %t0: !t, %r0: !r, %u0: !t, %amp: f64) {
  %a:2 = pulse.call @x90(%d0, %t0, %amp) : (!d, !t, f64) -> (!d, !t)
  %b:2 = pulse.call @x90(%a#0, %a#1, %amp) <{duration_cycles = 40 : i64, start_cycles = 0 : i64}> : (!d, !t, f64) -> (!d, !t)
  %c:3 = pulse.call @measure(%r0, %u0) <{duration_cycles = 450 : i64, start_cycles = 40 : i64}> : (!r, !t) -> (!r, !t, !m)
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %e:2 = scf.for %k = %c0 to %c8 step %c1 iter_args(%d = %b#0, %t = %b#1) -> (!d, !t) {
    %f:2 = pulse.call @x90(%d, %t, %amp) <{duration_cycles = 40 : i64, period_cycles = array<i64: 561>, start_cycles = 490 : i64}> : (!d, !t, f64) -> (!d, !t)
    scf.yield %f#0, %f#1 : !d, !t
  }
  return
}
