// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// RUN: cudaq-pulse-opt %s --split-input-file --verify-diagnostics --pulse-expand-calls

// What pulse-expand-calls refuses.

!d = !pulse.drive_line
!t = !pulse.tone

func.func private @x(%d: !d, %t: !t) -> (!d, !t) attributes {pulse.length_cycles = 0 : i64} {
  return %d, %t : !d, !t
}

func.func @not_placed(%d0: !d, %t0: !t) {
  // expected-error @+1 {{'pulse.call' op is not placed; schedule it before writing it out}}
  %a:2 = pulse.call @x(%d0, %t0) : (!d, !t) -> (!d, !t)
  return
}

// -----

!r = !pulse.readout_line
!t = !pulse.tone
!w = !pulse.waveform
!m = !pulse.measurement

func.func private @numbered(%r: !r, %t: !t) -> (!r, !t, !m) attributes {pulse.length_cycles = 400 : i64} {
  %c400 = arith.constant 400 : i64
  %a = arith.constant 0.05 : f64
  %z = arith.constant 0.0 : f64
  %x = pulse.square %c400, %a, %z : i64, f64, f64 -> !w
  %r1, %t1, %m = pulse.readout %r, %x, %t, iq <{duration_cycles = 400 : i64, start_cycles = 0 : i64}> {pulse.readout_number = 0 : i64} : !r, !w, !t -> !r, !t, !m
  return %r1, %t1, %m : !r, !t, !m
}

func.func @numbered_readouts(%r0: !r, %u0: !t) {
  // expected-error @+1 {{'pulse.call' op calls @numbered, whose readouts are numbered in the kernel (pulse.readout_number); such a call cannot be written out}}
  %a:3 = pulse.call @numbered(%r0, %u0) <{duration_cycles = 400 : i64, start_cycles = 0 : i64}> : (!r, !t) -> (!r, !t, !m)
  return
}

// -----

!d = !pulse.drive_line
!t = !pulse.tone

func.func private @blocky(%d: !d, %t: !t) -> (!d, !t) attributes {pulse.length_cycles = 0 : i64} {
  %e, %u = pulse.schedule_block asap (%d, %t) : (!d, !t) -> (!d, !t) {
  ^bb0(%a: !d, %b: !t):
    pulse.yield %a, %b : !d, !t
  }
  return %e, %u : !d, !t
}

func.func @block_in_callee(%d0: !d, %t0: !t) {
  // expected-error @+1 {{'pulse.call' op calls @blocky, which holds a pulse.schedule_block; write its blocks out first}}
  %a:2 = pulse.call @blocky(%d0, %t0) <{duration_cycles = 0 : i64, start_cycles = 0 : i64}> : (!d, !t) -> (!d, !t)
  return
}
