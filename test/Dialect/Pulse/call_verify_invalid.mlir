// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// RUN: cudaq-pulse-opt %s --split-input-file --verify-diagnostics --pulse-verify

// What pulse-verify refuses in a program with calls.

!d = !pulse.drive_line
!t = !pulse.tone
!w = !pulse.waveform

func.func private @x90(%d: !d, %t: !t) -> (!d, !t) attributes {pulse.length_cycles = 40 : i64} {
  %c40 = arith.constant 40 : i64
  %a = arith.constant 0.1 : f64
  %z = arith.constant 0.0 : f64
  %x = pulse.square %c40, %a, %z : i64, f64, f64 -> !w
  // expected-error @+1 {{operation overlaps or precedes its physical-line predecessor: start 20 < prior end 40}}
  %d1, %t1 = pulse.drive %d, %x, %t <{duration_cycles = 40 : i64, start_cycles = 0 : i64}> : !d, !w, !t -> !d, !t
  return %d1, %t1 : !d, !t
}

func.func @overlap() {
  %q = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit
  %d0, %t0 = pulse.get_drive_line %q : (!pulse.qudit) -> (!pulse.drive_line, !pulse.tone)
  %a:2 = pulse.call @x90(%d0, %t0) <{duration_cycles = 40 : i64, start_cycles = 0 : i64}> : (!d, !t) -> (!d, !t)
  %b:2 = pulse.call @x90(%a#0, %a#1) <{duration_cycles = 40 : i64, start_cycles = 20 : i64}> : (!d, !t) -> (!d, !t)
  return
}

// -----

!d = !pulse.drive_line
!t = !pulse.tone
!w = !pulse.waveform

func.func private @x90(%d: !d, %t: !t) -> (!d, !t) attributes {pulse.length_cycles = 40 : i64} {
  %c40 = arith.constant 40 : i64
  %a = arith.constant 0.1 : f64
  %z = arith.constant 0.0 : f64
  %x = pulse.square %c40, %a, %z : i64, f64, f64 -> !w
  // expected-error @+1 {{operation overlaps or precedes its physical-line predecessor: start 60 < prior end 100}}
  %d1, %t1 = pulse.drive %d, %x, %t <{duration_cycles = 40 : i64, start_cycles = 0 : i64}> : !d, !w, !t -> !d, !t
  return %d1, %t1 : !d, !t
}

// A call that lasts longer than its callee holds the callee's lines to its
// end: written out, a wait holds the line from 40 to 100.
func.func @held_to_the_call_end() {
  %q = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit
  %d0, %t0 = pulse.get_drive_line %q : (!pulse.qudit) -> (!pulse.drive_line, !pulse.tone)
  %a:2 = pulse.call @x90(%d0, %t0) <{duration_cycles = 100 : i64, start_cycles = 0 : i64}> : (!d, !t) -> (!d, !t)
  %b:2 = pulse.call @x90(%a#0, %a#1) <{duration_cycles = 40 : i64, start_cycles = 60 : i64}> : (!d, !t) -> (!d, !t)
  return
}

// -----

!d = !pulse.drive_line
!t = !pulse.tone

func.func private @f(%d: !d, %t: !t) -> (!d, !t) attributes {pulse.length_cycles = 0 : i64} {
  // expected-error @+1 {{'pulse.call' op calls @f, which calls back into it}}
  %a:2 = pulse.call @f(%d, %t) <{duration_cycles = 0 : i64, start_cycles = 0 : i64}> : (!d, !t) -> (!d, !t)
  return %a#0, %a#1 : !d, !t
}

// -----

!d = !pulse.drive_line
!t = !pulse.tone

func.func private @swap(%a: !d, %ta: !t, %b: !d, %tb: !t) -> (!d, !t, !d, !t) attributes {pulse.length_cycles = 0 : i64} {
  // expected-error @+2 {{@swap returns as result 2 a '!pulse.drive_line' that is not its argument of that type in the same place}}
  // expected-error @+1 {{@swap returns as result 0 a '!pulse.drive_line' that is not its argument of that type in the same place; a function pulse.call runs returns each line, tone and link it takes, in the order it takes them}}
  return %b, %ta, %a, %tb : !d, !t, !d, !t
}

func.func @order(%q0: !pulse.qudit, %q1: !pulse.qudit) {
  %d0, %t0 = pulse.get_drive_line %q0 : (!pulse.qudit) -> (!pulse.drive_line, !pulse.tone)
  %d1, %t1 = pulse.get_drive_line %q1 : (!pulse.qudit) -> (!pulse.drive_line, !pulse.tone)
  %a:4 = pulse.call @swap(%d0, %t0, %d1, %t1) <{duration_cycles = 0 : i64, start_cycles = 0 : i64}> : (!d, !t, !d, !t) -> (!d, !t, !d, !t)
  return
}

// -----

func.func private @fetches(%q: i64) attributes {pulse.length_cycles = 0 : i64} {
  %qq = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit
  // expected-error @+1 {{@fetches is run by pulse.call, and takes its lines and links as arguments; it cannot acquire one}}
  %d, %t = pulse.get_drive_line %qq : (!pulse.qudit) -> (!pulse.drive_line, !pulse.tone)
  return
}

func.func @acquires() {
  %n = arith.constant 0 : i64
  pulse.call @fetches(%n) <{duration_cycles = 0 : i64, start_cycles = 0 : i64}> : (i64) -> ()
  return
}

// -----

!d = !pulse.drive_line
!t = !pulse.tone
!w = !pulse.waveform

// expected-error @+1 {{records a length of 50 cycles, but its operations end at 40}}
func.func private @x90(%d: !d, %t: !t) -> (!d, !t) attributes {pulse.length_cycles = 50 : i64} {
  %c40 = arith.constant 40 : i64
  %a = arith.constant 0.1 : f64
  %z = arith.constant 0.0 : f64
  %x = pulse.square %c40, %a, %z : i64, f64, f64 -> !w
  %d1, %t1 = pulse.drive %d, %x, %t <{duration_cycles = 40 : i64, start_cycles = 0 : i64}> : !d, !w, !t -> !d, !t
  return %d1, %t1 : !d, !t
}

func.func @wrong_length(%q: !pulse.qudit) {
  %d0, %t0 = pulse.get_drive_line %q : (!pulse.qudit) -> (!pulse.drive_line, !pulse.tone)
  %a:2 = pulse.call @x90(%d0, %t0) <{duration_cycles = 50 : i64, start_cycles = 0 : i64}> : (!d, !t) -> (!d, !t)
  return
}

// -----

!d = !pulse.drive_line
!t = !pulse.tone
!w = !pulse.waveform

func.func private @x90(%d: !d, %t: !t) -> (!d, !t) attributes {pulse.length_cycles = 40 : i64} {
  %c40 = arith.constant 40 : i64
  %a = arith.constant 0.1 : f64
  %z = arith.constant 0.0 : f64
  %x = pulse.square %c40, %a, %z : i64, f64, f64 -> !w
  // expected-error @+1 {{operation is not placed, but @x90 records a length (pulse.length_cycles)}}
  %d1, %t1 = pulse.drive %d, %x, %t : !d, !w, !t -> !d, !t
  return %d1, %t1 : !d, !t
}

func.func @callee_unscheduled(%q: !pulse.qudit) {
  %d0, %t0 = pulse.get_drive_line %q : (!pulse.qudit) -> (!pulse.drive_line, !pulse.tone)
  %a:2 = pulse.call @x90(%d0, %t0) <{duration_cycles = 40 : i64, start_cycles = 0 : i64}> : (!d, !t) -> (!d, !t)
  return
}

// -----

!d = !pulse.drive_line
!r = !pulse.readout_line
!t = !pulse.tone
!w = !pulse.waveform
!m = !pulse.measurement

func.func private @measure(%r: !r, %t: !t) -> (!r, !t, !m) attributes {pulse.length_cycles = 400 : i64} {
  %c400 = arith.constant 400 : i64
  %a = arith.constant 0.05 : f64
  %z = arith.constant 0.0 : f64
  %x = pulse.square %c400, %a, %z : i64, f64, f64 -> !w
  %r1, %t1, %m = pulse.readout %r, %x, %t, iq <{duration_cycles = 400 : i64, start_cycles = 0 : i64}> : !r, !w, !t -> !r, !t, !m
  return %r1, %t1, %m : !r, !t, !m
}

func.func @branch_before_outcome(%q: !pulse.qudit, %p: !pulse.qudit) {
  %d0, %t0 = pulse.get_drive_line %p : (!pulse.qudit) -> (!pulse.drive_line, !pulse.tone)
  %r0, %u0 = pulse.get_readout_line %q : (!pulse.qudit) -> (!pulse.readout_line, !pulse.tone)
  %b:3 = pulse.call @measure(%r0, %u0) <{duration_cycles = 400 : i64, start_cycles = 0 : i64}> : (!r, !t) -> (!r, !t, !m)
  %bit = pulse.is_set %b#2 : !m -> i1
  %c40 = arith.constant 40 : i64
  %a = arith.constant 0.1 : f64
  %z = arith.constant 0.0 : f64
  %x = pulse.square %c40, %a, %z : i64, f64, f64 -> !w
  %e:2 = scf.if %bit -> (!d, !t) {
    // expected-error @+1 {{operation in a measurement-conditioned branch starts at 200, before its measurement completes at 400}}
    %f, %g = pulse.drive %d0, %x, %t0 <{duration_cycles = 40 : i64, start_cycles = 200 : i64}> : !d, !w, !t -> !d, !t
    scf.yield %f, %g : !d, !t
  } else {
    scf.yield %d0, %t0 : !d, !t
  }
  return
}

// -----

!r = !pulse.readout_line
!t = !pulse.tone
!w = !pulse.waveform
!m = !pulse.measurement
!d = !pulse.drive_line

// A measurement handed through a callee completes where its readout does.
func.func private @keep(%m: !m) -> !m attributes {pulse.length_cycles = 0 : i64} {
  return %m : !m
}

func.func @through_a_callee(%q: !pulse.qudit, %p: !pulse.qudit) {
  %r0, %u0 = pulse.get_readout_line %q : (!pulse.qudit) -> (!pulse.readout_line, !pulse.tone)
  %d0, %t0 = pulse.get_drive_line %p : (!pulse.qudit) -> (!pulse.drive_line, !pulse.tone)
  %c400 = arith.constant 400 : i64
  %c40 = arith.constant 40 : i64
  %a = arith.constant 0.05 : f64
  %z = arith.constant 0.0 : f64
  %y = pulse.square %c400, %a, %z : i64, f64, f64 -> !w
  %x = pulse.square %c40, %a, %z : i64, f64, f64 -> !w
  %r1, %u1, %m = pulse.readout %r0, %y, %u0, iq <{duration_cycles = 400 : i64, start_cycles = 0 : i64}> : !r, !w, !t -> !r, !t, !m
  %k = pulse.call @keep(%m) <{duration_cycles = 0 : i64, start_cycles = 0 : i64}> : (!m) -> !m
  %bit = pulse.is_set %k : !m -> i1
  %e:2 = scf.if %bit -> (!d, !t) {
    // expected-error @+1 {{operation in a measurement-conditioned branch starts at 100, before its measurement completes at 400}}
    %f, %g = pulse.drive %d0, %x, %t0 <{duration_cycles = 40 : i64, start_cycles = 100 : i64}> : !d, !w, !t -> !d, !t
    scf.yield %f, %g : !d, !t
  } else {
    scf.yield %d0, %t0 : !d, !t
  }
  return
}

// -----

!r = !pulse.readout_line
!t = !pulse.tone
!w = !pulse.waveform
!m = !pulse.measurement
!d = !pulse.drive_line

// A branch in a callee on a measurement it is given waits for the readout.
func.func private @cond(%m: !m, %d: !d, %t: !t) -> (!d, !t) attributes {pulse.length_cycles = 40 : i64} {
  %c40 = arith.constant 40 : i64
  %a = arith.constant 0.05 : f64
  %z = arith.constant 0.0 : f64
  %x = pulse.square %c40, %a, %z : i64, f64, f64 -> !w
  %b = pulse.is_set %m : !m -> i1
  %e:2 = scf.if %b -> (!d, !t) {
    // expected-error @+1 {{operation in a measurement-conditioned branch starts at 0, before its measurement completes at 400}}
    %f, %g = pulse.drive %d, %x, %t <{duration_cycles = 40 : i64, start_cycles = 0 : i64}> : !d, !w, !t -> !d, !t
    scf.yield %f, %g : !d, !t
  } else {
    scf.yield %d, %t : !d, !t
  }
  return %e#0, %e#1 : !d, !t
}

func.func @branch_in_callee(%q: !pulse.qudit, %p: !pulse.qudit) {
  %r0, %u0 = pulse.get_readout_line %q : (!pulse.qudit) -> (!pulse.readout_line, !pulse.tone)
  %d0, %t0 = pulse.get_drive_line %p : (!pulse.qudit) -> (!pulse.drive_line, !pulse.tone)
  %c400 = arith.constant 400 : i64
  %a = arith.constant 0.05 : f64
  %z = arith.constant 0.0 : f64
  %y = pulse.square %c400, %a, %z : i64, f64, f64 -> !w
  %r1, %u1, %m = pulse.readout %r0, %y, %u0, iq <{duration_cycles = 400 : i64, start_cycles = 0 : i64}> : !r, !w, !t -> !r, !t, !m
  %e:2 = pulse.call @cond(%m, %d0, %t0) <{duration_cycles = 40 : i64, start_cycles = 0 : i64}> : (!m, !d, !t) -> (!d, !t)
  return
}

// -----

!d = !pulse.drive_line
!t = !pulse.tone
!w = !pulse.waveform

func.func private @x90(%d: !d, %t: !t) -> (!d, !t) attributes {pulse.length_cycles = 40 : i64} {
  %c40 = arith.constant 40 : i64
  %a = arith.constant 0.1 : f64
  %z = arith.constant 0.0 : f64
  %x = pulse.square %c40, %a, %z : i64, f64, f64 -> !w
  // expected-error @+1 {{a scheduled loop needs constant bounds}}
  %d1, %t1 = pulse.drive %d, %x, %t <{duration_cycles = 40 : i64, start_cycles = 0 : i64}> : !d, !w, !t -> !d, !t
  return %d1, %t1 : !d, !t
}

func.func @loop_without_constant_bounds(%q: !pulse.qudit, %n: index) {
  %d0, %t0 = pulse.get_drive_line %q : (!pulse.qudit) -> (!pulse.drive_line, !pulse.tone)
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %f:2 = scf.for %k = %c0 to %n step %c1 iter_args(%d = %d0, %t = %t0) -> (!d, !t) {
    %g:2 = pulse.call @x90(%d, %t) <{duration_cycles = 40 : i64, period_cycles = array<i64: 40>, start_cycles = 0 : i64}> : (!d, !t) -> (!d, !t)
    scf.yield %g#0, %g#1 : !d, !t
  }
  return
}

// -----

!d = !pulse.drive_line
!t = !pulse.tone

func.func private @drops(%a: !d, %ta: !t, %b: !d, %tb: !t) -> (!d, !t) attributes {pulse.length_cycles = 0 : i64} {
  // expected-error @+2 {{@drops takes 2 of '!pulse.drive_line' and returns 1; a function pulse.call runs returns each line, tone and link it takes}}
  // expected-error @+1 {{@drops takes 2 of '!pulse.tone' and returns 1; a function pulse.call runs returns each line, tone and link it takes}}
  return %a, %ta : !d, !t
}

func.func @drops_lines(%q0: !pulse.qudit, %q1: !pulse.qudit) {
  %d0, %t0 = pulse.get_drive_line %q0 : (!pulse.qudit) -> (!pulse.drive_line, !pulse.tone)
  %d1, %t1 = pulse.get_drive_line %q1 : (!pulse.qudit) -> (!pulse.drive_line, !pulse.tone)
  %a:2 = pulse.call @drops(%d0, %t0, %d1, %t1) <{duration_cycles = 0 : i64, start_cycles = 0 : i64}> : (!d, !t, !d, !t) -> (!d, !t)
  return
}

// -----

!d = !pulse.drive_line
!t = !pulse.tone

// A swap in one arm of a branch is a swap.
func.func private @maybe_swap(%n: i64, %a: !d, %b: !d) -> (!d, !d) attributes {pulse.length_cycles = 0 : i64} {
  %zero = arith.constant 0 : i64
  %c = arith.cmpi ne, %n, %zero : i64
  %r:2 = scf.if %c -> (!d, !d) {
    scf.yield %a, %b : !d, !d
  } else {
    scf.yield %b, %a : !d, !d
  }
  // expected-error @+2 {{@maybe_swap returns as result 1 a '!pulse.drive_line' that is not its argument of that type in the same place}}
  // expected-error @+1 {{@maybe_swap returns as result 0 a '!pulse.drive_line' that is not its argument of that type in the same place}}
  return %r#0, %r#1 : !d, !d
}

func.func @swaps(%q0: !pulse.qudit, %q1: !pulse.qudit, %n: i64) {
  %d0, %t0 = pulse.get_drive_line %q0 : (!pulse.qudit) -> (!pulse.drive_line, !pulse.tone)
  %d1, %t1 = pulse.get_drive_line %q1 : (!pulse.qudit) -> (!pulse.drive_line, !pulse.tone)
  %a:2 = pulse.call @maybe_swap(%n, %d0, %d1) <{duration_cycles = 0 : i64, start_cycles = 0 : i64}> : (i64, !d, !d) -> (!d, !d)
  return
}

// -----

!d = !pulse.drive_line
!r = !pulse.readout_line
!t = !pulse.tone
!w = !pulse.waveform
!m = !pulse.measurement

func.func private @measure_and_drive(%r: !r, %u: !t, %d: !d, %t: !t) -> (!d, !t, !r, !t, !m) attributes {pulse.length_cycles = 400 : i64} {
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

// The drive line is held to the call's end (500), not only to the
// callee's drive (40).
func.func @drive_in_the_hold(%q: !pulse.qudit) {
  %r0, %u0 = pulse.get_readout_line %q : (!pulse.qudit) -> (!pulse.readout_line, !pulse.tone)
  %d0, %t0 = pulse.get_drive_line %q : (!pulse.qudit) -> (!pulse.drive_line, !pulse.tone)
  %a:5 = pulse.call @measure_and_drive(%r0, %u0, %d0, %t0) <{duration_cycles = 500 : i64, start_cycles = 0 : i64}> : (!r, !t, !d, !t) -> (!d, !t, !r, !t, !m)
  %c40 = arith.constant 40 : i64
  %b = arith.constant 0.1 : f64
  %z = arith.constant 0.0 : f64
  %x = pulse.square %c40, %b, %z : i64, f64, f64 -> !w
  // expected-error @+1 {{operation overlaps or precedes its physical-line predecessor: start 200 < prior end 500}}
  %e, %f = pulse.drive %a#0, %x, %a#3 <{duration_cycles = 40 : i64, start_cycles = 200 : i64}> : !d, !w, !t -> !d, !t
  return
}

// -----

!d = !pulse.drive_line
!t = !pulse.tone

func.func private @blocky(%d: !d, %t: !t) -> (!d, !t) attributes {pulse.length_cycles = 0 : i64} {
  // expected-error @+1 {{@blocky is run by pulse.call; write its schedule blocks out first}}
  %e, %u = pulse.schedule_block asap (%d, %t) : (!d, !t) -> (!d, !t) {
  ^bb0(%a: !d, %b: !t):
    pulse.yield %a, %b : !d, !t
  }
  return %e, %u : !d, !t
}

func.func @block_in_callee(%q: !pulse.qudit) {
  %d0, %t0 = pulse.get_drive_line %q : (!pulse.qudit) -> (!pulse.drive_line, !pulse.tone)
  %a:2 = pulse.call @blocky(%d0, %t0) <{duration_cycles = 0 : i64, start_cycles = 0 : i64}> : (!d, !t) -> (!d, !t)
  return
}

// -----

!d = !pulse.drive_line
!t = !pulse.tone

func.func private @pass(%d: !d, %t: !t) -> (!d, !t) attributes {pulse.length_cycles = 0 : i64} {
  return %d, %t : !d, !t
}

// The program's warnings are those of the program written out, each given
// once: @pass plays nothing on the line it is handed.
func.func @warned_once(%q: !pulse.qudit, %p: !pulse.qudit) {
  // expected-warning @+1 {{drive line allocated but never used}}
  %d0, %t0 = pulse.get_drive_line %q : (!pulse.qudit) -> (!pulse.drive_line, !pulse.tone)
  // expected-warning @+1 {{drive line allocated but never used}}
  %e0, %u0 = pulse.get_drive_line %p : (!pulse.qudit) -> (!pulse.drive_line, !pulse.tone)
  %a:2 = pulse.call @pass(%d0, %t0) <{duration_cycles = 0 : i64, start_cycles = 0 : i64}> : (!d, !t) -> (!d, !t)
  return
}
