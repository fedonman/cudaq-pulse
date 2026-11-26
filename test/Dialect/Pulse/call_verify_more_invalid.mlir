// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// RUN: cudaq-pulse-opt %s --split-input-file --verify-diagnostics --pulse-verify

// More of what pulse-verify refuses in a program with calls: mutual
// recursion, a callee loop without constant bounds, links, and a frame
// change placed past the length a callee records.

!d = !pulse.drive_line
!t = !pulse.tone

func.func private @f(%d: !d, %t: !t) -> (!d, !t) attributes {pulse.length_cycles = 0 : i64} {
  %a:2 = pulse.call @g(%d, %t) <{duration_cycles = 0 : i64, start_cycles = 0 : i64}> : (!d, !t) -> (!d, !t)
  return %a#0, %a#1 : !d, !t
}

func.func private @g(%d: !d, %t: !t) -> (!d, !t) attributes {pulse.length_cycles = 0 : i64} {
  // expected-error @+1 {{'pulse.call' op calls @f, which calls back into it}}
  %a:2 = pulse.call @f(%d, %t) <{duration_cycles = 0 : i64, start_cycles = 0 : i64}> : (!d, !t) -> (!d, !t)
  return %a#0, %a#1 : !d, !t
}

// -----

!d = !pulse.drive_line
!t = !pulse.tone
!w = !pulse.waveform

func.func private @loops(%d: !d, %t: !t, %n: i64) -> (!d, !t) attributes {pulse.length_cycles = 0 : i64} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %stop = arith.index_cast %n : i64 to index
  // expected-error @+1 {{@loops records a length, and a loop in it needs constant bounds}}
  %r:2 = scf.for %k = %c0 to %stop step %c1 iter_args(%x = %d, %y = %t) -> (!d, !t) {
    scf.yield %x, %y : !d, !t
  }
  return %r#0, %r#1 : !d, !t
}

func.func @loop_bound_in_callee(%q: !pulse.qudit, %n: i64) {
  %d0, %t0 = pulse.get_drive_line %q : (!pulse.qudit) -> (!pulse.drive_line, !pulse.tone)
  %a:2 = pulse.call @loops(%d0, %t0, %n) <{duration_cycles = 0 : i64, start_cycles = 0 : i64}> : (!d, !t, i64) -> (!d, !t)
  return
}

// -----

func.func private @acquires_link() attributes {pulse.length_cycles = 0 : i64} {
  // expected-error @+1 {{@acquires_link is run by pulse.call, and takes its lines and links as arguments; it cannot acquire one}}
  %l = classical.get_link {device = 0 : i64} : !classical.link
  return
}

func.func @link_in_callee() {
  pulse.call @acquires_link() <{duration_cycles = 0 : i64, start_cycles = 0 : i64}> : () -> ()
  return
}

// -----

!k = !classical.link

func.func private @swaps_links(%a: !k, %b: !k) -> (!k, !k) attributes {pulse.length_cycles = 0 : i64} {
  // expected-error @+2 {{@swaps_links returns as result 1 a '!classical.link' that is not its argument of that type in the same place}}
  // expected-error @+1 {{@swaps_links returns as result 0 a '!classical.link' that is not its argument of that type in the same place}}
  return %b, %a : !k, !k
}

func.func @links_out_of_order() {
  %a = classical.get_link {device = 0 : i64} : !classical.link
  %b = classical.get_link {device = 1 : i64} : !classical.link
  %r:2 = pulse.call @swaps_links(%a, %b) <{duration_cycles = 0 : i64, start_cycles = 0 : i64}> : (!k, !k) -> (!k, !k)
  return
}

// -----

!d = !pulse.drive_line
!t = !pulse.tone
!w = !pulse.waveform

// A frame change past the recorded length ends the callee later than it says.
// expected-error @+1 {{records a length of 8 cycles, but its operations end at 1000}}
func.func private @late_frame(%d: !d, %t: !t) -> (!d, !t) attributes {pulse.length_cycles = 8 : i64} {
  %n = arith.constant 8 : i64
  %a = arith.constant 0.1 : f64
  %z = arith.constant 0.0 : f64
  %p = arith.constant 0.5 : f64
  %x = pulse.square %n, %a, %z : i64, f64, f64 -> !w
  %d1, %t1 = pulse.drive %d, %x, %t <{duration_cycles = 8 : i64, start_cycles = 0 : i64}> : !d, !w, !t -> !d, !t
  %t2 = pulse.shift_phase %t1, %p <{duration_cycles = 0 : i64, start_cycles = 1000 : i64}> : !pulse.tone, f64 -> !pulse.tone
  return %d1, %t2 : !d, !t
}

func.func @calls_late_frame(%q: !pulse.qudit) {
  %d0, %t0 = pulse.get_drive_line %q : (!pulse.qudit) -> (!pulse.drive_line, !pulse.tone)
  %a:2 = pulse.call @late_frame(%d0, %t0) <{duration_cycles = 8 : i64, start_cycles = 0 : i64}> : (!d, !t) -> (!d, !t)
  return
}
