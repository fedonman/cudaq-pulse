// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// RUN: cudaq-pulse-opt %s --split-input-file --verify-diagnostics

!d = !pulse.drive_line
!t = !pulse.tone

func.func @no_such_function(%d0: !d, %t0: !t) {
  // expected-error @+1 {{calls 'missing', which is not a function of this module}}
  %a:2 = pulse.call @missing(%d0, %t0) : (!d, !t) -> (!d, !t)
  return
}

// -----

!d = !pulse.drive_line
!t = !pulse.tone

func.func private @declared(%d: !d, %t: !t) -> (!d, !t)

func.func @declaration(%d0: !d, %t0: !t) {
  // expected-error @+1 {{calls the declaration @declared; a call needs the callee's body}}
  %a:2 = pulse.call @declared(%d0, %t0) : (!d, !t) -> (!d, !t)
  return
}

// -----

!d = !pulse.drive_line
!t = !pulse.tone

func.func private @pass(%d: !d, %t: !t) -> (!d, !t) attributes {pulse.length_cycles = 0 : i64} {
  return %d, %t : !d, !t
}

func.func @wrong_type(%d0: !d, %t0: !t) {
  // expected-error @+1 {{does not match @pass's type '(!pulse.drive_line, !pulse.tone) -> (!pulse.drive_line, !pulse.tone)'}}
  %a = pulse.call @pass(%d0, %t0) : (!d, !t) -> !d
  return
}

// -----

!d = !pulse.drive_line
!t = !pulse.tone

func.func private @unscheduled(%d: !d, %t: !t) -> (!d, !t) {
  return %d, %t : !d, !t
}

func.func @placed_without_schedule(%d0: !d, %t0: !t) {
  // expected-error @+1 {{is placed, but @unscheduled has no schedule of its own (pulse.length_cycles)}}
  %a:2 = pulse.call @unscheduled(%d0, %t0) <{duration_cycles = 0 : i64, start_cycles = 0 : i64}> : (!d, !t) -> (!d, !t)
  return
}

// -----

!d = !pulse.drive_line
!t = !pulse.tone

func.func private @long(%d: !d, %t: !t) -> (!d, !t) attributes {pulse.length_cycles = 40 : i64} {
  return %d, %t : !d, !t
}

func.func @shorter_than_callee(%d0: !d, %t0: !t) {
  // expected-error @+1 {{lasts 20 cycles, but @long is 40 cycles long}}
  %a:2 = pulse.call @long(%d0, %t0) <{duration_cycles = 20 : i64, start_cycles = 0 : i64}> : (!d, !t) -> (!d, !t)
  return
}

// -----

!d = !pulse.drive_line
!t = !pulse.tone

func.func private @long(%d: !d, %t: !t) -> (!d, !t) attributes {pulse.length_cycles = 40 : i64} {
  return %d, %t : !d, !t
}

func.func @start_without_duration(%d0: !d, %t0: !t) {
  // expected-error @+1 {{is placed with start_cycles and duration_cycles together, or with neither}}
  %a:2 = pulse.call @long(%d0, %t0) <{start_cycles = 0 : i64}> : (!d, !t) -> (!d, !t)
  return
}

// -----

!d = !pulse.drive_line
!t = !pulse.tone

func.func private @long(%d: !d, %t: !t) -> (!d, !t) attributes {pulse.length_cycles = 40 : i64} {
  return %d, %t : !d, !t
}

func.func @negative_duration(%d0: !d, %t0: !t) {
  // expected-error @+1 {{lasts -1 cycles}}
  %a:2 = pulse.call @long(%d0, %t0) <{duration_cycles = -1 : i64, start_cycles = 0 : i64}> : (!d, !t) -> (!d, !t)
  return
}

// -----

func.func private @on_qudit(%q: !pulse.qudit) attributes {pulse.length_cycles = 0 : i64} {
  return
}

func.func @qudit_argument(%q: !pulse.qudit) {
  // expected-error @+1 {{argument 0 is a '!pulse.qudit'; a call takes lines, tones, measurements, links, f64 and i64}}
  pulse.call @on_qudit(%q) : (!pulse.qudit) -> ()
  return
}

// -----

func.func private @gives_number() -> i64 attributes {pulse.length_cycles = 0 : i64} {
  %c = arith.constant 1 : i64
  return %c : i64
}

func.func @number_result() {
  // expected-error @+1 {{result #0 must be variadic of}}
  %n = pulse.call @gives_number() : () -> i64
  return
}

// -----

!d = !pulse.drive_line
!t = !pulse.tone

func.func private @negative(%d: !d, %t: !t) -> (!d, !t) attributes {pulse.length_cycles = -5 : i64} {
  return %d, %t : !d, !t
}

func.func @negative_length(%d0: !d, %t0: !t) {
  // expected-error @+1 {{calls @negative, which records a length of -5 cycles}}
  %a:2 = pulse.call @negative(%d0, %t0) : (!d, !t) -> (!d, !t)
  return
}
