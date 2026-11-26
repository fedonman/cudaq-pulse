// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// RUN: cudaq-pulse-opt %s --split-input-file --pulse-schedule-asap --pulse-classical-check --verify-diagnostics
// RUN: cudaq-pulse-opt %s --split-input-file --pulse-schedule-alap --pulse-classical-check --verify-diagnostics
// RUN: cudaq-pulse-opt %s --split-input-file --pulse-schedule-rcp --pulse-classical-check --verify-diagnostics

// Under pulse.program_limits a wait whose length is an answer counts at its
// highest length for the lines it holds back, those of its timeline, and
// for no other line. The schedule passes hold the program as scheduled to
// the limits, and pulse-classical-check, which follows them for a module
// with classical calls, the program with those waits at their highest.

// Line e plays 300 cycles and nothing ties it to line d, whose wait of 16 to
// 1000 cycles starts at 101 and whose last drive ends at 157 as scheduled:
// d can last 157 + 984 = 1141 cycles and e 300, so the program fits 1141.
module attributes {pulse.control_system = #pulse.control_system<clock_period_fs = 1000000>, pulse.program_limits = [{controller = 0 : i64, cycles = 1141 : i64, name = "controllers[0]", duration = "1141 ns"}]} {
  classical.callable @tau {dispatch_key = 1 : i64, device = 0 : i64,
      args = [], results = ["i64"], latency_min_ns = 0.0,
      latency_nominal_ns = 50.0, latency_max_ns = 100.0, mode = "bounded",
      submit = "eager", dispatch_key_override}

  func.func @apart_fits() {
    %q = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit
    %p = pulse.qudit_alloc {qudit = 1 : i64} : !pulse.qudit
    %d0, %t0 = pulse.get_drive_line %q : (!pulse.qudit) -> (!pulse.drive_line, !pulse.tone)
    %e0, %u0 = pulse.get_drive_line %p : (!pulse.qudit) -> (!pulse.drive_line, !pulse.tone)
    %c40 = arith.constant 40 : i64
    %c300 = arith.constant 300 : i64
    %a = arith.constant 0.05 : f64
    %z = arith.constant 0.0 : f64
    %x = pulse.square %c40, %a, %z : i64, f64, f64 -> !pulse.waveform
    %long = pulse.square %c300, %a, %z : i64, f64, f64 -> !pulse.waveform
    %e1, %u1 = pulse.drive %e0, %long, %u0 : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
    %d1, %t1 = pulse.drive %d0, %x, %t0 : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
    %l = classical.get_link {device = 0 : i64} : !classical.link
    %l1, %h = classical.submit %l, @tau() : () -> (!classical.link, !classical.future)
    %n = classical.wait %h, @tau : i64
    %dur = pulse.duration_from_int %n : (i64) -> !pulse.duration
    %d2 = pulse.wait %d1, %dur {classical.variable, pulse.variable_wait, pulse.min_cycles = 16 : i64, pulse.max_cycles = 1000 : i64} : (!pulse.drive_line, !pulse.duration) -> !pulse.drive_line
    %d3, %t3 = pulse.drive %d2, %x, %t1 : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
    return
  }
}

// -----

// The same program does not fit one cycle less, and the error gives line
// d's length, not the 300 cycles of line e with the wait's 984 on top.
module attributes {pulse.control_system = #pulse.control_system<clock_period_fs = 1000000>, pulse.program_limits = [{controller = 0 : i64, cycles = 1140 : i64, name = "controllers[0]", duration = "1140 ns"}]} {
  classical.callable @tau {dispatch_key = 1 : i64, device = 0 : i64,
      args = [], results = ["i64"], latency_min_ns = 0.0,
      latency_nominal_ns = 50.0, latency_max_ns = 100.0, mode = "bounded",
      submit = "eager", dispatch_key_override}

  // expected-error@+1 {{kernel 'main' lasts up to 1141 cycles (157 as scheduled, the waits of run-time length that hold it back add up to 984), more than controllers[0]'s max_program_duration of 1140 ns (1140 cycles); let its operations overlap, shorten them or raise max_program_duration}}
  func.func @apart_too_long() {
    %q = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit
    %p = pulse.qudit_alloc {qudit = 1 : i64} : !pulse.qudit
    %d0, %t0 = pulse.get_drive_line %q : (!pulse.qudit) -> (!pulse.drive_line, !pulse.tone)
    %e0, %u0 = pulse.get_drive_line %p : (!pulse.qudit) -> (!pulse.drive_line, !pulse.tone)
    %c40 = arith.constant 40 : i64
    %c300 = arith.constant 300 : i64
    %a = arith.constant 0.05 : f64
    %z = arith.constant 0.0 : f64
    %x = pulse.square %c40, %a, %z : i64, f64, f64 -> !pulse.waveform
    %long = pulse.square %c300, %a, %z : i64, f64, f64 -> !pulse.waveform
    %e1, %u1 = pulse.drive %e0, %long, %u0 : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
    %d1, %t1 = pulse.drive %d0, %x, %t0 : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
    %l = classical.get_link {device = 0 : i64} : !classical.link
    %l1, %h = classical.submit %l, @tau() : () -> (!classical.link, !classical.future)
    %n = classical.wait %h, @tau : i64
    %dur = pulse.duration_from_int %n : (i64) -> !pulse.duration
    %d2 = pulse.wait %d1, %dur {classical.variable, pulse.variable_wait, pulse.min_cycles = 16 : i64, pulse.max_cycles = 1000 : i64} : (!pulse.drive_line, !pulse.duration) -> !pulse.drive_line
    %d3, %t3 = pulse.drive %d2, %x, %t1 : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
    return
  }
}

// -----

// A sync after the wait ties line e to it: one timeline, in which the wait
// starts when e ends (300) and the last drive ends at 356 as scheduled.
module attributes {pulse.control_system = #pulse.control_system<clock_period_fs = 1000000>, pulse.program_limits = [{controller = 0 : i64, cycles = 1339 : i64, name = "controllers[0]", duration = "1339 ns"}]} {
  classical.callable @tau {dispatch_key = 1 : i64, device = 0 : i64,
      args = [], results = ["i64"], latency_min_ns = 0.0,
      latency_nominal_ns = 50.0, latency_max_ns = 100.0, mode = "bounded",
      submit = "eager", dispatch_key_override}

  // expected-error@+1 {{kernel 'main' lasts up to 1340 cycles (356 as scheduled, the waits of run-time length that hold it back add up to 984), more than controllers[0]'s max_program_duration of 1339 ns (1339 cycles); let its operations overlap, shorten them or raise max_program_duration}}
  func.func @tied() {
    %q = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit
    %p = pulse.qudit_alloc {qudit = 1 : i64} : !pulse.qudit
    %d0, %t0 = pulse.get_drive_line %q : (!pulse.qudit) -> (!pulse.drive_line, !pulse.tone)
    %e0, %u0 = pulse.get_drive_line %p : (!pulse.qudit) -> (!pulse.drive_line, !pulse.tone)
    %c40 = arith.constant 40 : i64
    %c300 = arith.constant 300 : i64
    %a = arith.constant 0.05 : f64
    %z = arith.constant 0.0 : f64
    %x = pulse.square %c40, %a, %z : i64, f64, f64 -> !pulse.waveform
    %long = pulse.square %c300, %a, %z : i64, f64, f64 -> !pulse.waveform
    %e1, %u1 = pulse.drive %e0, %long, %u0 : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
    %d1, %t1 = pulse.drive %d0, %x, %t0 : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
    %l = classical.get_link {device = 0 : i64} : !classical.link
    %l1, %h = classical.submit %l, @tau() : () -> (!classical.link, !classical.future)
    %n = classical.wait %h, @tau : i64
    %dur = pulse.duration_from_int %n : (i64) -> !pulse.duration
    %d2 = pulse.wait %d1, %dur {classical.variable, pulse.variable_wait, pulse.min_cycles = 16 : i64, pulse.max_cycles = 1000 : i64} : (!pulse.drive_line, !pulse.duration) -> !pulse.drive_line
    %d3, %t3 = pulse.drive %d2, %x, %t1 : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
    %d4, %e2 = pulse.sync %d3, %e1 : !pulse.drive_line, !pulse.drive_line -> !pulse.drive_line, !pulse.drive_line
    return
  }
}

// -----

// Without a highest length the wait has nothing to hold to the limit: an
// error at the wait.
module attributes {pulse.control_system = #pulse.control_system<clock_period_fs = 1000000>, pulse.program_limits = [{controller = 0 : i64, cycles = 1140 : i64, name = "controllers[0]", duration = "1140 ns"}]} {
  classical.callable @tau {dispatch_key = 1 : i64, device = 0 : i64,
      args = [], results = ["i64"], latency_min_ns = 0.0,
      latency_nominal_ns = 50.0, latency_max_ns = 100.0, mode = "bounded",
      submit = "eager", dispatch_key_override}

  func.func @no_highest() {
    %q = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit
    %p = pulse.qudit_alloc {qudit = 1 : i64} : !pulse.qudit
    %d0, %t0 = pulse.get_drive_line %q : (!pulse.qudit) -> (!pulse.drive_line, !pulse.tone)
    %e0, %u0 = pulse.get_drive_line %p : (!pulse.qudit) -> (!pulse.drive_line, !pulse.tone)
    %c40 = arith.constant 40 : i64
    %c300 = arith.constant 300 : i64
    %a = arith.constant 0.05 : f64
    %z = arith.constant 0.0 : f64
    %x = pulse.square %c40, %a, %z : i64, f64, f64 -> !pulse.waveform
    %long = pulse.square %c300, %a, %z : i64, f64, f64 -> !pulse.waveform
    %e1, %u1 = pulse.drive %e0, %long, %u0 : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
    %d1, %t1 = pulse.drive %d0, %x, %t0 : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
    %l = classical.get_link {device = 0 : i64} : !classical.link
    %l1, %h = classical.submit %l, @tau() : () -> (!classical.link, !classical.future)
    %n = classical.wait %h, @tau : i64
    %dur = pulse.duration_from_int %n : (i64) -> !pulse.duration
    // expected-error@+1 {{kernel 'main' runs under controllers[0]'s max_program_duration of 1140 ns, but this wait in it has no highest length; bound it with wait(..., bounds=(lowest, highest)), or drop max_program_duration}}
    %d2 = pulse.wait %d1, %dur {classical.variable, pulse.variable_wait, pulse.min_cycles = 16 : i64} : (!pulse.drive_line, !pulse.duration) -> !pulse.drive_line
    %d3, %t3 = pulse.drive %d2, %x, %t1 : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
    return
  }
}

// -----

// The answer of a blocking classical function has no latest time: under a
// limit, with a drive after it, an error at the wait for it.
module attributes {pulse.control_system = #pulse.control_system<clock_period_fs = 1000000>, pulse.program_limits = [{controller = 0 : i64, cycles = 100000 : i64, name = "controllers[0]", duration = "100000 ns"}]} {
  classical.callable @slow {dispatch_key = 1 : i64, device = 0 : i64,
      args = [], results = ["i64"], latency_min_ns = 0.0,
      latency_nominal_ns = 50.0, latency_max_ns = 100.0, mode = "blocking",
      submit = "eager", dispatch_key_override}

  func.func @blocking() {
    %q = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit
    %d0, %t0 = pulse.get_drive_line %q : (!pulse.qudit) -> (!pulse.drive_line, !pulse.tone)
    %c40 = arith.constant 40 : i64
    %a = arith.constant 0.05 : f64
    %z = arith.constant 0.0 : f64
    %x = pulse.square %c40, %a, %z : i64, f64, f64 -> !pulse.waveform
    %l = classical.get_link {device = 0 : i64} : !classical.link
    %l1, %h = classical.submit %l, @slow() : () -> (!classical.link, !classical.future)
    // expected-error@+1 {{kernel 'main' runs under controllers[0]'s max_program_duration of 100000 ns, but it waits for an answer of a blocking classical function, which has no bound; use latency_mode="bounded" or "timeout", or drop max_program_duration}}
    %n = classical.wait %h, @slow : i64
    %d1, %t1 = pulse.drive %d0, %x, %t0 : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
    return
  }
}

// -----

// With nothing after the answer the program ends with it as scheduled.
module attributes {pulse.control_system = #pulse.control_system<clock_period_fs = 1000000>, pulse.program_limits = [{controller = 0 : i64, cycles = 100000 : i64, name = "controllers[0]", duration = "100000 ns"}]} {
  classical.callable @slow {dispatch_key = 1 : i64, device = 0 : i64,
      args = [], results = ["i64"], latency_min_ns = 0.0,
      latency_nominal_ns = 50.0, latency_max_ns = 100.0, mode = "blocking",
      submit = "eager", dispatch_key_override}

  func.func @blocking_last() {
    %q = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit
    %d0, %t0 = pulse.get_drive_line %q : (!pulse.qudit) -> (!pulse.drive_line, !pulse.tone)
    %c40 = arith.constant 40 : i64
    %a = arith.constant 0.05 : f64
    %z = arith.constant 0.0 : f64
    %x = pulse.square %c40, %a, %z : i64, f64, f64 -> !pulse.waveform
    %d1, %t1 = pulse.drive %d0, %x, %t0 : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
    %l = classical.get_link {device = 0 : i64} : !classical.link
    %l1, %h = classical.submit %l, @slow() : () -> (!classical.link, !classical.future)
    %n = classical.wait %h, @slow : i64
    return
  }
}
