// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// RUN: cudaq-pulse-opt %s '--pulse-schedule-rcp=max-drives=1' --pulse-classical-check --verify-diagnostics

// With more lines than lanes the schedule keeps no timelines apart: a wait
// whose length is an answer then holds back every line, and under
// pulse.program_limits the whole program counts with it at its highest
// length, also a line that nothing else ties to the wait. Here the two
// lines take the one lane in turn: line e to 300, then line d's drive, its
// wait of 16 to 1000 cycles and its last drive, which ends at 396.

module attributes {pulse.control_system = #pulse.control_system<clock_period_fs = 1000000>, pulse.program_limits = [{controller = 0 : i64, cycles = 1140 : i64, name = "controllers[0]", duration = "1140 ns"}]} {
  classical.callable @tau {dispatch_key = 1 : i64, device = 0 : i64,
      args = [], results = ["i64"], latency_min_ns = 0.0,
      latency_nominal_ns = 50.0, latency_max_ns = 100.0, mode = "bounded",
      submit = "eager", dispatch_key_override}

  // expected-error@+1 {{kernel 'main' lasts up to 1380 cycles (396 as scheduled, the waits of run-time length that hold it back add up to 984), more than controllers[0]'s max_program_duration of 1140 ns (1140 cycles); let its operations overlap, shorten them or raise max_program_duration}}
  func.func @one_lane() {
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
