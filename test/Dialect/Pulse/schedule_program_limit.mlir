// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// RUN: cudaq-pulse-opt %s --split-input-file --pulse-schedule-asap --verify-diagnostics
// RUN: cudaq-pulse-opt %s --split-input-file --pulse-schedule-alap --verify-diagnostics

// pulse.program_limits on a module, a controller's longest program, bounds
// the scheduled program: operations that play together on two lines count
// once, and operations that follow each other on one line add up.

!d = !pulse.drive_line
!t = !pulse.tone
!w = !pulse.waveform

// Two 80-cycle drives on two lines at the same time last 80 cycles.
module attributes {pulse.program_limits = [{controller = 0 : i64, cycles = 100 : i64, name = "controllers[0]", duration = "50 ns"}]} {
  func.func @together(%d0: !d, %t0: !t, %e0: !d, %u0: !t) {
    %c = arith.constant 80 : i64
    %a = arith.constant 0.1 : f64
    %s = arith.constant 0.0 : f64
    %x = pulse.square %c, %a, %s : i64, f64, f64 -> !w
    %d1, %t1 = pulse.drive %d0, %x, %t0 : !d, !w, !t -> !d, !t
    %e1, %u1 = pulse.drive %e0, %x, %u0 : !d, !w, !t -> !d, !t
    return
  }
}

// -----

!d = !pulse.drive_line
!t = !pulse.tone
!w = !pulse.waveform

// The same two drives on one line last 160 cycles: past the limit, and the
// function carries the error for whoever ran the schedule.
module attributes {pulse.program_limits = [{controller = 0 : i64, cycles = 100 : i64, name = "controllers[0]", duration = "50 ns"}]} {
  // expected-error@+1 {{kernel 'main' lasts 160 cycles, more than controllers[0]'s max_program_duration of 50 ns (100 cycles); let its operations overlap, shorten them or raise max_program_duration}}
  func.func @in_turn(%d0: !d, %t0: !t) {
    %c = arith.constant 80 : i64
    %a = arith.constant 0.1 : f64
    %s = arith.constant 0.0 : f64
    %x = pulse.square %c, %a, %s : i64, f64, f64 -> !w
    %d1, %t1 = pulse.drive %d0, %x, %t0 : !d, !w, !t -> !d, !t
    %d2, %t2 = pulse.drive %d1, %x, %t1 : !d, !w, !t -> !d, !t
    return
  }
}
