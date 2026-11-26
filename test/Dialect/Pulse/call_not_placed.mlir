// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// RUN: cudaq-pulse-opt %s --verify-diagnostics --pulse-schedule-asap
// RUN: cudaq-pulse-opt %s --verify-diagnostics --pulse-schedule-alap
// RUN: cudaq-pulse-opt %s --verify-diagnostics --pulse-schedule-rcp

// The schedule passes refuse a call until they place calls.

!d = !pulse.drive_line
!t = !pulse.tone

func.func private @x90(%d: !d, %t: !t) -> (!d, !t) attributes {pulse.length_cycles = 0 : i64} {
  return %d, %t : !d, !t
}

func.func @not_placed_yet(%d0: !d, %t0: !t) -> (!d, !t) {
  // expected-error @+1 {{the schedule passes do not place a pulse.call yet; write the ops of @x90 in its place before scheduling}}
  %a:2 = pulse.call @x90(%d0, %t0) : (!d, !t) -> (!d, !t)
  return %a#0, %a#1 : !d, !t
}
