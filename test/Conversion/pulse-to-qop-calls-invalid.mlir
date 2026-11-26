// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// RUN: cudaq-pulse-opt --pulse-to-qop --verify-diagnostics %s

// The simulation walks a call's pulses once it is written out
// (pulse-expand-calls); a call still in the program is refused, not
// skipped.

!d = !pulse.drive_line
!t = !pulse.tone
!w = !pulse.waveform

func.func private @x90(%d: !d, %t: !t) -> (!d, !t)
    attributes {pulse.length_cycles = 40 : i64, pulse.schedule_state = #pulse.schedule_state<timed>} {
  %c40 = arith.constant 40 : i64
  %a = arith.constant 0.1 : f64
  %z = arith.constant 0.0 : f64
  %x = pulse.square %c40, %a, %z : i64, f64, f64 -> !w
  %d1, %t1 = pulse.drive %d, %x, %t <{duration_cycles = 40 : i64, start_cycles = 0 : i64}> : !d, !w, !t -> !d, !t
  return %d1, %t1 : !d, !t
}

func.func @main() attributes {pulse.schedule_state = #pulse.schedule_state<timed>} {
  %q = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit
  %d0, %t0 = pulse.get_drive_line %q : (!pulse.qudit) -> (!pulse.drive_line, !pulse.tone)
  // expected-error @+1 {{a pulse.call is run once it is written out; run pulse-expand-calls first}}
  %a:2 = pulse.call @x90(%d0, %t0) <{duration_cycles = 40 : i64, start_cycles = 0 : i64}> : (!d, !t) -> (!d, !t)
  return
}
