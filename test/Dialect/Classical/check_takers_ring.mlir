// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// A block nothing reaches may hold values that use each other in a ring. The
// walk from a call's answer to the controllers that play what uses it ends at
// such a ring (the call's controllers are then not known) and does not go
// round it.
// RUN: cudaq-pulse-opt --pulse-classical-check %s > /dev/null

module attributes {pulse.qudit_count = 3 : i64,
    pulse.control_system = #pulse.control_system<clock_period_fs = 1000000,
      lines = [#pulse.line<drive, qudits = [0]>, #pulse.line<drive, qudits = [1], controller = 1>,
               #pulse.line<drive, qudits = [2], controller = 2>, #pulse.line<readout, qudits = [0]>,
               #pulse.line<readout, qudits = [1], controller = 1>,
               #pulse.line<readout, qudits = [2], controller = 2>],
      controllers = [
        #pulse.controller<>,
        #pulse.controller<blocks_on_calls>,
        #pulse.controller<blocks_on_calls>]>} {

classical.callable @decode {dispatch_key = 1, device = 0 : i64,
    args = ["bits:1"], results = ["f64"], latency_min_ns = 0.0,
    latency_nominal_ns = 100.0, latency_max_ns = 100.0, mode = "bounded",
    submit = "eager", dispatch_key_override}

func.func @ring_after_return() {
  %c10 = arith.constant 10 : i64
  %c40 = arith.constant 40 : i64
  %c400 = arith.constant 400 : i64
  %a = arith.constant 0.05 : f64
  %z = arith.constant 0.0 : f64
  %ro = pulse.square %c10, %a, %z : i64, f64, f64 -> !pulse.waveform
  %x = pulse.square %c40, %a, %z : i64, f64, f64 -> !pulse.waveform
  %long = pulse.square %c400, %a, %z : i64, f64, f64 -> !pulse.waveform
  %q0 = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit
  %r0, %rt0 = pulse.get_readout_line %q0 : (!pulse.qudit) -> (!pulse.readout_line, !pulse.tone)
  %q1 = pulse.qudit_alloc {qudit = 1 : i64} : !pulse.qudit
  %d1, %dt1 = pulse.get_drive_line %q1 : (!pulse.qudit) -> (!pulse.drive_line, !pulse.tone)
  %q2 = pulse.qudit_alloc {qudit = 2 : i64} : !pulse.qudit
  %d2, %dt2 = pulse.get_drive_line %q2 : (!pulse.qudit) -> (!pulse.drive_line, !pulse.tone)
  %d1a, %dt1a = pulse.drive %d1, %x, %dt1 : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
  %r0a, %rt0a, %m0 = pulse.readout %r0, %ro, %rt0, iq : !pulse.readout_line, !pulse.waveform, !pulse.tone -> !pulse.readout_line, !pulse.tone, !pulse.measurement
  %link = classical.get_link {device = 0 : i64} : !classical.link
  %link1, %f0 = classical.submit %link, @decode(%m0) : (!pulse.measurement) -> (!classical.link, !classical.future)
  %d2a, %dt2a = pulse.drive %d2, %long, %dt2 : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
  %x0 = classical.wait %f0, @decode : f64
  %w0 = pulse.square %c40, %x0, %z : i64, f64, f64 -> !pulse.waveform
  %d1b, %dt1b = pulse.drive %d1a, %w0, %dt1a : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
  return
^dead:
  %p = arith.addf %x0, %p2 : f64
  %p2 = arith.addf %p, %z : f64
  return
}

}
