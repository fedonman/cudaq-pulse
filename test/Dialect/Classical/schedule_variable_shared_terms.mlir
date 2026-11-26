// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// A variable wait whose length is arithmetic that shares its terms: an
// answer doubled forty times, each sum adding the sum before it to itself.
// The scheduler asks whether the length is a number yet, and computes each
// sum once: followed term by term, the forty sums are 2^40 walks and the
// schedule does not finish.
// RUN: cudaq-pulse-opt --pulse-schedule-asap %s | FileCheck %s
// RUN: cudaq-pulse-opt --pulse-schedule-alap %s | FileCheck %s
// RUN: cudaq-pulse-opt --pulse-schedule-rcp %s | FileCheck %s

module attributes {pulse.control_system = #pulse.control_system<clock_period_fs = 1000000>} {

classical.callable @tau {dispatch_key = 1 : i64, device = 0 : i64,
    args = [], results = ["i64"], latency_min_ns = 0.0,
    latency_nominal_ns = 50.0, latency_max_ns = 100.0, mode = "bounded",
    submit = "eager", dispatch_key_override}

// CHECK-LABEL: func.func @doubled
// CHECK: pulse.wait {{.*}}duration_cycles = 16 : i64, {{.*}}classical.variable
func.func @doubled() {
  %q = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit
  %d0, %t0 = pulse.get_drive_line %q : (!pulse.qudit) -> (!pulse.drive_line, !pulse.tone)
  %c40 = arith.constant 40 : i64
  %a = arith.constant 0.05 : f64
  %z = arith.constant 0.0 : f64
  %x = pulse.square %c40, %a, %z : i64, f64, f64 -> !pulse.waveform
  %d1, %t1 = pulse.drive %d0, %x, %t0 : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
  %l = classical.get_link {device = 0 : i64} : !classical.link
  %l1, %h = classical.submit %l, @tau() : () -> (!classical.link, !classical.future)
  %s0 = classical.wait %h, @tau : i64
  %s1 = arith.addi %s0, %s0 : i64
  %s2 = arith.addi %s1, %s1 : i64
  %s3 = arith.addi %s2, %s2 : i64
  %s4 = arith.addi %s3, %s3 : i64
  %s5 = arith.addi %s4, %s4 : i64
  %s6 = arith.addi %s5, %s5 : i64
  %s7 = arith.addi %s6, %s6 : i64
  %s8 = arith.addi %s7, %s7 : i64
  %s9 = arith.addi %s8, %s8 : i64
  %s10 = arith.addi %s9, %s9 : i64
  %s11 = arith.addi %s10, %s10 : i64
  %s12 = arith.addi %s11, %s11 : i64
  %s13 = arith.addi %s12, %s12 : i64
  %s14 = arith.addi %s13, %s13 : i64
  %s15 = arith.addi %s14, %s14 : i64
  %s16 = arith.addi %s15, %s15 : i64
  %s17 = arith.addi %s16, %s16 : i64
  %s18 = arith.addi %s17, %s17 : i64
  %s19 = arith.addi %s18, %s18 : i64
  %s20 = arith.addi %s19, %s19 : i64
  %s21 = arith.addi %s20, %s20 : i64
  %s22 = arith.addi %s21, %s21 : i64
  %s23 = arith.addi %s22, %s22 : i64
  %s24 = arith.addi %s23, %s23 : i64
  %s25 = arith.addi %s24, %s24 : i64
  %s26 = arith.addi %s25, %s25 : i64
  %s27 = arith.addi %s26, %s26 : i64
  %s28 = arith.addi %s27, %s27 : i64
  %s29 = arith.addi %s28, %s28 : i64
  %s30 = arith.addi %s29, %s29 : i64
  %s31 = arith.addi %s30, %s30 : i64
  %s32 = arith.addi %s31, %s31 : i64
  %s33 = arith.addi %s32, %s32 : i64
  %s34 = arith.addi %s33, %s33 : i64
  %s35 = arith.addi %s34, %s34 : i64
  %s36 = arith.addi %s35, %s35 : i64
  %s37 = arith.addi %s36, %s36 : i64
  %s38 = arith.addi %s37, %s37 : i64
  %s39 = arith.addi %s38, %s38 : i64
  %s40 = arith.addi %s39, %s39 : i64
  %dur = pulse.duration_from_int %s40 : (i64) -> !pulse.duration
  %d2 = pulse.wait %d1, %dur {classical.variable, pulse.variable_wait, pulse.min_cycles = 16 : i64, pulse.max_cycles = 1000 : i64} : (!pulse.drive_line, !pulse.duration) -> !pulse.drive_line
  %d3, %t3 = pulse.drive %d2, %x, %t1 : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
  return
}

}
