// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// RUN: cudaq-pulse-opt --pulse-fuse-pulses %s | FileCheck %s

// CHECK-LABEL: func.func @fuse_adjacent_squares
func.func @fuse_adjacent_squares() {
  %q = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit
  %d, %t = pulse.get_drive_line %q : (!pulse.qudit) -> (!pulse.drive_line, !pulse.tone)
  // Two adjacent square pulses with same amplitude should fuse
  %duration = arith.constant 50 : i64
  %amplitude = arith.constant 2.000000e-01 : f64
  %zero = arith.constant 0.000000e+00 : f64
  %wf1 = pulse.square %duration, %amplitude, %zero
      : i64, f64, f64 -> !pulse.waveform
  %d2, %t2 = pulse.drive %d, %wf1, %t : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
  %wf2 = pulse.square %duration, %amplitude, %zero
      : i64, f64, f64 -> !pulse.waveform
  %d3, %t3 = pulse.drive %d2, %wf2, %t2 : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
  // CHECK: arith.constant 100
  // CHECK: pulse.square
  // CHECK: fused
  return
}

// A frame change between the pulses, folded into the second drive by
// virtual Z, keeps them apart.
// CHECK-LABEL: func.func @keep_folded_frame_change
// CHECK-COUNT-2: pulse.drive
// CHECK-NOT: fused
func.func @keep_folded_frame_change() {
  %q = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit
  %d, %t = pulse.get_drive_line %q : (!pulse.qudit) -> (!pulse.drive_line, !pulse.tone)
  %duration = arith.constant 50 : i64
  %amplitude = arith.constant 2.000000e-01 : f64
  %zero = arith.constant 0.000000e+00 : f64
  %wf = pulse.square %duration, %amplitude, %zero
      : i64, f64, f64 -> !pulse.waveform
  %d2, %t2 = pulse.drive %d, %wf, %t : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
  %d3, %t3 = pulse.drive %d2, %wf, %t2 {frame_phase_offset = 3.1415926535897931 : f64} : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
  return
}

// The fused drive keeps the first drive's frame change.
// CHECK-LABEL: func.func @fused_drive_keeps_leading_frame_change
// CHECK: pulse.drive {{.*}}frame_phase_offset = 5.000000e-01
// CHECK-SAME: fused
func.func @fused_drive_keeps_leading_frame_change() {
  %q = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit
  %d, %t = pulse.get_drive_line %q : (!pulse.qudit) -> (!pulse.drive_line, !pulse.tone)
  %duration = arith.constant 50 : i64
  %amplitude = arith.constant 2.000000e-01 : f64
  %zero = arith.constant 0.000000e+00 : f64
  %wf = pulse.square %duration, %amplitude, %zero
      : i64, f64, f64 -> !pulse.waveform
  %d2, %t2 = pulse.drive %d, %wf, %t {frame_phase_offset = 5.000000e-01 : f64} : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
  %d3, %t3 = pulse.drive %d2, %wf, %t2 : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
  return
}

// A drive written into an idle window (idle replacement) stays apart from
// the kernel's own drive next to it; two written drives fuse.
// CHECK-LABEL: func.func @keep_written_drive_apart
// CHECK-COUNT-2: pulse.drive
// CHECK-NOT: fused
func.func @keep_written_drive_apart() {
  %q = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit
  %d, %t = pulse.get_drive_line %q : (!pulse.qudit) -> (!pulse.drive_line, !pulse.tone)
  %duration = arith.constant 50 : i64
  %amplitude = arith.constant 2.000000e-01 : f64
  %zero = arith.constant 0.000000e+00 : f64
  %wf = pulse.square %duration, %amplitude, %zero
      : i64, f64, f64 -> !pulse.waveform
  %d2, %t2 = pulse.drive %d, %wf, %t : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
  %d3, %t3 = pulse.drive %d2, %wf, %t2 {pulse.written} : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
  return
}

// CHECK-LABEL: func.func @fuse_written_drives
// CHECK: arith.constant 100
// CHECK: pulse.drive {{.*}}fused{{.*}}pulse.written
func.func @fuse_written_drives() {
  %q = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit
  %d, %t = pulse.get_drive_line %q : (!pulse.qudit) -> (!pulse.drive_line, !pulse.tone)
  %duration = arith.constant 50 : i64
  %amplitude = arith.constant 2.000000e-01 : f64
  %zero = arith.constant 0.000000e+00 : f64
  %wf = pulse.square %duration, %amplitude, %zero
      : i64, f64, f64 -> !pulse.waveform
  %d2, %t2 = pulse.drive %d, %wf, %t {pulse.written} : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
  %d3, %t3 = pulse.drive %d2, %wf, %t2 {pulse.written} : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
  return
}
