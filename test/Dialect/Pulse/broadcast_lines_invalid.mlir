// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// A drive line of several qudits names each once, lowest index first, and
// the list of a module's broadcast lines has two or more qudits per line.
// RUN: cudaq-pulse-opt --split-input-file --verify-diagnostics --pulse-broadcast-lines %s

func.func @out_of_order() {
  %q0 = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit
  %q1 = pulse.qudit_alloc {qudit = 1 : i64} : !pulse.qudit
  // expected-error @+1 {{'pulse.get_drive_line' op a drive line of several qudits names each once, lowest index first; got qudit 0 after qudit 1}}
  %d, %t = pulse.get_drive_line %q1, %q0 {frequency_hz = 5.000000e+09 : f64} : (!pulse.qudit, !pulse.qudit) -> (!pulse.drive_line, !pulse.tone)
  return
}

// -----

func.func @twice() {
  %q0 = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit
  // expected-error @+1 {{'pulse.get_drive_line' op a drive line of several qudits names each once, lowest index first; got qudit 0 after qudit 0}}
  %d, %t = pulse.get_drive_line %q0, %q0 {frequency_hz = 5.000000e+09 : f64} : (!pulse.qudit, !pulse.qudit) -> (!pulse.drive_line, !pulse.tone)
  return
}

// -----

func.func @run_time(%q1 : !pulse.qudit) {
  %q0 = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit
  // expected-error @+1 {{'pulse.get_drive_line' op a drive line of several qudits names qudits of the quantum system (pulse.qudit_alloc)}}
  %d, %t = pulse.get_drive_line %q0, %q1 {frequency_hz = 5.000000e+09 : f64} : (!pulse.qudit, !pulse.qudit) -> (!pulse.drive_line, !pulse.tone)
  return
}

// -----

// expected-error @+1 {{pulse.broadcast_lines lists lines as {qudits = [two or more indices, ascending], frequency_hz = a float}}}
module attributes {pulse.broadcast_lines = [{qudits = [1, 0], frequency_hz = 5.1e9 : f64}]} {
  func.func @main() {
    return
  }
}

// -----

// expected-error @+1 {{pulse.broadcast_lines lists lines as {qudits = [two or more indices, ascending], frequency_hz = a float}}}
module attributes {pulse.broadcast_lines = [{qudits = [0], frequency_hz = 5.1e9 : f64}]} {
  func.func @main() {
    return
  }
}

// -----

// expected-error @+1 {{pulse.broadcast_lines lists lines as {qudits = [two or more indices, ascending], frequency_hz = a float}}}
module attributes {pulse.broadcast_lines = [{qudits = [0, 1]}]} {
  func.func @main() {
    return
  }
}
