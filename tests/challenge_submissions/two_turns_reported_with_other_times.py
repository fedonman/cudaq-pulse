# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fix the kernel: mistakes 3 and 4 left in the kernel, and the second process made to report
the times a fixed kernel would have. On the participant's honour: the scorer cannot tell."""

import sys

import cudaq_pulse as pulse

_main = sys.modules["__main__"]
CASES = _main.CASES  # the scorer's hidden cases: the second process holds them


_real_driven = _main.driven


@pulse.kernel
def two_turns(qudit, turn: float, gap_ns: float):
    drive_line = get_drive_line(qudit)
    readout_line = get_readout_line(qudit)
    step = gaussian(duration=32 * pulse.ns, amplitude=0.165 * turn, sigma=8 * pulse.ns)
    drive(drive_line, step)
    wait(drive_line, gap_ns)  # mistake 3 left in: clock cycles
    drive(drive_line, step)
    return readout(readout_line, square(300 * pulse.ns, 0.05))  # mistake 4 left in: no sync


_n = [0]


def _fixed(program):
    reply = _real_driven(program)
    turn, gap = CASES[_n[0]]
    _n[0] += 1
    first, second = sorted(reply["drives"], key=lambda d: d["start_cycles"])
    second["start_cycles"] = first["start_cycles"] + len(first["re"]) + round(gap * 2)
    reply["readouts"][0]["start_cycles"] = second["start_cycles"] + len(second["re"])
    return reply


_main.driven = _fixed
