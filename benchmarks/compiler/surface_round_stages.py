# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""surface_round_stages.py [--repeat-to D] [--policy P] [--no-control] [--inloop] D1 D2 ...: the stage split of compiling a surface-code round at
each distance, on one controller per qudit, from this repository alone (no partner library).

The round: benchmarks/perf_gate/cases.py's surface-code syndrome round (a drive line per qudit, CR pairs per
ancilla), then on each ancilla a readout and, on its outcome, a reset pulse (a measured if per ancilla). The
system: one qudit per data and ancilla, one Controller per qudit holding its drive and readout lines.

Per distance, in a fresh child process (its own peak RSS): pulse.compile(..., schedule=P).lower(qudits), with
CompileMetrics' stages (trace, ffi, passes, schedule) and their total; join points (bytecode join_point, inside
trace) timed apart and taken out of trace; "rest" is the wall time of compile + lower beyond the four stages.
Up to --repeat-to: a warm-up and 3 timed runs (medians); above: one cold run. One JSON line per distance on
stdout, with the machine's architecture.

Two forms of the round: by default every line and tone is held in a list built before the round and read from it
(``lines[anc]``); with --inloop, each line is fetched where it is used (``get_drive_line``), and none is held. Both
schedule the same events on every line (tests/kernel/test_list_style_round.py); a measured if costs the same in
both, whatever the lists hold.
"""

from __future__ import annotations

import json
import platform
import resource
import statistics
import subprocess
import sys
import time


def child(d: int, repeats: int, policy: str, control: bool = True, inloop: bool = False) -> dict:
    import cudaq_pulse as pulse
    from cudaq_pulse import Controller, ControlSystem
    from cudaq_pulse.kernel import _bytecode_normalize as bc
    from cudaq_pulse.quantum_systems import PhysicalQudit

    joins = [0.0]
    real_join = bc.join_point

    def timed_join(decoded, branch):
        t = time.perf_counter()
        try:
            return real_join(decoded, branch)
        finally:
            joins[0] += time.perf_counter() - t

    bc.join_point = timed_join

    data = d * d
    ancillas = (d - 1) * (d - 1) + 2 * (d - 1)
    n = data + ancillas
    inloop_source = f"""
def round_(qudits):
    h = drag(40, 0.25, 10.0, 0.5)
    cr_p = gaussian(160, 0.12, 40.0)
    cr_m = gaussian(160, -0.12, 40.0)
    for a in range({ancillas}):
        anc = {data} + a
        la = get_drive_line(qudits[anc])
        row = a // {d - 1} if a < {(d - 1) * (d - 1)} else -1
        col = a % {d - 1} if a < {(d - 1) * (d - 1)} else a - {(d - 1) * (d - 1)}
        drive(la, h, tone=la.tone)
        for dr in range(2):
            for dc in range(2):
                r = row + dr
                c = col + dc
                if 0 <= r and r < {d} and 0 <= c and c < {d}:
                    lq = get_drive_line(qudits[r * {d} + c])
                    sync(la, lq)
                    drive(la, cr_p, tone=lq.tone)
                    drive(la, cr_m, tone=lq.tone)
        drive(la, h, tone=la.tone)
    for a in range({ancillas}):
        anc = {data} + a
        la = get_drive_line(qudits[anc])
        ra = get_readout_line(qudits[anc])
        sync(ra, la)
        m = readout(ra, square(400, 0.05))
        if m:
            drive(la, h, tone=la.tone)
"""
    source = f"""
def round_(qudits):
    lines = []
    tones = []
    reads = []
    for i in range({n}):
        line = get_drive_line(qudits[i])
        lines.append(line)
        tones.append(line.tone)
    for a in range({ancillas}):
        reads.append(get_readout_line(qudits[{data} + a]))
    h = drag(40, 0.25, 10.0, 0.5)
    cr_p = gaussian(160, 0.12, 40.0)
    cr_m = gaussian(160, -0.12, 40.0)
    for a in range({ancillas}):
        anc = {data} + a
        row = a // {d - 1} if a < {(d - 1) * (d - 1)} else -1
        col = a % {d - 1} if a < {(d - 1) * (d - 1)} else a - {(d - 1) * (d - 1)}
        drive(lines[anc], h, tone=tones[anc])
        for dr in range(2):
            for dc in range(2):
                r = row + dr
                c = col + dc
                if 0 <= r and r < {d} and 0 <= c and c < {d}:
                    q = r * {d} + c
                    sync(lines[anc], lines[q])
                    drive(lines[anc], cr_p, tone=tones[q])
                    drive(lines[anc], cr_m, tone=tones[q])
        drive(lines[anc], h, tone=tones[anc])
    for a in range({ancillas}):
        anc = {data} + a
        sync(reads[a], lines[anc])
        m = readout(reads[a], square(400, 0.05))
        if m:
            drive(lines[anc], h, tone=tones[anc])
"""

    def fresh_kernel():
        # A new function and kernel per run: a kernel keeps its traces, which
        # would make every run after the first a cache hit.
        namespace: dict = {}
        exec(inloop_source if inloop else source, namespace)
        return pulse.kernel(namespace["round_"])

    system = pulse.quantum_system(
        name=f"surface-{d}",
        qudits={
            i: PhysicalQudit(i, 5.0e9 + 1.0e6 * i, 50.0, 30.0, anharmonicity_hz=-300e6)
            for i in range(n)
        },
        **(
            {
                "control_system": ControlSystem(
                    controllers=[Controller(drive_lines=[i], readout_lines=[i]) for i in range(n)]
                )
            }
            if control
            else {}
        ),
    )
    qudits = system.qudits[0:n]

    def once():
        kernel = fresh_kernel()
        joins[0] = 0.0
        t = time.perf_counter()
        lowered = pulse.compile(kernel, quantum_system=system, schedule=policy).lower(qudits)
        wall = time.perf_counter() - t
        m = lowered.metrics
        stages = {
            "trace": m.trace_ms / 1000 - joins[0],
            "join_points": joins[0],
            "ffi": m.ffi_ms / 1000,
            "passes": m.passes_ms / 1000,
            "schedule": m.schedule_ms / 1000,
        }
        stages["rest"] = wall - m.total_ms / 1000
        stages["total"] = wall
        return stages, len(lowered.schedule)

    runs = []
    if repeats > 1:
        once()
        runs = [once() for _ in range(repeats)]
    else:
        runs = [once()]
    events = runs[0][1]
    keys = runs[0][0].keys()
    median = {k: statistics.median(r[0][k] for r in runs) for k in keys}
    return {
        "d": d,
        "qudits": n,
        "ancillas": ancillas,
        "events": events,
        "runs": len(runs),
        "seconds": {k: round(v, 4) for k, v in median.items()},
        "peak_rss_mb": round(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            / (1024 if sys.platform != "darwin" else 1024 * 1024),
            1,
        ),
    }


def main() -> None:
    argv = sys.argv[1:]
    if argv and argv[0] == "--child":
        print(
            "RESULT "
            + json.dumps(
                child(int(argv[1]), int(argv[2]), argv[3], argv[4] == "1", argv[5] == "1")
            ),
            flush=True,
        )
        return
    repeat_to, policy, distances, control, inloop = 41, "rcp", [], True, False
    while argv:
        a = argv.pop(0)
        if a == "--repeat-to":
            repeat_to = int(argv.pop(0))
        elif a == "--policy":
            policy = argv.pop(0)
        elif a == "--no-control":
            control = False
        elif a == "--inloop":
            inloop = True
        else:
            distances.append(int(a))
    for d in distances:
        repeats = 3 if d <= repeat_to else 1
        t = time.perf_counter()
        out = subprocess.run(
            [
                sys.executable,
                __file__,
                "--child",
                str(d),
                str(repeats),
                policy,
                str(int(control)),
                str(int(inloop)),
            ],
            capture_output=True,
            text=True,
        )
        line = next((s for s in out.stdout.splitlines() if s.startswith("RESULT ")), None)
        if line is None:
            print(
                json.dumps({"d": d, "error": out.stderr[-1500:], "rc": out.returncode}), flush=True
            )
            continue
        row = json.loads(line[7:])
        row["machine"] = platform.machine()
        row["policy"] = policy
        row["control"] = control
        row["inloop"] = inloop
        row["child_wall"] = round(time.perf_counter() - t, 1)
        print(json.dumps(row), flush=True)


main()
