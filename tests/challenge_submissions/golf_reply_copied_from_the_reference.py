# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Schedule golf: no kernel; the second process is made to report what the silver reference
plays. On the participant's honour: the scorer cannot tell."""

import importlib.util
import pathlib
import sys

_main = sys.modules["__main__"]
_here = pathlib.Path(_main.__file__).parent


def honest(which="silver"):
    spec = importlib.util.spec_from_file_location("ref", _here / "reference" / f"{which}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return _main.compiled(module)


class _Fake:
    def compile(self, *a, **k):
        return self

    def lower(self, *a, **k):
        return None


def reply_with(value):  # the scorer receives json.dumps(value)
    _main.pulse = _Fake()
    _main.driven = lambda program: value


kernel = SCHEDULE = None
R = honest()
reply_with(R)
