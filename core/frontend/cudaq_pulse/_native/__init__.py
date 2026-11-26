# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Thin wrapper around the C++ nanobind extension."""

from __future__ import annotations

try:
    from ._cudaq_pulse_native import *  # noqa: F403
except ImportError as _error:
    raise ImportError(
        "The cudaq-pulse native extension (_cudaq_pulse_native) could not be "
        "imported. Install the cudaq-pulse wheel, or build from source with "
        "`pip install .` (see CONTRIBUTING.md)."
    ) from _error
