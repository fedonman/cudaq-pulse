# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_mlir_dialect_library(CudaqPulsePulse
  PulseAttrs.cpp
  PulseCallOp.cpp
  PulseDialect.cpp
  PulseOps.cpp
  PulseTypes.cpp

  DEPENDS
  CudaqPulsePulseIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRIR
  MLIRSupport
  MLIRBytecodeOpInterface
  MLIRFunctionInterfaces
)
