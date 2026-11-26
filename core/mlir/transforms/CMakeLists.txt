# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# BroadcastLines.cpp registers its pass itself, and its object file goes
# into the native module and cudaq-pulse-opt directly, not through this
# library: the object files of the other passes, and of the binaries that
# list the passes, are then the same with it as without it.
set(LLVM_OPTIONAL_SOURCES BroadcastLines.cpp)
add_library(CudaqPulseBroadcastLines OBJECT BroadcastLines.cpp)
add_dependencies(CudaqPulseBroadcastLines CudaqPulsePulseIncGen)
set_target_properties(CudaqPulseBroadcastLines PROPERTIES POSITION_INDEPENDENT_CODE ON)
llvm_update_compile_flags(CudaqPulseBroadcastLines)

add_mlir_dialect_library(CudaqPulsePulseTransforms
  PulseVerify.cpp
  PulseCallCheck.cpp
  PulseProgramLimits.cpp
  Canonicalize.cpp
  VirtualZ.cpp
  Fusion.cpp
  Rewrite.cpp
  ScheduleAlap.cpp
  PulseExpandCalls.cpp
  ClassicalCheck.cpp
  FrameChanges.cpp
  LowerSelects.cpp

  DEPENDS
  CudaqPulsePulseIncGen
  CudaqPulseClassicalIncGen

  LINK_LIBS PUBLIC
  CudaqPulsePulse
  CudaqPulseClassical
  MLIRIR
  MLIRPass
  MLIRSupport
  MLIRFuncDialect
  MLIRArithDialect
  MLIRSCFDialect
  MLIRTransformUtils
)
