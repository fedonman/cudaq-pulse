# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set(LLVM_TARGET_DEFINITIONS PulseOps.td)
# pulse.call is generated on its own (PulseCallOp.*.inc, compiled in
# PulseCallOp.cpp), so the other ops' generated code is as it was. The
# pattern has no `$` anchor: tablegen's command is not VERBATIM, so a `$`
# would reach the build tool. No other op's name starts with pulse.call.
mlir_tablegen(PulseOps.h.inc -gen-op-decls -op-exclude-regex=^pulse[.]call)
mlir_tablegen(PulseOps.cpp.inc -gen-op-defs -op-exclude-regex=^pulse[.]call)
mlir_tablegen(PulseCallOp.h.inc -gen-op-decls -op-include-regex=^pulse[.]call)
mlir_tablegen(PulseCallOp.cpp.inc -gen-op-defs -op-include-regex=^pulse[.]call)
mlir_tablegen(PulseDialect.h.inc -gen-dialect-decls)
mlir_tablegen(PulseDialect.cpp.inc -gen-dialect-defs)

set(LLVM_TARGET_DEFINITIONS PulseTypes.td)
mlir_tablegen(PulseTypes.h.inc -gen-typedef-decls)
mlir_tablegen(PulseTypes.cpp.inc -gen-typedef-defs)
mlir_tablegen(PulseEnums.h.inc -gen-enum-decls)
mlir_tablegen(PulseEnums.cpp.inc -gen-enum-defs)

set(LLVM_TARGET_DEFINITIONS PulseAttrs.td)
mlir_tablegen(PulseAttrs.h.inc -gen-attrdef-decls)
mlir_tablegen(PulseAttrs.cpp.inc -gen-attrdef-defs)
mlir_tablegen(PulseAttrEnums.h.inc -gen-enum-decls)
mlir_tablegen(PulseAttrEnums.cpp.inc -gen-enum-defs)

add_public_tablegen_target(CudaqPulsePulseIncGen)

# The dialect reference page, built for the documentation (pulse-docs).
set(LLVM_TARGET_DEFINITIONS PulseOps.td)
mlir_tablegen(PulseDialect.md -gen-dialect-doc -dialect=pulse)
add_custom_target(CudaqPulsePulseDocGen
  DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/PulseDialect.md)
