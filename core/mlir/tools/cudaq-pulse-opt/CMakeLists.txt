# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set(LLVM_LINK_COMPONENTS Support)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

add_llvm_executable(cudaq-pulse-opt
  cudaq-pulse-opt.cpp
  # The pulse-broadcast-lines pass, which registers itself (transforms/).
  $<TARGET_OBJECTS:CudaqPulseBroadcastLines>
)

# --version prints the package's version (cmake/CudaqPulseVersion.cmake) and
# the LLVM it was built with.
file(CONFIGURE
  OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/include/CudaqPulseOptVersion.h"
  CONTENT "#define CUDAQ_PULSE_VERSION \"@CUDAQ_PULSE_PACKAGE_VERSION@\"\n#define CUDAQ_PULSE_LLVM_VERSION \"@LLVM_PACKAGE_VERSION@\"\n"
  @ONLY)
target_include_directories(cudaq-pulse-opt PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/include")

target_link_libraries(cudaq-pulse-opt PRIVATE
  CudaqPulsePulse
  CudaqPulseClassical
  CudaqPulsePulseTransforms
  CudaqPulseQOp
  CudaqPulseCuDensityMat
  CudaqPulsePulseToQOp
  CudaqPulseQOpToCuDensityMat
  CudaqPulseCuDensityMatToLLVM
  MLIRFuncDialect
  MLIRArithDialect
  MLIRSCFDialect
  MLIRUBDialect
  MLIRArithToLLVM
  MLIRFuncToLLVM
  MLIRLLVMDialect
  MLIRReconcileUnrealizedCasts
  MLIRRegisterAllPasses
  MLIROptLib
  MLIRPass
  MLIRIR
  MLIRParser
  MLIRSupport
  MLIRTransformUtils
)

# MLIR is linked in statically, so the driver loads no library of its own.
# add_llvm_executable's run paths (../lib, and on macOS the toolchain's
# absolute lib directory) would only point an installed driver at the
# build machine's directories.
set_target_properties(cudaq-pulse-opt PROPERTIES
  BUILD_RPATH ""
  INSTALL_RPATH ""
  SKIP_BUILD_RPATH ON)
llvm_update_compile_flags(cudaq-pulse-opt)
# Pass plugins (--load-pass-plugin) resolve MLIR symbols from this executable.
export_executable_symbols_for_plugins(cudaq-pulse-opt)
mlir_check_all_link_libraries(cudaq-pulse-opt)

# The cudaq-pulse-tools wheel (packaging/cudaq-pulse-tools) installs this
# component into its scripts, which pip puts on PATH.
if(SKBUILD)
  set(_cudaq_pulse_opt_destination "${SKBUILD_SCRIPTS_DIR}")
else()
  set(_cudaq_pulse_opt_destination "${CMAKE_INSTALL_BINDIR}")
endif()
install(
  PROGRAMS "$<TARGET_FILE:cudaq-pulse-opt>"
  DESTINATION "${_cudaq_pulse_opt_destination}"
  COMPONENT CudaqPulseOpt
)
# install(TARGETS) would strip with `strip -u -r` on macOS, which keeps only
# the symbols the driver itself references, and pass plugins could no longer
# resolve MLIR against it. `strip -x` keeps the global symbols.
if(IS_ABSOLUTE "${_cudaq_pulse_opt_destination}")
  set(_cudaq_pulse_opt_installed "\$ENV{DESTDIR}${_cudaq_pulse_opt_destination}/cudaq-pulse-opt")
else()
  set(_cudaq_pulse_opt_installed
    "\$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/${_cudaq_pulse_opt_destination}/cudaq-pulse-opt")
endif()
if(APPLE)
  set(_cudaq_pulse_opt_strip_flags "-x")
else()
  set(_cudaq_pulse_opt_strip_flags "")
endif()
if(CMAKE_STRIP)
  set(_cudaq_pulse_opt_strip "
    execute_process(
      COMMAND \"${CMAKE_STRIP}\" ${_cudaq_pulse_opt_strip_flags} \"${_cudaq_pulse_opt_installed}\"
      COMMAND_ERROR_IS_FATAL ANY)")
else()
  set(_cudaq_pulse_opt_strip "
    message(WARNING \"Installing cudaq-pulse-opt unstripped: CMake found no strip \"
      \"tool. Configure with -DCMAKE_STRIP=<path to strip> to strip it.\")")
endif()
install(CODE "
  if(CMAKE_INSTALL_DO_STRIP)${_cudaq_pulse_opt_strip}
  endif()"
  COMPONENT CudaqPulseOpt
)

include(CudaqPulseNotices)
cudaq_pulse_third_party_notices(
  NOTICES "${CUDAQ_PULSE_SOURCE_DIR}/packaging/cudaq-pulse-tools/THIRD_PARTY_NOTICES"
  COMPONENT CudaqPulseOpt
  TARGETS cudaq-pulse-opt)
