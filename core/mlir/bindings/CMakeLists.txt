# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# STABLE_ABI takes effect only on CPython >= 3.12 with Python::SABIModule,
# which is found when scikit-build-core builds a cp312-abi3 wheel.
if(SKBUILD_SABI_VERSION AND NOT TARGET Python::SABIModule)
  message(FATAL_ERROR
    "scikit-build-core is tagging this wheel abi3 (Python ${SKBUILD_SABI_VERSION}), "
    "but FindPython did not provide Development.SABIModule, so the extension "
    "would not be built against the stable ABI.")
endif()
# Off by default: on a free-threaded interpreter the module then re-enables
# the GIL when imported. The free-threaded CI job and pyproject.toml's
# override for free-threaded wheels (cp314t) turn it on. A free-threaded
# build gets no stable ABI either way: nanobind drops STABLE_ABI there.
option(CUDAQ_PULSE_FREE_THREADED
  "Declare the extension safe to run without the GIL (free-threaded CPython only)" OFF)
set(_cudaq_pulse_free_threaded "")
if(CUDAQ_PULSE_FREE_THREADED)
  set(_cudaq_pulse_free_threaded FREE_THREADED)
elseif(SKBUILD AND Python_SOABI MATCHES "^cpython-[0-9]+t")
  message(FATAL_ERROR
    "This wheel is for the free-threaded CPython ${Python_VERSION} "
    "(${Python_SOABI}), and CUDAQ_PULSE_FREE_THREADED is OFF, so importing it "
    "would turn the GIL back on. pyproject.toml's override for free-threaded "
    "builds (if.abi-flags = \"t\") should have set it.")
endif()

# The stamp a pickled program records of the build that compiled it, from the
# native sources and the LLVM they build with (see cmake/write_build_stamp.cmake).
file(GLOB_RECURSE _cudaq_pulse_native_sources CONFIGURE_DEPENDS
  "${CUDAQ_PULSE_SOURCE_DIR}/core/mlir/*.cpp"
  "${CUDAQ_PULSE_SOURCE_DIR}/core/mlir/*.h"
  "${CUDAQ_PULSE_SOURCE_DIR}/core/mlir/*.td")
set(_build_stamp_sources "${CMAKE_CURRENT_BINARY_DIR}/build_stamp_sources.txt")
list(JOIN _cudaq_pulse_native_sources "\n" _build_stamp_list)
file(CONFIGURE OUTPUT "${_build_stamp_sources}" CONTENT "${_build_stamp_list}\n")
set(_build_stamp "${CMAKE_CURRENT_BINARY_DIR}/build_stamp.cpp")
add_custom_command(
  OUTPUT "${_build_stamp}"
  COMMAND "${CMAKE_COMMAND}" "-DSOURCES=${_build_stamp_sources}"
    "-DLLVM_VERSION=${LLVM_PACKAGE_VERSION}" "-DOUTPUT=${_build_stamp}"
    -P "${CUDAQ_PULSE_SOURCE_DIR}/cmake/write_build_stamp.cmake"
  DEPENDS ${_cudaq_pulse_native_sources} "${_build_stamp_sources}"
    "${CUDAQ_PULSE_SOURCE_DIR}/cmake/write_build_stamp.cmake"
  COMMENT "Stamping the native extension's build")

set(_cudaq_pulse_nostrip)
if(CUDAQ_PULSE_KEEP_SYMBOLS)
  set(_cudaq_pulse_nostrip NOSTRIP)
endif()
nanobind_add_module(
  _cudaq_pulse_native
  NB_STATIC
  STABLE_ABI
  ${_cudaq_pulse_free_threaded}
  ${_cudaq_pulse_nostrip}
  bindings.cpp
  "${_build_stamp}"
  ContextDialects.cpp
  LengthInference.cpp
  ModuleLoader.cpp
  ProgramLimitError.cpp
  SegmentBoundaries.cpp
  # The pulse-broadcast-lines pass, which registers itself (transforms/).
  $<TARGET_OBJECTS:CudaqPulseBroadcastLines>
)

target_link_libraries(_cudaq_pulse_native PRIVATE
  CudaqPulsePulse
  CudaqPulseClassical
  CudaqPulsePulseTransforms
  CudaqPulseQOp
  CudaqPulseCuDensityMat
  CudaqPulsePulseToQOp
  CudaqPulseQOpToCuDensityMat
  CudaqPulseCuDensityMatToLLVM
  MLIRExecutionEngine
  MLIRArithToLLVM
  MLIRFuncToLLVM
  MLIRReconcileUnrealizedCasts
  MLIRParser
  MLIRBytecodeWriter
  MLIRSCFDialect
  MLIRUBDialect
  MLIRTargetLLVMIRExport
  MLIRBuiltinToLLVMIRTranslation
  MLIRLLVMToLLVMIRTranslation
)

set_target_properties(_cudaq_pulse_native PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python/cudaq_pulse/_native"
)

# Export only PyInit__cudaq_pulse_native. Exporting the embedded LLVM/MLIR
# (about 60k symbols, many weak) made dyld bind and coalesce each of them
# at import, which took over a second on macOS, and let them interpose with
# any other LLVM in the process. HandleLLVMOptions links every module with a
# flat namespace and dynamic lookup of undefined symbols, for plugins; the
# extension uses the two-level namespace, and nanobind's linker response file
# leaves only the Python C API to resolve at load time.
string(REPLACE "-Wl,-flat_namespace -Wl,-undefined -Wl,dynamic_lookup" ""
  CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS}")
if(APPLE)
  target_link_options(_cudaq_pulse_native PRIVATE
    "LINKER:-exported_symbols_list,${CMAKE_CURRENT_SOURCE_DIR}/exports.darwin.txt")
else()
  target_link_options(_cudaq_pulse_native PRIVATE
    "LINKER:--version-script=${CMAKE_CURRENT_SOURCE_DIR}/exports.linux.map"
    "LINKER:--exclude-libs,ALL")
endif()
set_property(TARGET _cudaq_pulse_native APPEND PROPERTY LINK_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/exports.darwin.txt"
  "${CMAKE_CURRENT_SOURCE_DIR}/exports.linux.map")

set(_native_dir "${CMAKE_BINARY_DIR}/python/cudaq_pulse/_native")
nanobind_add_stub(
  _cudaq_pulse_native_stub
  MODULE _cudaq_pulse_native
  OUTPUT "${_native_dir}/_cudaq_pulse_native.pyi"
  PYTHON_PATH "${_native_dir}"
  DEPENDS _cudaq_pulse_native
)

install(
  TARGETS _cudaq_pulse_native
  LIBRARY DESTINATION cudaq_pulse/_native
  COMPONENT CudaqPulse
)
install(
  FILES "${_native_dir}/_cudaq_pulse_native.pyi"
  DESTINATION cudaq_pulse/_native
  COMPONENT CudaqPulse
)
