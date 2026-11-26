// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// pulse.call: generated on its own (PulseCallOp.*.inc), so the other ops'
// generated code is as it was before it.

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

#include "cudaq-pulse/Dialect/Pulse/PulseAttrs.h"
#include "cudaq-pulse/Dialect/Pulse/PulseDialect.h.inc"
#include "cudaq-pulse/Dialect/Pulse/PulseEnums.h.inc"
#include "cudaq-pulse/Dialect/Pulse/PulseTimingFormat.h"

#define GET_TYPEDEF_CLASSES
#include "cudaq-pulse/Dialect/Pulse/PulseTypes.h.inc"

#define GET_OP_CLASSES
#include "cudaq-pulse/Dialect/Pulse/PulseCallOp.h.inc"

#include <atomic>

using namespace mlir;

namespace pulse {

/// The types AnyTimedType admits: lines, tones, measurements, and the
/// classical dialect's links and call handles.
static bool isTimedValueType(Type type) {
  return isa<DriveLineType, ReadoutLineType, ToneType, MeasurementType>(type) ||
         type.getDialect().getNamespace() == "classical";
}

/// Set when a pulse.call is verified: the passes look for calls only in a
/// process that has one (callsVerified). Every module the passes see is
/// verified first (parsed, or loaded), and no code builds a pulse.call; code
/// that builds one and runs the passes without verifying it would have its
/// calls go unseen.
static std::atomic<bool> callVerified{false};

bool callsVerified() { return callVerified.load(std::memory_order_relaxed); }

LogicalResult CallOp::verify() {
  callVerified.store(true, std::memory_order_relaxed);
  // A call takes lines, tones, measurements, links and numbers; qudits and
  // other values do not cross into a function scheduled on its own.
  for (auto [index, arg] : llvm::enumerate(getArgs())) {
    Type type = arg.getType();
    if (isTimedValueType(type) || type.isF64() || type.isInteger(64))
      continue;
    return emitOpError("argument ")
           << index << " is a " << type
           << "; a call takes lines, tones, measurements, links, f64 and i64";
  }
  if (std::optional<int64_t> duration = getDurationCycles();
      duration && *duration < 0)
    return emitOpError("lasts ") << *duration << " cycles";
  if (getStartCycles().has_value() != getDurationCycles().has_value())
    return emitOpError("is placed with start_cycles and duration_cycles "
                       "together, or with neither");
  return success();
}

LogicalResult CallOp::verifySymbolUses(SymbolTableCollection &symbols) {
  auto callee = symbols.lookupNearestSymbolFrom<FunctionOpInterface>(
      *this, getCalleeAttr());
  if (!callee)
    return emitOpError("calls '")
           << getCallee() << "', which is not a function of this module";
  if (callee.isExternal())
    return emitOpError("calls the declaration @")
           << getCallee() << "; a call needs the callee's body";
  if (TypeRange(callee.getArgumentTypes()) != getArgs().getTypes() ||
      TypeRange(callee.getResultTypes()) != getResultTypes())
    return emitOpError("does not match @")
           << getCallee() << "'s type " << callee.getFunctionType();
  // A placed call needs the callee's own schedule: its length.
  auto length = callee->getAttrOfType<IntegerAttr>(kLengthCyclesAttr);
  if (length && length.getInt() < 0)
    return emitOpError("calls @")
           << getCallee() << ", which records a length of " << length.getInt()
           << " cycles";
  if (!getStartCycles())
    return success();
  if (!length)
    return emitOpError("is placed, but @")
           << getCallee() << " has no schedule of its own ("
           << kLengthCyclesAttr << ")";
  if (*getDurationCycles() < length.getInt())
    return emitOpError("lasts ")
           << *getDurationCycles() << " cycles, but @" << getCallee() << " is "
           << length.getInt() << " cycles long";
  return success();
}

} // namespace pulse

#define GET_OP_CLASSES
#include "cudaq-pulse/Dialect/Pulse/PulseCallOp.cpp.inc"
