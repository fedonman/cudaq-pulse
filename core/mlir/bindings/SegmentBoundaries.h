//===- SegmentBoundaries.h - The ops that can open a segment --------------===//
//
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
//
// The walk behind PulseModule::segment_boundaries(), which asks only where
// segments open. It is in a translation unit of its own to keep it out of
// bindings.cpp, where it would be one more instantiation of
// SegmentWalk::run beside the walks that list a program's events.
//
//===----------------------------------------------------------------------===//

#ifndef CUDAQ_PULSE_BINDINGS_SEGMENTBOUNDARIES_H
#define CUDAQ_PULSE_BINDINGS_SEGMENTBOUNDARIES_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include <cstdint>

namespace cudaq_pulse {

/// `visit(op, iteration)` for the timed ops, syncs and classical calls of
/// `block` in program order, as the event listings visit them (once per
/// iteration of the loops around them, both arms of each branch), but for
/// the iterations of a loop that holds no op with the attribute `variable`
/// or `blocking`: only such an op opens a segment, so no iteration of that
/// loop opens one, and none of its ops is visited. A loop that holds one is
/// walked once per iteration, as each may open segments.
void walkSegmentBoundaries(
    mlir::Block &block, mlir::StringAttr variable, mlir::StringAttr blocking,
    llvm::function_ref<void(mlir::Operation *, llvm::ArrayRef<int64_t>)> visit);

} // namespace cudaq_pulse

#endif // CUDAQ_PULSE_BINDINGS_SEGMENTBOUNDARIES_H
