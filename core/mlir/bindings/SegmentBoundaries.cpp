//===- SegmentBoundaries.cpp - The ops that can open a segment ------------===//
//
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "SegmentBoundaries.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Visitors.h"

// The dialects' generated classes, in the order their definitions need,
// and after them the header that reads a loop's trip count.
// clang-format off
#include "cudaq-pulse/Dialect/Pulse/PulseDialect.h.inc"
#include "cudaq-pulse/Dialect/Pulse/PulseEnums.h.inc"

#define GET_TYPEDEF_CLASSES
#include "cudaq-pulse/Dialect/Pulse/PulseTypes.h.inc"

#define GET_OP_CLASSES
#include "cudaq-pulse/Dialect/Pulse/PulseAttrs.h"
#include "cudaq-pulse/Dialect/Pulse/PulseOps.h.inc"

#include "cudaq-pulse/Dialect/Classical/ClassicalDialect.h.inc"
#define GET_TYPEDEF_CLASSES
#include "cudaq-pulse/Dialect/Classical/ClassicalTypes.h.inc"
#define GET_OP_CLASSES
#include "cudaq-pulse/Dialect/Classical/ClassicalOps.h.inc"

#include "cudaq-pulse/Analysis/Timing.h"
// clang-format on

namespace {

struct Walk {
  mlir::StringAttr variable, blocking;
  llvm::function_ref<void(mlir::Operation *, llvm::ArrayRef<int64_t>)> visit;
  llvm::SmallVector<int64_t, 2> iteration;
  /// Whether each loop met holds an op that can open a segment.
  llvm::SmallDenseMap<mlir::Operation *, bool, 4> opening;

  bool opens(mlir::scf::ForOp loop) {
    auto [it, inserted] = opening.try_emplace(loop, false);
    if (inserted)
      it->second = loop->walk([&](mlir::Operation *inner) {
                         return inner->getDiscardableAttr(variable) ||
                                        inner->getDiscardableAttr(blocking)
                                    ? mlir::WalkResult::interrupt()
                                    : mlir::WalkResult::advance();
                       })
                       .wasInterrupted();
    return it->second;
  }

  void run(mlir::Block &block) {
    for (mlir::Operation &op : block) {
      if (mlir::isa<pulse::DriveOp, pulse::ReadoutOp, pulse::WaitOp,
                    pulse::SyncOp, classical::SubmitOp, classical::WaitOp>(
              op)) {
        visit(&op, iteration);
      } else if (auto loop = mlir::dyn_cast<mlir::scf::ForOp>(op)) {
        if (!opens(loop))
          continue;
        int64_t trips = pulse::constantTripCount(loop).value_or(0);
        iteration.push_back(0);
        for (int64_t k = 0; k < trips; ++k) {
          iteration.back() = k;
          run(*loop.getBody());
        }
        iteration.pop_back();
      } else {
        for (mlir::Region &region : op.getRegions())
          for (mlir::Block &nested : region)
            run(nested);
      }
    }
  }
};

} // namespace

void cudaq_pulse::walkSegmentBoundaries(
    mlir::Block &block, mlir::StringAttr variable, mlir::StringAttr blocking,
    llvm::function_ref<void(mlir::Operation *, llvm::ArrayRef<int64_t>)>
        visit) {
  Walk walk{variable, blocking, visit, llvm::SmallVector<int64_t, 2>(),
            llvm::SmallDenseMap<mlir::Operation *, bool, 4>()};
  walk.run(block);
}
