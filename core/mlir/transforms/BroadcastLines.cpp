// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Broadcast drive lines: a drive line whose one tone reaches several qudits.
//
// A program is traced with such a line acquired from its lowest qudit. The
// module then lists the broadcast lines of its control system,
//
//   pulse.broadcast_lines = [{qudits = [0, 1, 2], frequency_hz = 5.1e9}]
//
// and this pass gives each `pulse.get_drive_line` of a listed line's lowest
// qudit the line's other qudits and the frequency of its tone, and drops the
// list. Only a program compiled for a control system with a broadcast line
// runs it.

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include "mlir/Pass/PassRegistry.h"

#include "cudaq-pulse/Dialect/Pulse/PulseDialect.h.inc"
#include "cudaq-pulse/Dialect/Pulse/PulseEnums.h.inc"

#define GET_TYPEDEF_CLASSES
#include "cudaq-pulse/Dialect/Pulse/PulseTypes.h.inc"
#define GET_OP_CLASSES
#include "cudaq-pulse/Dialect/Pulse/PulseOps.h.inc"

namespace {

constexpr llvm::StringLiteral kBroadcastLines = "pulse.broadcast_lines";

struct BroadcastLine {
  llvm::SmallVector<int64_t, 4> qudits;
  double frequencyHz;
};

struct BroadcastLinesPass
    : public mlir::PassWrapper<BroadcastLinesPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(BroadcastLinesPass)

  llvm::StringRef getArgument() const override {
    return "pulse-broadcast-lines";
  }
  llvm::StringRef getDescription() const override {
    return "Give each broadcast drive line its qudits and its tone's "
           "frequency";
  }

  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();
    auto listed = module->getAttrOfType<mlir::ArrayAttr>(kBroadcastLines);
    if (!listed) {
      if (module->hasAttr(kBroadcastLines)) {
        module.emitError(kBroadcastLines) << " is a list of lines";
        signalPassFailure();
      }
      return;
    }
    // By lowest qudit, the line.
    llvm::DenseMap<int64_t, BroadcastLine> lines;
    for (mlir::Attribute entry : listed) {
      auto fields = mlir::dyn_cast<mlir::DictionaryAttr>(entry);
      auto qudits =
          fields ? fields.getAs<mlir::ArrayAttr>("qudits") : mlir::ArrayAttr();
      auto frequency = fields ? fields.getAs<mlir::FloatAttr>("frequency_hz")
                              : mlir::FloatAttr();
      BroadcastLine line;
      bool valid = qudits && frequency && qudits.size() > 1;
      for (mlir::Attribute qudit :
           valid ? qudits.getValue() : llvm::ArrayRef<mlir::Attribute>()) {
        auto index = mlir::dyn_cast<mlir::IntegerAttr>(qudit);
        if (!index || index.getInt() < 0 ||
            (!line.qudits.empty() && index.getInt() <= line.qudits.back())) {
          valid = false;
          break;
        }
        line.qudits.push_back(index.getInt());
      }
      if (!valid) {
        module.emitError(kBroadcastLines)
            << " lists lines as {qudits = [two or more indices, ascending], "
               "frequency_hz = a float}";
        return signalPassFailure();
      }
      line.frequencyHz = frequency.getValueAsDouble();
      int64_t lowest = line.qudits.front();
      lines[lowest] = std::move(line);
    }

    llvm::SmallVector<pulse::GetDriveLineOp> acquisitions;
    module.walk([&](pulse::GetDriveLineOp op) {
      auto alloc = op.getQudit().getDefiningOp<pulse::QuditAllocOp>();
      if (alloc && op.getOthers().empty() &&
          lines.count(alloc.getQuditAttr().getInt()))
        acquisitions.push_back(op);
    });
    for (pulse::GetDriveLineOp op : acquisitions) {
      auto first = op.getQudit().getDefiningOp<pulse::QuditAllocOp>();
      const BroadcastLine &line = lines[first.getQuditAttr().getInt()];
      mlir::OpBuilder builder(op);
      llvm::SmallVector<mlir::Value, 4> operands{op.getQudit()};
      for (int64_t qudit : llvm::ArrayRef(line.qudits).drop_front()) {
        // The qudit's allocation when one stands before the line in its
        // block, else a new one.
        mlir::Value found;
        for (mlir::Operation &earlier : *op->getBlock()) {
          if (&earlier == op.getOperation())
            break;
          auto alloc = mlir::dyn_cast<pulse::QuditAllocOp>(earlier);
          if (alloc && alloc.getQuditAttr().getInt() == qudit) {
            found = alloc.getResult();
            break;
          }
        }
        if (!found) {
          auto alloc = pulse::QuditAllocOp::create(
              builder, op->getLoc(), first.getResult().getType(),
              builder.getI64IntegerAttr(qudit));
          found = alloc.getResult();
        }
        operands.push_back(found);
      }
      op->setOperands(operands);
      op->setAttr("frequency_hz", builder.getF64FloatAttr(line.frequencyHz));
    }
    module->removeAttr(kBroadcastLines);
  }
};

// Registered where it is defined, as a source of each binary that runs it:
// the list of passes the other binaries share (registerPulsePasses) stays as
// it is.
mlir::PassRegistration<BroadcastLinesPass> registration;

} // namespace
