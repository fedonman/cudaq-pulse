// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Lower the values a condition selects while the program runs: each
// `arith.select` of numbers or waveforms, as a measurement-conditioned if
// leaves on a name the arms set differently.
//
// A use of a select stays where the target takes that parameter at run time:
// the amplitude of a waveform or the factor of `pulse.scale` ("amplitude"),
// the phase of `pulse.shift_phase` and `pulse.set_phase` ("phase"), and the
// frequency of `pulse.shift_frequency` and `pulse.set_frequency`
// ("frequency"), for the kinds the `runtime` option lists. Arithmetic between
// the select and the op is looked through.
//
// Every other op a select reaches (a duration, a waveform's shape, a waveform
// chosen whole, anything that moves ops in time) runs in both arms of an
// `scf.if` on the select's condition instead: the first op with effects the
// select's value reaches is copied into each arm, with the pure ops between
// them, the select replaced by that arm's value. That op is the one a
// program writing the op in both arms of the if would have; a value that
// leaves a region copies the op that holds the region. Selects that are
// left without uses are erased.
//
// A function with a select carries `pulse.selected_values`: scheduling then
// starts every op once all its operands are available, so an op that reads a
// selected value, kept or copied into the arms, waits for the condition's
// measurement either way.
//
// A select of an i1 is a condition on measurement results a kernel kept in a
// value: an if on it becomes ifs on the measurements, in the order the
// condition reads them. Each copy of the if folds the condition with the
// arm's value, and takes its arm where that decides it, so a measurement is
// tested only where the condition still depends on it: the ifs a kernel
// that writes the condition in the if itself gets. Those selects alone do
// not mark the function.

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/StringSet.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/FoldUtils.h"

#include "cudaq-pulse/Transforms/Passes.h"

#include "cudaq-pulse/Dialect/Pulse/PulseDialect.h.inc"
#include "cudaq-pulse/Dialect/Pulse/PulseEnums.h.inc"

#define GET_TYPEDEF_CLASSES
#include "cudaq-pulse/Dialect/Pulse/PulseTypes.h.inc"
#define GET_OP_CLASSES
#include "cudaq-pulse/Dialect/Pulse/PulseOps.h.inc"

#include <string>
#include <vector>

namespace {

constexpr llvm::StringLiteral kSelectedValues = "pulse.selected_values";

/// The run-time parameter `use` sets, or "" for one that only a copy of its
/// op in each arm can take.
llvm::StringRef kindOf(mlir::OpOperand &use) {
  mlir::Operation *op = use.getOwner();
  auto amplitudeIf = [&](mlir::OpOperand &amplitude) {
    return &use == &amplitude ? llvm::StringRef("amplitude") : "";
  };
  if (auto square = mlir::dyn_cast<pulse::SquarePulseOp>(op))
    return &use == &square.getDurationMutable() ? "" : "amplitude";
  if (auto wf = mlir::dyn_cast<pulse::GaussianPulseOp>(op))
    return amplitudeIf(wf.getAmplitudeMutable());
  if (auto wf = mlir::dyn_cast<pulse::GaussianSquarePulseOp>(op))
    return amplitudeIf(wf.getAmplitudeMutable());
  if (auto wf = mlir::dyn_cast<pulse::DRAGPulseOp>(op))
    return amplitudeIf(wf.getAmplitudeMutable());
  if (auto wf = mlir::dyn_cast<pulse::CosinePulseOp>(op))
    return amplitudeIf(wf.getAmplitudeMutable());
  if (auto wf = mlir::dyn_cast<pulse::TanhRampOp>(op))
    return amplitudeIf(wf.getAmplitudeMutable());
  if (auto scale = mlir::dyn_cast<pulse::PulseScaleOp>(op))
    return amplitudeIf(scale.getScaleMutable());
  if (auto frame = mlir::dyn_cast<pulse::ShiftPhaseOp>(op))
    return &use == &frame.getPhaseRadMutable() ? "phase" : "";
  if (auto frame = mlir::dyn_cast<pulse::SetPhaseOp>(op))
    return &use == &frame.getPhaseRadMutable() ? "phase" : "";
  if (auto frame = mlir::dyn_cast<pulse::ShiftFrequencyOp>(op))
    return &use == &frame.getFrequencyHzMutable() ? "frequency" : "";
  if (auto frame = mlir::dyn_cast<pulse::SetFrequencyOp>(op))
    return &use == &frame.getFrequencyHzMutable() ? "frequency" : "";
  return "";
}

/// Arithmetic a selected number passes through on its way to the op that
/// takes it.
bool isArithmetic(mlir::Operation *op) {
  return op->getDialect() && op->getDialect()->getNamespace() == "arith" &&
         op->getNumRegions() == 0 && mlir::isMemoryEffectFree(op);
}

/// An op that reading a selected value only computes from it: one without
/// effects, or that only allocates its result (a new tone).
bool computesOnly(mlir::Operation *op) {
  return op->getNumRegions() == 0 &&
         !op->hasTrait<mlir::OpTrait::IsTerminator>() &&
         (mlir::isMemoryEffectFree(op) ||
          mlir::hasSingleEffect<mlir::MemoryEffects::Allocate>(op));
}

class Lowering {
public:
  Lowering(mlir::func::FuncOp function, llvm::ArrayRef<std::string> kinds)
      : function(function) {
    for (const std::string &kind : kinds)
      runtime.insert(kind);
  }

  mlir::LogicalResult run() {
    bool selects = false, conditions = false;
    function.walk([&](mlir::arith::SelectOp select) {
      if (!select.getType().isInteger(1)) {
        selects = true;
        return mlir::WalkResult::interrupt();
      }
      conditions = true;
      return mlir::WalkResult::advance();
    });
    if (conditions) {
      // The true and false a condition is built from, as constants that
      // fold with it in each copy.
      mlir::OperationFolder folder(function.getContext());
      function.walk([&](mlir::arith::CmpIOp compare) {
        if (mlir::matchPattern(compare.getLhs(), mlir::m_Constant()) &&
            mlir::matchPattern(compare.getRhs(), mlir::m_Constant()))
          (void)folder.tryToFold(compare);
      });
    }
    if (selects)
      function->setAttr(kSelectedValues,
                        mlir::UnitAttr::get(function.getContext()));
    while (true) {
      mlir::arith::SelectOp select;
      mlir::Operation *anchor = nullptr;
      function.walk([&](mlir::arith::SelectOp candidate) {
        // A select an elif nests in another is copied after it, into the
        // arm that reads it only.
        if (llvm::any_of(candidate->getUses(), [&](mlir::OpOperand &use) {
              auto outer =
                  mlir::dyn_cast<mlir::arith::SelectOp>(use.getOwner());
              return outer && use.getOperandNumber() != 0 && firstAnchor(outer);
            }))
          return mlir::WalkResult::advance();
        if (mlir::Operation *found = firstAnchor(candidate)) {
          select = candidate;
          anchor = found;
          return mlir::WalkResult::interrupt();
        }
        return mlir::WalkResult::advance();
      });
      if (failed)
        return mlir::failure();
      if (!anchor)
        break;
      if (mlir::failed(duplicate(anchor, select)))
        return mlir::failure();
      eraseDead();
    }
    eraseDead();
    return mlir::success();
  }

private:
  /// The ops `value` reaches through arithmetic, as the uses that take it.
  void takers(mlir::Value value, llvm::SmallVectorImpl<mlir::OpOperand *> &out,
              llvm::DenseSet<mlir::Value> &seen) {
    // The values being followed, the innermost last, each with the use it
    // is at: a value's uses in their order, and what an arithmetic use
    // computes followed to its end before the next use.
    struct Following {
      mlir::Value::use_iterator at, end;
    };
    llvm::SmallVector<Following, 16> following;
    auto follow = [&](mlir::Value next) {
      if (seen.insert(next).second)
        following.push_back({next.use_begin(), next.use_end()});
    };
    follow(value);
    while (!following.empty()) {
      Following &last = following.back();
      if (last.at == last.end) {
        following.pop_back();
        continue;
      }
      mlir::OpOperand &use = *last.at++;
      mlir::Operation *owner = use.getOwner();
      if (isArithmetic(owner) && !(mlir::isa<mlir::arith::SelectOp>(owner) &&
                                   use.getOperandNumber() == 0)) {
        // The first result is followed first: it is pushed last.
        for (mlir::Value result : llvm::reverse(owner->getResults()))
          follow(result);
        continue;
      }
      out.push_back(&use);
    }
  }

  /// The first op that must run in both arms of an if on `select`'s
  /// condition: the first op with effects that a use the target cannot take
  /// at run time reaches.
  mlir::Operation *firstAnchor(mlir::arith::SelectOp select) {
    llvm::SmallVector<mlir::OpOperand *> uses;
    llvm::DenseSet<mlir::Value> seen;
    takers(select.getResult(), uses, seen);
    for (mlir::OpOperand *use : uses) {
      if (runtime.contains(kindOf(*use)))
        continue;
      if (mlir::Operation *anchor = anchorOf(use->getOwner()))
        return anchor;
    }
    return nullptr;
  }

  /// The op with effects that the value of `op` reaches first.
  mlir::Operation *anchorOf(mlir::Operation *op) {
    llvm::SmallVector<mlir::Operation *> work{op};
    llvm::DenseSet<mlir::Operation *> seen;
    while (!work.empty()) {
      mlir::Operation *current = work.pop_back_val();
      if (!seen.insert(current).second)
        continue;
      if (computesOnly(current)) {
        for (mlir::Value result : current->getResults())
          for (mlir::Operation *user : result.getUsers())
            work.push_back(user);
        continue;
      }
      if (current->hasTrait<mlir::OpTrait::IsTerminator>()) {
        mlir::Operation *parent = current->getParentOp();
        if (!parent || mlir::isa<mlir::func::FuncOp>(parent)) {
          current->emitError("a value a measurement selects is returned; "
                             "use it in the program instead");
          failed = true;
          return nullptr;
        }
        return parent;
      }
      return current;
    }
    return nullptr;
  }

  /// The condition of `select`, where `anchor` is: a measurement's
  /// `pulse.is_set` made inside the anchor is made again before it.
  mlir::Value conditionAt(mlir::Operation *anchor,
                          mlir::arith::SelectOp select) {
    mlir::Value condition = select.getCondition();
    mlir::Operation *def = condition.getDefiningOp();
    if (!def || !anchor->isAncestor(def))
      return condition;
    auto isSet = mlir::dyn_cast<pulse::IsSetOp>(def);
    if (isSet) {
      mlir::Operation *measured = isSet.getMeasurement().getDefiningOp();
      if (!measured || !anchor->isAncestor(measured)) {
        mlir::OpBuilder builder(anchor);
        return pulse::IsSetOp::create(builder, isSet.getLoc(),
                                      builder.getI1Type(),
                                      isSet.getMeasurement())
            .getResult();
      }
    }
    anchor->emitError("a value a measurement selects reaches this op, "
                      "which also makes the measurement; use the value "
                      "inside the branch where it is chosen");
    failed = true;
    return {};
  }

  /// Copy `anchor` into both arms of an if on `select`'s condition, with
  /// the pure ops between them; each arm reads that arm's value instead.
  mlir::LogicalResult duplicate(mlir::Operation *anchor,
                                mlir::arith::SelectOp select) {
    mlir::Value condition = conditionAt(anchor, select);
    if (!condition)
      return mlir::failure();
    bool inside = anchor->isAncestor(select);
    llvm::SetVector<mlir::Operation *> between;
    if (!inside)
      collectBetween(anchor, select, between);

    mlir::OpBuilder builder(anchor);
    // A condition's measurement is tested where its if is, as an if on it
    // tests it.
    if (auto isSet = condition.getDefiningOp<pulse::IsSetOp>();
        isSet && select.getType().isInteger(1))
      condition = pulse::IsSetOp::create(
          builder, isSet.getLoc(), builder.getI1Type(), isSet.getMeasurement());
    auto branch = mlir::scf::IfOp::create(builder, anchor->getLoc(),
                                          anchor->getResultTypes(), condition,
                                          /*withElseRegion=*/true);
    for (bool taken : {true, false}) {
      mlir::Block *arm = taken ? branch.thenBlock() : branch.elseBlock();
      if (!arm->empty())
        arm->getTerminator()->erase();
      mlir::OpBuilder inArm = mlir::OpBuilder::atBlockEnd(arm);
      mlir::Value chosen =
          taken ? select.getTrueValue() : select.getFalseValue();
      mlir::IRMapping mapping;
      if (!inside)
        mapping.map(select.getResult(), chosen);
      for (mlir::Operation *op : between)
        inArm.clone(*op, mapping);
      mlir::Operation *copy = inArm.clone(*anchor, mapping);
      if (inside) {
        mlir::Value copied = mapping.lookup(select.getResult());
        copied.replaceAllUsesWith(mapping.lookupOrDefault(chosen));
        copied.getDefiningOp()->erase();
      }
      mlir::scf::YieldOp::create(inArm, anchor->getLoc(), copy->getResults());
      if (select.getType().isInteger(1))
        decide(arm);
    }
    anchor->replaceAllUsesWith(branch.getResults());
    for (mlir::Operation *op : between)
      dead.insert(op);
    dead.insert(select);
    if (mlir::Operation *def = select.getCondition().getDefiningOp())
      if (!anchor->isAncestor(def))
        dead.insert(def);
    dead.remove_if([&](mlir::Operation *op) { return anchor->isAncestor(op); });
    anchor->erase();
    return mlir::success();
  }

  /// Fold the pure ops of `arm`, a copy for one value of a condition's
  /// select, and replace an if they decide with the arm it takes.
  void decide(mlir::Block *arm) {
    mlir::OperationFolder folder(function.getContext());
    for (mlir::Operation &op : llvm::make_early_inc_range(*arm))
      if (isArithmetic(&op))
        (void)folder.tryToFold(&op);
    for (mlir::Operation &op : llvm::make_early_inc_range(*arm)) {
      auto branch = mlir::dyn_cast<mlir::scf::IfOp>(op);
      llvm::APInt taken;
      if (!branch || !mlir::matchPattern(branch.getCondition(),
                                         mlir::m_ConstantInt(&taken)))
        continue;
      mlir::Block *body =
          taken.isZero() ? branch.elseBlock() : branch.thenBlock();
      if (!body) {
        branch.erase();
        continue;
      }
      auto yield = mlir::cast<mlir::scf::YieldOp>(body->getTerminator());
      branch->replaceAllUsesWith(yield.getOperands());
      arm->getOperations().splice(mlir::Block::iterator(branch),
                                  body->getOperations(), body->begin(),
                                  mlir::Block::iterator(yield));
      branch.erase();
    }
  }

  /// The pure ops outside `anchor` that compute what it reads from
  /// `select`, in an order that defines each before its uses.
  void collectBetween(mlir::Operation *anchor, mlir::arith::SelectOp select,
                      llvm::SetVector<mlir::Operation *> &between) {
    // What the select's value reaches through pure ops.
    llvm::DenseSet<mlir::Operation *> reached;
    llvm::SmallVector<mlir::Operation *> work;
    for (mlir::Operation *user : select->getUsers())
      work.push_back(user);
    while (!work.empty()) {
      mlir::Operation *op = work.pop_back_val();
      if (!computesOnly(op) || anchor->isAncestor(op) ||
          !reached.insert(op).second)
        continue;
      for (mlir::Value result : op->getResults())
        for (mlir::Operation *user : result.getUsers())
          work.push_back(user);
    }
    // Of those, the ones the anchor reads, operands first.
    // The ops being placed, the innermost last, each with the operand it
    // is at: an op goes in once those its operands need are in.
    llvm::SmallVector<std::pair<mlir::Operation *, unsigned>, 16> placing;
    auto visit = [&](mlir::Value value) {
      auto open = [&](mlir::Value read) {
        mlir::Operation *def = read.getDefiningOp();
        if (def && reached.contains(def) && !between.contains(def))
          placing.emplace_back(def, 0);
      };
      open(value);
      while (!placing.empty()) {
        auto &[def, at] = placing.back();
        if (between.contains(def)) {
          placing.pop_back();
        } else if (at < def->getNumOperands()) {
          open(def->getOperand(at++));
        } else {
          between.insert(def);
          placing.pop_back();
        }
      }
    };
    anchor->walk([&](mlir::Operation *op) {
      for (mlir::Value operand : op->getOperands())
        visit(operand);
    });
  }

  /// Erase the ops copied into arms, the selects they read and their
  /// conditions, once nothing uses them.
  void eraseDead() {
    // The ops still to look at, and those erased: an op is looked at again
    // when one that used it goes, so each is looked at once per use.
    llvm::SmallVector<mlir::Operation *, 16> todo(dead.begin(), dead.end());
    llvm::DenseSet<mlir::Operation *> erased;
    while (!todo.empty()) {
      mlir::Operation *op = todo.pop_back_val();
      if (erased.contains(op) || !op->use_empty())
        continue;
      for (mlir::Value operand : op->getOperands())
        if (mlir::Operation *def = operand.getDefiningOp())
          if (computesOnly(def) &&
              !mlir::isa<pulse::QuditAllocOp, pulse::GetDriveLineOp,
                         pulse::GetReadoutLineOp>(def)) {
            dead.insert(def);
            todo.push_back(def);
          } else if (dead.contains(def)) {
            // A condition's op, dead whatever it is: looked at again too.
            todo.push_back(def);
          }
      // An op with regions (a condition's own branch) takes the uses of
      // the ops in them with it: every dead op is looked at again.
      if (op->getNumRegions())
        todo.append(dead.begin(), dead.end());
      erased.insert(op);
      op->erase();
    }
    dead.remove_if([&](mlir::Operation *op) { return erased.contains(op); });
  }

  mlir::func::FuncOp function;
  llvm::StringSet<> runtime;
  llvm::SetVector<mlir::Operation *> dead;
  bool failed = false;
};

struct PulseLowerSelectsPass
    : public mlir::PassWrapper<PulseLowerSelectsPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PulseLowerSelectsPass)

  PulseLowerSelectsPass() = default;
  PulseLowerSelectsPass(const PulseLowerSelectsPass &other)
      : PassWrapper(other) {}
  explicit PulseLowerSelectsPass(llvm::ArrayRef<std::string> kinds) {
    runtime = kinds;
  }

  llvm::StringRef getArgument() const override { return "pulse-lower-selects"; }
  llvm::StringRef getDescription() const override {
    return "Keep the selected values a target takes at run time; run every "
           "other op that reads one in both arms of an if";
  }
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::scf::SCFDialect>();
  }

  void runOnOperation() override {
    for (const std::string &kind : runtime)
      if (kind != "amplitude" && kind != "phase" && kind != "frequency") {
        getOperation().emitError("unknown run-time parameter '")
            << kind << "'; choose amplitude, phase or frequency";
        return signalPassFailure();
      }
    std::vector<std::string> kinds(runtime.begin(), runtime.end());
    if (mlir::failed(Lowering(getOperation(), kinds).run()))
      signalPassFailure();
  }

  ListOption<std::string> runtime{
      *this, "runtime",
      llvm::cl::desc("Parameters the target takes at run time: amplitude, "
                     "phase, frequency")};
};

} // namespace

namespace pulse {
std::unique_ptr<mlir::Pass> createPulseLowerSelectsPass() {
  return std::make_unique<PulseLowerSelectsPass>();
}

std::unique_ptr<mlir::Pass>
createPulseLowerSelectsPass(llvm::ArrayRef<std::string> runtime) {
  return std::make_unique<PulseLowerSelectsPass>(runtime);
}
} // namespace pulse
