// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// The Pulse dialect's attributes (PulseAttrs.td) and the module and function
// attributes made of them, read by name.

#pragma once

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Operation.h"

#include <algorithm>
#include <optional>

#include "cudaq-pulse/Dialect/Pulse/PulseAttrEnums.h.inc"

#define GET_ATTRDEF_CLASSES
#include "cudaq-pulse/Dialect/Pulse/PulseAttrs.h.inc"

namespace pulse {

/// The module attribute that describes the quantum system a program plays on.
inline constexpr llvm::StringLiteral kQuantumSystemAttr =
    "pulse.quantum_system";

/// A quantum system of which a compiled module knows the frequencies of
/// qudits 0 to `frequencies.size() - 1`.
inline QuantumSystemAttr quantumSystemAt(mlir::MLIRContext *context,
                                         llvm::ArrayRef<double> frequencies) {
  return QuantumSystemAttr::get(
      context,
      frequencies.empty() ? mlir::DenseF64ArrayAttr()
                          : mlir::DenseF64ArrayAttr::get(context, frequencies),
      {}, {}, {}, {}, {}, {});
}

/// The module attribute that describes the control system a program plays
/// on: its clock, lines and controllers.
inline constexpr llvm::StringLiteral kControlSystemAttr =
    "pulse.control_system";

inline ControlSystemAttr controlSystemOf(mlir::Operation *module) {
  return module->getAttrOfType<ControlSystemAttr>(kControlSystemAttr);
}

/// The program clock of `module` in GHz; 1 without a control system.
inline double clockGHz(mlir::Operation *module) {
  ControlSystemAttr system = controlSystemOf(module);
  return system ? system.clockGHz() : 1.0;
}

/// A control system of a `clockGHz` clock and nothing else; null when the
/// clock has no exact period.
inline ControlSystemAttr controlSystemAt(mlir::MLIRContext *context,
                                         double clockGHz) {
  auto period = ControlSystemAttr::periodFs(clockGHz);
  if (!period)
    return {};
  return ControlSystemAttr::get(context, period->first, period->second, {}, {});
}

/// Whether some controller of `module` keeps its tones' frames from one shot
/// to the next.
inline bool persistentFrames(mlir::Operation *module) {
  ControlSystemAttr system = controlSystemOf(module);
  return system && llvm::any_of(system.getControllers(), [](auto controller) {
           return controller.getPersistentFrames();
         });
}

/// Whether some controller of `module` plays nothing while one of its calls
/// is out.
inline bool blocksOnCalls(mlir::Operation *module) {
  ControlSystemAttr system = controlSystemOf(module);
  return system && llvm::any_of(system.getControllers(), [](auto controller) {
           return controller.getBlocksOnCalls();
         });
}

/// The longest a program of `module` may be, in cycles, if it is bounded.
inline std::optional<int64_t> programBudget(mlir::Operation *module) {
  ControlSystemAttr system = controlSystemOf(module);
  std::optional<int64_t> budget;
  if (system)
    for (ControllerAttr controller : system.getControllers())
      if (int64_t cycles = controller.getMaxDurationCycles())
        budget = budget ? std::min(*budget, cycles) : cycles;
  return budget;
}

/// The function attribute holding a function's length in clock cycles once
/// it is scheduled on its own: the end of its last timed op. A `pulse.call`
/// lasts at least that long.
inline constexpr llvm::StringLiteral kLengthCyclesAttr = "pulse.length_cycles";

/// The function attribute that says how far a function is scheduled.
inline constexpr llvm::StringLiteral kScheduleStateAttr =
    "pulse.schedule_state";

/// How far `function` is scheduled: unscheduled without the attribute.
inline ScheduleState scheduleStateOf(mlir::Operation *function) {
  if (auto attr =
          function->getAttrOfType<ScheduleStateAttr>(kScheduleStateAttr))
    return attr.getValue();
  return ScheduleState::Unscheduled;
}

} // namespace pulse
