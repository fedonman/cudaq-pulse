//===- ContextDialects.h - The dialects of a shared context ---------------===//
//
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
//
// Which dialects a context that modules share is made with, and how a module
// reaches a context that has the others. These are in a translation unit of
// their own so that the code compiling a kernel is built as it is without
// them.
//
//===----------------------------------------------------------------------===//

#ifndef CUDAQ_PULSE_BINDINGS_CONTEXTDIALECTS_H
#define CUDAQ_PULSE_BINDINGS_CONTEXTDIALECTS_H

#include "llvm/ADT/STLFunctionalExtras.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/OwningOpRef.h"

namespace mlir {
class PassManager;
} // namespace mlir

namespace cudaq_pulse {

/// The dialects of a context that the modules of compiled kernels share:
/// every dialect an operation on such a module can meet or make in the
/// module's own context. The dialects left out are the ones that only the
/// lowering to LLVM makes; that lowering works on a copy in a context of the
/// calling thread (copyForLowering()), and a pipeline that would
/// make them in a shared context is refused (requireLoadedDialects()).
///
/// MLIR registers a dialect's operations in tables of the context that other
/// threads read without a lock, so a context that more than one thread can
/// reach must never load a dialect. A shared context therefore loads all of
/// these when it is made, before any module is built in it, and its registry
/// names no other: text or bytecode that names another dialect does not parse
/// in it, rather than loading that dialect under the threads using it.
mlir::DialectRegistry sharedContextDialects();

/// `module` copied, through bytecode as a pickled module is, into a context
/// with every dialect of `registry()` loaded that the calling thread keeps
/// for its lowerings: no other thread reaches that context, so nothing in it
/// needs a lock, and it runs passes on the calling thread only. Making one
/// costs far more than a lowering needs of it, so the thread keeps it, and
/// replaces it once it has served 256 copies or 2^20 packed words of them
/// (bytes / 8), a shared context's default limits, so that what the copies
/// uniqued in it is freed. The copy and anything else made in its context
/// must be gone before the thread's next call. Throws std::runtime_error if
/// the module cannot be written or read back. The caller holds whatever
/// keeps `module` from changing.
mlir::OwningOpRef<mlir::ModuleOp>
copyForLowering(mlir::ModuleOp module,
                llvm::function_ref<mlir::DialectRegistry()> registry);

/// Throws std::runtime_error if a pass of `passes` declares a dialect that
/// `context` has not loaded: running it would load the dialect into a
/// context other threads may be working in. `what` starts the message.
void requireLoadedDialects(const mlir::PassManager &passes,
                           mlir::MLIRContext *context, const char *what);

} // namespace cudaq_pulse

#endif // CUDAQ_PULSE_BINDINGS_CONTEXTDIALECTS_H
