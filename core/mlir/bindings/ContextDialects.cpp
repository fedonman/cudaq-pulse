//===- ContextDialects.cpp - The dialects of a shared context ------------===//
//
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "ContextDialects.h"

#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"

#include "llvm/Support/raw_ostream.h"

#include <memory>
#include <stdexcept>
#include <string>

#include "cudaq-pulse/Dialect/Classical/ClassicalDialect.h.inc"
#include "cudaq-pulse/Dialect/Pulse/PulseDialect.h.inc"
#include "cudaq-pulse/Dialect/QOp/QOpDialect.h.inc"

namespace cudaq_pulse {

mlir::DialectRegistry sharedContextDialects() {
  mlir::DialectRegistry registry;
  // What a kernel's packed program decodes to and the passes of a compile
  // make.
  registry.insert<pulse::PulseDialect>();
  registry.insert<classical::ClassicalDialect>();
  registry.insert<mlir::arith::ArithDialect>();
  registry.insert<mlir::func::FuncDialect>();
  registry.insert<mlir::scf::SCFDialect>();
  registry.insert<mlir::ub::UBDialect>();
  // What simulation lowers a copy to in the module's own context.
  registry.insert<qop::QOpDialect>();
  // Left out: the cuDensityMat and LLVM dialects, which only the lowering to
  // LLVM makes. Loading the LLVM dialect is most of loading them all.
  return registry;
}

mlir::OwningOpRef<mlir::ModuleOp>
copyForLowering(mlir::ModuleOp module,
                llvm::function_ref<mlir::DialectRegistry()> registry) {
  std::string bytes;
  llvm::raw_string_ostream os(bytes);
  if (mlir::failed(mlir::writeBytecodeToFile(
          module, os, mlir::BytecodeWriterConfig("cudaq-pulse"))))
    throw std::runtime_error("cannot write the module as bytecode");

  // The defaults of a shared context: 256 modules, 2^20 packed words, a
  // module counted as its bytecode's bytes / 8 as a pickled one is.
  constexpr int kMostUses = 256;
  constexpr size_t kMostWords = size_t(1) << 20;
  struct Kept {
    std::unique_ptr<mlir::MLIRContext> context;
    int uses = 0;
    size_t words = 0;
  };
  thread_local Kept kept;
  if (!kept.context || ++kept.uses > kMostUses || kept.words >= kMostWords) {
    // Without threads of its own: its passes run on the calling thread, and
    // a forked child can go on using it.
    kept.context = std::make_unique<mlir::MLIRContext>(
        registry(), mlir::MLIRContext::Threading::DISABLED);
    kept.context->loadAllAvailableDialects();
    kept.uses = 1;
    kept.words = 0;
  }
  kept.words += bytes.size() / 8;

  // The module was valid or not where it came from; what runs on the copy
  // verifies it as it would have verified the original.
  mlir::ParserConfig config(kept.context.get(), /*verifyAfterParse=*/false);
  auto copy = mlir::parseSourceString<mlir::ModuleOp>(bytes, config);
  if (!copy)
    throw std::runtime_error("cannot read the module back from bytecode");
  return copy;
}

void requireLoadedDialects(const mlir::PassManager &passes,
                           mlir::MLIRContext *context, const char *what) {
  mlir::DialectRegistry needed;
  passes.getDependentDialects(needed);
  for (llvm::StringRef name : needed.getDialectNames())
    if (!context->getLoadedDialect(name))
      throw std::runtime_error(
          std::string(what) + ": a pass makes operations of the '" +
          name.str() +
          "' dialect, which the context of a compiled module does not have; "
          "lower the module to that level instead (mlir(level)), or run the "
          "pass on a module parsed from its text");
}

} // namespace cudaq_pulse
