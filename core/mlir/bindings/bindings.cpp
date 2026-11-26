// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// nanobind Python bindings for cudaq-pulse MLIR dialects and passes.

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <map>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <thread>
#include <unordered_map>

#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ExecutionEngine/Orc/Mangling.h"
#include "llvm/Support/TargetSelect.h"

#include <dlfcn.h>
#include <pthread.h>

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

#include "ContextDialects.h"
#include "LengthInference.h"
#include "ModuleLoader.h"
#include "SegmentBoundaries.h"
#include "cudaq-pulse/Analysis/ClassicalTiming.h"
#include "cudaq-pulse/Analysis/Execution.h"
#include "cudaq-pulse/Analysis/LineGrid.h"
#include "cudaq-pulse/Analysis/LineTrace.h"
#include "cudaq-pulse/Analysis/Timing.h"
#include "cudaq-pulse/Dialect/Classical/Answers.h"
#include "cudaq-pulse/Runtime/WaveformCallback.h"
#include "cudaq-pulse/Transforms/Passes.h"

#include "cudaq-pulse/Dialect/CuDensityMat/CuDensityMatDialect.h.inc"
#include "cudaq-pulse/Dialect/QOp/QOpDialect.h.inc"

namespace nb = nanobind;

// Generated at build time by cmake/write_build_stamp.cmake.
extern const char *const cudaqPulseBuildStamp;

namespace {

/// Item `index` of `list`, which has more items than that. Without the GIL
/// another thread could replace and free a borrowed item, so the item is
/// then a new reference; with it, the item is borrowed from `list`, which
/// the caller keeps alive. Wheels build against the limited API
/// (Py_LIMITED_API), where NB_LIST_GET_ITEM is PyList_GetItem, which can
/// fail.
#ifdef Py_GIL_DISABLED
nb::object listItem(nb::handle list, size_t index) {
  PyObject *item =
      PyList_GetItemRef(list.ptr(), static_cast<Py_ssize_t>(index));
  if (!item)
    throw nb::python_error();
  return nb::steal(item);
}
#else
nb::handle listItem(nb::handle list, size_t index) {
  PyObject *item = NB_LIST_GET_ITEM(list.ptr(), static_cast<Py_ssize_t>(index));
#ifdef Py_LIMITED_API
  if (!item)
    throw nb::python_error();
#endif
  return item;
}
#endif

mlir::DialectRegistry makeRegistry() {
  mlir::DialectRegistry reg;
  reg.insert<pulse::PulseDialect>();
  reg.insert<classical::ClassicalDialect>();
  reg.insert<qop::QOpDialect>();
  reg.insert<cudm::CuDensityMatDialect>();
  reg.insert<mlir::arith::ArithDialect>();
  reg.insert<mlir::func::FuncDialect>();
  reg.insert<mlir::scf::SCFDialect>();
  reg.insert<mlir::ub::UBDialect>();
  reg.insert<mlir::LLVM::LLVMDialect>();
  return reg;
}

// Packed-buffer opcodes; they must match the opcodes in
// core/frontend/cudaq_pulse/kernel/packed_ir_builder.py.
constexpr int kOpAllocDrive = 0;
constexpr int kOpAllocReadout = 1;
constexpr int kOpAllocTone = 2;
constexpr int kOpWfGaussian = 3;
constexpr int kOpWfSquare = 4;
constexpr int kOpWfDrag = 5;
constexpr int kOpWfCosine = 6;
constexpr int kOpWfTanhRamp = 7;
constexpr int kOpWfGaussSquare = 8;
constexpr int kOpWfCustom = 9;
constexpr int kOpDrive = 10;
constexpr int kOpReadout = 11;
constexpr int kOpSync = 12;
constexpr int kOpWait = 13;
constexpr int kOpShiftPhase = 14;
constexpr int kOpSetPhase = 15;
constexpr int kOpShiftFreq = 16;
constexpr int kOpSetFreq = 17;
constexpr int kOpParam = 18;
constexpr int kOpNumConst = 19;
constexpr int kOpNumBinary = 20;
constexpr int kOpNumNeg = 21;
constexpr int kOpNumCast = 22;
constexpr int kOpWfCustomSamples = 23;
constexpr int kOpWfAdd = 24;
constexpr int kOpWfSub = 25;
constexpr int kOpWfMul = 26;
constexpr int kOpWfScale = 27;
constexpr int kOpWfNeg = 28;
constexpr int kOpForBegin = 29;
constexpr int kOpForEnd = 30;
constexpr int kOpIfBegin = 31;
constexpr int kOpElse = 32;
constexpr int kOpIfEnd = 33;
constexpr int kOpBlockBegin = 34;
constexpr int kOpBlockEnd = 35;
constexpr int kOpClassicalDeclare = 36;
constexpr int kOpClassicalSubmit = 37;
constexpr int kOpClassicalWait = 38;
constexpr int kOpClassicalCompare = 39;
constexpr int kOpVariableWait = 40;
constexpr int kOpSelect = 41;
constexpr int kOpMaxCycles = 42;
constexpr int kOpSite = 43;
// A payload length of 0xFF means the length is in the next word.
constexpr int kExtendedPayload = 0xFF;

/// a + b and a * b, or nothing where they overflow. Each builtin is the
/// condition of its own if: a call's arguments run in no fixed order, so
/// one argument must not read what another writes.
template <typename T>
std::optional<T> checkedAdd(T a, T b) {
  T r{};
  if (__builtin_add_overflow(a, b, &r))
    return std::nullopt;
  return r;
}

template <typename T>
std::optional<T> checkedMul(T a, T b) {
  T r{};
  if (__builtin_mul_overflow(a, b, &r))
    return std::nullopt;
  return r;
}

/// A literal in a packed program that its op cannot take. `word` is the
/// offset of the record that holds it, which the frontend maps back to the
/// kernel line that wrote it.
struct PackedValueError : std::invalid_argument {
  PackedValueError(const std::string &what, size_t word)
      : std::invalid_argument(what), word(word) {}
  size_t word;
};

/// What the exports cannot say yet; NotImplementedError in Python.
struct NotSupportedYet : std::runtime_error {
  using std::runtime_error::runtime_error;
};

/// Refuse a readout whose mode the exports do not support yet: only iq.
inline void requireExportedMode(mlir::Operation *op) {
  if (auto readout = mlir::dyn_cast<pulse::ReadoutOp>(op);
      readout && readout.getMode() != pulse::ReadoutMode::IQ) [[unlikely]]
    throw NotSupportedYet("readout mode " +
                          pulse::stringifyReadoutMode(readout.getMode()).str() +
                          " is not supported by the exports yet; only iq is");
}

/// `value` as Python's repr() writes a float.
[[gnu::cold]] std::string pythonFloat(double value) {
  if (std::isnan(value))
    return "nan";
  if (std::isinf(value))
    return value > 0 ? "inf" : "-inf";
  char text[32];
  for (int digits = 1; digits <= 17; ++digits) {
    std::snprintf(text, sizeof text, "%.*g", digits, value);
    if (std::strtod(text, nullptr) == value)
      break;
  }
  std::string result(text);
  if (result.find_first_of(".en") == std::string::npos)
    result += ".0";
  return result;
}

// Out of line, so that the decode loop builds no strings until one fails.
[[noreturn, gnu::cold, gnu::noinline]] void
rejectPacked(const std::string &what, const char *rule, const std::string &got,
             size_t word) {
  throw PackedValueError(what + " " + rule + ", got " + got, word);
}

/// The value of a line or tone as it enters `arm`, an arm of an if being
/// built: the version of it that no op of the arm made. `entered` holds,
/// for a result of an if that lies in an arm of another, the value that
/// entered that if, so that the enclosing if does not walk its arms again.
inline mlir::Value
enteringArm(mlir::Value value, mlir::Block *arm,
            const llvm::DenseMap<mlir::Value, mlir::Value> &entered) {
  while (value) {
    mlir::Operation *made = value.getDefiningOp();
    if (!made || !arm ||
        (made->getBlock() != arm && !arm->findAncestorOpInBlock(*made)))
      break;
    if (!entered.empty()) {
      auto inner = entered.find(value);
      if (inner != entered.end()) {
        value = inner->second;
        continue;
      }
    }
    mlir::Value before = pulse::linearPredecessor(value);
    if (!before)
      break;
    value = before;
  }
  return value;
}

/// What a line or tone value is a version of: its line (a line's own tone
/// stands for its line), or, for a tone that is a frame of its own, the
/// pulse.tone that made the frame. Null when neither is found.
inline mlir::Operation *lineOrFrame(pulse::AllocationTracer &trace,
                                    mlir::Value value) {
  if (mlir::Operation *line = trace(value))
    return line;
  while (mlir::Value before = pulse::linearPredecessor(value))
    value = before;
  mlir::Operation *made = value.getDefiningOp();
  return made && mlir::isa<pulse::ToneOp>(made) ? made : nullptr;
}

/// Whether `op` is in `block` or in a region of an op of it.
inline bool within(mlir::Operation *op, mlir::Block *block) {
  return block &&
         (op->getBlock() == block || block->findAncestorOpInBlock(*op));
}

/// Whether two operands are one number: one value, or equal constants.
inline bool sameNumber(mlir::Value a, mlir::Value b) {
  llvm::APFloat x(0.0), y(0.0);
  return a == b || (mlir::matchPattern(a, mlir::m_ConstantFloat(&x)) &&
                    mlir::matchPattern(b, mlir::m_ConstantFloat(&y)) &&
                    x.bitwiseIsEqual(y));
}

/// Whether `tone` is, on every path that reaches it, a frame made at the
/// frequency and phase `like` is made at and only played since: no op
/// changed its phase or frequency. An if's result is so when what each of
/// its arms yields there is; a loop's, when what the loop began with and
/// what its body gives back are; a schedule block's, when what its body
/// yields is. Every value is looked at once.
inline bool asMade(mlir::Value tone, mlir::Operation *like) {
  llvm::SmallVector<mlir::Value, 16> open{tone};
  llvm::DenseSet<mlir::Value> seen;
  while (!open.empty()) {
    mlir::Value value = open.pop_back_val();
    if (!value)
      return false;
    if (!seen.insert(value).second)
      continue;
    if (auto argument = mlir::dyn_cast<mlir::BlockArgument>(value)) {
      mlir::Operation *parent = argument.getOwner()->getParentOp();
      unsigned place = argument.getArgNumber();
      if (auto block = mlir::dyn_cast_or_null<pulse::ScheduleBlockOp>(parent)) {
        open.push_back(block.getInputs()[place]);
        continue;
      }
      // An iteration begins with what the loop began with, or with what
      // the iteration before it gave back.
      auto loop = mlir::dyn_cast_or_null<mlir::scf::ForOp>(parent);
      if (!loop || place == 0)
        return false;
      open.push_back(loop.getInitArgs()[place - 1]);
      open.push_back(loop.getBody()->getTerminator()->getOperand(place - 1));
      continue;
    }
    mlir::Operation *made = value.getDefiningOp();
    unsigned index = mlir::cast<mlir::OpResult>(value).getResultNumber();
    if (mlir::isa<pulse::ToneOp>(made)) {
      if (!sameNumber(made->getOperand(0), like->getOperand(0)) ||
          !sameNumber(made->getOperand(1), like->getOperand(1)))
        return false;
    } else if (auto branch = mlir::dyn_cast<mlir::scf::IfOp>(made)) {
      open.push_back(branch.thenYield().getOperand(index));
      open.push_back(branch.elseYield().getOperand(index));
    } else if (auto loop = mlir::dyn_cast<mlir::scf::ForOp>(made)) {
      open.push_back(loop.getInitArgs()[index]);
      open.push_back(loop.getBody()->getTerminator()->getOperand(index));
    } else if (auto block = mlir::dyn_cast<pulse::ScheduleBlockOp>(made)) {
      open.push_back(
          block.getBody().front().getTerminator()->getOperand(index));
    } else if (mlir::isa<pulse::DriveOp, pulse::ReadoutOp, pulse::IQAcquireOp>(
                   made)) {
      open.push_back(pulse::linearPredecessor(value));
    } else {
      return false;
    }
  }
  return true;
}

/// Whether `then` and `other`, which the two arms of an if yield in one
/// result, are versions of different lines, or of different tones. Looked
/// up only when the arms did not begin with one value there: both arms
/// begin with the same version of every line.
///
/// Whether an op in `arm` changes the tone `entered`, a version from
/// outside the arm, or a later version of it: any op given it there other
/// than a play. A play's tone result is followed, and so is what a loop,
/// an if or a schedule block inside the arm makes of it.
inline bool changedIn(mlir::Value entered, mlir::Block *arm) {
  if (!arm)
    return false;
  llvm::SmallVector<mlir::Value, 8> open{entered};
  llvm::DenseSet<mlir::Value> seen;
  while (!open.empty()) {
    mlir::Value value = open.pop_back_val();
    if (!seen.insert(value).second)
      continue;
    for (mlir::OpOperand &use : value.getUses()) {
      mlir::Operation *user = use.getOwner();
      if (!within(user, arm))
        continue;
      unsigned place = use.getOperandNumber();
      if (mlir::isa<pulse::DriveOp, pulse::ReadoutOp, pulse::IQAcquireOp>(
              user)) {
        for (mlir::OpResult result : user->getResults())
          if (mlir::isa<pulse::ToneType>(result.getType()))
            open.push_back(result);
      } else if (mlir::isa<mlir::scf::YieldOp, pulse::YieldOp>(user)) {
        // What a region inside the arm yields is its op's result there,
        // and a loop's next iteration begins with it.
        mlir::Operation *parent = user->getParentOp();
        if (!parent || place >= parent->getNumResults())
          return true;
        open.push_back(parent->getResult(place));
        if (auto loop = mlir::dyn_cast<mlir::scf::ForOp>(parent))
          open.push_back(loop.getRegionIterArgs()[place]);
      } else if (auto loop = mlir::dyn_cast<mlir::scf::ForOp>(user)) {
        for (auto [init, argument, result] :
             llvm::zip(loop.getInitArgs(), loop.getRegionIterArgs(),
                       loop.getResults()))
          if (init == value) {
            open.push_back(argument);
            open.push_back(result);
          }
      } else if (auto block = mlir::dyn_cast<pulse::ScheduleBlockOp>(user)) {
        for (auto [input, argument] : llvm::zip(
                 block.getInputs(), block.getBody().front().getArguments()))
          if (input == value)
            open.push_back(argument);
      } else {
        return true;
      }
    }
  }
  return false;
}

/// The first read of a transition's tone inside an arm makes a pulse.tone
/// there, so two arms may yield frames of two ops for what the kernel
/// reads as one tone. A read still in view at the if (`kept` tells: made
/// in a block open around the if) is the one a read of the same tone
/// returns, in the arms too: such a frame and another such, or one that an
/// arm of this if made, are two tones of the kernel, also at one
/// frequency. Otherwise: what reads the result follows the frame the first
/// arm yields, as it is on each path. That is the other path's tone too
/// when the other arm's frame is made alike and only played, and the first
/// arm's is either its own (made in that arm, whatever the arm then does to
/// it) or made alike, only played, and not changed by the other arm
/// (`entered` is the version of it the if began with). A frame an op
/// changed on the other path is another tone.
inline bool differentLines(mlir::Value then, mlir::Block *thenArm,
                           mlir::Value entered, mlir::Value other,
                           mlir::Block *elseArm,
                           llvm::function_ref<bool(mlir::Operation *)> kept) {
  pulse::AllocationTracer trace;
  mlir::Operation *a = lineOrFrame(trace, then);
  mlir::Operation *b = lineOrFrame(trace, other);
  if (!a || !b || a == b)
    return false;
  if (!mlir::isa<pulse::ToneOp>(a) || !mlir::isa<pulse::ToneOp>(b))
    return true;
  bool ownA = within(a, thenArm), ownB = within(b, elseArm);
  bool keptA = kept(a), keptB = kept(b);
  if ((keptA && (keptB || ownB)) || (keptB && ownA))
    return true;
  return !(asMade(other, a) &&
           (ownA || (asMade(then, a) && !changedIn(entered, elseArm))));
}

/// A result of an if that is no value: its arms leave versions of
/// different lines there, or, when `tone`, of different tones. `word` is the
/// if's record and `place` the result's place among the record's results.
struct MixedResult {
  size_t word;
  int64_t place;
  bool tone;
};

// The frontend reads what follows "arms:" as the result, by place, that
// holds different lines after the if (compiler.py, _rejected_literal),
// ",tone" after it as the result being a tone, and ",loop" as a loop having
// to carry that result from one iteration to the next.
[[noreturn, gnu::cold, gnu::noinline]] void rejectArms(const MixedResult &from,
                                                       bool carried = false) {
  throw PackedValueError("arms:" + std::to_string(from.place) +
                             (from.tone ? ",tone" : "") +
                             (carried ? ",loop" : ""),
                         from.word);
}

// A record names a value no record made: a result of an if that is no
// value (the error is at that if), or a malformed buffer. Out of line, so
// that the lookup every operand goes through is a find and a compare.
[[noreturn, gnu::cold, gnu::noinline]] void
unknownValue(const llvm::DenseMap<int64_t, MixedResult> &mixed, int64_t vid) {
  auto lost = mixed.find(vid);
  if (lost != mixed.end())
    rejectArms(lost->second);
  throw std::runtime_error("packed: SSA value not found for vid=" +
                           std::to_string(vid));
}

/// Whether `value` is what an op that takes one line or one tone and gives
/// it back made of `given`: the op is given `given`, of `value`'s type.
/// Such an op has one operand of each of those types, so no other check is
/// needed; a sync, a loop, a block or an if is not asked here. A drive,
/// the op most arms hold, is asked first, by the places of its line and
/// its tone. True only for a line or a tone: these ops are given nothing
/// else that they give back.
inline bool madeFrom(mlir::Value value, mlir::Value given) {
  mlir::Operation *op = value.getDefiningOp();
  if (!op)
    return false;
  if (mlir::isa<pulse::DriveOp>(op))
    return (op->getOperand(0) == given || op->getOperand(2) == given) &&
           given.getType() == value.getType();
  if (!mlir::isa<pulse::ReadoutOp, pulse::WaitOp, pulse::ShiftPhaseOp,
                 pulse::IQAcquireOp, pulse::ShiftFrequencyOp, pulse::SetPhaseOp,
                 pulse::SetFrequencyOp>(op))
    return false;
  for (mlir::Value operand : op->getOperands())
    if (operand == given)
      return operand.getType() == value.getType();
  return false;
}

// A line passed to one sync twice: it would be consumed twice, and a line
// is in step with itself. `first` and `second` count from one, as the
// call's lines do once its lists are flattened.
[[noreturn, gnu::cold, gnu::noinline]] void
rejectRepeatedLine(int64_t first, int64_t second, size_t word) {
  throw PackedValueError("sync(): lines " + std::to_string(first) + " and " +
                             std::to_string(second) +
                             " are the same line; a line is in step with "
                             "itself, so pass it once",
                         word);
}

[[noreturn, gnu::cold, gnu::noinline]] void
rejectFloat(const char *what, const char *rule, double got, size_t word) {
  rejectPacked(what, rule, pythonFloat(got), word);
}

[[noreturn, gnu::cold, gnu::noinline]] void
rejectSample(int64_t index, double got, size_t word) {
  rejectPacked("custom_samples() sample " + std::to_string(index),
               "must be finite", pythonFloat(got), word);
}

/// Collects the error diagnostics emitted while it is alive, so that a
/// failure can report them in the Python exception instead of on stderr.
class DiagnosticCollector {
public:
  explicit DiagnosticCollector(mlir::MLIRContext *ctx)
      : owner_(std::this_thread::get_id()),
        handler_(ctx, [this](mlir::Diagnostic &diagnostic) {
          // Modules from specialize() share a context; another thread's
          // pipeline may be running on it too.
          if (diagnostic.getSeverity() != mlir::DiagnosticSeverity::Error ||
              std::this_thread::get_id() != owner_)
            return mlir::failure();
          llvm::raw_string_ostream os(messages_);
          // An op numbered for idle replacement reads as one with no
          // location, as the same op compiled without the rule does.
          mlir::Location where = diagnostic.getLocation();
          if (auto site = mlir::dyn_cast<mlir::FileLineColLoc>(where);
              site && site.getFilename().getValue() == "pulse.site")
            where = mlir::UnknownLoc::get(where.getContext());
          os << "\n  " << where << ": " << diagnostic;
          for (mlir::Diagnostic &note : diagnostic.getNotes())
            os << "\n    note: " << note;
          return mlir::success();
        }) {}

  [[noreturn]] void fail(const std::string &what) const {
    throw std::runtime_error(what + messages_);
  }

private:
  std::string messages_;
  std::thread::id owner_;
  mlir::ScopedDiagnosticHandler handler_;
};

/// Parse `text`, raising with the parser diagnostics on failure.
mlir::OwningOpRef<mlir::ModuleOp> parseModule(const std::string &text,
                                              mlir::MLIRContext *ctx,
                                              const std::string &what) {
  DiagnosticCollector diagnostics(ctx);
  auto module = mlir::parseSourceString<mlir::ModuleOp>(text, ctx);
  if (!module)
    diagnostics.fail(what);
  return module;
}

/// Run `pm` on `op`, raising with the collected diagnostics on failure.
void runPipeline(mlir::PassManager &pm, mlir::Operation *op,
                 const std::string &what) {
  DiagnosticCollector diagnostics(pm.getContext());
  if (mlir::failed(pm.run(op)))
    diagnostics.fail(what);
}

/// Whether CUDAQ_PULSE_VERIFY=1 asks for the verifications that only catch a
/// bug of the compiler itself: of the module run_passes() leaves, which
/// scheduling verifies next anyway, and of a specialization rescheduled from
/// a verified module. The test suite sets it; read once per process.
bool verifyEverything() {
  static const bool on = [] {
    const char *text = std::getenv("CUDAQ_PULSE_VERIFY");
    if (!text || !*text || std::strcmp(text, "0") == 0)
      return false;
    if (std::strcmp(text, "1") == 0)
      return true;
    throw std::invalid_argument(
        std::string("CUDAQ_PULSE_VERIFY must be 0 or 1, not '") + text + "'");
  }();
  return on;
}

/// Run `pm` on `op` and, with `verify`, verify `op` once at the end, where
/// the pass manager would verify all of it after each pass that changed it.
void runPipelineVerifyingOnce(mlir::PassManager &pm, mlir::Operation *op,
                              const std::string &what, bool verify = true) {
  DiagnosticCollector diagnostics(pm.getContext());
  pm.enableVerifier(false);
  if (mlir::failed(pm.run(op)) || (verify && mlir::failed(mlir::verify(op))))
    diagnostics.fail(what);
}

/// The discardable attributes the walks over a module read, by their
/// interned names: reading an attribute by its interned name compares
/// pointers, where reading it by its spelling compares strings with each
/// inherent and discardable name of the op, and interning a name is a
/// lookup in the context's uniquer. None of them is inherent on any op.
struct AttrNames {
  explicit AttrNames(mlir::MLIRContext *context)
      : timeline(mlir::StringAttr::get(context, classical::kTimelineAttr)),
        variable(mlir::StringAttr::get(context, "classical.variable")),
        blocking(mlir::StringAttr::get(context, "classical.blocking")),
        device(mlir::StringAttr::get(context, "classical.device")),
        instance(mlir::StringAttr::get(context, classical::kInstanceAttr)),
        budget(mlir::StringAttr::get(context, "classical.budget_cycles")),
        variableWait(mlir::StringAttr::get(context, "pulse.variable_wait")),
        minCycles(mlir::StringAttr::get(context, "pulse.min_cycles")),
        maxCycles(mlir::StringAttr::get(context, "pulse.max_cycles")),
        framePhaseOffset(mlir::StringAttr::get(context, "frame_phase_offset")),
        phaseOffset(mlir::StringAttr::get(context, "phase_offset")) {}
  mlir::StringAttr timeline, variable, blocking, device, instance, budget,
      variableWait, minCycles, maxCycles, framePhaseOffset, phaseOffset;

  static bool has(mlir::Operation *op, mlir::StringAttr name) {
    return static_cast<bool>(op->getDiscardableAttr(name));
  }
};

/// What each context made by makeContext keeps for its next modules.
///
/// Its scheduling pass managers, by the options that built them, for modules
/// scheduled alike, and its run_passes() pass managers, by their pipeline:
/// building one, and its first run on a multithreaded context, which copies
/// its nested passes for each thread, cost as much as scheduling a small
/// module. One being run is taken out, so that no two threads run it at
/// once.
///
/// Its AttrNames, interned once for the life of the context, and the latest
/// texts set_attributes() parsed in it.
class ContextStore {
public:
  /// (policy, rcp's four resource limits or zeros, whether the classical
  /// calls are checked, whether the frame-changes pass runs).
  using Options =
      std::tuple<std::string, int64_t, int64_t, int64_t, int64_t, bool, bool>;

  explicit ContextStore(mlir::MLIRContext *context) : context(context) {
    std::lock_guard<std::mutex> guard(lock());
    contexts()[context] = this;
  }
  ~ContextStore() {
    std::lock_guard<std::mutex> guard(lock());
    contexts().erase(context);
  }
  ContextStore(const ContextStore &) = delete;
  ContextStore &operator=(const ContextStore &) = delete;

  /// The frozen pattern set of `context` kept under `key`, made by `make`
  /// the first time (outside the lock: two threads may both make it, and
  /// the first kept is the one every caller gets). A context without a
  /// store keeps none: each call makes the set anew.
  static std::shared_ptr<const mlir::FrozenRewritePatternSet>
  frozen(mlir::MLIRContext *context, const std::string &key,
         llvm::function_ref<mlir::RewritePatternSet()> make) {
    {
      std::lock_guard<std::mutex> guard(lock());
      auto found = contexts().find(context);
      if (found != contexts().end()) {
        auto &sets = found->second->patternSets;
        if (auto it = sets.find(key); it != sets.end())
          return it->second;
      }
    }
    auto made = std::make_shared<const mlir::FrozenRewritePatternSet>(make());
    std::lock_guard<std::mutex> guard(lock());
    auto found = contexts().find(context);
    if (found == contexts().end())
      return made;
    auto &sets = found->second->patternSets;
    if (sets.size() >= kMaxPatternSets && !sets.count(key))
      sets.clear();
    return sets.emplace(key, std::move(made)).first->second;
  }

  /// A kept pass manager of `context` built with `options`, or null.
  static std::unique_ptr<mlir::PassManager> take(mlir::MLIRContext *context,
                                                 const Options &options) {
    std::lock_guard<std::mutex> guard(lock());
    auto found = contexts().find(context);
    if (found == contexts().end())
      return nullptr;
    auto &kept = found->second->kept;
    for (auto it = kept.begin(); it != kept.end(); ++it)
      if (it->first == options) {
        std::unique_ptr<mlir::PassManager> passes = std::move(it->second);
        kept.erase(it);
        return passes;
      }
    return nullptr;
  }

  /// Keep `passes`, built with `options`, if `context` keeps pass managers:
  /// the latest few.
  static void keep(mlir::MLIRContext *context, Options options,
                   std::unique_ptr<mlir::PassManager> passes) {
    std::unique_ptr<mlir::PassManager> dropped; // destroyed unlocked
    std::lock_guard<std::mutex> guard(lock());
    auto found = contexts().find(context);
    if (found == contexts().end()) {
      dropped = std::move(passes);
      return;
    }
    auto &kept = found->second->kept;
    if (kept.size() == kMaxKept) {
      dropped = std::move(kept.front().second);
      kept.erase(kept.begin());
    }
    kept.emplace_back(std::move(options), std::move(passes));
  }

  /// A kept run_passes() pass manager of `context` for `pipeline`, or null.
  static std::unique_ptr<mlir::PassManager>
  takePipeline(mlir::MLIRContext *context,
               const std::vector<std::string> &pipeline) {
    std::lock_guard<std::mutex> guard(lock());
    auto found = contexts().find(context);
    if (found == contexts().end())
      return nullptr;
    auto &kept = found->second->pipelines;
    for (auto it = kept.begin(); it != kept.end(); ++it)
      if (it->first == pipeline) {
        std::unique_ptr<mlir::PassManager> passes = std::move(it->second);
        kept.erase(it);
        return passes;
      }
    return nullptr;
  }

  /// Keep `passes`, built for `pipeline`, if `context` keeps pass managers:
  /// the latest few.
  static void keepPipeline(mlir::MLIRContext *context,
                           std::vector<std::string> pipeline,
                           std::unique_ptr<mlir::PassManager> passes) {
    std::unique_ptr<mlir::PassManager> dropped; // destroyed unlocked
    std::lock_guard<std::mutex> guard(lock());
    auto found = contexts().find(context);
    if (found == contexts().end()) {
      dropped = std::move(passes);
      return;
    }
    auto &kept = found->second->pipelines;
    if (kept.size() == kMaxKept) {
      dropped = std::move(kept.front().second);
      kept.erase(kept.begin());
    }
    kept.emplace_back(std::move(pipeline), std::move(passes));
  }

  /// The AttrNames of `context`: those it keeps, or else interned in `own`.
  static const AttrNames &names(mlir::MLIRContext *context,
                                std::optional<AttrNames> &own) {
    {
      std::lock_guard<std::mutex> guard(lock());
      auto found = contexts().find(context);
      if (found != contexts().end()) {
        auto &kept = found->second->attrNames;
        if (!kept)
          kept = std::make_unique<AttrNames>(context);
        return *kept;
      }
    }
    return own.emplace(context);
  }

  /// `text` parsed as an attribute of `context`: one of the latest texts
  /// parsed in it, when it keeps them, is not parsed again. An attribute is
  /// immutable, so the same text is the same attribute for the life of the
  /// context.
  static mlir::Attribute parsed(mlir::MLIRContext *context,
                                const std::string &text) {
    {
      std::lock_guard<std::mutex> guard(lock());
      auto found = contexts().find(context);
      if (found != contexts().end()) {
        auto &texts = found->second->texts;
        if (auto it = texts.find(text); it != texts.end())
          return it->second;
      }
    }
    mlir::Attribute attr = mlir::parseAttribute(text, context);
    if (!attr)
      return attr;
    std::lock_guard<std::mutex> guard(lock());
    auto found = contexts().find(context);
    if (found != contexts().end()) {
      auto &texts = found->second->texts;
      if (texts.size() >= kMaxTexts)
        texts.clear();
      texts.emplace(text, attr);
    }
    return attr;
  }

  /// The lock is held across fork, so that a child inherits it unlocked.
  static void handleForks() {
    pthread_atfork([] { lock().lock(); }, [] { lock().unlock(); },
                   [] { lock().unlock(); });
  }

private:
  static constexpr size_t kMaxKept = 4;
  mlir::MLIRContext *context;
  std::vector<std::pair<Options, std::unique_ptr<mlir::PassManager>>> kept;
  std::vector<
      std::pair<std::vector<std::string>, std::unique_ptr<mlir::PassManager>>>
      pipelines;
  std::unique_ptr<AttrNames> attrNames;
  static constexpr size_t kMaxTexts = 64;
  std::unordered_map<std::string, mlir::Attribute> texts;
  // Frozen pattern sets by key (canonicalPatternsFor, foldBinding).
  static constexpr size_t kMaxPatternSets = 64;
  std::unordered_map<std::string,
                     std::shared_ptr<const mlir::FrozenRewritePatternSet>>
      patternSets;

  // Never destroyed, as the shared pool's last context is not.
  static std::mutex &lock() {
    static auto *mutex = new std::mutex();
    return *mutex;
  }
  static llvm::DenseMap<mlir::MLIRContext *, ContextStore *> &contexts() {
    static auto *all =
        new llvm::DenseMap<mlir::MLIRContext *, ContextStore *>();
    return *all;
  }
};

// ---- The canonicalization patterns a binding's module can match ----
//
// Folding a binding's module (foldBinding) runs the canonicalizer's patterns
// on it. MLIR sorts every pattern of the set by benefit each time it applies
// them, so a module takes only the patterns rooted at an op it holds or can
// come to hold: its own op names, closed under the ops each kept pattern
// makes. A pattern declares those (RewritePattern::getGeneratedOps); the
// table below lists them for the patterns that declare none.

/// The ops a canonicalization pattern that declares none can make, by
/// "root|debug name" (an empty list: it makes none, only moving, inlining or
/// erasing ops). Read from the patterns' sources at the pinned MLIR
/// (llvmorg-22.1.8): RE-READ THIS TABLE WHEN THE TOOLCHAIN IS BUMPED. A
/// pattern that declares nothing and is not listed here makes every module
/// take the whole set; tests/test_canonical_patterns.py fails and names it.
const llvm::StringMap<llvm::SmallVector<llvm::StringRef, 4>> &
undeclaredMakes() {
  static const auto *table = [] {
    auto *t = new llvm::StringMap<llvm::SmallVector<llvm::StringRef, 4>>();
    auto add = [&](llvm::StringRef key,
                   std::initializer_list<llvm::StringRef> made) {
      (*t)[key] = llvm::SmallVector<llvm::StringRef, 4>(made);
    };
    add("arith.cmpf|CmpFIntToFPConst", {"arith.cmpi", "arith.constant"});
    add("arith.index_castui|{anonymous}::IndexCastUIOfIndexCastUI", {});
    add("arith.index_cast|{anonymous}::IndexCastOfIndexCast", {});
    add("arith.select|SelectToExtUI",
        {"arith.constant", "arith.extui", "arith.xori"});
    add("func.call_indirect|", {"func.call"});
    add("scf.execute_region|MultiBlockExecuteInliner", {"cf.br"});
    add("scf.execute_region|SingleBlockExecuteInliner", {});
    for (llvm::StringRef root : {"scf.execute_region", "scf.for", "scf.if",
                                 "scf.index_switch", "scf.while"})
      for (llvm::StringRef name :
           {"{anonymous}::MakeRegionBranchOpSuccessorInputsDead",
            "{anonymous}::RemoveDeadRegionBranchOpSuccessorInputs",
            "{anonymous}::RemoveDuplicateSuccessorInputUses"})
        add((root + "|" + name).str(), {});
    add("scf.forall|{anonymous}::FoldTensorCastOfOutputIntoForallOp",
        {"scf.forall", "tensor.cast"});
    add("scf.forall|{anonymous}::ForallOpControlOperandsFolder", {});
    add("scf.forall|{anonymous}::ForallOpIterArgsFolder", {});
    add("scf.forall|{anonymous}::ForallOpReplaceConstantInductionVar", {});
    add("scf.forall|{anonymous}::ForallOpSingleOrZeroIterationDimsFolder",
        {"scf.forall"});
    add("scf.for|{anonymous}::ForOpTensorCastFolder", {"tensor.cast"});
    add("scf.for|{anonymous}::SimplifyTrivialLoops", {});
    add("scf.if|{anonymous}::CombineIfs", {"scf.if", "scf.yield"});
    add("scf.if|{anonymous}::CombineNestedIfs",
        {"scf.if", "scf.yield", "arith.andi", "arith.select"});
    add("scf.if|{anonymous}::ConditionPropagation", {"arith.constant"});
    add("scf.if|{anonymous}::ConvertTrivialIfToSelect",
        {"scf.if", "scf.yield", "arith.select"});
    add("scf.if|{anonymous}::RemoveEmptyElseBranch", {});
    add("scf.if|{anonymous}::RemoveStaticCondition", {});
    add("scf.if|{anonymous}::ReplaceIfYieldWithConditionOrValue",
        {"arith.xori", "arith.constant"});
    add("scf.index_switch|FoldConstantCase", {});
    add("scf.parallel|{anonymous}::MergeNestedParallelLoops", {"scf.parallel"});
    add("scf.parallel|{anonymous}::ParallelOpSingleOrZeroIterationDimsFolder",
        {"scf.parallel"});
    add("scf.while|{anonymous}::WhileCmpCond", {"arith.constant"});
    add("scf.while|{anonymous}::WhileConditionTruth", {"arith.constant"});
    add("scf.while|{anonymous}::WhileMoveIfDown", {"scf.while"});
    add("scf.while|{anonymous}::WhileOpAlignBeforeArgs",
        {"scf.condition", "scf.while"});
    add("tensor.dim|{anonymous}::DimOfForallOp", {});
    return t;
  }();
  return *table;
}

/// "root|debug name" of `pattern`, "*" for a pattern rooted at any op. A
/// debug name is the pattern's type name as the compiler spells it: an
/// anonymous namespace is "{anonymous}" for GCC and "(anonymous namespace)"
/// for Clang, spelled the first way here.
std::string patternKey(const mlir::RewritePattern &pattern) {
  std::optional<mlir::OperationName> root = pattern.getRootKind();
  std::string name = pattern.getDebugName().str();
  for (size_t at;
       (at = name.find("(anonymous namespace)")) != std::string::npos;)
    name.replace(at, std::strlen("(anonymous namespace)"), "{anonymous}");
  return (root ? root->getStringRef().str() : std::string("*")) + "|" + name;
}

/// The canonicalization patterns of `context`'s loaded dialects and
/// registered ops, as the canonicalizer gathers them.
mlir::RewritePatternSet allCanonicalPatterns(mlir::MLIRContext *context) {
  mlir::RewritePatternSet set(context);
  for (mlir::Dialect *dialect : context->getLoadedDialects())
    dialect->getCanonicalizationPatterns(set);
  for (mlir::RegisteredOperationName op : context->getRegisteredOperations())
    op.getCanonicalizationPatterns(set, context);
  return set;
}

/// The patterns of `all` that declare no ops they make and are not in
/// undeclaredMakes(), by key.
std::vector<std::string> unlistedPatterns(mlir::RewritePatternSet &all) {
  std::vector<std::string> unlisted;
  for (const std::unique_ptr<mlir::RewritePattern> &pattern :
       all.getNativePatterns())
    if (pattern->getGeneratedOps().empty() &&
        !undeclaredMakes().count(patternKey(*pattern)))
      unlisted.push_back(patternKey(*pattern));
  return unlisted;
}

/// Of `all`, the patterns a module holding the ops `held` can match while it
/// is canonicalized: those rooted at any op, and those rooted at an op of
/// `held` closed under what each kept pattern makes. All of `all` if a
/// pattern on the way declares nothing and is not listed.
mlir::RewritePatternSet reachablePatterns(mlir::MLIRContext *context,
                                          mlir::RewritePatternSet all,
                                          const llvm::StringSet<> &held) {
#if MLIR_ENABLE_PDL_IN_PATTERNMATCH
  // Patterns in PDL are not sorted out here: all of them, then.
  if (all.getPDLPatterns().getModule())
    return all;
#endif
  llvm::StringSet<> roots = held;
  // What folding makes: the constants it materializes.
  roots.insert("arith.constant");
  roots.insert("ub.poison");
  auto &patterns = all.getNativePatterns();
  for (bool grew = true; grew;) {
    grew = false;
    for (const std::unique_ptr<mlir::RewritePattern> &pattern : patterns) {
      std::optional<mlir::OperationName> root = pattern->getRootKind();
      if (root && !roots.contains(root->getStringRef()))
        continue;
      llvm::ArrayRef<mlir::OperationName> declared = pattern->getGeneratedOps();
      if (!declared.empty()) {
        for (mlir::OperationName made : declared)
          grew |= roots.insert(made.getStringRef()).second;
        continue;
      }
      auto listed = undeclaredMakes().find(patternKey(*pattern));
      if (listed == undeclaredMakes().end())
        return all;
      for (llvm::StringRef made : listed->second)
        grew |= roots.insert(made).second;
    }
  }
  mlir::RewritePatternSet kept(context);
  for (std::unique_ptr<mlir::RewritePattern> &pattern : patterns) {
    std::optional<mlir::OperationName> root = pattern->getRootKind();
    if (!root || roots.contains(root->getStringRef()))
      kept.add(std::move(pattern));
  }
  return kept;
}

/// Whether CUDAQ_PULSE_FULL_CANONICAL=1 asks folding to take every
/// canonicalization pattern, as the canonicalizer does: the tests compare
/// what it makes with what the patterns a module can match make. Read once
/// per process.
bool fullCanonicalSet() {
  static const bool on = [] {
    const char *text = std::getenv("CUDAQ_PULSE_FULL_CANONICAL");
    return text && std::strcmp(text, "1") == 0;
  }();
  return on;
}

/// The canonicalization patterns `module` can match while it is folded,
/// frozen once per context for each set of op names.
std::shared_ptr<const mlir::FrozenRewritePatternSet>
canonicalPatternsFor(mlir::MLIRContext *context, mlir::ModuleOp module) {
  if (fullCanonicalSet())
    return ContextStore::frozen(
        context,
        "canonical:*:" + std::to_string(context->getLoadedDialects().size()),
        [&] { return allCanonicalPatterns(context); });
  std::set<std::string> names;
  module.walk([&](mlir::Operation *op) {
    names.insert(op->getName().getStringRef().str());
  });
  // Patterns of a dialect loaded since may be rooted at these ops too.
  std::string key =
      "canonical:" + std::to_string(context->getLoadedDialects().size()) + ":";
  llvm::StringSet<> held;
  for (const std::string &name : names) {
    key += name + ",";
    held.insert(name);
  }
  return ContextStore::frozen(context, key, [&] {
    return reachablePatterns(context, allCanonicalPatterns(context), held);
  });
}

/// A new context with `registry`'s dialects and its ContextStore, destroyed
/// before it.
std::shared_ptr<mlir::MLIRContext>
makeContext(const mlir::DialectRegistry &registry) {
  struct Owner {
    explicit Owner(const mlir::DialectRegistry &registry) : context(registry) {}
    mlir::MLIRContext context;
    ContextStore store{&context};
  };
  auto owner = std::make_shared<Owner>(registry);
  return std::shared_ptr<mlir::MLIRContext>(owner, &owner->context);
}

/// The pass registered as `name`, or null. The registry is filled once, on
/// first use from any thread.
const mlir::PassInfo *lookupPass(llvm::StringRef name) {
  static const bool registered = [] {
    pulse::registerPulsePasses();
    mlir::registerLoopInvariantCodeMotionPass();
    return true;
  }();
  (void)registered;
  return mlir::PassInfo::lookup(name);
}

/// The passes of `entry`, split at the commas outside option braces.
llvm::SmallVector<llvm::StringRef, 4> splitPipeline(llvm::StringRef entry) {
  llvm::SmallVector<llvm::StringRef, 4> passes;
  int depth = 0;
  size_t start = 0;
  for (size_t i = 0; i < entry.size(); ++i) {
    if (entry[i] == '{')
      ++depth;
    else if (entry[i] == '}')
      --depth;
    else if (entry[i] == ',' && depth == 0) {
      passes.push_back(entry.slice(start, i).trim());
      start = i + 1;
    }
  }
  passes.push_back(entry.drop_front(start).trim());
  return passes;
}

inline double i2f(int64_t bits) {
  double d;
  std::memcpy(&d, &bits, sizeof(double));
  return d;
}

/// Whether a branch that yields a value of `type` can leave something known
/// only at run time: anything but a line, a tone or a link.
static bool yieldsValue(mlir::Type type) {
  return !mlir::isa<pulse::DriveLineType, pulse::ReadoutLineType,
                    pulse::ToneType, pulse::QuditType, classical::LinkType>(
      type);
}

/// The native frequency of a runtime qudit's line stands in the arguments
/// of `main` a controller program depends on as its qudit's qref argument
/// plus this: 0 for a drive line (the 0-1 frequency), the number of qref
/// arguments for a readout line (the readout frequency).
static int64_t nativeOffset(mlir::Operation *line) {
  if (!mlir::isa<pulse::GetReadoutLineOp>(line))
    return 0;
  auto function = line->getParentOfType<mlir::func::FuncOp>();
  return llvm::count_if(function.getArgumentTypes(), [](mlir::Type type) {
    return mlir::isa<pulse::QuditType>(type);
  });
}

/// The segments of a scheduled program, in program order. A variable wait or a
/// blocking wait ends the current segment of its timeline (the lines it holds
/// back) and opens the next one, which starts after it: segment k > 0 is
/// opened by the k-th such wait, and follows `after`, the segment its timeline
/// was in. Without timelines every segment follows the one before.
struct SegmentWalk {
  /// (kind, line or link, start, lowest, highest, after) per boundary: see
  /// PulseModule::segment_boundaries().
  using Boundary =
      std::tuple<std::string, int64_t, int64_t, int64_t, int64_t, int64_t>;

  explicit SegmentWalk(mlir::ModuleOp module)
      : names(ContextStore::names(module.getContext(), ownNames)) {
    module.walk([&](mlir::Operation *op) {
      if (mlir::isa<pulse::GetDriveLineOp, pulse::GetReadoutLineOp>(op))
        lineIndex.try_emplace(op, static_cast<int64_t>(lineIndex.size()));
    });
  }

  std::optional<AttrNames> ownNames;
  const AttrNames &names;
  llvm::DenseMap<mlir::Operation *, int64_t> lineIndex;
  pulse::AllocationTracer trace;
  std::map<int64_t, int64_t> current;
  std::vector<Boundary> boundaries;
  /// The boundaries of the loop iterations a summarizing walk does not
  /// visit, which number the segments after them as if it did.
  int64_t skipped = 0;

  /// The segments opened so far.
  int64_t count() const {
    return static_cast<int64_t>(boundaries.size()) + skipped;
  }

  /// The segments opened and each timeline's segment at a point of the walk.
  struct Mark {
    int64_t opened = 0;
    std::map<int64_t, int64_t> current;
  };
  Mark mark() const { return {count(), current}; }

  /// Whether the iterations of a loop from each of `marks` to the next each
  /// open as many segments and move each timeline's segment as far: then, as
  /// iterations that play alike visit the same boundaries, every later one
  /// does too, and an event of copy j is in its first copy's segment moved j
  /// times as far.
  static bool steady(llvm::ArrayRef<Mark> marks) {
    auto at = [](const Mark &mark, int64_t timeline) {
      auto found = mark.current.find(timeline);
      return found == mark.current.end() ? int64_t{0} : found->second;
    };
    for (size_t k = 2; k < marks.size(); ++k) {
      const Mark &a = marks[k - 2], &b = marks[k - 1], &c = marks[k];
      if (c.opened - b.opened != b.opened - a.opened)
        return false;
      for (const Mark *mark : {&a, &b, &c})
        for (const auto &[timeline, segment] : mark->current)
          if (at(c, timeline) - at(b, timeline) !=
              at(b, timeline) - at(a, timeline))
            return false;
    }
    return true;
  }

  /// Count `times` more iterations of a loop as visited, each opening the
  /// segments and moving each timeline's segment as much as the iteration
  /// walked since `from` did; false past 2**63 segments.
  bool skip(int64_t times, const Mark &from) {
    std::optional<int64_t> more = checkedMul(times, count() - from.opened);
    if (!more || !checkedAdd(skipped, *more))
      return false;
    for (auto &[timeline, segment] : current) {
      auto found = from.current.find(timeline);
      int64_t was = found == from.current.end() ? 0 : found->second;
      std::optional<int64_t> moved = checkedMul(times, segment - was);
      if (!moved || !checkedAdd(segment, *moved))
        return false;
      segment += *moved;
    }
    skips.push_back({static_cast<int64_t>(boundaries.size()), count() + 1,
                     count() - from.opened, times});
    skipped += *more;
    return true;
  }

  /// Of each loop skip() counted iterations of: how many boundaries were
  /// visited before them, the first segment they open, how many each
  /// opens, and how many they are.
  std::vector<std::array<int64_t, 4>> skips;

  int64_t timelineOf(mlir::Operation *op) {
    if (AttrNames::has(op, names.timeline))
      return classical::intAttr(op, names.timeline);
    for (mlir::Value operand : op->getOperands())
      if (mlir::isa<pulse::DriveLineType, pulse::ReadoutLineType,
                    pulse::ToneType>(operand.getType())) {
        mlir::Operation *allocation = trace(operand);
        return allocation ? classical::intAttr(allocation, names.timeline) : 0;
      }
    return 0;
  }

  /// The segment `op` runs in, in iteration `iteration`; a boundary opens the
  /// next. A variable wait runs in the segment it ends, a blocking wait's
  /// answer is taken in the one it opens.
  int64_t visit(mlir::Operation *op, llvm::ArrayRef<int64_t> iteration) {
    int64_t &segment = current[timelineOf(op)];
    if (!AttrNames::has(op, names.variable) &&
        !AttrNames::has(op, names.blocking))
      return segment;
    auto start = pulse::scheduledStart(op, iteration);
    if (!start)
      return segment;
    int64_t after = segment;
    if (mlir::isa<pulse::WaitOp>(op)) {
      mlir::Operation *allocation = trace(op->getOperand(0));
      boundaries.emplace_back(
          "wait", allocation ? lineIndex.lookup(allocation) : -1, *start,
          classical::intAttr(op, names.minCycles),
          classical::intAttr(op, names.maxCycles, -1), after);
    } else {
      boundaries.emplace_back("answer", classical::intAttr(op, names.device),
                              *start, 0, -1, after);
    }
    segment = count();
    return mlir::isa<pulse::WaitOp>(op) ? after : segment;
  }

  /// Visit every op of `block` once per loop iteration, both arms of each
  /// branch, as the event listings do; `fn(op, segment)` for each timed op,
  /// call and sync.
  template <typename Fn>
  void run(mlir::Block &block, Fn &&fn) {
    for (mlir::Operation &op : block) {
      if (mlir::isa<pulse::DriveOp, pulse::ReadoutOp, pulse::WaitOp,
                    pulse::SyncOp, classical::SubmitOp, classical::WaitOp>(
              op)) {
        fn(&op, visit(&op, iteration));
      } else if (auto loop = mlir::dyn_cast<mlir::scf::ForOp>(op)) {
        int64_t trips = pulse::constantTripCount(loop).value_or(0);
        iteration.push_back(0);
        for (int64_t k = 0; k < trips; ++k) {
          iteration.back() = k;
          run(*loop.getBody(), fn);
        }
        iteration.pop_back();
      } else {
        for (mlir::Region &region : op.getRegions())
          for (mlir::Block &nested : region)
            run(nested, fn);
      }
    }
  }

  llvm::SmallVector<int64_t, 2> iteration;
};

#include "ControllerExpressions.h"

/// The walk behind PulseModule::controller_events(), a check's walk of a
/// module that computes nothing at run time (its events are those of
/// ControllerWalk::Walk::Every): every timed op, sync and classical call once
/// per loop iteration, both arms of every branch, with the frame of its tone
/// and the branches it lies in; with `summarize`, a long rolled loop whose
/// iterations repeat walked three times. What it computes at run time is
/// in the export's vocabulary (ControllerExpressions).
struct ControllerEventWalk : ControllerExpressions {
  using WaveformNode =
      std::tuple<std::string, std::vector<double>, std::vector<int64_t>>;
  using Condition = std::tuple<int64_t, int64_t, std::vector<int64_t>,
                               std::vector<int64_t>, std::string>;
  using Event =
      std::tuple<std::string, int64_t, int64_t, std::optional<int64_t>,
                 std::optional<int64_t>, std::vector<int64_t>,
                 std::vector<WaveformNode>, double, double,
                 std::vector<Condition>, std::vector<int64_t>,
                 std::vector<int64_t>, std::string, std::string, double>;

  /// A tone's frame as the simulator tracks it (pulse::ToneState): `phase`
  /// at `timeNs`, the end of its latest drive or the place of a later frame
  /// change; a tone detuned from its frame frequency gains phase from there
  /// on.
  struct Frame {
    double frequencyHz = 0.0;
    double frameFrequencyHz = 0.0;
    double phase = 0.0;
    double timeNs = 0.0;

    double phaseAt(double timeNs) const {
      double detuning = frequencyHz - frameFrequencyHz;
      if (detuning == 0.0)
        return phase;
      return phase + detuning * (2.0 * M_PI) * 1.0e-9 * (timeNs - this->timeNs);
    }
    /// The frame at a frame change at `start` cycles, if that is later: it
    /// runs at its old frequency until then.
    void advanceTo(std::optional<int64_t> start, double clockGHz) {
      if (!start)
        return;
      double changeNs = static_cast<double>(*start) / clockGHz;
      if (changeNs > timeNs) {
        phase = phaseAt(changeNs);
        timeNs = changeNs;
      }
    }
    /// Whether two arms leave the frame the same: without a detuning, the
    /// time of the last drive does not matter.
    bool operator==(const Frame &other) const {
      auto same = [](double a, double b) {
        return a == b || (std::isnan(a) && std::isnan(b));
      };
      return same(frequencyHz, other.frequencyHz) &&
             same(frameFrequencyHz, other.frameFrequencyHz) &&
             same(phase, other.phase) &&
             (same(timeNs, other.timeNs) || frequencyHz == frameFrequencyHz);
    }
  };

  ControllerEventWalk(mlir::MLIRContext *context, double clockGHz,
                      bool track = false)
      : names(ContextStore::names(context, ownNames)), clockGHz(clockGHz),
        track(track) {}

  std::optional<AttrNames> ownNames;
  const AttrNames &names;
  double clockGHz;
  llvm::DenseMap<mlir::Operation *, int64_t> lineIndex;
  std::vector<Event> result;

  /// With `track`, the arguments of `main` each event's tone frequency,
  /// frame phase and waveform depend on, parallel to `result`. A qref
  /// argument stands for its qudit's native frequency (see nativeOffset).
  bool track;
  using Dependencies = std::array<std::vector<int64_t>, 3>;
  std::vector<Dependencies> dependencies;
  /// With `track`, parallel to `result`: the qref argument (see
  /// nativeOffset) whose native frequency an event's tone plays (its line was
  /// never retuned), or -1.
  std::vector<int64_t> natives;
  /// What a tone's frequency, the frequency of its frame (where it
  /// started) and its phase depend on.
  struct FrameDependencies {
    std::vector<int64_t> frequency, frame, phase;
    bool retuned = false;
  };
  std::map<mlir::Operation *, FrameDependencies> frameDependencies;

  /// The arguments of `main` *value* is computed from.
  static void argumentsOf(mlir::Value value, std::vector<int64_t> &out) {
    llvm::SmallVector<mlir::Value> work{value};
    llvm::DenseSet<mlir::Value> seen;
    while (!work.empty()) {
      mlir::Value v = work.pop_back_val();
      if (!seen.insert(v).second)
        continue;
      if (auto argument = mlir::dyn_cast<mlir::BlockArgument>(v)) {
        mlir::Block *owner = argument.getOwner();
        if (owner->isEntryBlock() &&
            mlir::isa_and_nonnull<mlir::func::FuncOp>(owner->getParentOp()))
          out.push_back(argument.getArgNumber());
        continue;
      }
      mlir::Operation *def = v.getDefiningOp();
      if (def && def->getNumRegions() == 0)
        work.append(def->getOperands().begin(), def->getOperands().end());
    }
  }

  static void merge(std::vector<int64_t> &into,
                    const std::vector<int64_t> &from) {
    into.insert(into.end(), from.begin(), from.end());
    std::sort(into.begin(), into.end());
    into.erase(std::unique(into.begin(), into.end()), into.end());
  }

  FrameDependencies &frameDependenciesOf(mlir::Value tone) {
    mlir::Operation *root = toneRoots(tone);
    auto [it, inserted] = frameDependencies.try_emplace(root);
    if (inserted && root) {
      if (mlir::isa<pulse::GetDriveLineOp, pulse::GetReadoutLineOp>(root) &&
          !root->hasAttr("frequency_hz")) {
        argumentsOf(root->getOperand(0), it->second.frequency);
        for (int64_t &argument : it->second.frequency)
          argument += nativeOffset(root);
      } else if (root->getName().getStringRef() == "pulse.tone" &&
                 root->getNumOperands() == 2) {
        argumentsOf(root->getOperand(0), it->second.frequency);
        argumentsOf(root->getOperand(1), it->second.phase);
      }
      merge(it->second.frequency, {});
      merge(it->second.phase, {});
      it->second.frame = it->second.frequency;
    }
    return it->second;
  }

  pulse::AllocationTracer trace;
  pulse::ToneRoots toneRoots;
  llvm::SmallVector<int64_t, 2> iteration;
  std::vector<Condition> branches;
  int64_t nextBranch = 0;
  llvm::DenseMap<mlir::Operation *, Frame> frames;
  // The time each line has reached (the end of its latest event), or none
  // when an event on it is not scheduled.
  llvm::DenseMap<mlir::Operation *, std::optional<int64_t>> reached;
  // While the arms of scf.ifs are walked (ifDepth of them): each change to
  // frames and reached, with what it replaced, oldest first. A branch undoes
  // the changes of its then arm before its else arm, and merges after it
  // what either arm changed, instead of copying every tone's frame and every
  // line's reach at each branch.
  struct FrameUndo {
    mlir::Operation *root;
    bool had;
    Frame old;
  };
  struct ReachedUndo {
    mlir::Operation *allocation;
    bool had;
    std::optional<int64_t> old;
  };
  llvm::SmallVector<FrameUndo, 16> frameUndo;
  llvm::SmallVector<ReachedUndo, 16> reachedUndo;
  int ifDepth = 0;
  /// What an arm left of what it changed (or, of what only the other arm
  /// changed, what it had before), each root and allocation once, as notes
  /// whose `had` and `old` say whether it left one and what. An arm changes
  /// few as a rule, so these are lists, and small sets find a root or an
  /// allocation in them by a linear search until they hold many.
  struct ArmLeft {
    llvm::SmallVector<FrameUndo, 4> frames;
    llvm::SmallVector<ReachedUndo, 4> reached;
    llvm::SmallPtrSet<mlir::Operation *, 8> roots, allocations;
  };

  static Frame initialFrame(mlir::Operation *root) {
    if (!root)
      return {};
    if (auto frequency = root->getAttrOfType<mlir::FloatAttr>("frequency_hz"))
      return {frequency.getValueAsDouble(), frequency.getValueAsDouble(), 0.0,
              0.0};
    // A runtime qudit's line: its frequency is the carrier's.
    if (mlir::isa<pulse::GetDriveLineOp, pulse::GetReadoutLineOp>(root))
      return {kUnknown, kUnknown, 0.0, 0.0};
    if (root->getName().getStringRef() == "pulse.tone" &&
        root->getNumOperands() == 2) {
      double frequency = numberOf(root->getOperand(0));
      return {frequency, frequency, numberOf(root->getOperand(1)), 0.0};
    }
    return {};
  }

  /// The frame of `tone`'s root, from its initial frame. A caller that
  /// changes it passes `change`, for an scf.if arm to undo.
  Frame &frameOf(mlir::Value tone, bool change = false) {
    mlir::Operation *root = toneRoots(tone);
    auto [it, inserted] = frames.try_emplace(root);
    if (inserted) {
      it->second = initialFrame(root);
      if (ifDepth > 0)
        frameUndo.push_back({root, false, {}});
    } else if (change && ifDepth > 0) {
      frameUndo.push_back({root, true, it->second});
    }
    return it->second;
  }

  /// Sets the frame of `root`, noting what it was within an scf.if arm.
  void setFrame(mlir::Operation *root, const Frame &frame) {
    auto [it, inserted] = frames.try_emplace(root, frame);
    if (ifDepth > 0)
      frameUndo.push_back({root, !inserted, inserted ? Frame() : it->second});
    if (!inserted)
      it->second = frame;
  }

  /// Sets the time `allocation`'s line has reached, noting what it was
  /// within an scf.if arm.
  void setReached(mlir::Operation *allocation, std::optional<int64_t> time) {
    auto [it, inserted] = reached.try_emplace(allocation, time);
    reachedTo(it, inserted, time);
  }

  /// Sets the time the line at `it` has reached, which `inserted` says was
  /// added just now, noting what it was within an scf.if arm.
  void reachedTo(
      llvm::DenseMap<mlir::Operation *, std::optional<int64_t>>::iterator it,
      bool inserted, std::optional<int64_t> time) {
    if (!inserted && it->second == time)
      return;
    if (ifDepth > 0)
      reachedUndo.push_back(
          {it->first, !inserted, inserted ? std::nullopt : it->second});
    it->second = time;
  }

  /// Undoes the changes to frames and reached since the marks, newest
  /// first, and returns what they had come to.
  ArmLeft undoArm(size_t frameMark, size_t reachedMark) {
    ArmLeft left;
    while (frameUndo.size() > frameMark) {
      FrameUndo undo = frameUndo.pop_back_val();
      auto found = frames.find(undo.root);
      bool has = found != frames.end();
      if (left.roots.insert(undo.root).second)
        left.frames.push_back({undo.root, has, has ? found->second : Frame()});
      if (!undo.had) {
        if (has)
          frames.erase(found);
      } else if (has) {
        found->second = undo.old;
      } else {
        frames.try_emplace(undo.root, undo.old);
      }
    }
    while (reachedUndo.size() > reachedMark) {
      ReachedUndo undo = reachedUndo.pop_back_val();
      auto found = reached.find(undo.allocation);
      bool has = found != reached.end();
      if (left.allocations.insert(undo.allocation).second)
        left.reached.push_back(
            {undo.allocation, has, has ? found->second : std::nullopt});
      if (!undo.had) {
        if (has)
          reached.erase(found);
      } else if (has) {
        found->second = undo.old;
      } else {
        reached.try_emplace(undo.allocation, undo.old);
      }
    }
    return left;
  }

  /// After both arms of an scf.if: a frame both leave alike stays, and one
  /// they leave apart is unknown from here on, a frame an arm has none of
  /// being its initial frame; a line has reached the later of the two
  /// times, or none if either arm leaves it none. Only what either arm
  /// changed can differ: `then` is what the then arm left of its changes,
  /// and the else arm's changes are noted from the marks on.
  void mergeArms(const ArmLeft &then, size_t frameMark, size_t reachedMark) {
    // Where the then arm changed nothing, it left what the else arm's
    // oldest note of a change has from before the branch.
    ArmLeft before;
    for (size_t i = frameMark; i < frameUndo.size(); ++i)
      if (!then.roots.contains(frameUndo[i].root) &&
          before.roots.insert(frameUndo[i].root).second)
        before.frames.push_back(frameUndo[i]);
    for (size_t i = reachedMark; i < reachedUndo.size(); ++i)
      if (!then.allocations.contains(reachedUndo[i].allocation) &&
          before.allocations.insert(reachedUndo[i].allocation).second)
        before.reached.push_back(reachedUndo[i]);
    if (ifDepth == 0) {
      frameUndo.resize(frameMark);
      reachedUndo.resize(reachedMark);
    }
    auto mergeFrame = [&](const FrameUndo &theirs) {
      auto found = frames.find(theirs.root);
      bool has = found != frames.end();
      if (!has && !theirs.had)
        return;
      Frame ours = has ? found->second : initialFrame(theirs.root);
      Frame other = theirs.had ? theirs.old : initialFrame(theirs.root);
      if (!(other == ours))
        ours = {kUnknown, kUnknown, kUnknown, ours.timeNs};
      else
        ours.timeNs = std::max(ours.timeNs, other.timeNs);
      if (!has)
        return setFrame(theirs.root, ours);
      if (ifDepth > 0)
        frameUndo.push_back({theirs.root, true, found->second});
      found->second = ours;
    };
    for (const FrameUndo &left : then.frames)
      mergeFrame(left);
    for (const FrameUndo &was : before.frames)
      mergeFrame(was);
    auto mergeReached = [&](const ReachedUndo &theirs) {
      if (!theirs.had)
        return;
      auto [it, inserted] = reached.try_emplace(theirs.allocation, theirs.old);
      if (inserted)
        return reachedTo(it, true, theirs.old);
      std::optional<int64_t> ours = it->second;
      reachedTo(it, false,
                ours && theirs.old
                    ? std::optional<int64_t>(std::max(*ours, *theirs.old))
                    : std::nullopt);
    };
    for (const ReachedUndo &left : then.reached)
      mergeReached(left);
    for (const ReachedUndo &was : before.reached)
      mergeReached(was);
  }

  /// Whether `value` depends on a readout or an answer.
  static bool fromRunTime(mlir::Value value) {
    mlir::Operation *first = value.getDefiningOp();
    if (!first || first->hasTrait<mlir::OpTrait::ConstantLike>())
      return false;
    llvm::SmallVector<mlir::Value> work{value};
    llvm::DenseSet<mlir::Value> seen;
    while (!work.empty()) {
      mlir::Value v = work.pop_back_val();
      if (!seen.insert(v).second)
        continue;
      mlir::Operation *def = v.getDefiningOp();
      if (!def)
        continue;
      if (mlir::isa<pulse::ReadoutOp, classical::WaitOp, mlir::scf::IfOp>(def))
        return true;
      work.append(def->getOperands().begin(), def->getOperands().end());
    }
    return false;
  }

  /// Whether a measurement reaches anything but branch conditions and
  /// submissions; true past a few ops, which only costs a slower walk.
  static bool measurementIsUsed(mlir::Value value, int depth = 0) {
    for (mlir::Operation *user : value.getUsers()) {
      if (mlir::isa<mlir::scf::IfOp, classical::SubmitOp>(user))
        continue;
      if (depth == 4 ||
          (!mlir::isa<pulse::IsSetOp, pulse::OutcomeOp>(user) &&
           !mlir::isa<mlir::arith::ArithDialect>(user->getDialect())))
        return true;
      for (mlir::Value next : user->getResults())
        if (measurementIsUsed(next, depth + 1))
          return true;
    }
    return false;
  }

  /// The tones whose frame a controller tracks at run time: those with a
  /// frame change computed from a readout or an answer, or made in a branch.
  llvm::DenseSet<mlir::Operation *> tracked;
  /// Whether to track them. A check reads none of their frame changes, and
  /// listing them can take memory cubic in the program's length: a change
  /// that a drive in one arm takes stays pending on the other arm's path,
  /// gaining a condition at every later branch.
  bool trackFrames = true;
  /// Whether any event can have something known only at run time: a tracked
  /// tone, a classical call, or a measurement used other than as a branch
  /// condition.
  bool detail = false;

  void scan(mlir::Operation *op) {
    if (mlir::isa<pulse::ShiftPhaseOp, pulse::SetPhaseOp,
                  pulse::ShiftFrequencyOp, pulse::SetFrequencyOp>(op)) {
      if (trackFrames && (fromRunTime(op->getOperand(1)) ||
                          op->getParentOfType<mlir::scf::IfOp>())) {
        tracked.insert(toneRoots(op->getOperand(0)));
        detail = true;
      }
    } else if (auto branch = mlir::dyn_cast<mlir::scf::IfOp>(op)) {
      // A virtual Z folded into a drive in an arm is a frame change there.
      if (trackFrames)
        branch->walk([&](pulse::DriveOp drive) {
          if (drive->hasAttr("frame_phase_offset")) {
            tracked.insert(toneRoots(drive->getOperand(2)));
            detail = true;
          }
        });
    } else if (mlir::isa<classical::SubmitOp>(op)) {
      detail = true;
    } else if (mlir::isa<pulse::ReadoutOp>(op) && !detail) {
      detail = measurementIsUsed(op->getResult(2));
    }
  }

  // Each tracked tone's id, in the order tones are first played, and the
  // frame changes on it since its latest drive or readout.
  struct FrameUpdate {
    uint64_t seq;
    llvm::StringRef op;
    int32_t value; // an expression
    std::vector<Condition> conditions;
  };
  llvm::DenseMap<mlir::Operation *, int64_t> toneIds;
  std::map<mlir::Operation *, std::vector<FrameUpdate>> pending;
  uint64_t nextUpdate = 0;

  /// What an event has that is known only at run time: on a tracked tone
  /// (`tone` at least 0), the frame changes since its latest drive or readout
  /// and its own phase offset; the waveform operands computed from a readout
  /// or an answer, as (node, operand index, expression); a variable wait's
  /// length (at least 0) and bounds; a submission's operands.
  struct Known {
    int64_t event = 0;
    int64_t tone = -1;
    std::vector<FrameUpdate> updates;
    std::optional<double> offset;
    std::vector<std::tuple<int64_t, int64_t, int32_t>> waveform;
    int32_t length = -1;
    std::optional<int64_t> minCycles, maxCycles;
    std::optional<std::vector<int32_t>> operands;
  };
  /// In event order, only for the events that have any.
  std::vector<Known> known;

  /// A frame change's value: its number, else its expression.
  int32_t frameValue(mlir::Value value) {
    double constant = numberOf(value);
    return std::isnan(constant) ? expression(value)
                                : add(Expression::Number, constant);
  }

  void frameUpdate(mlir::Operation *root, llvm::StringRef op, int32_t value) {
    pending[root].push_back({nextUpdate++, op, value, branches});
  }

  // The run-time work of the walk stays out of line, so that the per-op
  // path of a walk without tracked tones or detail is unchanged.

  /// A frame change `op` (named `name`), on a tone that may be tracked.
  LLVM_ATTRIBUTE_NOINLINE void frameChange(mlir::Operation *op,
                                           llvm::StringRef name) {
    if (mlir::Operation *root = toneRoots(op->getOperand(0));
        tracked.contains(root))
      frameUpdate(root, name, frameValue(op->getOperand(1)));
  }

  // The frame changes pending before each open branch, and after its then
  // arm.
  std::vector<std::pair<decltype(pending), decltype(pending)>> pendingArms;

  LLVM_ATTRIBUTE_NOINLINE void enterBranch() {
    pendingArms.emplace_back(pending, decltype(pending){});
  }

  LLVM_ATTRIBUTE_NOINLINE void enterElse() {
    auto &[before, then] = pendingArms.back();
    then = std::move(pending);
    pending = before;
  }

  LLVM_ATTRIBUTE_NOINLINE void leaveBranch(const Condition &arms) {
    auto [before, then] = std::move(pendingArms.back());
    pendingArms.pop_back();
    mergePending(before, then, arms);
  }

  /// The frame changes pending after a branch: those of both arms, by
  /// program order. One pending before the branch that a drive or readout
  /// in one arm took is pending only on the other arm's path.
  void
  mergePending(std::map<mlir::Operation *, std::vector<FrameUpdate>> &before,
               std::map<mlir::Operation *, std::vector<FrameUpdate>> &then,
               const Condition &branch) {
    std::set<mlir::Operation *> roots;
    for (auto *side : {&before, &then, &pending})
      for (auto &entry : *side)
        roots.insert(entry.first);
    auto seqs = [](std::map<mlir::Operation *, std::vector<FrameUpdate>> &side,
                   mlir::Operation *root) {
      std::set<uint64_t> out;
      auto found = side.find(root);
      if (found != side.end())
        for (auto &update : found->second)
          out.insert(update.seq);
      return out;
    };
    for (mlir::Operation *root : roots) {
      std::set<uint64_t> old = seqs(before, root), inThen = seqs(then, root),
                         inElse = seqs(pending, root);
      std::map<uint64_t, FrameUpdate> merged;
      auto take =
          [&](std::map<mlir::Operation *, std::vector<FrameUpdate>> &side,
              bool arm, const std::set<uint64_t> &other) {
            auto found = side.find(root);
            if (found == side.end())
              return;
            for (auto &update : found->second) {
              if (merged.count(update.seq))
                continue;
              if (old.count(update.seq) && !other.count(update.seq)) {
                update.conditions.push_back(branch);
                std::get<1>(update.conditions.back()) = arm;
              }
              merged.emplace(update.seq, std::move(update));
            }
          };
      take(then, true, inElse);
      take(pending, false, inThen);
      std::vector<FrameUpdate> out;
      for (auto &[seq, update] : merged)
        out.push_back(std::move(update));
      if (out.empty())
        pending.erase(root);
      else
        pending[root] = std::move(out);
    }
  }

  static int64_t waveform(mlir::Value value, std::vector<WaveformNode> &nodes) {
    mlir::Operation *def = value.getDefiningOp();
    if (!def) {
      nodes.emplace_back("argument", std::vector<double>{},
                         std::vector<int64_t>{});
      return static_cast<int64_t>(nodes.size()) - 1;
    }
    std::vector<double> numbers;
    std::vector<int64_t> children;
    for (mlir::Value operand : def->getOperands()) {
      if (mlir::isa<pulse::WaveformType>(operand.getType()))
        children.push_back(waveform(operand, nodes));
      else
        numbers.push_back(numberOf(operand));
    }
    if (auto samples = def->getAttrOfType<mlir::ArrayAttr>("samples"))
      for (mlir::Attribute sample : samples)
        if (auto number = mlir::dyn_cast<mlir::FloatAttr>(sample))
          numbers.push_back(number.getValueAsDouble());
    std::string name = def->getName().stripDialect().str();
    if (auto custom = def->getAttrOfType<mlir::FlatSymbolRefAttr>("callee"))
      name += ":" + custom.getValue().str();
    nodes.emplace_back(name, std::move(numbers), std::move(children));
    return static_cast<int64_t>(nodes.size()) - 1;
  }

  /// The operands of `value`'s waveform nodes (numbered as `waveform` does,
  /// `nodes` its result) computed from a readout or an answer, as (node,
  /// operand index, operand).
  static void
  runTimeOperands(mlir::Value value, const std::vector<WaveformNode> &nodes,
                  int64_t &next,
                  std::vector<std::tuple<int64_t, int64_t, mlir::Value>> &out) {
    mlir::Operation *def = value.getDefiningOp();
    if (!def) {
      ++next;
      return;
    }
    llvm::SmallVector<std::pair<int64_t, mlir::Value>, 4> numbers;
    for (mlir::Value operand : def->getOperands()) {
      if (mlir::isa<pulse::WaveformType>(operand.getType()))
        runTimeOperands(operand, nodes, next, out);
      else
        numbers.emplace_back(static_cast<int64_t>(numbers.size()), operand);
    }
    int64_t node = next++;
    const std::vector<double> &known = std::get<1>(nodes[node]);
    for (auto &[index, operand] : numbers)
      if (std::isnan(known[index]) && fromRunTime(operand))
        out.emplace_back(node, index, operand);
  }

  /// The readout events and answer events `value` is computed from.
  void inputs(mlir::Value value, std::vector<int64_t> &readouts,
              std::vector<int64_t> &answers) {
    llvm::SmallVector<mlir::Value> work{value};
    llvm::DenseSet<mlir::Value> seen;
    while (!work.empty()) {
      mlir::Value v = work.pop_back_val();
      if (!seen.insert(v).second)
        continue;
      mlir::Operation *def = v.getDefiningOp();
      if (!def || mlir::isa<classical::SubmitOp>(def))
        continue;
      if (mlir::isa<pulse::ReadoutOp, classical::WaitOp>(def)) {
        auto found = latest.find(def);
        if (found != latest.end()) {
          auto &list = mlir::isa<pulse::ReadoutOp>(def) ? readouts : answers;
          if (llvm::find(list, found->second) == list.end())
            list.push_back(found->second);
        }
        continue;
      }
      if (def->getNumRegions() == 0)
        work.append(def->getOperands().begin(), def->getOperands().end());
    }
  }

  int64_t lineOf(mlir::Operation *allocation) const {
    return allocation ? lineIndex.lookup(allocation) : -1;
  }

  // The walk is compiled twice: RunTime is false for a module that computes
  // nothing at run time, so that its walk has none of the run-time work.

  template <bool RunTime>
  void block(mlir::Block &body) {
    for (mlir::Operation &op : body)
      step<RunTime>(&op);
  }

  template <bool RunTime>
  void timed(mlir::Operation *op) {
    requireExportedMode(op);
    mlir::Operation *allocation = trace(op->getOperand(0));
    int64_t qubit = allocation ? pulse::quditOf(allocation) : -1;
    auto start = pulse::scheduledStart(op, iteration);
    auto duration = pulse::scheduledDuration(op);
    std::vector<WaveformNode> nodes;
    double frequency = kUnknown, phase = kUnknown;
    if (!mlir::isa<pulse::WaitOp>(op)) {
      waveform(op->getOperand(1), nodes);
      bool drive = mlir::isa<pulse::DriveOp>(op);
      Frame &frame = frameOf(op->getOperand(2), drive);
      double offset = 0.0;
      if (drive) {
        if (auto shift = mlir::dyn_cast_or_null<mlir::FloatAttr>(
                op->getDiscardableAttr(names.framePhaseOffset)))
          frame.phase += shift.getValueAsDouble();
        if (auto own = mlir::dyn_cast_or_null<mlir::FloatAttr>(
                op->getDiscardableAttr(names.phaseOffset)))
          offset = own.getValueAsDouble();
      }
      frequency = frame.frequencyHz;
      if (start)
        phase = frame.phaseAt(static_cast<double>(*start) / clockGHz) + offset;
      if (drive) {
        if (start && duration) {
          double endNs = static_cast<double>(*start + *duration) / clockGHz;
          frame.phase = frame.phaseAt(endNs);
          frame.timeNs = endNs;
        } else {
          frame = {kUnknown, kUnknown, kUnknown, frame.timeNs};
        }
      }
    }
    if (track) {
      Dependencies found;
      int64_t native = -1;
      if (!mlir::isa<pulse::WaitOp>(op)) {
        FrameDependencies &frame = frameDependenciesOf(op->getOperand(2));
        mlir::Operation *root = toneRoots(op->getOperand(2));
        if (!frame.retuned && root &&
            mlir::isa<pulse::GetDriveLineOp, pulse::GetReadoutLineOp>(root) &&
            !root->hasAttr("frequency_hz"))
          if (auto argument =
                  mlir::dyn_cast<mlir::BlockArgument>(root->getOperand(0)))
            if (argument.getOwner()->isEntryBlock() &&
                mlir::isa_and_nonnull<mlir::func::FuncOp>(
                    argument.getOwner()->getParentOp()))
              native = argument.getArgNumber() + nativeOffset(root);
        found[0] = frame.frequency;
        found[1] = frame.phase;
        // A retuned tone gains phase at its detuning from the frame.
        if (frame.retuned) {
          merge(found[1], frame.frequency);
          merge(found[1], frame.frame);
        }
        argumentsOf(op->getOperand(1), found[2]);
        merge(found[2], {});
      }
      dependencies.push_back(std::move(found));
      natives.push_back(native);
    }
    if (mlir::isa<pulse::ReadoutOp>(op))
      latest[op] = static_cast<int64_t>(result.size());
    auto [line, inserted] = reached.try_emplace(allocation, 0);
    reachedTo(line, inserted,
              start && duration && line->second
                  ? std::optional<int64_t>(
                        std::max(*line->second, *start + *duration))
                  : std::nullopt);
    if constexpr (RunTime)
      if (detail)
        runTime(op, nodes);
    segmentOf(op);
    result.emplace_back(
        op->getName().stripDialect().str(), lineOf(allocation), qubit, start,
        duration, std::vector<int64_t>(iteration.begin(), iteration.end()),
        std::move(nodes), frequency, phase, branches, std::vector<int64_t>{},
        std::vector<int64_t>{}, std::string(), std::string(), kUnknown);
  }

  /// What the next event, a drive, readout or wait, has that is known only
  /// at run time.
  LLVM_ATTRIBUTE_NOINLINE void runTime(mlir::Operation *op,
                                       const std::vector<WaveformNode> &nodes) {
    Known next;
    next.event = static_cast<int64_t>(result.size());
    if (mlir::isa<pulse::WaitOp>(op)) {
      if (!fromRunTime(op->getOperand(1)) ||
          !AttrNames::has(op, names.variableWait))
        return;
      next.length = expression(op->getOperand(1));
      if (auto bound = mlir::dyn_cast_or_null<mlir::IntegerAttr>(
              op->getDiscardableAttr(names.maxCycles)))
        next.maxCycles = bound.getInt();
      if (auto bound = mlir::dyn_cast_or_null<mlir::IntegerAttr>(
              op->getDiscardableAttr(names.minCycles)))
        next.minCycles = bound.getInt();
      known.push_back(std::move(next));
      return;
    }
    bool unknown = llvm::any_of(nodes, [](const WaveformNode &node) {
      return llvm::any_of(std::get<1>(node),
                          [](double v) { return std::isnan(v); });
    });
    if (unknown) {
      std::vector<std::tuple<int64_t, int64_t, mlir::Value>> operands;
      int64_t count = 0;
      runTimeOperands(op->getOperand(1), nodes, count, operands);
      for (auto &[node, index, operand] : operands)
        next.waveform.emplace_back(node, index, expression(operand));
    }
    mlir::Operation *root =
        tracked.empty() ? nullptr : toneRoots(op->getOperand(2));
    if (root && tracked.contains(root)) {
      next.tone =
          toneIds.try_emplace(root, static_cast<int64_t>(toneIds.size()))
              .first->second;
      if (mlir::isa<pulse::DriveOp>(op)) {
        if (auto shift = mlir::dyn_cast_or_null<mlir::FloatAttr>(
                op->getDiscardableAttr(names.framePhaseOffset)))
          frameUpdate(root, "shift_phase",
                      add(Expression::Number, shift.getValueAsDouble()));
        if (auto own = mlir::dyn_cast_or_null<mlir::FloatAttr>(
                op->getDiscardableAttr(names.phaseOffset)))
          next.offset = own.getValueAsDouble();
      }
      if (auto found = pending.find(root); found != pending.end()) {
        next.updates = std::move(found->second);
        pending.erase(found);
      }
    }
    if (next.tone >= 0 || !next.waveform.empty())
      known.push_back(std::move(next));
  }

  /// The operands of a submission, in argument order.
  LLVM_ATTRIBUTE_NOINLINE void submission(classical::SubmitOp request) {
    Known next;
    next.event = static_cast<int64_t>(result.size());
    next.operands.emplace();
    for (mlir::Value arg : request.getArgs())
      next.operands->push_back(expression(arg));
    known.push_back(std::move(next));
  }

  void sync(pulse::SyncOp op) {
    std::vector<int64_t> lines;
    std::optional<int64_t> time = 0;
    for (mlir::Value value : op.getLines()) {
      mlir::Operation *allocation = trace(value);
      lines.push_back(lineOf(allocation));
      auto found = reached.find(allocation);
      if (found == reached.end())
        continue;
      if (!found->second || !time)
        time = std::nullopt;
      else
        time = std::max(*time, *found->second);
    }
    if (time)
      for (mlir::Value value : op.getLines())
        setReached(trace(value), time);
    if (track) {
      dependencies.emplace_back();
      natives.push_back(-1);
    }
    segmentOf(op);
    result.emplace_back(
        "sync", -1, -1, time,
        time ? std::optional<int64_t>(0) : std::optional<int64_t>(),
        std::vector<int64_t>(iteration.begin(), iteration.end()),
        std::vector<WaveformNode>{}, kUnknown, kUnknown, branches,
        std::move(lines), std::vector<int64_t>{}, std::string(), std::string(),
        kUnknown);
  }

  /// A submission (with the readout events whose measurements it sends) or
  /// the wait for its answer, on its link.
  template <bool RunTime>
  void call(mlir::Operation *op) {
    bool submit = mlir::isa<classical::SubmitOp>(op);
    std::vector<int64_t> refs, answers;
    if (submit)
      for (mlir::Value operand : op->getOperands())
        inputs(operand, refs, answers);
    std::string callee;
    if (auto symbol = op->getAttrOfType<mlir::FlatSymbolRefAttr>("callee"))
      callee = symbol.getValue().str();
    std::string mode;
    double timeoutNs = kUnknown;
    if (!submit) {
      latest[op] = static_cast<int64_t>(result.size());
      auto [wait, timeout] = waitMode(op);
      mode = wait;
      timeoutNs = timeout;
    }
    if constexpr (RunTime)
      if (auto request = mlir::dyn_cast<classical::SubmitOp>(op))
        submission(request);
    if (track) {
      dependencies.emplace_back();
      natives.push_back(-1);
    }
    // A call has no qudit: its slot holds the controller whose instance of
    // a per-controller link runs it (-1: none).
    segmentOf(op);
    result.emplace_back(
        submit ? "submit" : "answer", classical::intAttr(op, names.device),
        classical::intAttr(op, names.instance, -1),
        pulse::scheduledStart(op, iteration), pulse::scheduledDuration(op),
        std::vector<int64_t>(iteration.begin(), iteration.end()),
        std::vector<WaveformNode>{}, kUnknown, kUnknown, branches,
        std::vector<int64_t>{}, std::move(refs), std::move(callee),
        std::move(mode), timeoutNs);
  }

  template <bool RunTime>
  void step(mlir::Operation *op) {
    llvm::StringRef name = op->getName().getStringRef();
    if (mlir::isa<pulse::DriveOp, pulse::ReadoutOp, pulse::WaitOp>(op)) {
      timed<RunTime>(op);
      return;
    }
    if (auto barrier = mlir::dyn_cast<pulse::SyncOp>(op)) {
      sync(barrier);
      return;
    }
    if (mlir::isa<classical::SubmitOp, classical::WaitOp>(op)) {
      call<RunTime>(op);
      return;
    }
    if (name == "pulse.shift_phase" || name == "pulse.set_phase" ||
        name == "pulse.shift_frequency" || name == "pulse.set_frequency") {
      if (track) {
        FrameDependencies &frame = frameDependenciesOf(op->getOperand(0));
        std::vector<int64_t> found;
        argumentsOf(op->getOperand(1), found);
        bool phase = name == "pulse.shift_phase" || name == "pulse.set_phase";
        auto &field = phase ? frame.phase : frame.frequency;
        if (name.starts_with("pulse.set_"))
          field.clear();
        merge(field, found);
        frame.retuned |= !phase;
      }
      double value = numberOf(op->getOperand(1));
      Frame &frame = frameOf(op->getOperand(0), true);
      if constexpr (RunTime)
        if (!tracked.empty())
          frameChange(op, name.drop_front(6));
      frame.advanceTo(pulse::scheduledStart(op, iteration), clockGHz);
      if (name == "pulse.shift_phase")
        frame.phase += value;
      else if (name == "pulse.set_phase")
        frame.phase = value;
      else if (name == "pulse.shift_frequency")
        frame.frequencyHz += value;
      else
        frame.frequencyHz = value;
      return;
    }
    if (auto loop = mlir::dyn_cast<mlir::scf::ForOp>(op)) {
      int64_t trips = pulse::constantTripCount(loop).value_or(0);
      iteration.push_back(0);
      if (summarize && trips > kSummarizeAfter &&
          (!RunTime || tracked.empty())) {
        summarizeLoop<RunTime>(*loop.getBody(), trips);
        iteration.pop_back();
        return;
      }
      for (int64_t k = 0; k < trips; ++k) {
        iteration.back() = k;
        block<RunTime>(*loop.getBody());
      }
      iteration.pop_back();
      return;
    }
    if (auto branch = mlir::dyn_cast<mlir::scf::IfOp>(op)) {
      mlir::Value condition = branch.getCondition();
      std::vector<int64_t> readouts, answers;
      inputs(condition, readouts, answers);
      std::string test = "expression";
      if (mlir::Operation *def = condition.getDefiningOp()) {
        if (mlir::isa<pulse::IsSetOp>(def) && readouts.size() == 1)
          test = "is_set";
        else if (mlir::isa<classical::WaitOp>(def) && answers.size() == 1)
          test = "answer";
      }
      int64_t id = nextBranch++;
      size_t frameMark = frameUndo.size(), reachedMark = reachedUndo.size();
      ++ifDepth;
      bool tracking = RunTime && !tracked.empty();
      if (tracking)
        enterBranch();
      auto dependenciesBefore = frameDependencies;
      branches.emplace_back(id, 1, std::move(readouts), std::move(answers),
                            std::move(test));
      block<RunTime>(*branch.thenBlock());
      ArmLeft thenLeft = undoArm(frameMark, reachedMark);
      auto dependenciesThen = std::move(frameDependencies);
      if (tracking)
        enterElse();
      frameDependencies = std::move(dependenciesBefore);
      std::get<1>(branches.back()) = 0;
      if (!branch.getElseRegion().empty())
        block<RunTime>(*branch.elseBlock());
      if (tracking)
        leaveBranch(branches.back());
      branches.pop_back();
      --ifDepth;
      mergeArms(thenLeft, frameMark, reachedMark);
      // After the branch, a frame depends on what either arm set.
      for (auto &[root, theirs] : dependenciesThen) {
        auto &ours = frameDependencies[root];
        merge(ours.frequency, theirs.frequency);
        merge(ours.frame, theirs.frame);
        merge(ours.phase, theirs.phase);
        ours.retuned |= theirs.retuned;
      }
      return;
    }
    for (mlir::Region &region : op->getRegions())
      for (mlir::Block &nested : region)
        block<RunTime>(nested);
  }

  /// With `segmentWalk` (a module with classical calls): the segment of each
  /// event, parallel to `result`.
  std::optional<SegmentWalk> segmentWalk;
  std::vector<int64_t> segments;

  void segmentOf(mlir::Operation *op) {
    if (segmentWalk)
      segments.push_back(segmentWalk->visit(op, iteration));
  }

  SegmentWalk::Mark segmentsNow() const {
    return segmentWalk ? segmentWalk->mark() : SegmentWalk::Mark{};
  }

  /// With `summarize` (for a check, which tracks no tone's frame changes): a
  /// rolled loop of more than kSummarizeAfter iterations whose third
  /// iteration plays its second again one period later, computing at run
  /// time what it computes (sameRunTime), is walked three times, and
  /// `repeats` records that the third stands for every later one. Frame
  /// phases are not followed past such a loop.
  bool summarize = false;
  static constexpr int64_t kSummarizeAfter = 64;
  /// The events a summarizing walk plays at most, loops that do not repeat
  /// played out.
  static constexpr size_t kMaxEvents = size_t(1) << 22;
  /// (begin, end, copies, period, depth, branch, branches): result[begin,
  /// end) is the first of `copies` copies. Copy j starts j * period cycles
  /// later, has entry `depth` of its iteration j higher and the ids of its
  /// branches from `branch` on j * branches higher; a readout or answer
  /// event it refers to in the iteration before it or in itself is j copies
  /// later, and an event after the loop refers to the last copy.
  using Repeat = std::array<int64_t, 7>;
  std::vector<Repeat> repeats;
  /// With `unsynced` too: a loop whose lines each play their events of
  /// iteration 1 again at a period of their own, which syncs only lines of
  /// one period, also repeats. Its Repeat has period -1, and each line's
  /// period is a (repeat index, line, period) here; a line with none moves
  /// no event. A classical call's submit or answer, i events into the body,
  /// is on a line of its own, kCallLine - i.
  bool unsynced = false;
  std::vector<std::array<int64_t, 3>> linePeriods;
  static constexpr int64_t kCallLine = std::numeric_limits<int64_t>::min() / 2;

  struct Mark {
    size_t event, repeat;
    int64_t branch;
    size_t known;
  };
  Mark mark() const {
    return {result.size(), repeats.size(), nextBranch, known.size()};
  }

  template <bool RunTime>
  LLVM_ATTRIBUTE_NOINLINE void summarizeLoop(mlir::Block &body, int64_t trips) {
    Mark zero = mark();
    std::array<SegmentWalk::Mark, 4> segmentStates;
    segmentStates[0] = segmentsNow();
    iteration.back() = 0;
    block<RunTime>(body);
    Mark one = mark();
    segmentStates[1] = segmentsNow();
    iteration.back() = 1;
    block<RunTime>(body);
    Mark two = mark();
    segmentStates[2] = segmentsNow();
    auto framesOne = frames;
    auto reachedOne = reached;
    auto latestOne = latest;
    iteration.back() = 2;
    block<RunTime>(body);
    Mark three = mark();
    segmentStates[3] = segmentsNow();
    std::map<int64_t, int64_t> periods;
    auto period = repeating(zero, one, two, three, framesOne, reachedOne,
                            latestOne, periods);
    if constexpr (RunTime)
      if (period && !sameRunTime(zero, one, two, three))
        period.reset();
    if (period && segmentWalk && !SegmentWalk::steady(segmentStates))
      period.reset();
    // The segments the copies after the first open, as if visited.
    if (period && segmentWalk &&
        !segmentWalk->skip(trips - 3, segmentStates[2]))
      throw std::overflow_error("a rolled loop opens more than 2**63 segments");
    if (period && *period < 0) {
      summarizeUnsynced(two, three, trips, reachedOne, periods);
      return;
    }
    if (period) {
      int64_t copies = trips - 2;
      int64_t stride = three.branch - two.branch;
      auto overflowed = [] {
        return std::overflow_error("a rolled loop runs past 2**63 cycles or "
                                   "branches");
      };
      std::optional<int64_t> shift = checkedMul(copies - 1, *period);
      std::optional<int64_t> branches = checkedMul(copies - 1, stride);
      if (!shift || !branches)
        throw overflowed();
      std::optional<int64_t> next = checkedAdd(nextBranch, *branches);
      if (!next)
        throw overflowed();
      nextBranch = *next;
      for (auto &[allocation, time] : reached)
        if (time && *time != *reachedOne[allocation]) {
          std::optional<int64_t> later = checkedAdd(*time, *shift);
          if (!later)
            throw overflowed();
          time = *later;
        }
      repeats.push_back({static_cast<int64_t>(two.event),
                         static_cast<int64_t>(three.event), copies, *period,
                         static_cast<int64_t>(iteration.size() - 1), two.branch,
                         stride});
      return;
    }
    for (int64_t k = 3; k < trips; ++k) {
      // A walk at run time plays out a loop that does not repeat, as the
      // walk of every iteration does.
      if (!RunTime && result.size() > kMaxEvents)
        throw std::length_error(
            "the controller check plays more than " +
            std::to_string(kMaxEvents) +
            " events of a rolled loop whose iterations do not repeat one "
            "period apart; give the loop fewer iterations, or sync its lines "
            "so that every iteration plays alike");
      iteration.back() = k;
      block<RunTime>(body);
    }
  }

  /// Whether iteration 2 (after `two`) computes at run time what iteration
  /// 1 (after `one`) does, as `repeating` compares their events: the same
  /// waveform operands, wait lengths and bounds and submitted values, term
  /// by term, each readout or answer they refer to in the iteration before
  /// or in their own one iteration later; and a phase known where the
  /// other's is.
  LLVM_ATTRIBUTE_NOINLINE bool sameRunTime(Mark zero, Mark one, Mark two,
                                           Mark three) const {
    if (three.known - two.known != two.known - one.known)
      return false;
    const auto len = static_cast<int64_t>(two.event - one.event);
    const auto low = static_cast<int64_t>(zero.event);
    const auto high = static_cast<int64_t>(two.event);
    auto alike = [&](int32_t a, int32_t b) {
      return sameExpression(a, b, low, high, len);
    };
    for (size_t i = 0; i < two.event - one.event; ++i)
      if (std::isnan(std::get<8>(result[one.event + i])) !=
          std::isnan(std::get<8>(result[two.event + i])))
        return false;
    for (size_t i = 0; i < two.known - one.known; ++i) {
      const Known &a = known[one.known + i];
      const Known &b = known[two.known + i];
      if (b.event != a.event + len || a.tone >= 0 || b.tone >= 0 ||
          a.minCycles != b.minCycles || a.maxCycles != b.maxCycles ||
          (a.length < 0) != (b.length < 0) ||
          (a.length >= 0 && !alike(a.length, b.length)) ||
          a.waveform.size() != b.waveform.size() ||
          a.operands.has_value() != b.operands.has_value())
        return false;
      for (size_t k = 0; k < a.waveform.size(); ++k) {
        const auto &[nodeA, indexA, valueA] = a.waveform[k];
        const auto &[nodeB, indexB, valueB] = b.waveform[k];
        if (nodeA != nodeB || indexA != indexB || !alike(valueA, valueB))
          return false;
      }
      if (a.operands) {
        if (a.operands->size() != b.operands->size())
          return false;
        for (size_t k = 0; k < a.operands->size(); ++k)
          if (!alike((*a.operands)[k], (*b.operands)[k]))
            return false;
      }
    }
    return true;
  }

  /// The later copies of a loop whose lines repeat at `periods`, as
  /// summarizeLoop records those of one period: each line's reach moves by
  /// its own period, which iteration 2 moved it by.
  LLVM_ATTRIBUTE_NOINLINE void summarizeUnsynced(
      Mark two, Mark three, int64_t trips,
      const llvm::DenseMap<mlir::Operation *, std::optional<int64_t>>
          &reachedOne,
      const std::map<int64_t, int64_t> &periods) {
    int64_t copies = trips - 2;
    int64_t stride = three.branch - two.branch;
    auto overflowed = [] {
      return std::overflow_error("a rolled loop runs past 2**63 cycles or "
                                 "branches");
    };
    std::optional<int64_t> branches = checkedMul(copies - 1, stride);
    if (!branches)
      throw overflowed();
    std::optional<int64_t> next = checkedAdd(nextBranch, *branches);
    if (!next)
      throw overflowed();
    nextBranch = *next;
    for (auto &[allocation, time] : reached) {
      const std::optional<int64_t> &before = reachedOne.at(allocation);
      if (!time || *time == *before)
        continue;
      std::optional<int64_t> shift = checkedMul(copies - 1, *time - *before);
      if (!shift)
        throw overflowed();
      std::optional<int64_t> later = checkedAdd(*time, *shift);
      if (!later)
        throw overflowed();
      time = *later;
    }
    const auto index = static_cast<int64_t>(repeats.size());
    for (auto [line, period] : periods) {
      if (!checkedMul(copies - 1, period))
        throw overflowed();
      linePeriods.push_back({index, line, period});
    }
    repeats.push_back({static_cast<int64_t>(two.event),
                       static_cast<int64_t>(three.event), copies, -1,
                       static_cast<int64_t>(iteration.size() - 1), two.branch,
                       stride});
  }

  /// Whether repeats `a` and `b`, of lines of their own periods, have the
  /// same lines at the same periods.
  bool samePeriods(int64_t a, int64_t b) const {
    std::vector<std::pair<int64_t, int64_t>> first, second;
    for (const auto &[index, line, period] : linePeriods) {
      if (index == a)
        first.emplace_back(line, period);
      else if (index == b)
        second.emplace_back(line, period);
    }
    return first == second;
  }

  /// The period of each line of the events [one, two), by how far iteration
  /// 2 moved their starts: those of a line by one period, a sync by that of
  /// each line it syncs. None when a line's events move by more than one
  /// period, or a sync syncs lines of two.
  LLVM_ATTRIBUTE_NOINLINE std::optional<std::map<int64_t, int64_t>>
  periodsOf(Mark one, Mark two) const {
    std::map<int64_t, int64_t> periods;
    auto agree = [&](int64_t line, int64_t shift) {
      auto [it, inserted] = periods.try_emplace(line, shift);
      return inserted || it->second == shift;
    };
    const size_t len = two.event - one.event;
    for (size_t i = 0; i < len; ++i) {
      const Event &a = result[one.event + i];
      const Event &b = result[two.event + i];
      if (!std::get<3>(a) || std::get<1>(a) < 0)
        continue;
      // A call's line is its link, which it shares with other calls: each
      // submit and answer has a period of its own, the line kCallLine - i
      // for the i-th event of the body.
      int64_t line = std::get<1>(a);
      if (std::get<0>(a) == "submit" || std::get<0>(a) == "answer")
        line = kCallLine - static_cast<int64_t>(i);
      if (!agree(line, *std::get<3>(b) - *std::get<3>(a)))
        return std::nullopt;
    }
    for (size_t i = 0; i < len; ++i) {
      const Event &a = result[one.event + i];
      const Event &b = result[two.event + i];
      if (!std::get<3>(a) || std::get<1>(a) >= 0)
        continue;
      const std::vector<int64_t> &synced = std::get<10>(a);
      if (synced.empty())
        return std::nullopt;
      for (int64_t line : synced)
        if (!agree(line, *std::get<3>(b) - *std::get<3>(a)))
          return std::nullopt;
    }
    return periods;
  }

  /// Whether iteration 2 of the loop being walked, after `two`, plays
  /// iteration 1 (after `one`) again and leaves the walk as iteration 1 did,
  /// shifted by one iteration; its period, or 0 with nothing scheduled.
  /// With `unsynced`, -1 when its lines repeat at the `periods` of their
  /// own (periodsOf).
  std::optional<int64_t>
  repeating(Mark zero, Mark one, Mark two, Mark three,
            const llvm::DenseMap<mlir::Operation *, Frame> &framesOne,
            const llvm::DenseMap<mlir::Operation *, std::optional<int64_t>>
                &reachedOne,
            const llvm::DenseMap<mlir::Operation *, int64_t> &latestOne,
            std::map<int64_t, int64_t> &periods) {
    const int64_t len = static_cast<int64_t>(two.event - one.event);
    const int64_t stride = two.branch - one.branch;
    if (three.event - two.event != two.event - one.event ||
        three.repeat - two.repeat != two.repeat - one.repeat ||
        three.branch - two.branch != stride)
      return std::nullopt;
    const size_t depth = iteration.size() - 1;
    auto same = [](double a, double b) {
      return a == b || (std::isnan(a) && std::isnan(b));
    };
    auto ref = [&](int64_t a, int64_t b) {
      bool moves = a >= static_cast<int64_t>(zero.event) &&
                   a < static_cast<int64_t>(two.event);
      return b == (moves ? a + len : a);
    };
    auto refs = [&](const std::vector<int64_t> &a,
                    const std::vector<int64_t> &b) {
      if (a.size() != b.size())
        return false;
      for (size_t k = 0; k < a.size(); ++k)
        if (!ref(a[k], b[k]))
          return false;
      return true;
    };
    auto branch = [&](int64_t a, int64_t b) {
      return b == (a >= zero.branch ? a + stride : a);
    };
    auto nodes = [&](const std::vector<WaveformNode> &a,
                     const std::vector<WaveformNode> &b) {
      if (a.size() != b.size())
        return false;
      for (size_t k = 0; k < a.size(); ++k) {
        auto &[nameA, numbersA, childrenA] = a[k];
        auto &[nameB, numbersB, childrenB] = b[k];
        if (nameA != nameB || childrenA != childrenB ||
            numbersA.size() != numbersB.size())
          return false;
        for (size_t n = 0; n < numbersA.size(); ++n)
          if (!same(numbersA[n], numbersB[n]))
            return false;
      }
      return true;
    };
    std::optional<int64_t> period;
    bool mixed = false;
    for (size_t i = 0; i < static_cast<size_t>(len); ++i) {
      const Event &a = result[one.event + i];
      const Event &b = result[two.event + i];
      if (std::get<0>(a) != std::get<0>(b) ||
          std::get<1>(a) != std::get<1>(b) ||
          std::get<2>(a) != std::get<2>(b) ||
          std::get<4>(a) != std::get<4>(b) ||
          std::get<10>(a) != std::get<10>(b) ||
          std::get<12>(a) != std::get<12>(b) ||
          std::get<13>(a) != std::get<13>(b) ||
          !same(std::get<7>(a), std::get<7>(b)) ||
          !same(std::get<14>(a), std::get<14>(b)) ||
          !nodes(std::get<6>(a), std::get<6>(b)) ||
          !refs(std::get<11>(a), std::get<11>(b)) ||
          (track && natives[one.event + i] != natives[two.event + i]))
        return std::nullopt;
      auto &startA = std::get<3>(a);
      auto &startB = std::get<3>(b);
      if (startA.has_value() != startB.has_value())
        return std::nullopt;
      if (startA) {
        int64_t shift = *startB - *startA;
        if (shift < 0)
          return std::nullopt;
        if (period && shift != *period) [[unlikely]] {
          if (!unsynced)
            return std::nullopt;
          mixed = true;
        }
        period = shift;
      }
      auto &iterationA = std::get<5>(a);
      auto &iterationB = std::get<5>(b);
      if (iterationA.size() != iterationB.size())
        return std::nullopt;
      for (size_t k = 0; k < iterationA.size(); ++k)
        if (iterationB[k] != iterationA[k] + (k == depth ? 1 : 0))
          return std::nullopt;
      auto &conditionsA = std::get<9>(a);
      auto &conditionsB = std::get<9>(b);
      if (conditionsA.size() != conditionsB.size())
        return std::nullopt;
      for (size_t k = 0; k < conditionsA.size(); ++k) {
        auto &[idA, armA, readoutsA, answersA, testA] = conditionsA[k];
        auto &[idB, armB, readoutsB, answersB, testB] = conditionsB[k];
        if (!branch(idA, idB) || armA != armB || testA != testB ||
            !refs(readoutsA, readoutsB) || !refs(answersA, answersB))
          return std::nullopt;
      }
    }
    for (size_t i = 0; i < two.repeat - one.repeat; ++i) {
      const Repeat &a = repeats[one.repeat + i];
      const Repeat &b = repeats[two.repeat + i];
      if (b[0] != a[0] + len || b[1] != a[1] + len || b[2] != a[2] ||
          b[3] != a[3] || b[4] != a[4] || !branch(a[5], b[5]) || b[6] != a[6])
        return std::nullopt;
      if (a[3] < 0 && !samePeriods(static_cast<int64_t>(one.repeat + i),
                                   static_cast<int64_t>(two.repeat + i)))
        return std::nullopt;
    }
    if (mixed) {
      auto found = periodsOf(one, two);
      if (!found)
        return std::nullopt;
      periods = std::move(*found);
    }
    int64_t shift = period.value_or(0);
    if (frames.size() != framesOne.size() ||
        reached.size() != reachedOne.size() ||
        latest.size() != latestOne.size())
      return std::nullopt;
    for (auto &[root, frame] : frames) {
      auto found = framesOne.find(root);
      if (found == framesOne.end() ||
          !same(frame.frequencyHz, found->second.frequencyHz) ||
          !same(frame.frameFrequencyHz, found->second.frameFrequencyHz))
        return std::nullopt;
    }
    for (auto &[allocation, time] : reached) {
      auto found = reachedOne.find(allocation);
      if (found == reachedOne.end() ||
          time.has_value() != found->second.has_value())
        return std::nullopt;
      if (!time || *time == *found->second)
        continue;
      if (mixed) {
        auto line = periods.find(lineOf(allocation));
        if (line == periods.end() || *time != *found->second + line->second)
          return std::nullopt;
      } else if (*time != *found->second + shift) {
        return std::nullopt;
      }
    }
    for (auto &[op, index] : latest) {
      auto found = latestOne.find(op);
      if (found == latestOne.end() || !ref(found->second, index))
        return std::nullopt;
    }
    if (mixed)
      return -1;
    return shift;
  }
};

/// The lines and events of a ControllerEventWalk as controller.export() gives
/// them, built as Python objects in one pass; needs the GIL. A number the
/// format has always written as JSON text and read back (an expression's
/// constant, a frame change's value, a phase offset) is an int where that
/// text has no fraction or exponent.
class ControllerObjects {
public:
  /// `parameters`: the parameter names of each waveform shape, in operand
  /// order; `amplitudes`: the shapes with an overall amplitude.
  ControllerObjects(nb::handle parameters, nb::handle amplitudes)
      : keys(sharedKeys()) {
    for (nb::handle shape : parameters) {
      nb::object names = parameters[shape];
      Shape &info = shapes[nb::cast<std::string>(shape)];
      info.amplitude = PySequence_Contains(amplitudes.ptr(), shape.ptr()) == 1;
      for (nb::handle name : names) {
        auto text = nb::cast<std::string>(name);
        int index = static_cast<int>(info.names.size());
        if (text == "amplitude")
          info.real = index;
        else if (text == "amplitude_imag")
          info.imag = index;
        else if (text == "factor")
          info.factor = index;
        info.names.push_back(nb::borrow(name));
      }
    }
  }

  nb::list lines(const std::vector<std::tuple<std::string, int64_t>> &lines) {
    nb::list out = list(lines.size());
    for (size_t i = 0; i < lines.size(); ++i) {
      nb::object line = dict();
      put(line, keys.line, integer(static_cast<int64_t>(i)));
      put(line, keys.role, string(std::get<0>(lines[i])));
      put(line, keys.qudit, integer(std::get<1>(lines[i])));
      set(out, i, std::move(line));
    }
    return out;
  }

  nb::list events(const ControllerEventWalk &walk) {
    // Nothing built here is garbage until it is returned: a collection now
    // would only walk the heap.
    CollectorPause pause;
    this->walk = &walk;
    const std::vector<int64_t> *segments =
        walk.segmentWalk ? &walk.segments : nullptr;
    nb::list out = list(walk.result.size());
    auto known = walk.known.begin();
    for (size_t i = 0; i < walk.result.size(); ++i) {
      const ControllerEventWalk::Known *extra = nullptr;
      if (known != walk.known.end() && known->event == static_cast<int64_t>(i))
        extra = &*known++;
      set(out, i, event(i, walk.result[i], extra, segments));
    }
    return out;
  }

private:
  struct CollectorPause {
    int enabled = PyGC_Disable();
    ~CollectorPause() {
      if (enabled)
        PyGC_Enable();
    }
  };

  struct Shape {
    std::vector<nb::object> names;
    int real = -1, imag = -1, factor = -1;
    bool amplitude = false;
  };
  llvm::StringMap<Shape> shapes;
  llvm::StringMap<nb::object> strings;
  llvm::DenseMap<uint64_t, nb::object> floats;
  const ControllerEventWalk *walk = nullptr;

  static nb::object key(const char *text) {
    return check(PyUnicode_InternFromString(text));
  }
  struct Keys {
    nb::object amplitude = key("amplitude"), answer = key("answer"),
               answers = key("answers"), arm = key("arm"),
               branch = key("branch"), conditions = key("conditions"),
               constant = key("const"), controller = key("controller"),
               duration = key("duration_cycles"), end = key("end_cycles"),
               segment = key("segment"), frameUpdates = key("frame_updates"),
               frequency = key("frequency_hz"), function = key("function"),
               id = key("id"), im = key("im"), index = key("index"),
               iteration = key("iteration"), kind = key("kind"),
               length = key("length"), line = key("line"), lines = key("lines"),
               link = key("link"), maxCycles = key("max_cycles"),
               measurements = key("measurements"),
               minCycles = key("min_cycles"), name = key("name"),
               of = key("of"), op = key("op"), operands = key("operands"),
               params = key("params"), phase = key("phase_rad"),
               phaseOffset = key("phase_offset_rad"), qudit = key("qudit"),
               re = key("re"), readout = key("readout"),
               readouts = key("readouts"), result = key("result"),
               role = key("role"), runtime = key("runtime"),
               samples = key("samples"), shape = key("shape"),
               start = key("start_cycles"), term = key("term"),
               table = key("terms"), test = key("test"),
               timeout = key("timeout_ns"), tone = key("tone"),
               type = key("type"), unknown = key("unknown"),
               value = key("value"), values = key("values"), wait = key("wait"),
               waveform = key("waveform");
    using Expression = ControllerExpressions::Expression;
    std::array<nb::object, Expression::Terms> terms;
    std::array<nb::object, Expression::Types> types;

    Keys() {
      for (size_t i = 0; i < terms.size(); ++i)
        terms[i] = key(Expression::names[i]);
      for (size_t i = 0; i < types.size(); ++i)
        types[i] = key(Expression::typeNames[i]);
    }
  };
  /// Interned once, for the life of the process.
  static const Keys &sharedKeys() {
    static const Keys *keys = new Keys();
    return *keys;
  }
  const Keys &keys;

  static nb::object check(PyObject *object) {
    if (!object)
      throw nb::python_error();
    return nb::steal(object);
  }
  static nb::object dict() { return check(PyDict_New()); }
  static nb::list list(size_t size) {
    return nb::steal<nb::list>(
        check(PyList_New(static_cast<Py_ssize_t>(size))).release());
  }
  static void put(nb::handle dict, nb::handle key, nb::handle value) {
    if (PyDict_SetItem(dict.ptr(), key.ptr(), value.ptr()) != 0)
      throw nb::python_error();
  }
  static void set(nb::handle list, size_t index, nb::object value) {
    if (PyList_SetItem(list.ptr(), static_cast<Py_ssize_t>(index),
                       value.release().ptr()) != 0)
      throw nb::python_error();
  }
  static nb::object integer(int64_t value) {
    return check(PyLong_FromLongLong(value));
  }
  /// A float, one object per value: an export repeats few numbers many
  /// times.
  nb::object real(double value) {
    auto bits = std::bit_cast<uint64_t>(value);
    // Two NaNs are the map's own empty and erased keys.
    if (bits >= llvm::DenseMapInfo<uint64_t>::getTombstoneKey())
      return check(PyFloat_FromDouble(value));
    auto [it, inserted] = floats.try_emplace(bits);
    if (inserted)
      it->second = check(PyFloat_FromDouble(value));
    return it->second;
  }
  /// A number, None where it is NaN.
  nb::object number(double value) {
    return std::isnan(value) ? nb::none() : real(value);
  }
  /// A number as the JSON text "%.17g" of it reads back.
  nb::object textNumber(double value) {
    if (std::isfinite(value) && value == std::trunc(value) &&
        std::fabs(value) < 1e17)
      return integer(static_cast<int64_t>(value));
    return real(value);
  }
  static nb::object optional(std::optional<int64_t> value) {
    return value ? integer(*value) : nb::none();
  }
  static nb::object integers(const std::vector<int64_t> &values) {
    nb::list out = list(values.size());
    for (size_t i = 0; i < values.size(); ++i)
      set(out, i, integer(values[i]));
    return out;
  }
  nb::object string(llvm::StringRef text) {
    auto [it, inserted] = strings.try_emplace(text);
    if (inserted)
      it->second = check(PyUnicode_FromStringAndSize(
          text.data(), static_cast<Py_ssize_t>(text.size())));
    return it->second;
  }

  /// A branch an event lies in, keys in the order of the event's conditions,
  /// else (for a frame change's) sorted.
  nb::object condition(const ControllerEventWalk::Condition &condition,
                       bool sorted) {
    const auto &[id, arm, readouts, answers, test] = condition;
    nb::object out = dict();
    if (sorted) {
      put(out, keys.answers, integers(answers));
      put(out, keys.arm, nb::bool_(arm != 0));
      put(out, keys.branch, integer(id));
      put(out, keys.readouts, integers(readouts));
    } else {
      put(out, keys.branch, integer(id));
      put(out, keys.arm, nb::bool_(arm != 0));
      put(out, keys.readouts, integers(readouts));
      put(out, keys.answers, integers(answers));
    }
    put(out, keys.test, string(test));
    return out;
  }

  nb::object conditions(const std::vector<ControllerEventWalk::Condition> &all,
                        bool sorted) {
    nb::list out = list(all.size());
    for (size_t i = 0; i < all.size(); ++i)
      set(out, i, condition(all[i], sorted));
    return out;
  }

  nb::object expression(int32_t index) {
    using Expression = ControllerEventWalk::Expression;
    return walk->written(
        index, [&](int32_t at) { return leaf(at); },
        [&](int n) {
          nb::object out = dict();
          put(out, keys.term, integer(n));
          return out;
        },
        [&](const Expression &e, nb::handle of, nb::handle table) {
          nb::object out = dict();
          if (table.is_valid())
            put(out, keys.table, table);
          put(out, keys.op, keys.terms[e.term]);
          put(out, keys.type, keys.types[e.type]);
          put(out, keys.of, of);
          return out;
        });
  }

  /// What is no term of an expression.
  nb::object leaf(int32_t index) {
    using Expression = ControllerEventWalk::Expression;
    const Expression &e = walk->expressions[index];
    if (e.kind == Expression::Number)
      return textNumber(e.number);
    nb::object out = dict();
    switch (e.kind) {
    case Expression::Constant:
      if (e.type == Expression::Bool)
        put(out, keys.constant, nb::bool_(e.number != 0.0));
      else
        put(out, keys.constant, textNumber(e.number));
      break;
    case Expression::Unknown:
      put(out, keys.unknown, string(e.name));
      break;
    case Expression::Readout:
      put(out, keys.readout, integer(e.event));
      break;
    case Expression::Answer:
      put(out, keys.answer, integer(e.event));
      put(out, keys.result, integer(e.result));
      break;
    default:
      break;
    }
    return out;
  }

  /// The names of a node's numbers, if its shape names them all.
  const Shape *named(llvm::StringRef shape, size_t numbers) const {
    auto found = shapes.find(shape);
    if (found == shapes.end() || found->second.names.size() != numbers)
      return nullptr;
    return &found->second;
  }

  /// The waveform expression from node `index` down, with the operands
  /// computed at run time (`runtime`, sorted) on their nodes.
  nb::object
  tree(const std::vector<ControllerEventWalk::WaveformNode> &nodes,
       int64_t index,
       llvm::ArrayRef<std::tuple<int64_t, int64_t, int32_t>> runtime) {
    const auto &[name, numbers, children] = nodes[index];
    auto [shape, custom] = llvm::StringRef(name).split(':');
    nb::object node = dict();
    put(node, keys.shape, string(shape));
    if (!custom.empty())
      put(node, keys.name, string(custom));
    const Shape *names = named(shape, numbers.size());
    if (shape == "custom_samples") {
      nb::list samples = list(numbers.size());
      for (size_t i = 0; i < numbers.size(); ++i)
        set(samples, i, real(numbers[i]));
      put(node, keys.samples, samples);
    } else if (names) {
      nb::object params = dict();
      for (size_t i = 0; i < numbers.size(); ++i)
        put(params, names->names[i], number(numbers[i]));
      put(node, keys.params, params);
    } else if (!numbers.empty()) {
      nb::list values = list(numbers.size());
      for (size_t i = 0; i < numbers.size(); ++i)
        set(values, i, number(numbers[i]));
      put(node, keys.values, values);
    }
    if (!children.empty()) {
      nb::list of = list(children.size());
      for (size_t i = 0; i < children.size(); ++i)
        set(of, i, tree(nodes, children[i], runtime));
      put(node, keys.of, of);
    }
    nb::object computed;
    for (const auto &[at, operand, expr] : runtime) {
      if (at != index)
        continue;
      if (!computed)
        computed = dict();
      put(computed,
          names ? nb::handle(names->names[operand])
                : nb::handle(string(std::to_string(operand))),
          expression(expr));
    }
    if (computed)
      put(node, keys.runtime, computed);
    return node;
  }

  /// The overall complex amplitude of node `index`, through scale and neg,
  /// in Python's complex arithmetic; None where it is unknown.
  nb::object
  amplitude(const std::vector<ControllerEventWalk::WaveformNode> &nodes,
            int64_t index) {
    const auto &[name, numbers, children] = nodes[index];
    llvm::StringRef shape = llvm::StringRef(name).split(':').first;
    const Shape *names = named(shape, numbers.size());
    if (shape == "scale" || shape == "neg") {
      if (children.empty())
        return nb::none();
      nb::object inner = amplitude(nodes, children[0]);
      if (inner.is_none())
        return inner;
      if (shape == "neg")
        return check(PyNumber_Negative(inner.ptr()));
      if (!names || names->factor < 0 || std::isnan(numbers[names->factor]))
        return nb::none();
      return check(
          PyNumber_Multiply(inner.ptr(), real(numbers[names->factor]).ptr()));
    }
    if (shape == "custom_samples") {
      if (numbers.empty())
        return nb::none();
      double peak = numbers[0];
      for (double sample : numbers)
        if (std::fabs(sample) > std::fabs(peak))
          peak = sample;
      return check(PyComplex_FromDoubles(peak, 0.0));
    }
    if (!names || !names->amplitude || names->real < 0)
      return nb::none();
    double re = numbers[names->real];
    double im = names->imag < 0 ? 0.0 : numbers[names->imag];
    if (std::isnan(re) || std::isnan(im))
      return nb::none();
    return check(PyComplex_FromDoubles(re, im));
  }

  nb::object
  frameUpdates(const std::vector<ControllerEventWalk::FrameUpdate> &all) {
    nb::list out = list(all.size());
    for (size_t i = 0; i < all.size(); ++i) {
      const auto &update = all[i];
      nb::object item = dict();
      put(item, keys.conditions, conditions(update.conditions, true));
      put(item, keys.id, integer(static_cast<int64_t>(update.seq)));
      put(item, keys.op, string(update.op));
      put(item, keys.value, expression(update.value));
      set(out, i, std::move(item));
    }
    return out;
  }

  nb::object event(size_t index, const ControllerEventWalk::Event &item,
                   const ControllerEventWalk::Known *extra,
                   const std::vector<int64_t> *segments) {
    const auto &[kind, line, qudit, start, duration, iteration, nodes,
                 frequency, phase, branches, joined, refs, callee, mode,
                 timeout] = item;
    bool sync = kind == "sync";
    bool call = kind == "submit" || kind == "answer";
    nb::object event = dict();
    put(event, keys.index, integer(static_cast<int64_t>(index)));
    put(event, keys.kind, string(kind));
    put(event, keys.line, sync ? nb::none() : integer(line));
    put(event, keys.qudit, sync ? nb::none() : integer(call ? -1 : qudit));
    put(event, keys.start, optional(start));
    put(event, keys.duration, optional(duration));
    put(event, keys.end,
        start && duration ? integer(*start + *duration) : nb::none());
    put(event, keys.iteration, integers(iteration));
    put(event, keys.segment,
        integer(segments && start ? (*segments)[index] : 0));
    put(event, keys.conditions, conditions(branches, false));
    if (kind == "drive" || kind == "readout") {
      llvm::ArrayRef<std::tuple<int64_t, int64_t, int32_t>> runtime;
      std::vector<std::tuple<int64_t, int64_t, int32_t>> sortedRuntime;
      if (extra && !extra->waveform.empty()) {
        sortedRuntime = extra->waveform;
        llvm::stable_sort(sortedRuntime, [](const auto &a, const auto &b) {
          return std::make_pair(std::get<0>(a), std::get<1>(a)) <
                 std::make_pair(std::get<0>(b), std::get<1>(b));
        });
        runtime = sortedRuntime;
      }
      int64_t root = static_cast<int64_t>(nodes.size()) - 1;
      put(event, keys.waveform,
          nodes.empty() ? nb::none() : tree(nodes, root, runtime));
      nb::object overall = nodes.empty() ? nb::none() : amplitude(nodes, root);
      if (overall.is_none()) {
        put(event, keys.amplitude, overall);
      } else {
        nb::object parts = dict();
        put(parts, keys.re, real(PyComplex_RealAsDouble(overall.ptr())));
        put(parts, keys.im, real(PyComplex_ImagAsDouble(overall.ptr())));
        put(event, keys.amplitude, parts);
      }
      put(event, keys.frequency, number(frequency));
      put(event, keys.phase, number(phase));
    } else if (sync) {
      put(event, keys.lines, integers(joined));
    } else if (call) {
      put(event, keys.link, integer(line));
      if (qudit >= 0)
        put(event, keys.controller, integer(qudit));
      put(event, keys.function, string(callee));
      if (kind == "submit") {
        put(event, keys.measurements, integers(refs));
      } else {
        put(event, keys.wait, string(mode));
        put(event, keys.timeout, number(timeout));
      }
    }
    if (!extra)
      return event;
    if (extra->tone >= 0) {
      put(event, keys.tone, integer(extra->tone));
      put(event, keys.frameUpdates, frameUpdates(extra->updates));
      if (extra->offset)
        put(event, keys.phaseOffset, textNumber(*extra->offset));
    }
    if (extra->length >= 0) {
      put(event, keys.length, expression(extra->length));
      put(event, keys.minCycles, optional(extra->minCycles));
      put(event, keys.maxCycles, optional(extra->maxCycles));
    }
    if (extra->operands) {
      nb::list operands = list(extra->operands->size());
      for (size_t i = 0; i < extra->operands->size(); ++i)
        set(operands, i, expression((*extra->operands)[i]));
      put(event, keys.operands, operands);
    }
    return event;
  }
};

/// The carrier of each runtime-qudit slot of a specialization, and the
/// device's frequencies by carrier (0-1, and of readout lines' tones), which
/// the caller keeps alive.
struct Slots {
  std::vector<int64_t> carriers;
  const double *frequencies = nullptr;
  size_t count = 0;
  const double *readoutFrequencies = nullptr;
  size_t readoutCount = 0;
};
#include "ControllerProgram.h"

// -----------------------------------------------------------------------
// Simulation windows
// -----------------------------------------------------------------------
// A path through a program with classical calls is simulated by binding
// each answer and scheduling the program again. In a program whose lines
// share one timeline, a run-time wait (a variable wait, or a blocking wait)
// holds back every line: nothing after it starts before it ends, nothing
// before it ends later, and it keeps the time the ops before it give it.
// So from such a wait on, the schedule is that of the rest of the program
// scheduled alone and moved to start where the wait ends, and up to the
// next one whose length is not bound yet it is that of the program cut
// there. A window is that part: the top-level ops of the unrolled program
// between two such waits, after the allocations, lines and links, with the
// answers it reads from before it as constants and each tone's frame where
// it starts. Binding an answer on a path schedules only its window, so a
// long loop of calls costs the same per answer at any length.

/// What windows are cut from: the unrolled simulation module, scheduled,
/// before any answer is bound, and where each window may start and end.
struct SimulationProgram {
  mlir::OwningOpRef<mlir::ModuleOp> module;
  /// The top-level ops of main, its terminator excluded.
  llvm::SmallVector<mlir::Operation *> ops;
  /// The readouts and calls, in walk order, before each top-level op (one
  /// more entry than ops).
  std::vector<int64_t> readoutsBefore, submitsBefore;
  /// The top-level op each call is in, by call index.
  std::vector<int64_t> submitAt;
  /// The call each classical.wait takes the answer of.
  llvm::DenseMap<mlir::Operation *, int64_t> callOfWait;
  /// Allocations, lines and links: every window starts with them, in this
  /// order, so lines are numbered as in the program.
  llvm::SmallVector<int64_t> roots;
  /// The op each top-level line, tone or link value continues, and those ops.
  llvm::DenseMap<mlir::Value, mlir::Operation *> rootOf;
  llvm::DenseSet<mlir::Operation *> rootOps;
  /// Top-level run-time waits, with the calls whose answers their length
  /// depends on.
  std::map<int64_t, llvm::SmallVector<int64_t, 1>> barriers;
  /// Barriers a call is in flight across, and barriers some value a window
  /// cannot start with crosses.
  llvm::DenseSet<int64_t> crossedByCall, crossed;
  llvm::DenseMap<mlir::Operation *, int64_t> position;
  bool eligible = false;
};

static bool isThreadedType(mlir::Type type) {
  return mlir::isa<pulse::DriveLineType, pulse::ReadoutLineType,
                   pulse::ToneType, classical::LinkType>(type);
}

static bool isReadoutOp(mlir::Operation *op) {
  return mlir::isa<pulse::ReadoutOp>(op) ||
         op->getName().getStringRef() == "pulse.iq_acquire";
}

/// The windows of `source`, a simulation module before any answer is
/// bound; not eligible where a cut could change its schedule.
static std::shared_ptr<const SimulationProgram>
simulationProgram(mlir::ModuleOp source) {
  auto program = std::make_shared<SimulationProgram>();
  program->module = mlir::OwningOpRef<mlir::ModuleOp>(source.clone());
  mlir::ModuleOp module = *program->module;
  auto main = module.lookupSymbol<mlir::func::FuncOp>("main");
  if (!main || !main.getBody().hasOneBlock() || main.getNumArguments() != 0 ||
      main.getBody().front().getTerminator()->getNumOperands() != 0 ||
      main->hasAttr("pulse.selected_values"))
    return program;
  // Grids, several controllers and line delays place lines apart from the
  // time a wait ends; a program budget is of the whole shot.
  pulse::ClockAttributes clocks(module);
  if (pulse::LineGrid(clocks).enabled() ||
      (clocks.system && clocks.system.getControllers().size() > 1) ||
      clocks.delayed() || pulse::programBudget(module) ||
      module->hasAttr("pulse.program_limits") ||
      pulse::persistentFrames(module) || pulse::blocksOnCalls(module))
    return program;
  mlir::WalkResult unsupported = main.walk([](mlir::Operation *op) {
    if (mlir::isa<pulse::ScheduleBlockOp>(op) ||
        op->hasAttr(classical::kTimelineAttr) ||
        op->hasAttr("classical.bound") || op->hasAttr("classical.answers"))
      return mlir::WalkResult::interrupt();
    return mlir::WalkResult::advance();
  });
  if (unsupported.wasInterrupted())
    return program;

  mlir::Block &body = main.getBody().front();
  for (mlir::Operation &op : body.without_terminator()) {
    program->position[&op] = static_cast<int64_t>(program->ops.size());
    program->ops.push_back(&op);
  }
  const int64_t count = static_cast<int64_t>(program->ops.size());
  auto topOf = [&](mlir::Operation *op) -> int64_t {
    while (op && op->getBlock() != &body)
      op = op->getParentOp();
    return op ? program->position.lookup(op) : -1;
  };

  // Readouts and calls in the walk order the module's own numbering uses.
  llvm::DenseMap<mlir::Operation *, int64_t> callIndex;
  program->readoutsBefore.assign(count + 1, 0);
  program->submitsBefore.assign(count + 1, 0);
  for (int64_t i = 0; i < count; ++i) {
    int64_t readouts = 0;
    program->ops[i]->walk([&](mlir::Operation *op) {
      if (isReadoutOp(op))
        ++readouts;
      else if (mlir::isa<classical::SubmitOp>(op)) {
        callIndex[op] = static_cast<int64_t>(program->submitAt.size());
        program->submitAt.push_back(i);
      }
    });
    program->readoutsBefore[i + 1] = program->readoutsBefore[i] + readouts;
    program->submitsBefore[i + 1] =
        static_cast<int64_t>(program->submitAt.size());
  }
  bool unknownFuture = false;
  main.walk([&](classical::WaitOp wait) {
    auto submit = wait.getFuture().getDefiningOp<classical::SubmitOp>();
    if (!submit) {
      unknownFuture = true;
      return;
    }
    program->callOfWait[wait] = callIndex.lookup(submit);
  });
  if (unknownFuture)
    return program;

  // The op each line, tone and link continues.
  for (int64_t i = 0; i < count; ++i) {
    mlir::Operation *op = program->ops[i];
    if (mlir::isa<pulse::GetDriveLineOp, pulse::GetReadoutLineOp,
                  pulse::QuditAllocOp>(op) ||
        op->getName().getStringRef() == "classical.get_link")
      program->roots.push_back(i);
    for (mlir::Value result : op->getResults()) {
      if (!isThreadedType(result.getType()))
        continue;
      llvm::SmallVector<mlir::Value, 8> chain;
      mlir::Operation *root = nullptr;
      mlir::Value current = result;
      while (true) {
        auto known = program->rootOf.find(current);
        if (known != program->rootOf.end()) {
          root = known->second;
          break;
        }
        chain.push_back(current);
        mlir::Value previous = pulse::linearPredecessor(current);
        if (!previous) {
          root = current.getDefiningOp();
          break;
        }
        current = previous;
      }
      if (root && root->getBlock() != &body)
        root = nullptr;
      for (mlir::Value value : chain)
        program->rootOf[value] = root;
      if (root)
        program->rootOps.insert(root);
    }
  }

  // Run-time waits and the calls their length depends on.
  for (int64_t i = 0; i < count; ++i) {
    mlir::Operation *op = program->ops[i];
    llvm::SmallVector<int64_t, 1> calls;
    if (mlir::isa<classical::WaitOp>(op) && op->hasAttr("classical.blocking")) {
      calls.push_back(program->callOfWait.lookup(op));
    } else if (mlir::isa<pulse::WaitOp>(op) &&
               op->hasAttr("classical.variable")) {
      llvm::SmallVector<mlir::Value> work{op->getOperand(1)};
      llvm::DenseSet<mlir::Value> seen;
      bool known = true;
      while (!work.empty() && known) {
        mlir::Value value = work.pop_back_val();
        if (!seen.insert(value).second)
          continue;
        mlir::Operation *def = value.getDefiningOp();
        if (!def || def->getNumRegions() != 0) {
          known = false;
          break;
        }
        if (mlir::isa<classical::WaitOp>(def)) {
          calls.push_back(program->callOfWait.lookup(def));
          continue;
        }
        work.append(def->getOperands().begin(), def->getOperands().end());
      }
      if (!known)
        continue;
    } else {
      continue;
    }
    program->barriers[i] = std::move(calls);
  }

  // What crosses each top-level position: a call in flight, or a value a
  // window starting after it could not be given.
  std::vector<int64_t> calls(count + 1, 0), values(count + 1, 0);
  auto across = [&](std::vector<int64_t> &marks, int64_t from, int64_t to) {
    if (from < to) {
      ++marks[from];
      --marks[to];
    }
  };
  main.walk([&](classical::SubmitOp submit) {
    int64_t from = topOf(submit), to = from;
    for (classical::WaitOp wait : classical::waitsOf(submit))
      to = std::max(to, topOf(wait));
    across(calls, from, to);
  });
  llvm::DenseMap<mlir::Value, bool> importable;
  std::function<bool(mlir::Value)> canImport = [&](mlir::Value value) {
    auto known = importable.find(value);
    if (known != importable.end())
      return known->second;
    bool can = false;
    mlir::Operation *def = value.getDefiningOp();
    mlir::Type type = value.getType();
    if (!def) {
      can = false;
    } else if (isThreadedType(type)) {
      can = program->rootOf.lookup(value) != nullptr;
    } else if (mlir::isa<classical::WaitOp>(def)) {
      // Bound by the time a window starts after it, as a constant.
      can = type.isF64() || type.isSignlessInteger(64);
    } else if (def->getNumRegions() == 0 && !isReadoutOp(def) &&
               !mlir::isa<classical::SubmitOp>(def) &&
               !mlir::isa<pulse::MeasurementType>(type) &&
               !pulse::isClassicalType(type) &&
               llvm::none_of(def->getOperandTypes(), isThreadedType)) {
      // Computed from constants and answers alone (not from where a line
      // stands, as pulse.get_current_time is).
      can = llvm::all_of(def->getOperands(), canImport);
    }
    importable[value] = can;
    return can;
  };
  for (int64_t i = 0; i < count; ++i)
    for (mlir::Value result : program->ops[i]->getResults()) {
      int64_t last = i;
      for (mlir::Operation *user : result.getUsers())
        last = std::max(last, topOf(user));
      if (last > i && !canImport(result))
        across(values, i, last);
    }
  int64_t inFlight = 0, unimportable = 0;
  for (int64_t i = 0; i < count; ++i) {
    inFlight += calls[i];
    unimportable += values[i];
    if (inFlight > 0)
      program->crossedByCall.insert(i);
    if (unimportable > 0)
      program->crossed.insert(i);
  }
  program->eligible = true;
  return program;
}

/// Answers a window no longer holds the calls of, newest first: what a
/// later window reads from before it.
struct RetiredAnswer {
  int64_t call;
  std::vector<double> values;
  std::shared_ptr<const RetiredAnswer> next;
};

// -----------------------------------------------------------------------
// PulseModule: opaque handle around an in-memory mlir::ModuleOp
// -----------------------------------------------------------------------
// The wrapper owns the MLIRContext and the ModuleOp; run_passes() looks
// passes up by their cudaq-pulse-opt names.
class PulseModule {
public:
  PulseModule(std::shared_ptr<mlir::MLIRContext> ctx,
              mlir::OwningOpRef<mlir::ModuleOp> mod)
      : ctx_(std::move(ctx)), module_(std::move(mod)),
        lock_(std::make_shared<std::recursive_mutex>()) {}

  std::string print() {
    std::lock_guard<std::recursive_mutex> guard(*lock_);
    std::string result;
    llvm::raw_string_ostream os(result);
    mlir::OpPrintingFlags flags;
    // The printer verifies a module first, to print an invalid one in the
    // generic form; one verified since it last changed is valid.
    if (verified_ == version_.load()) {
      if (verifyEverything() && mlir::failed(mlir::verify(*module_)))
        throw std::runtime_error(
            "print: a module noted as verified does not verify");
      flags.assumeVerified();
    }
    module_->print(os, flags);
    return result;
  }

  /// Run the passes named in `pipeline`, in order: each entry is a pass as
  /// cudaq-pulse-opt names it, with its options in braces, or several
  /// separated by commas. A pass that runs on functions runs on each
  /// function of the module. Pattern passes in a row (pulse-fold-constants,
  /// pulse-remove-unused, pulse-merge-waits, pulse-simplify-control-flow,
  /// pulse-fold-phases, pulse-fuse-pulses) run as one rewrite with all their
  /// patterns: one walk of the module, to where none of them applies.
  void run_passes(const std::vector<std::string> &pipeline) {
    std::lock_guard<std::recursive_mutex> guard(*lock_);
    changed();
    std::unique_ptr<mlir::PassManager> passes =
        ContextStore::takePipeline(ctx_.get(), pipeline);
    bool build = !passes;
    if (build)
      passes = std::make_unique<mlir::PassManager>(
          ctx_.get(), mlir::ModuleOp::getOperationName(),
          mlir::OpPassManager::Nesting::Implicit);
    mlir::PassManager &pm = *passes;
    llvm::SmallVector<const pulse::RewriteStage *, 4> stages;
    auto flush = [&] {
      if (build && !stages.empty())
        pm.addNestedPass<mlir::func::FuncOp>(
            pulse::createPulseRewritePass(stages));
      stages.clear();
    };
    for (const std::string &entry : pipeline)
      for (llvm::StringRef pass : splitPipeline(entry)) {
        auto [name, options] = pass.split('{');
        if (const pulse::RewriteStage *stage = pulse::findRewriteStage(name);
            stage && options.empty()) {
          if (!llvm::is_contained(stages, stage))
            stages.push_back(stage);
          if (stage->onBinding && !llvm::is_contained(rewrites_, stage)) {
            rewrites_.push_back(stage);
            rewritePatterns_.reset();
          }
          continue;
        }
        flush();
        if (!build)
          continue;
        const mlir::PassInfo *info = lookupPass(name);
        if (!info)
          throw std::runtime_error("Unknown pass: " + name.str());
        if (mlir::failed(info->addToPipeline(
                pm, options.empty() ? options : options.drop_back(),
                [](const llvm::Twine &) { return mlir::failure(); })))
          throw std::runtime_error("Cannot add pass: " + pass.str());
      }
    flush();
    if (build)
      cudaq_pulse::requireLoadedDialects(pm, ctx_.get(), "run_passes");
    programs_.clear();
    bool verify = verifyEverything();
    runPipelineVerifyingOnce(pm, *module_, "MLIR pass pipeline failed", verify);
    if (verify)
      verified();
    ContextStore::keepPipeline(ctx_.get(), pipeline, std::move(passes));
  }

  /// Keep the selects whose uses all set a parameter of a kind in
  /// `runtime`; run every other op that reads one in both arms of an if.
  /// Returns whether a select is kept.
  bool lower_selects(const std::vector<std::string> &runtime) {
    std::lock_guard<std::recursive_mutex> guard(*lock_);
    mlir::PassManager pm(ctx_.get());
    pm.addNestedPass<mlir::func::FuncOp>(
        pulse::createPulseLowerSelectsPass(runtime));
    programs_.clear();
    runPipeline(pm, *module_, "lowering selects failed");
    return keepsSelects();
  }

  /// This module, scheduled, with every op that reads a run-time select in
  /// both arms of an if on its condition at the time it was scheduled: each
  /// measurement path plays the values it selects. For simulation.
  PulseModule resolve_selects() {
    std::lock_guard<std::recursive_mutex> guard(*lock_);
    auto resolve = [&](mlir::ModuleOp module) {
      auto copy = mlir::OwningOpRef<mlir::ModuleOp>(module.clone());
      mlir::PassManager pm(ctx_.get());
      pm.addNestedPass<mlir::func::FuncOp>(
          pulse::createPulseLowerSelectsPass({}));
      runPipeline(pm, *copy, "resolving selects failed");
      return copy;
    };
    PulseModule result(ctx_, resolve(*module_));
    result.loops_ = loops_;
    if (source_)
      result.source_ = resolve(*source_);
    return result;
  }

  /// The module and, if kept, its source before scheduling, as MLIR
  /// bytecode (the source empty if not kept), with what the builder and
  /// run_passes() noted of them: what PulseModuleBuilder::load() takes.
  std::tuple<std::string, std::string, bool, std::vector<std::string>>
  serialize() {
    std::lock_guard<std::recursive_mutex> guard(*lock_);
    auto write = [](mlir::ModuleOp module) {
      std::string bytes;
      llvm::raw_string_ostream os(bytes);
      if (mlir::failed(mlir::writeBytecodeToFile(
              module, os, mlir::BytecodeWriterConfig("cudaq-pulse"))))
        throw std::runtime_error("cannot write the module as bytecode");
      return bytes;
    };
    std::vector<std::string> rewrites;
    for (const pulse::RewriteStage *stage : rewrites_)
      rewrites.push_back(stage->argument.str());
    return {write(*module_), source_ ? write(*source_) : std::string(),
            runTime_, std::move(rewrites)};
  }

  std::string run_full_lowering() {
    std::lock_guard<std::recursive_mutex> guard(*lock_);
    // Lower a copy so inspection does not consume the scheduled Pulse module;
    // callers may lower repeatedly and still execute the original artifact.
    // The copy is in a context this thread keeps for lowering: the module's
    // context is shared and has no LLVM dialect, and loading one there would
    // race with the threads working in it (ContextDialects.h).
    auto lowered = cudaq_pulse::copyForLowering(*module_, makeRegistry);
    mlir::PassManager pm(lowered->getContext());
    pm.addPass(pulse::createPulseToQOpPass());
    pm.addPass(qop::createQOpToCuDensityMatPass());
    pm.addPass(cudm::createCuDensityMatToLLVMPass());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createArithToLLVMConversionPass());
    pm.addPass(mlir::createConvertFuncToLLVMPass());
    pm.addPass(mlir::createReconcileUnrealizedCastsPass());
    runPipeline(pm, *lowered, "MLIR full lowering pipeline failed");
    std::string result;
    llvm::raw_string_ostream os(result);
    lowered->print(os);
    return result;
  }

  void schedule(const std::string &policy, int64_t maxDrives = 4,
                int64_t maxReadouts = 2, int64_t readoutLatency = 0,
                int64_t switchPenalty = 0) {
    std::lock_guard<std::recursive_mutex> guard(*lock_);
    changed();
    programs_.clear();
    keepSource();
    runSchedule(*module_, true, policy, maxDrives, maxReadouts, readoutLatency,
                switchPenalty, "pulse scheduling failed");
    verified();
  }

  using Event = std::tuple<std::string, int64_t, int64_t,
                           std::optional<int64_t>, std::optional<int64_t>>;

  /// (kind, line, qubit, start_cycles, duration_cycles) for every drive,
  /// readout and wait, once per loop iteration and in program order. Both arms
  /// of a conditional are listed. Lines are numbered in allocation order. A
  /// classical call adds a "submit" and an "answer" event, whose line is the
  /// link id and whose qubit is -1.
  std::vector<Event> events() {
    std::lock_guard<std::recursive_mutex> guard(*lock_);
    llvm::DenseMap<mlir::Operation *, int64_t> lineIndex;
    module_->walk([&](mlir::Operation *op) {
      if (mlir::isa<pulse::GetDriveLineOp, pulse::GetReadoutLineOp>(op))
        lineIndex.try_emplace(op, static_cast<int64_t>(lineIndex.size()));
    });
    std::vector<Event> result;
    llvm::SmallVector<int64_t, 2> iteration;
    pulse::AllocationTracer trace;
    std::optional<AttrNames> ownNames;
    const AttrNames &names = ContextStore::names(ctx_.get(), ownNames);
    std::function<void(mlir::Block &)> collect = [&](mlir::Block &block) {
      for (mlir::Operation &op : block) {
        if (mlir::isa<pulse::DriveOp, pulse::ReadoutOp, pulse::WaitOp>(op)) {
          requireExportedMode(&op);
          mlir::Operation *allocation = trace(op.getOperand(0));
          int64_t line = allocation ? lineIndex.lookup(allocation) : -1;
          int64_t qubit = allocation ? pulse::quditOf(allocation) : -1;
          result.emplace_back(op.getName().stripDialect().str(), line, qubit,
                              pulse::scheduledStart(&op, iteration),
                              pulse::scheduledDuration(&op));
        } else if (mlir::isa<classical::SubmitOp, classical::WaitOp>(op)) {
          // A classical call's events are on its link; they have no qudit.
          result.emplace_back(mlir::isa<classical::SubmitOp>(op) ? "submit"
                                                                 : "answer",
                              classical::intAttr(&op, names.device), -1,
                              pulse::scheduledStart(&op, iteration),
                              pulse::scheduledDuration(&op));
        } else if (auto loop = mlir::dyn_cast<mlir::scf::ForOp>(op)) {
          int64_t trips = pulse::constantTripCount(loop).value_or(0);
          iteration.push_back(0);
          for (int64_t k = 0; k < trips; ++k) {
            iteration.back() = k;
            collect(*loop.getBody());
          }
          iteration.pop_back();
        } else {
          for (mlir::Region &region : op.getRegions())
            for (mlir::Block &nested : region)
              collect(nested);
        }
      }
    };
    collect(*module_->getBody());
    return result;
  }

  /// events(), or, when it lists more than `limit` events, a 1-tuple of
  /// their loops: a list of nodes in program order, each (kind, line, qubit,
  /// start_cycles, duration_cycles, shifts) for an op at the first iteration
  /// of its loops, `shifts` its start's shift per iteration of each loop
  /// around it, outermost first, or (trips, nodes) for a loop.
  nb::object event_tree(int64_t limit) {
    if (!loops_ && limit >= 0)
      return nb::cast(events());
    std::lock_guard<std::recursive_mutex> guard(*lock_);
    auto timed = [](mlir::Operation &op) {
      return mlir::isa<pulse::DriveOp, pulse::ReadoutOp, pulse::WaitOp,
                       classical::SubmitOp, classical::WaitOp>(op);
    };
    auto add = [](uint64_t a, uint64_t b) {
      return checkedAdd(a, b).value_or(UINT64_MAX);
    };
    std::function<uint64_t(mlir::Block &)> count =
        [&](mlir::Block &block) -> uint64_t {
      uint64_t n = 0;
      for (mlir::Operation &op : block) {
        if (timed(op)) {
          n = add(n, 1);
        } else if (auto loop = mlir::dyn_cast<mlir::scf::ForOp>(op)) {
          uint64_t trips = static_cast<uint64_t>(
              std::max<int64_t>(0, pulse::constantTripCount(loop).value_or(0)));
          uint64_t body = count(*loop.getBody());
          n = add(n, checkedMul(trips, body).value_or(UINT64_MAX));
        } else {
          for (mlir::Region &region : op.getRegions())
            for (mlir::Block &nested : region)
              n = add(n, count(nested));
        }
      }
      return n;
    };
    if (limit >= 0 &&
        count(*module_->getBody()) <= static_cast<uint64_t>(limit))
      return nb::cast(events());
    llvm::DenseMap<mlir::Operation *, int64_t> lineIndex;
    module_->walk([&](mlir::Operation *op) {
      if (mlir::isa<pulse::GetDriveLineOp, pulse::GetReadoutLineOp>(op))
        lineIndex.try_emplace(op, static_cast<int64_t>(lineIndex.size()));
    });
    llvm::SmallVector<int64_t, 2> iteration;
    pulse::AllocationTracer trace;
    std::optional<AttrNames> ownNames;
    const AttrNames &names = ContextStore::names(ctx_.get(), ownNames);
    auto node = [&](mlir::Operation &op, std::string kind, int64_t line,
                    int64_t qubit) {
      std::optional<int64_t> start = pulse::scheduledStart(&op, iteration);
      nb::list shifts;
      if (start)
        for (int64_t &k : iteration) {
          k = 1;
          std::optional<int64_t> next = pulse::scheduledStart(&op, iteration);
          k = 0;
          shifts.append(next ? nb::cast(*next - *start) : nb::none());
        }
      return nb::make_tuple(std::move(kind), line, qubit, start,
                            pulse::scheduledDuration(&op), shifts);
    };
    std::function<void(mlir::Block &, nb::list &)> build =
        [&](mlir::Block &block, nb::list &out) {
          for (mlir::Operation &op : block) {
            if (mlir::isa<pulse::DriveOp, pulse::ReadoutOp, pulse::WaitOp>(
                    op)) {
              mlir::Operation *allocation = trace(op.getOperand(0));
              int64_t line = allocation ? lineIndex.lookup(allocation) : -1;
              int64_t qubit = allocation ? pulse::quditOf(allocation) : -1;
              out.append(
                  node(op, op.getName().stripDialect().str(), line, qubit));
            } else if (mlir::isa<classical::SubmitOp, classical::WaitOp>(op)) {
              out.append(node(
                  op, mlir::isa<classical::SubmitOp>(op) ? "submit" : "answer",
                  classical::intAttr(&op, names.device), -1));
            } else if (auto loop = mlir::dyn_cast<mlir::scf::ForOp>(op)) {
              int64_t trips = pulse::constantTripCount(loop).value_or(0);
              nb::list body;
              iteration.push_back(0);
              build(*loop.getBody(), body);
              iteration.pop_back();
              out.append(nb::make_tuple(std::max<int64_t>(trips, 0), body));
            } else {
              for (mlir::Region &region : op.getRegions())
                for (mlir::Block &nested : region)
                  build(nested, out);
            }
          }
        };
    nb::list root;
    build(*module_->getBody(), root);
    return nb::make_tuple(root);
  }

  // ── Sites: the traced ops of a program, for idle replacement ────────
  //
  // The frontend numbers every drive, readout, wait, sync and frame change
  // it traces. A number is kept as the op's location, which no pass reads
  // and the printed module does not show, so a numbered module compiles as
  // the same module without numbers; a copy of an op (a peeled or unrolled
  // loop iteration) keeps its number, and so does the op a rewrite makes
  // of several (the first of them). An op written into an idle window has
  // a negative number and, until the numbers are removed, the attribute
  // `pulse.written`: the rewrites keep it apart from the kernel's own ops
  // (a written wait is not merged, a written drive not fused with one of
  // the kernel's).

  /// The site number a packed site record gave `op`, if any.
  static std::optional<int64_t> siteOf(mlir::Operation *op) {
    auto loc = mlir::dyn_cast<mlir::FileLineColLoc>(op->getLoc());
    if (!loc || loc.getFilename().getValue() != "pulse.site")
      return std::nullopt;
    int64_t line = loc.getLine();
    return loc.getColumn() ? -line - 1 : line;
  }

  /// Remove the site numbers the packed site records gave, from the module and
  /// from its source before scheduling.
  void strip_sites() {
    std::lock_guard<std::recursive_mutex> guard(*lock_);
    mlir::Location unknown = mlir::UnknownLoc::get(ctx_.get());
    auto strip = [&](mlir::ModuleOp module) {
      module.walk([&](mlir::Operation *op) {
        if (siteOf(op))
          op->setLoc(unknown);
        op->removeAttr("pulse.written");
      });
    };
    strip(*module_);
    if (source_)
      strip(*source_);
  }

  /// The lines, (role, qudit) in allocation order, and the scheduled
  /// program as a list of nodes in program order, linear in its ops:
  /// - (kind, site, line, tone, start_cycles, duration_cycles, shifts,
  ///   barrier) for a drive, readout, wait or frame change ("drive",
  ///   "readout", "wait", "shift_phase", "set_phase", "shift_frequency",
  ///   "set_frequency"): its number (None without one), its line (-1 for a
  ///   frame change), the line whose tone it plays or changes (-1 for a wait
  ///   and for a tone that is no line's own), its start at the first
  ///   iteration of the loops around it and that start's shift per iteration
  ///   of each, outermost first, and whether its length is known only when
  ///   the program runs;
  /// - ("sync", site, lines);
  /// - ("answer",) for a blocking wait on a classical answer;
  /// - ("for", trips, nodes) for a loop;
  /// - ("if", nodes, nodes) for the two arms of a branch.
  nb::tuple site_tree() {
    std::lock_guard<std::recursive_mutex> guard(*lock_);
    llvm::DenseMap<mlir::Operation *, int64_t> lineIndex;
    nb::list lines;
    module_->walk([&](mlir::Operation *op) {
      if (!mlir::isa<pulse::GetDriveLineOp, pulse::GetReadoutLineOp>(op))
        return;
      lineIndex.try_emplace(op, static_cast<int64_t>(lineIndex.size()));
      lines.append(nb::make_tuple(
          mlir::isa<pulse::GetDriveLineOp>(op) ? "drive" : "readout",
          pulse::quditOf(op)));
    });
    llvm::SmallVector<int64_t, 2> iteration;
    pulse::AllocationTracer trace;
    std::optional<AttrNames> ownNames;
    const AttrNames &names = ContextStore::names(ctx_.get(), ownNames);
    auto indexOf = [&](mlir::Value value) -> int64_t {
      mlir::Operation *allocation = trace(value);
      if (!allocation)
        return -1;
      auto found = lineIndex.find(allocation);
      return found == lineIndex.end() ? -1 : found->second;
    };
    auto site = [](mlir::Operation &op) -> nb::object {
      std::optional<int64_t> number = siteOf(&op);
      return number ? nb::cast(*number) : nb::none();
    };
    std::function<void(mlir::Block &, nb::list &)> build =
        [&](mlir::Block &block, nb::list &out) {
          for (mlir::Operation &op : block) {
            bool played = mlir::isa<pulse::DriveOp, pulse::ReadoutOp>(op);
            bool wait = mlir::isa<pulse::WaitOp>(op);
            if (played || wait ||
                mlir::isa<pulse::ShiftPhaseOp, pulse::SetPhaseOp,
                          pulse::ShiftFrequencyOp, pulse::SetFrequencyOp>(op)) {
              std::optional<int64_t> start =
                  pulse::scheduledStart(&op, iteration);
              nb::list shifts;
              if (start)
                for (int64_t &k : iteration) {
                  k = 1;
                  std::optional<int64_t> next =
                      pulse::scheduledStart(&op, iteration);
                  k = 0;
                  shifts.append(next ? nb::cast(*next - *start) : nb::none());
                }
              out.append(nb::make_tuple(
                  op.getName().stripDialect().str(), site(op),
                  played || wait ? indexOf(op.getOperand(0)) : int64_t{-1},
                  wait ? int64_t{-1} : indexOf(op.getOperand(played ? 2 : 0)),
                  start, pulse::scheduledDuration(&op), shifts,
                  AttrNames::has(&op, names.variable)));
            } else if (mlir::isa<pulse::SyncOp>(op)) {
              nb::list joined;
              for (mlir::Value line : op.getOperands())
                joined.append(indexOf(line));
              out.append(nb::make_tuple("sync", site(op), joined));
            } else if (mlir::isa<classical::WaitOp>(op)) {
              if (AttrNames::has(&op, names.blocking))
                out.append(nb::make_tuple("answer"));
            } else if (auto loop = mlir::dyn_cast<mlir::scf::ForOp>(op)) {
              int64_t trips = pulse::constantTripCount(loop).value_or(0);
              nb::list body;
              iteration.push_back(0);
              build(*loop.getBody(), body);
              iteration.pop_back();
              out.append(
                  nb::make_tuple("for", std::max<int64_t>(trips, 0), body));
            } else if (auto branch = mlir::dyn_cast<mlir::scf::IfOp>(op)) {
              nb::list then, other;
              build(branch.getThenRegion().front(), then);
              if (!branch.getElseRegion().empty())
                build(branch.getElseRegion().front(), other);
              out.append(nb::make_tuple("if", then, other));
            } else {
              for (mlir::Region &region : op.getRegions())
                for (mlir::Block &nested : region)
                  build(nested, out);
            }
          }
        };
    nb::list root;
    build(*module_->getBody(), root);
    return nb::make_tuple(lines, root);
  }

  /// (role, qudit): a runtime qudit's line has qudit -1.
  using ControllerLine = std::tuple<std::string, int64_t>;

  /// The lines, (role, qubit, slot) in allocation order, and the timed ops of
  /// `events()`, in the same order, every sync and every classical call,
  /// with what a controller needs to play them: (kind, line, qubit,
  /// start_cycles, duration_cycles, loop iteration, waveform, tone frequency in
  /// Hz and frame phase in radians at the op's start, the branches it lies in,
  /// the lines a sync joins, references, callee, wait mode, timeout). A
  /// waveform is its expression's nodes, children before their parent: (op name
  /// without the dialect, its numeric operands or the samples of
  /// `custom_samples`, NaN where one is not a number, the indices of its
  /// waveform operands). A branch is (branch id, 1 for the then arm and 0 for
  /// the else arm, the readout events and the answer events the condition is
  /// computed from, "is_set" when it is one readout's `pulse.is_set`, "answer"
  /// when it is one answer, else "expression"). The frame is tracked through
  /// both arms of a branch; where the arms leave it different, or a frame
  /// change is not a number, it is NaN from there on. A sync starts when the
  /// last of its lines is free and lasts 0 cycles. A classical call's line is
  /// its link: the references of a submission are the readout events whose
  /// measurements it sends, the callee is its function, and an answer's wait is
  /// "bounded", "blocking" or "timeout", with its timeout in ns (NaN for
  /// none). Then what is known only at run time: a drive or readout on a
  /// tone whose frame depends on run time (a frame change computed from a
  /// readout or an answer, or made in a branch) has its tone's id and the
  /// frame changes on the tone since its latest drive or readout; a drive or
  /// readout has the expression of each waveform operand computed from a
  /// readout or an answer; a variable wait has the expression of its length
  /// and its bounds; a submission has the expression of each argument.
  ///
  /// Returned as controller.export() gives them: (lines, events), lists of
  /// dicts, with `parameters` the parameter names of each waveform shape and
  /// `amplitudes` the shapes with an overall amplitude; each event has its
  /// segment (all 0 without classical calls). A list `slots` gets each line's
  /// runtime-qudit slot (-1 for none). With a list `repeats`, for a check, a
  /// long rolled loop whose iterations repeat (and that holds no segment
  /// boundary) has its events once, no event has a tone id or frame
  /// changes, and `repeats` gets each such loop's ControllerEventWalk::Repeat;
  /// with a list `periods` too, its lines may repeat at periods of their
  /// own, and `periods` gets them (ControllerEventWalk::linePeriods). A
  /// list `segments` gets, of a module with classical calls, the segment
  /// boundaries the walk visited (as segment_boundaries() gives them) and
  /// then SegmentWalk::skips, the iterations it did not visit.
  nb::tuple controller_events(nb::handle parameters, nb::handle amplitudes,
                              nb::object slots, nb::object repeats,
                              nb::object periods, nb::object segments) {
    ControllerObjects objects(parameters, amplitudes);
    std::unique_ptr<ControllerEventWalk> walk;
    std::vector<ControllerLine> lines;
    std::vector<int64_t> found;
    bool once = !repeats.is_none();
    {
      nb::gil_scoped_release release;
      walk = controllerWalk(lines, slots.is_none() ? nullptr : &found, once,
                            once && !periods.is_none(), repeats.is_none());
    }
    lineSlots(found, slots);
    appendRepeats(*walk, repeats, periods);
    if (!segments.is_none() && walk->segmentWalk) {
      auto out = nb::borrow<nb::list>(segments);
      out.append(nb::cast(walk->segmentWalk->boundaries));
      out.append(nb::cast(walk->segmentWalk->skips));
    }
    return nb::make_tuple(objects.lines(lines), objects.events(*walk));
  }

  /// `walk`'s repeats into the list `repeats` and its line periods into the
  /// list `periods`, where it has any.
  static void appendRepeats(const ControllerEventWalk &walk, nb::handle repeats,
                            nb::handle periods) {
    if (!walk.repeats.empty()) {
      auto out = nb::borrow<nb::list>(repeats);
      for (const auto &repeat : walk.repeats)
        out.append(nb::cast(repeat));
    }
    if (!walk.linePeriods.empty()) {
      auto out = nb::borrow<nb::list>(periods);
      for (const auto &linePeriod : walk.linePeriods)
        out.append(nb::cast(linePeriod));
    }
  }

  /// For waits whose length floats set: (cycles op after op with each at its
  /// lowest length, the most the program may last, 1 if that is the
  /// controller's longest program, how often each such wait runs), and when
  /// such a wait is in an if, the loops and ifs around them (`excess` in
  /// the decoder).
  nb::tuple program_length() const {
    if (!length_.excess.empty())
      return nb::make_tuple(length_.cycles, length_.most, length_.controller,
                            nb::cast(length_.repeats),
                            nb::cast(length_.excess));
    return nb::make_tuple(length_.cycles, length_.most, length_.controller,
                          nb::cast(length_.repeats));
  }

  /// `found`, each line's runtime-qudit slot, into the list `slots`, unless
  /// it is None.
  static void lineSlots(const std::vector<int64_t> &found, nb::handle slots) {
    if (slots.is_none())
      return;
    auto out = nb::borrow<nb::list>(slots);
    for (int64_t slot : found)
      out.append(slot);
  }

  /// A ControllerEventWalk of the module, its lines in `lines` and, unless
  /// `slots` is null, each one's runtime-qudit slot (-1 for none) in
  /// `slots`; without the GIL.
  std::unique_ptr<ControllerEventWalk>
  controllerWalk(std::vector<ControllerLine> &lines,
                 std::vector<int64_t> *slots, bool summarize = false,
                 bool unsynced = false, bool trackFrames = true) {
    std::lock_guard<std::recursive_mutex> guard(*lock_);
    auto walk = std::make_unique<ControllerEventWalk>(ctx_.get(), clock());
    walk->summarize = summarize;
    walk->unsynced = unsynced;
    walk->trackFrames = trackFrames;
    if (classical::declaresClassicalCalls(*module_))
      walk->segmentWalk.emplace(*module_);
    if (runTime_)
      controllerEvents<true>(*walk, lines, slots);
    else
      controllerEvents<false>(*walk, lines, slots);
    return walk;
  }

  /// The slot of a line `op` allocates: its own, or that of the argument of
  /// `main` it is on; -1 for none.
  static int64_t lineSlot(mlir::Operation *op) {
    if (auto attr = op->getAttrOfType<mlir::IntegerAttr>("pulse.slot"))
      return attr.getInt();
    if (auto arg = mlir::dyn_cast<mlir::BlockArgument>(op->getOperand(0)))
      if (auto func =
              mlir::dyn_cast<mlir::func::FuncOp>(arg.getOwner()->getParentOp()))
        if (auto attr = func.getArgAttrOfType<mlir::IntegerAttr>(
                arg.getArgNumber(), "pulse.slot"))
          return attr.getInt();
    return -1;
  }

  template <bool RunTime>
  void controllerEvents(ControllerEventWalk &walk,
                        std::vector<ControllerLine> &lines,
                        std::vector<int64_t> *slots) {
    module_->walk([&](mlir::Operation *op) {
      if constexpr (RunTime)
        walk.scan(op);
      if (!mlir::isa<pulse::GetDriveLineOp, pulse::GetReadoutLineOp>(op))
        return;
      if (!walk.lineIndex
               .try_emplace(op, static_cast<int64_t>(walk.lineIndex.size()))
               .second)
        return;
      int64_t qubit = pulse::quditOf(op);
      lines.emplace_back(
          mlir::isa<pulse::GetDriveLineOp>(op) ? "drive" : "readout", qubit);
      if (slots)
        slots->push_back(lineSlot(op));
    });
    walk.block<RunTime>(*module_->getBody());
  }

  /// The walk of every execution with what each record depends on (see
  /// ControllerWalk::track).
  std::unique_ptr<ControllerWalk> trackedWalk() {
    auto walk =
        std::make_unique<ControllerWalk>(clock(), ControllerWalk::Walk::Every);
    walk->track = true;
    std::vector<ControllerLine> lines;
    walkController(*walk, lines, nullptr);
    return walk;
  }

  /// For each event of controller_program(), not rolled, in the same order,
  /// the arguments of `main` its tone frequency, frame phase and waveform
  /// are computed from; a qref argument stands for its qudit's native
  /// frequency (see nativeOffset).
  std::vector<ControllerWalk::Dependencies> controller_dependencies() {
    return std::move(trackedWalk()->dependencies);
  }

  /// For each event of controller_program(), not rolled, in the same order,
  /// the qref argument of `main` (see nativeOffset) whose native frequency
  /// its tone plays,
  /// because the tone is its line's and was never retuned; -1 for any other
  /// event. With `summarize` (and `unsynced`), for each event of
  /// controller_events()'s summarizing walk, which keeps a loop rolled only
  /// if each copy plays the natives of the first: (natives, repeats, line
  /// periods).
  nb::object controller_native_tones(bool summarize, bool unsynced) {
    if (!summarize)
      return nb::cast(std::move(trackedWalk()->natives));
    std::lock_guard<std::recursive_mutex> guard(*lock_);
    ControllerEventWalk walk(ctx_.get(), clock(), /*track=*/true);
    walk.summarize = true;
    walk.unsynced = unsynced;
    module_->walk([&](mlir::Operation *op) {
      if (mlir::isa<pulse::GetDriveLineOp, pulse::GetReadoutLineOp>(op))
        walk.lineIndex.try_emplace(op,
                                   static_cast<int64_t>(walk.lineIndex.size()));
    });
    walk.block<false>(*module_->getBody());
    return nb::make_tuple(nb::cast(walk.natives), nb::cast(walk.repeats),
                          nb::cast(walk.linePeriods));
  }

  /// For each event of controller_program() rolled, in the same order (a
  /// loop before its body): its kind, its trips for a loop (else -1), and
  /// what it depends on, as controller_dependencies() gives it; an event of
  /// a rolled loop's body depends on what any of its iterations does. On
  /// the module of a call's values, the events match where loops roll alike.
  std::vector<std::tuple<std::string, int64_t, ControllerWalk::Dependencies>>
  controller_rolled_dependencies() {
    ControllerWalk walk(clock(), ControllerWalk::Walk::Rolled);
    walk.runTime = runTime_;
    walk.track = true;
    std::vector<ControllerLine> lines;
    walkController(walk, lines, nullptr);
    std::vector<std::tuple<std::string, int64_t, ControllerWalk::Dependencies>>
        out;
    out.reserve(walk.result.size());
    for (size_t i = 0; i < walk.result.size(); ++i) {
      const ControllerRecord &r = walk.result[i];
      bool loop = isKind(r.kind, "loop");
      out.emplace_back(r.kind, loop ? r.trips : -1,
                       std::move(walk.dependencies[i]));
    }
    return out;
  }

  /// Walk the program for a controller (see ControllerProgram.h): its lines,
  /// (role, qubit) in allocation order, into `lines`, and its records into
  /// `walk`; with `waveforms`, the waveform of every drive and readout too.
  void walkController(ControllerWalk &walk, std::vector<ControllerLine> &lines,
                      ControllerWaveforms *waveforms,
                      std::vector<int64_t> *slots = nullptr) {
    std::lock_guard<std::recursive_mutex> guard(*lock_);
    module_->walk([&](mlir::Operation *op) {
      if (walk.runTime)
        walk.scan(op);
      if (!mlir::isa<pulse::GetDriveLineOp, pulse::GetReadoutLineOp>(op))
        return;
      if (!walk.lineIndex
               .try_emplace(op, static_cast<int64_t>(walk.lineIndex.size()))
               .second)
        return;
      int64_t qubit = pulse::quditOf(op);
      lines.emplace_back(
          mlir::isa<pulse::GetDriveLineOp>(op) ? "drive" : "readout", qubit);
      if (slots)
        slots->push_back(lineSlot(op));
    });
    walk.block(*module_->getBody());
    if (waveforms)
      for (const ControllerRecord &r : walk.result)
        if (r.played())
          waveforms->add(r.op, &walk.played);
  }

  /// The lines and the events of the program as the dictionaries of the
  /// controller export: every timed op, sync and
  /// classical call once per loop iteration, or, with `rolled`, with each
  /// loop whose iterations repeat kept as one "loop" event holding its body,
  /// in the compact encoding of format /2, its lines as (role, qubit). With
  /// `segments`, each event is in the segment of its timeline (see
  /// SegmentWalk), else in segment 0; `waveform`, when given, turns a
  /// waveform's nodes (op name without the dialect, its numbers with NaN where
  /// one is not a number, the indices of its waveform operands; children before
  /// their parent) into (expression, amplitude), once per distinct waveform;
  /// without it the expression is built here, its numbers named by
  /// `parameters`, and the amplitude is that of the shapes in `amplitudes`.
  /// What is known only at run time is an expression over readouts, answers
  /// and constants (see ControllerWalk), its operands named by `parameters`.
  /// Third, a dict, empty unless the program has either of: with frames that
  /// persist from shot to shot, "frame_state", the frame each tone ends the
  /// shot in; and "runtime_tones", each tone tracked at run time, by id, as
  /// (line, reset frequency, reset phase), -1 and None where not known
  /// (ControllerWalk::RuntimeTone).
  ///
  /// With `own`, (num, den, grid) and not `rolled`: the times in the cycles
  /// of a controller counting `num` of its cycles every `den` program
  /// cycles (see ownCycles; what an event computes at run time stays in
  /// program cycles, for the caller to convert), and fourth, the drives,
  /// readouts and waits off its grid of `grid` program cycles. A list
  /// `slots` gets each line's runtime-qudit slot (-1 for none), a list
  /// `ids` each event's `pulse.op_id` (see with_holds; -1 for none). Without
  /// `exportable`, for a check, a value the export's vocabulary has no term
  /// for is {"unknown": what it is computed from} rather than an error.
  nb::tuple
  controller_program(nb::handle parameters, nb::handle amplitudes,
                     bool segments, bool rolled, nb::object waveform,
                     std::optional<std::tuple<int64_t, int64_t, int64_t>> own,
                     nb::object slots, bool exportable, nb::object ids) {
    if (own && (rolled || std::get<0>(*own) < 1 || std::get<1>(*own) < 1 ||
                std::get<2>(*own) < 1))
      throw std::invalid_argument("controller_program: a controller's ratio "
                                  "and grid are positive, on a walk of every "
                                  "execution");
    ControllerWalk walk(clock(), rolled ? ControllerWalk::Walk::Rolled
                                        : ControllerWalk::Walk::Every);
    walk.runTime = runTime_;
    walk.persistent = pulse::persistentFrames(*module_);
    if (segments)
      walk.segments.emplace(*module_);
    std::vector<ControllerLine> lines;
    ControllerWaveforms waveforms;
    std::vector<OffGridEvent> offGrid;
    std::vector<int64_t> found;
    {
      nb::gil_scoped_release release;
      walkController(walk, lines, &waveforms,
                     slots.is_none() ? nullptr : &found);
      if (own) {
        offGrid = ownCycles(walk.result, std::get<0>(*own), std::get<1>(*own),
                            std::get<2>(*own));
        ownFrameUpdateCycles(walk, std::get<0>(*own), std::get<1>(*own));
      }
    }
    if (!slots.is_none()) {
      auto out = nb::borrow<nb::list>(slots);
      for (int64_t slot : found)
        out.append(slot);
    }
    if (!ids.is_none()) {
      auto out = nb::borrow<nb::list>(ids);
      for (const ControllerRecord &r : walk.result)
        out.append(r.op ? classical::intAttr(r.op, "pulse.op_id", -1) : -1);
    }
    nb::list roles;
    if (rolled) {
      for (auto &[role, qubit] : lines)
        roles.append(nb::make_tuple(role, qubit));
    } else {
      nb::str lineKey("line"), roleKey("role"), quditKey("qudit");
      for (size_t i = 0; i < lines.size(); ++i) {
        nb::dict line;
        line[lineKey] = i;
        line[roleKey] = std::get<0>(lines[i]);
        line[quditKey] = std::get<1>(lines[i]);
        roles.append(line);
      }
    }
    std::vector<ControllerWalk::FinalFrame> finals;
    if (walk.persistent) {
      int64_t end = 0;
      for (const ControllerRecord &r : walk.result)
        if (auto stop = r.stop())
          end = std::max(end, *stop);
      finals = walk.finalFrames(static_cast<double>(end) / walk.clockGHz);
    }
    ControllerEvents program(walk, waveforms, std::move(waveform), parameters,
                             amplitudes, rolled, exportable);
    nb::list events = program.build();
    nb::dict frames;
    if (!finals.empty())
      frames["frame_state"] = program.frameState(finals);
    if (!walk.toneIds.empty()) {
      auto number = [](double x) -> nb::object {
        return std::isnan(x) ? nb::none() : nb::cast(x);
      };
      nb::list tones;
      for (const ControllerWalk::RuntimeTone &tone : walk.runtimeTones())
        tones.append(nb::make_tuple(tone.line, number(tone.frequencyHz),
                                    number(tone.phase)));
      frames["runtime_tones"] = tones;
    }
    if (own)
      return nb::make_tuple(roles, events, frames, nb::cast(offGrid));
    return nb::make_tuple(roles, events, frames);
  }

  /// `controller_events()` for a controller counting `num` of its cycles
  /// every `den` program cycles: start and duration in its cycles, rounded
  /// up (what an event computes at run time stays in program cycles, for
  /// the caller to convert), with the drives, readouts and waits off its grid
  /// of `grid` program cycles, and (index, end) for the events whose end,
  /// rounded up in its cycles, is not its start plus its duration: (lines,
  /// events, off grid, ends). A list `repeats` (and `periods`) gets the
  /// repeats (and line periods) of a summarizing walk as controller_events()
  /// gives them (and no event a tone id or frame changes), in its cycles: of
  /// the loops whose every period is whole
  /// steps of its grid, so that each copy is on or off it as the first is;
  /// with a loop of another period, every iteration is played.
  nb::tuple controller_events_on(nb::handle parameters, nb::handle amplitudes,
                                 int64_t num, int64_t den, int64_t grid,
                                 nb::object slots, nb::object repeats,
                                 nb::object periods) {
    if (num < 1 || den < 1 || grid < 1)
      throw std::invalid_argument("controller_events_on: the ratio and grid "
                                  "must be positive");
    ControllerObjects objects(parameters, amplitudes);
    std::unique_ptr<ControllerEventWalk> walk;
    std::vector<ControllerLine> lines;
    std::vector<int64_t> found;
    std::vector<OffGridEvent> offGrid;
    std::vector<std::tuple<int64_t, int64_t>> ends;
    bool once = !repeats.is_none();
    {
      nb::gil_scoped_release release;
      walk = controllerWalk(lines, slots.is_none() ? nullptr : &found, once,
                            once && !periods.is_none(), repeats.is_none());
      if (!ownRepeats(*walk, num, den, grid)) {
        lines.clear();
        found.clear();
        walk = controllerWalk(lines, slots.is_none() ? nullptr : &found, false,
                              false, repeats.is_none());
      }
      ownEventCycles(*walk, num, den, grid, offGrid, ends);
    }
    lineSlots(found, slots);
    appendRepeats(*walk, repeats, periods);
    return nb::make_tuple(objects.lines(lines), objects.events(*walk),
                          nb::cast(offGrid), nb::cast(ends));
  }

  /// Whether every period of `walk`'s repeats is whole steps of `grid`
  /// program cycles and whole cycles of a controller counting `num` of them
  /// every `den` program cycles; if so, they are converted to its cycles.
  static bool ownRepeats(ControllerEventWalk &walk, int64_t num, int64_t den,
                         int64_t grid) {
    auto own = [&](int64_t period) -> std::optional<int64_t> {
      if (period % grid)
        return std::nullopt;
      std::optional<int64_t> cycles = checkedMul(period, num);
      if (!cycles || *cycles % den)
        return std::nullopt;
      return *cycles / den;
    };
    for (const auto &repeat : walk.repeats)
      if (repeat[3] > 0 && !own(repeat[3]))
        return false;
    for (const auto &[index, line, period] : walk.linePeriods)
      if (!own(period))
        return false;
    for (auto &repeat : walk.repeats)
      if (repeat[3] > 0)
        repeat[3] = *own(repeat[3]);
    for (auto &linePeriod : walk.linePeriods)
      linePeriod[2] = *own(linePeriod[2]);
    return true;
  }

  static void ownEventCycles(ControllerEventWalk &walk, int64_t num,
                             int64_t den, int64_t grid,
                             std::vector<OffGridEvent> &offGrid,
                             std::vector<std::tuple<int64_t, int64_t>> &ends) {
    auto own = [&](std::optional<int64_t> &cycles) {
      if (cycles)
        *cycles = (*cycles * num + den - 1) / den;
    };
    auto &events = walk.result;
    for (size_t index = 0; index < events.size(); ++index) {
      auto &event = events[index];
      const std::string &kind = std::get<0>(event);
      auto &start = std::get<3>(event);
      auto &duration = std::get<4>(event);
      if (grid > 1 && start && duration &&
          (kind == "drive" || kind == "readout" || kind == "wait") &&
          (*start % grid || *duration % grid))
        offGrid.emplace_back(static_cast<int64_t>(index), *start, *duration);
      if (num != den) {
        std::optional<int64_t> end;
        if (start && duration)
          end = *start + *duration;
        own(start);
        own(duration);
        own(end);
        if (end && *end != *start + *duration)
          ends.emplace_back(static_cast<int64_t>(index), *end);
      }
    }
  }

  /// The controller limits the schedule exceeds, as messages; -1 for a
  /// limit not set. A loop whose iterations repeat is checked at its first
  /// two iterations and its last. `memory` is (samples, granule, shared,
  /// scaling, synthesized shapes, hold). With `resolve`, a value not known
  /// counts only where every value exceeds the limit, and the second result
  /// says whether some value might; else it is false. A long loop whose
  /// iterations repeat out of step is left to the summarizing walk
  /// (ControllerWalk::declined): the messages are then one saying so.
  std::pair<std::vector<std::string>, bool> controller_check(
      int64_t issue, int64_t gap, int64_t samplesNum, int64_t samplesDen,
      const std::tuple<int64_t, int64_t, bool, bool, std::vector<std::string>,
                       bool> &memory,
      int64_t maxFrequenciesPerLine, int64_t maxEvents,
      const std::map<int64_t, int64_t> &detectors, int64_t detectorRecovery,
      int64_t feedbackLatency, bool answerTimeouts,
      const std::vector<std::string> &timeoutFunctions, bool resolve) {
    ControllerWalk walk(clock(), ControllerWalk::Walk::Sampled);
    std::vector<ControllerLine> lines;
    walkController(walk, lines, nullptr);
    if (walk.declined)
      return {{"a long loop whose iterations repeat out of step is checked "
               "from its summarizing walk"},
              false};
    ControllerLimitSpec limits;
    limits.issue = issue;
    limits.gap = gap;
    limits.samplesNum = samplesNum;
    limits.samplesDen = samplesDen;
    const auto &[samples, granule, shared, scaling, synthesized, hold] = memory;
    limits.waveformSamples = samples;
    limits.granule = granule;
    limits.sharedMemory = shared;
    limits.scaling = scaling;
    limits.synthesized = {synthesized.begin(), synthesized.end()};
    limits.hold = hold;
    limits.maxFrequenciesPerLine = maxFrequenciesPerLine;
    limits.maxEvents = maxEvents;
    limits.detectors = detectors;
    limits.detectorRecovery = detectorRecovery;
    limits.feedbackLatency = feedbackLatency;
    limits.answerTimeouts = answerTimeouts;
    limits.timeoutFunctions = {timeoutFunctions.begin(),
                               timeoutFunctions.end()};
    std::lock_guard<std::recursive_mutex> guard(*lock_);
    bool unresolved = false;
    auto found = checkControllerLimits(walk.result, limits,
                                       resolve ? &unresolved : nullptr);
    return {std::move(found), unresolved};
  }

  /// What each float argument of `main` reaches, in argument order: a
  /// sorted subset of amplitude (an overall factor of a waveform), phase,
  /// frequency, shape (another waveform parameter), timing (a length) and
  /// other (anything else, such as a classical call or a branch).
  std::vector<std::vector<std::string>> parameter_roles() {
    std::lock_guard<std::recursive_mutex> guard(*lock_);
    std::vector<std::vector<std::string>> out;
    auto func = source().lookupSymbol<mlir::func::FuncOp>("main");
    if (!func || func.getBody().empty())
      return out;
    static const std::map<std::string, std::set<unsigned>> amplitudes = {
        {"pulse.square", {1, 2}},       {"pulse.gaussian", {1}},
        {"pulse.gaussian_square", {1}}, {"pulse.drag", {1}},
        {"pulse.cosine", {1}},          {"pulse.tanh_ramp", {1}},
        {"pulse.scale", {0}},
    };
    for (mlir::BlockArgument arg : func.getBody().front().getArguments()) {
      if (!mlir::isa<mlir::FloatType>(arg.getType()))
        continue;
      std::set<std::string> roles;
      llvm::SmallVector<mlir::Value> work{arg};
      llvm::DenseSet<mlir::Value> seen;
      while (!work.empty()) {
        mlir::Value value = work.pop_back_val();
        if (!seen.insert(value).second)
          continue;
        for (mlir::OpOperand &use : value.getUses()) {
          mlir::Operation *user = use.getOwner();
          std::string name = user->getName().getStringRef().str();
          if (user->getNumResults() == 1 &&
              mlir::isa<pulse::WaveformType>(user->getResult(0).getType())) {
            // The index among the operands that are numbers.
            unsigned index = 0;
            for (unsigned i = 0; i < use.getOperandNumber(); ++i)
              if (!mlir::isa<pulse::WaveformType>(
                      user->getOperand(i).getType()))
                ++index;
            auto found = amplitudes.find(name);
            if (found != amplitudes.end() && found->second.count(index))
              roles.insert("amplitude");
            else
              roles.insert(index == 0 ? "timing" : "shape");
          } else if (name == "pulse.shift_phase" || name == "pulse.set_phase") {
            roles.insert("phase");
          } else if (name == "pulse.shift_frequency" ||
                     name == "pulse.set_frequency") {
            roles.insert("frequency");
          } else if (name == "pulse.tone") {
            roles.insert(use.getOperandNumber() == 0 ? "frequency" : "phase");
          } else if (name == "pulse.wait" ||
                     name == "pulse.duration_from_int") {
            roles.insert("timing");
          } else if (user->getNumRegions() == 0 &&
                     !user->hasTrait<mlir::OpTrait::IsTerminator>() &&
                     user->getNumResults() > 0 &&
                     llvm::all_of(user->getResultTypes(), [](mlir::Type t) {
                       return t.isIntOrIndexOrFloat();
                     })) {
            for (mlir::Value result : user->getResults())
              work.push_back(result);
          } else {
            roles.insert("other");
          }
        }
      }
      out.emplace_back(roles.begin(), roles.end());
    }
    return out;
  }

  /// (function, link, iteration, submit start, submit duration, due,
  /// taken, first use, deadline, segment) for every execution of a classical
  /// call, in program order; times in cycles. `taken` is the wait's start
  /// (absent for an answer never taken), `first use` the earliest start of a
  /// pulse computed from the answer, `deadline` the latest due time the
  /// function's deadline allows, and `segment` the segment it is submitted in.
  using ClassicalCall =
      std::tuple<std::string, int64_t, std::vector<int64_t>, int64_t, int64_t,
                 int64_t, std::optional<int64_t>, std::optional<int64_t>,
                 std::optional<int64_t>, int64_t, std::optional<int64_t>>;

  std::vector<ClassicalCall> classical_calls() {
    std::lock_guard<std::recursive_mutex> guard(*lock_);
    std::vector<ClassicalCall> calls;
    if (!classical::declaresClassicalCalls(*module_))
      return calls;
    double clock = classical::moduleClockGHz(*module_);
    mlir::SymbolTable symbols(*module_);
    // The timed ops computed from each wait's answer, found once.
    llvm::DenseMap<mlir::Operation *, llvm::SmallVector<mlir::Operation *>>
        consumers;
    auto consumersOf = [&](classical::WaitOp wait) {
      auto [it, inserted] = consumers.try_emplace(wait);
      if (!inserted)
        return it->second;
      llvm::SmallVector<mlir::Value> work(wait.getValues().begin(),
                                          wait.getValues().end());
      llvm::DenseSet<mlir::Operation *> seen;
      while (!work.empty()) {
        mlir::Value value = work.pop_back_val();
        for (mlir::Operation *user : value.getUsers()) {
          if (!seen.insert(user).second)
            continue;
          if (mlir::isa<pulse::DriveOp, pulse::ReadoutOp, pulse::WaitOp,
                        classical::SubmitOp>(user)) {
            it->second.push_back(user);
          } else if (user->getNumRegions() > 0) {
            user->walk([&it = it](mlir::Operation *inner) {
              if (mlir::isa<pulse::DriveOp, pulse::ReadoutOp, pulse::WaitOp,
                            classical::SubmitOp>(inner))
                it->second.push_back(inner);
            });
          } else {
            work.append(user->getResults().begin(), user->getResults().end());
          }
        }
      }
      return it->second;
    };
    SegmentWalk segments(*module_);
    const AttrNames &names = segments.names;
    auto startOf = [&](mlir::Operation *op, llvm::ArrayRef<int64_t> iteration) {
      llvm::SmallVector<int64_t, 2> at(iteration.begin(), iteration.end());
      at.resize(pulse::enclosingLoops(op).size(), 0);
      return pulse::scheduledStart(op, at);
    };
    llvm::SmallVector<int64_t, 2> iteration;
    llvm::DenseMap<mlir::Operation *, bool> busy;
    auto callsOrBoundaries = [&](mlir::scf::ForOp loop) {
      auto [it, inserted] = busy.try_emplace(loop, false);
      if (inserted)
        it->second =
            loop->walk([&](mlir::Operation *inner) {
                  return mlir::isa<classical::SubmitOp>(inner) ||
                                 AttrNames::has(inner, names.variable) ||
                                 AttrNames::has(inner, names.blocking)
                             ? mlir::WalkResult::interrupt()
                             : mlir::WalkResult::advance();
                })
                .wasInterrupted();
      return it->second;
    };
    std::function<void(mlir::Block &)> collect = [&](mlir::Block &block) {
      for (mlir::Operation &op : block) {
        if (auto submit = mlir::dyn_cast<classical::SubmitOp>(op)) {
          int64_t segment = segments.visit(&op, iteration);
          auto start = pulse::scheduledStart(&op, iteration);
          if (!start)
            continue;
          auto callable =
              symbols.lookup<classical::CallableOp>(submit.getCallee());
          int64_t duration = pulse::scheduledDuration(&op).value_or(1);
          int64_t due =
              *start + duration + classical::intAttr(&op, names.budget);
          std::optional<int64_t> taken, firstUse, deadline;
          for (mlir::Operation *user : submit.getFuture().getUsers())
            if (auto wait = mlir::dyn_cast<classical::WaitOp>(user)) {
              taken = startOf(wait, iteration);
              for (mlir::Operation *consumer : consumersOf(wait))
                if (auto at = startOf(consumer, iteration))
                  firstUse = firstUse ? std::min(*firstUse, *at) : *at;
            }
          if (callable)
            if (auto limit = callable.getDeadlineNs())
              deadline = *start +
                         classical::ceilCycles(limit->convertToDouble(), clock);
          calls.emplace_back(
              submit.getCallee().str(), classical::intAttr(&op, names.device),
              std::vector<int64_t>(iteration.begin(), iteration.end()), *start,
              duration, due, taken, firstUse, deadline, segment,
              AttrNames::has(&op, names.instance)
                  ? std::optional<int64_t>(
                        classical::intAttr(&op, names.instance))
                  : std::nullopt);
        } else if (mlir::isa<classical::WaitOp, pulse::WaitOp>(op)) {
          segments.visit(&op, iteration);
        } else if (auto loop = mlir::dyn_cast<mlir::scf::ForOp>(op)) {
          // A loop without calls or segment boundaries adds nothing.
          if (!callsOrBoundaries(loop))
            continue;
          int64_t trips = pulse::constantTripCount(loop).value_or(0);
          iteration.push_back(0);
          for (int64_t k = 0; k < trips; ++k) {
            iteration.back() = k;
            collect(*loop.getBody());
          }
          iteration.pop_back();
        } else {
          for (mlir::Region &region : op.getRegions())
            for (mlir::Block &nested : region)
              collect(nested);
        }
      }
    };
    collect(*module_->getBody());
    return calls;
  }

  /// The boundaries between segments, in program order, once per loop
  /// iteration: (kind, line or link, start, lowest, highest, after). A "wait"
  /// is a variable wait on a line, whose length lies in [lowest, highest]
  /// cycles (highest -1 if unbounded); an "answer" is a blocking wait on a
  /// link, whose length is the answer's latency (lowest 0, highest -1). The
  /// k-th boundary starts segment k when it ends; `after` is the segment it
  /// ends, of the lines it holds back.
  std::vector<SegmentWalk::Boundary> segment_boundaries() {
    std::lock_guard<std::recursive_mutex> guard(*lock_);
    if (!classical::declaresClassicalCalls(*module_))
      return {};
    SegmentWalk walk(*module_);
    // Only where segments open is asked for: the iterations of a loop that
    // opens none are not walked (SegmentBoundaries.h).
    cudaq_pulse::walkSegmentBoundaries(
        *module_->getBody(), walk.names.variable, walk.names.blocking,
        [&](mlir::Operation *op, llvm::ArrayRef<int64_t> iteration) {
          walk.visit(op, iteration);
        });
    return std::move(walk.boundaries);
  }

  /// The segment of each event of `events()` (with `syncs`, of
  /// `controller_program()` not rolled), in the same order.
  std::vector<int64_t> event_segments(bool syncs) {
    std::lock_guard<std::recursive_mutex> guard(*lock_);
    std::vector<int64_t> segments;
    SegmentWalk walk(*module_);
    walk.run(*module_->getBody(), [&](mlir::Operation *op, int64_t segment) {
      if (syncs || !mlir::isa<pulse::SyncOp>(op))
        segments.push_back(segment);
    });
    return segments;
  }

  // ── Simulating classical calls ──────────────────────────────────────
  //
  // Simulation binds the answer of each call as a constant, one execution
  // at a time along a measurement path, and schedules again: the timeline
  // of a path is static once its answers are known.

  /// This module with loops that hold calls unrolled and scheduled again
  /// with `policy`, for simulation.
  PulseModule simulation_module(const std::string &policy, int64_t maxDrives,
                                int64_t maxReadouts, int64_t readoutLatency,
                                int64_t switchPenalty) {
    std::lock_guard<std::recursive_mutex> guard(*lock_);
    auto copy = mlir::OwningOpRef<mlir::ModuleOp>(source().clone());
    mlir::PassManager unroll(ctx_.get());
    unroll.addNestedPass<mlir::func::FuncOp>(
        pulse::createPulseUnrollClassicalLoopsPass());
    runPipeline(unroll, *copy, "simulation: unrolling classical loops failed");
    PulseModule result(ctx_, std::move(copy));
    result.loops_ = loops_;
    result.keepSource();
    runSchedule(*result.module_, false, policy, maxDrives, maxReadouts,
                readoutLatency, switchPenalty, "simulation: scheduling failed");
    result.verified();
    result.simulation_ = true;
    return result;
  }

  /// This module with the lines of its control system replaced by `lines`
  /// (a list of `#pulse.line`, or empty for none), which carry the delays,
  /// and scheduled again with `policy`: everything before scheduling is
  /// kept.
  PulseModule with_line_delays(const std::string &lines,
                               const std::string &policy, int64_t maxDrives,
                               int64_t maxReadouts, int64_t readoutLatency,
                               int64_t switchPenalty) {
    std::lock_guard<std::recursive_mutex> guard(*lock_);
    auto copy = mlir::OwningOpRef<mlir::ModuleOp>(source().clone());
    llvm::SmallVector<pulse::LineAttr> parsed;
    if (!lines.empty()) {
      auto list = mlir::dyn_cast_or_null<mlir::ArrayAttr>(
          mlir::parseAttribute(lines, ctx_.get()));
      if (!list)
        throw std::invalid_argument("with_line_delays: not a list of lines");
      for (mlir::Attribute line : list) {
        auto parsedLine = mlir::dyn_cast<pulse::LineAttr>(line);
        if (!parsedLine)
          throw std::invalid_argument("with_line_delays: not a #pulse.line");
        parsed.push_back(parsedLine);
      }
    }
    pulse::ControlSystemAttr system = pulse::controlSystemOf(*copy);
    if (!system)
      system = pulse::controlSystemAt(ctx_.get(), 1.0);
    copy->getOperation()->setAttr(
        pulse::kControlSystemAttr,
        pulse::ControlSystemAttr::get(ctx_.get(),
                                      system.getClockPeriodFsNumerator(),
                                      system.getClockPeriodFsDenominator(),
                                      parsed, system.getControllers()));
    PulseModule result(ctx_, std::move(copy));
    result.loops_ = loops_;
    result.runTime_ = runTime_;
    result.keepSource();
    runSchedule(*result.module_, true, policy, maxDrives, maxReadouts,
                readoutLatency, switchPenalty, "pulse scheduling failed");
    result.verified();
    return result;
  }

  /// This module scheduled again from its module before scheduling, each
  /// drive, readout and frame change numbered in walk order
  /// (`pulse.op_id`, which controller_program() gives with `ids`) and held
  /// back as `holds` gives by number (`pulse.hold_cycles`: the cycles its
  /// line stays idle before it), every other hold removed.
  PulseModule with_holds(const std::map<int64_t, int64_t> &holds,
                         const std::string &policy, int64_t maxDrives,
                         int64_t maxReadouts, int64_t readoutLatency,
                         int64_t switchPenalty) {
    std::lock_guard<std::recursive_mutex> guard(*lock_);
    auto copy = mlir::OwningOpRef<mlir::ModuleOp>(source().clone());
    auto i64 = mlir::IntegerType::get(ctx_.get(), 64);
    int64_t next = 0;
    copy->walk([&](mlir::Operation *op) {
      if (!mlir::isa<pulse::DriveOp, pulse::ReadoutOp, pulse::ShiftPhaseOp,
                     pulse::SetPhaseOp, pulse::ShiftFrequencyOp,
                     pulse::SetFrequencyOp>(op))
        return;
      int64_t id = next++;
      op->setAttr("pulse.op_id", mlir::IntegerAttr::get(i64, id));
      auto hold = holds.find(id);
      if (hold != holds.end() && hold->second > 0)
        op->setAttr("pulse.hold_cycles",
                    mlir::IntegerAttr::get(i64, hold->second));
      else
        op->removeAttr("pulse.hold_cycles");
    });
    PulseModule result(ctx_, std::move(copy));
    result.loops_ = loops_;
    result.runTime_ = runTime_;
    result.keepSource();
    runSchedule(*result.module_, true, policy, maxDrives, maxReadouts,
                readoutLatency, switchPenalty, "pulse scheduling failed");
    result.verified();
    return result;
  }

  /// One argument of a call: (kind, readout index, iteration, value), kind 0
  /// for a measurement and 1 for a number.
  using CallArgument =
      std::tuple<int64_t, int64_t, std::vector<int64_t>, double>;
  /// (submit index, start ns, callee, arguments, answered, timeout ns or -1)
  using PendingCall = std::tuple<int64_t, double, std::string,
                                 std::vector<CallArgument>, bool, double>;

  /// The first call on `path` that has no answer bound yet, by start time.
  std::optional<PendingCall>
  next_call(const std::vector<std::vector<int64_t>> &path) {
    std::lock_guard<std::recursive_mutex> guard(*lock_);
    auto readouts = readoutOps();
    pulse::Path chosen = toPath(path, readouts);
    llvm::DenseMap<mlir::Operation *, int64_t> indexOf;
    module_->walk([&](classical::SubmitOp submit) {
      indexOf.try_emplace(submit, static_cast<int64_t>(indexOf.size()));
    });
    double clockGHz = clock();
    std::optional<PendingCall> next;
    pulse::ExecutionWalker walker(clockGHz, chosen);
    walker.onSubmit = [&](const pulse::Occurrence &occurrence) {
      mlir::Operation *op = occurrence.op;
      if (op->hasAttr("classical.bound"))
        return false;
      double start = occurrence.startCycles / clockGHz;
      if (next && std::get<1>(*next) <= start)
        return false;
      auto submit = mlir::cast<classical::SubmitOp>(op);
      std::vector<CallArgument> args;
      for (mlir::Value operand : submit.getArgs()) {
        if (auto key = walker.measurementKeyOf(operand)) {
          int64_t index = std::distance(readouts.begin(),
                                        llvm::find(readouts, key->readout));
          args.emplace_back(0, index, key->iteration, 0.0);
        } else if (auto value = pulse::ExecutionWalker::number(operand)) {
          args.emplace_back(1, -1, std::vector<int64_t>{}, *value);
        } else {
          throw std::runtime_error(
              "simulation: an argument of a classical call is neither a "
              "measurement nor a number known on this path");
        }
      }
      bool answered = false;
      double timeout = -1.0;
      for (classical::WaitOp wait : classical::waitsOf(submit)) {
        answered = true;
        if (auto t = wait.getTimeoutNs())
          timeout = t->convertToDouble();
      }
      next = PendingCall{indexOf.lookup(op), start,    submit.getCallee().str(),
                         std::move(args),    answered, timeout};
      return false;
    };
    auto function = getFuncOp();
    if (!function)
      throw std::runtime_error("no 'main' func in module");
    if (!walker.run(function))
      throw std::runtime_error(walker.error);
    return next;
  }

  /// Per call, by the submit index `next_call` gives, the ns from its start
  /// until its answer is due.
  std::vector<double> call_due_ns() {
    std::lock_guard<std::recursive_mutex> guard(*lock_);
    double clockGHz = clock();
    std::vector<double> due;
    module_->walk([&](classical::SubmitOp submit) {
      mlir::Operation *op = submit.getOperation();
      int64_t cycles = pulse::scheduledDuration(op).value_or(1) +
                       classical::intAttr(op, "classical.budget_cycles");
      due.push_back(static_cast<double>(cycles) / clockGHz);
    });
    return due;
  }

  /// This module with the answer `values` bound to execution `index` of a
  /// call (an empty `values` for a call whose answer is not taken), folded
  /// and scheduled again with `policy`. Given the measurement `path` taken
  /// to the call, a simulation module returns the window of the program
  /// that path plays next (see SimulationProgram); its readouts and calls
  /// are numbered from window_bases().
  PulseModule
  bind_call(int64_t index, const std::vector<double> &values,
            const std::string &policy, int64_t maxDrives, int64_t maxReadouts,
            int64_t readoutLatency, int64_t switchPenalty,
            const std::optional<std::vector<std::vector<int64_t>>> &path) {
    std::lock_guard<std::recursive_mutex> guard(*lock_);
    if (window_ && !path)
      throw std::invalid_argument(
          "bind_call: a simulation window binds along its path");
    if (path &&
        (window_ || (simulation_ && (policy == "asap" || policy == "alap"))))
      if (auto window =
              bindInWindow(index, values, *path, policy, maxDrives, maxReadouts,
                           readoutLatency, switchPenalty))
        return std::move(*window);
    bool unscheduled = static_cast<bool>(source_);
    auto copy = mlir::OwningOpRef<mlir::ModuleOp>(source().clone());
    classical::SubmitOp found;
    int64_t seen = 0;
    copy->walk([&](classical::SubmitOp submit) {
      if (seen++ == index)
        found = submit;
    });
    if (!found)
      throw std::invalid_argument("bind_call: no call execution " +
                                  std::to_string(index));
    if (unscheduled) {
      int64_t scheduled = 0;
      module_->walk([&](classical::SubmitOp) { ++scheduled; });
      if (scheduled != seen)
        throw std::runtime_error(
            "bind_call: the module has " + std::to_string(scheduled) +
            " call executions and its source " + std::to_string(seen));
    }
    found->setAttr("classical.bound", mlir::UnitAttr::get(ctx_.get()));
    bool rebound = false;
    for (classical::WaitOp wait : classical::waitsOf(found)) {
      if (wait.getValues().size() != values.size())
        throw std::invalid_argument(
            "bind_call: " + found.getCallee().str() + " answers " +
            std::to_string(wait.getValues().size()) + " values, got " +
            std::to_string(values.size()));
      bindConstants(wait, values);
      rebound = true;
    }
    PulseModule result(ctx_, std::move(copy));
    result.loops_ = loops_;
    if (rebound || unscheduled)
      result.schedule(policy, maxDrives, maxReadouts, readoutLatency,
                      switchPenalty);
    return result;
  }

  /// The readouts and calls of the program before this window: its own
  /// are numbered from there.
  std::pair<int64_t, int64_t> window_bases() {
    std::lock_guard<std::recursive_mutex> guard(*lock_);
    if (!window_)
      return {0, 0};
    return {program_->readoutsBefore[first_], program_->submitsBefore[first_]};
  }

  /// The number and loop iteration the kernel gave each readout execution
  /// [readout index, iteration...] of this module (in a window, numbered as
  /// in the program): what SimulationResult.measurements reports. Peeling
  /// and unrolling copy readouts; a copy keeps the kernel's.
  std::vector<std::pair<int64_t, std::vector<int64_t>>>
  readout_keys(const std::vector<std::vector<int64_t>> &entries) {
    std::lock_guard<std::recursive_mutex> guard(*lock_);
    auto readouts = readoutOps();
    int64_t base = window_bases().first;
    std::vector<std::pair<int64_t, std::vector<int64_t>>> keys;
    for (const auto &entry : entries) {
      int64_t index = entry.empty() ? -1 : entry.front() - base;
      if (index < 0 || index >= static_cast<int64_t>(readouts.size()))
        throw std::invalid_argument(
            "readout_keys entries are [readout index, iteration...]");
      keys.push_back(pulse::kernelReadoutKey(
          readouts[index], llvm::ArrayRef(entry).drop_front(), entry.front()));
    }
    return keys;
  }

  /// This module with every answer not bound yet replaced by zeros, for
  /// evolving up to the next call: those answers only affect what follows.
  PulseModule placeholders() {
    std::lock_guard<std::recursive_mutex> guard(*lock_);
    auto copy = mlir::OwningOpRef<mlir::ModuleOp>(module_->clone());
    llvm::SmallVector<classical::WaitOp> waits;
    copy->walk([&](classical::WaitOp wait) {
      if (!wait->hasAttr("classical.answers"))
        waits.push_back(wait);
    });
    for (classical::WaitOp wait : waits)
      bindConstants(wait, std::vector<double>(wait.getValues().size(), 0.0));
    PulseModule result(ctx_, std::move(copy));
    result.loops_ = loops_;
    return result;
  }

  using Measurement =
      std::tuple<int64_t, std::vector<int64_t>, int64_t, double, double>;

  /// The first measurement to complete on `path` (entries [readout index,
  /// iteration..., outcome]) that the path gives no outcome for, as
  /// (readout index, iteration, qubit, start ns, end ns).
  std::optional<Measurement>
  next_measurement(const std::vector<std::vector<int64_t>> &path) {
    std::lock_guard<std::recursive_mutex> guard(*lock_);
    auto readouts = readoutOps();
    pulse::Path chosen = toPath(path, readouts);
    double clockGHz = clock();
    std::optional<Measurement> next;
    pulse::ExecutionWalker walker(clockGHz, chosen);
    pulse::AllocationTracer trace;
    walker.onReadout = [&](const pulse::Occurrence &occurrence,
                           const pulse::MeasurementKey &key) {
      if (chosen.count(key))
        return false;
      double end =
          (occurrence.startCycles + occurrence.durationCycles) / clockGHz;
      if (next && std::get<4>(*next) <= end)
        return false;
      int64_t index =
          std::distance(readouts.begin(), llvm::find(readouts, occurrence.op));
      int64_t qubit = -1;
      if (auto *allocation = trace(occurrence.op->getOperand(0)))
        qubit = pulse::quditOf(allocation);
      next = Measurement{index, key.iteration, qubit,
                         occurrence.startCycles / clockGHz, end};
      return false;
    };
    auto function = getFuncOp();
    if (!function)
      throw std::runtime_error("no 'main' func in module");
    if (!walker.run(function))
      throw std::runtime_error(walker.error);
    return next;
  }

  using ReadoutDrive =
      std::tuple<int64_t, std::vector<int64_t>, double, double>;

  /// Every readout on `path` as (readout index, iteration, tone frequency in
  /// Hz, peak amplitude of its pulse); NaN where a value is not a number.
  std::vector<ReadoutDrive>
  readout_drives(const std::vector<std::vector<int64_t>> &path) {
    std::lock_guard<std::recursive_mutex> guard(*lock_);
    auto readouts = readoutOps();
    pulse::Path chosen = toPath(path, readouts);
    std::vector<ReadoutDrive> result;
    pulse::ExecutionWalker walker(clock(), chosen);
    walker.readoutTones = true;
    walker.onReadout = [&](const pulse::Occurrence &occurrence,
                           const pulse::MeasurementKey &key) {
      int64_t index =
          std::distance(readouts.begin(), llvm::find(readouts, occurrence.op));
      const pulse::Affine &frequency = occurrence.tone.frequencyHz;
      double hz = frequency.isConstant()
                      ? frequency.evaluate({})
                      : std::numeric_limits<double>::quiet_NaN();
      result.emplace_back(index, key.iteration, hz,
                          peakAmplitude(occurrence.op->getOperand(1)));
      return false;
    };
    auto function = getFuncOp();
    if (!function)
      throw std::runtime_error("no 'main' func in module");
    if (!walker.run(function))
      throw std::runtime_error(walker.error);
    return result;
  }

  using DriveFrame =
      std::tuple<int64_t, std::vector<int64_t>, int64_t, double, double>;
  using FinalTone = std::tuple<int64_t, std::string, double, double, double>;

  /// Every drive on `path` as (drive index, iteration, start cycles, tone
  /// frequency in Hz, phase in radians it plays with: its tone's frame
  /// phase at its start plus its own phase offset), and the frame each line's
  /// tone ends the walk in as (qudit, "drive" or "readout", frequency in
  /// Hz, phase in radians at `time` ns, `time`); NaN where a value is not a
  /// number. With `outcome` at least 0, a measurement the path gives no
  /// outcome has that one.
  std::pair<std::vector<DriveFrame>, std::vector<FinalTone>>
  drive_frames(const std::vector<std::vector<int64_t>> &path, int64_t outcome) {
    std::lock_guard<std::recursive_mutex> guard(*lock_);
    auto readouts = readoutOps();
    pulse::Path chosen = toPath(path, readouts);
    llvm::DenseMap<mlir::Operation *, int64_t> indexOf;
    module_->walk([&](pulse::DriveOp drive) {
      indexOf.try_emplace(drive, static_cast<int64_t>(indexOf.size()));
    });
    double clockGHz = clock();
    constexpr double nan = std::numeric_limits<double>::quiet_NaN();
    auto value = [&](const pulse::Affine &a) {
      return a.isConstant() ? a.evaluate({}) : nan;
    };
    std::vector<DriveFrame> drives;
    pulse::ExecutionWalker walker(clockGHz, chosen);
    walker.onDrive = [&](const pulse::Occurrence &occurrence) {
      const pulse::ToneState &tone = occurrence.tone;
      double startNs = static_cast<double>(occurrence.startCycles) / clockGHz;
      double offset = 0.0;
      for (llvm::StringRef name : {"phase_offset", "frame_phase_offset"})
        if (auto attr = occurrence.op->getAttrOfType<mlir::FloatAttr>(name))
          offset += attr.getValueAsDouble();
      pulse::Affine phase = tone.phase +
                            (tone.frequencyHz - tone.frameFrequencyHz) *
                                (2.0 * M_PI) * 1.0e-9 *
                                (startNs - tone.timeNs) +
                            pulse::Affine(offset);
      drives.emplace_back(indexOf.lookup(occurrence.op), occurrence.iteration,
                          occurrence.startCycles, value(tone.frequencyHz),
                          value(phase));
    };
    if (outcome >= 0)
      walker.onReadout = [&](const pulse::Occurrence &,
                             const pulse::MeasurementKey &key) {
        chosen.try_emplace(key, outcome);
        return false;
      };
    auto function = getFuncOp();
    if (!function)
      throw std::runtime_error("no 'main' func in module");
    if (!walker.run(function))
      throw std::runtime_error(walker.error);
    std::vector<FinalTone> finals;
    for (auto &[root, tone] : walker.toneStates()) {
      if (!mlir::isa_and_nonnull<pulse::GetDriveLineOp,
                                 pulse::GetReadoutLineOp>(root))
        continue;
      int64_t qudit = pulse::quditOf(root);
      if (qudit < 0)
        continue;
      finals.emplace_back(
          qudit, mlir::isa<pulse::GetDriveLineOp>(root) ? "drive" : "readout",
          value(tone.frequencyHz), value(tone.phase), tone.timeNs);
    }
    std::sort(finals.begin(), finals.end());
    return {std::move(drives), std::move(finals)};
  }

  /// The peak amplitude of a built-in waveform (its operand 1), through
  /// `pulse.scale` by a real number and `pulse.neg`; the numbers may be
  /// computed from constants and bound answers.
  static double peakAmplitude(mlir::Value waveform) {
    double factor = 1.0;
    for (mlir::Operation *op = waveform.getDefiningOp(); op;) {
      llvm::StringRef name = op->getName().getStringRef();
      if (name == "pulse.neg") {
        op = op->getOperand(0).getDefiningOp();
        continue;
      }
      if (name == "pulse.scale") {
        auto scale = pulse::boundNumber(op->getOperand(1));
        if (!scale)
          break;
        factor *= *scale;
        op = op->getOperand(0).getDefiningOp();
        continue;
      }
      if (op->getNumOperands() < 2)
        break;
      if (auto amplitude = pulse::boundNumber(op->getOperand(1)))
        return std::abs(*amplitude * factor);
      break;
    }
    return std::numeric_limits<double>::quiet_NaN();
  }

  using Factor = std::tuple<int64_t, int64_t, std::vector<double>>;
  using Term = std::tuple<double, double, int32_t, std::vector<double>,
                          std::vector<Factor>>;
  using OperatorProgram =
      std::tuple<std::vector<int64_t>, std::vector<Term>, std::vector<Term>,
                 double, double, int64_t, std::string>;

  /// The Hamiltonian and collapse operators pulse-to-qop builds, as data:
  /// (mode extents, Hamiltonian terms, collapse terms, t_start, t_end,
  /// step_count, integrator). A term is (coefficient real, imaginary,
  /// callback kind or 0, callback parameters, factors), and a factor is
  /// (mode, dimension, column-major interleaved complex matrix). Callback
  /// parameters that depend on the program's float arguments are evaluated
  /// at `arguments`. The lowering is kept per set of module attributes, so
  /// evaluating it again at other arguments does not lower again.
  OperatorProgram operator_program(const std::vector<double> &arguments) {
    std::lock_guard<std::recursive_mutex> guard(*lock_);
    const void *key =
        module_->getOperation()->getAttrDictionary().getAsOpaquePointer();
    auto found = programs_.find(key);
    if (found == programs_.end()) {
      if (programs_.size() >= kMaxPrograms)
        programs_.clear();
      found = programs_.emplace(key, lowerOperatorProgram()).first;
    }
    const ProgramTemplate &program = found->second;
    if (program.arguments > arguments.size())
      throw std::invalid_argument("operator_program: the drives depend on " +
                                  std::to_string(program.arguments) +
                                  " float arguments, got " +
                                  std::to_string(arguments.size()));
    auto bind = [&](const std::vector<TermTemplate> &terms) {
      std::vector<Term> bound;
      bound.reserve(terms.size());
      for (const TermTemplate &term : terms) {
        bound.push_back(term.term);
        auto &parameters = std::get<3>(bound.back());
        for (const SymbolicParameter &symbolic : term.symbolic)
          parameters[symbolic.slot] +=
              symbolic.coefficient * arguments[symbolic.argument];
      }
      return bound;
    };
    return {program.extents,        bind(program.hamiltonian),
            bind(program.collapse), program.tStart,
            program.tEnd,           program.steps,
            program.integrator};
  }

  /// Merge a dictionary attribute, written in MLIR syntax, into the module
  /// attributes, then `then`, another, unless it is empty.
  void set_attributes(const std::string &dictionary,
                      const std::string &then = {}) {
    std::lock_guard<std::recursive_mutex> guard(*lock_);
    changed();
    auto merge = [&](const std::string &text) {
      auto attr = ContextStore::parsed(ctx_.get(), text);
      auto entries = mlir::dyn_cast_or_null<mlir::DictionaryAttr>(attr);
      if (!entries)
        throw std::invalid_argument("expected an MLIR dictionary attribute");
      for (mlir::NamedAttribute entry : entries)
        module_->getOperation()->setAttr(entry.getName(), entry.getValue());
    };
    merge(dictionary);
    if (!then.empty())
      merge(then);
  }

  std::vector<std::string> param_names() {
    std::lock_guard<std::recursive_mutex> guard(*lock_);
    std::vector<std::string> names;
    if (auto attr = module_->getOperation()->getAttrOfType<mlir::ArrayAttr>(
            "pulse.param_names")) {
      for (auto a : attr)
        names.push_back(mlir::cast<mlir::StringAttr>(a).getValue().str());
    }
    return names;
  }

  /// specialize(), shared once its arguments recur: the latest few are
  /// kept, while neither this module nor the one kept has changed in place
  /// since. Called with the GIL held; it is released to build.
  nb::object specialize_shared(
      const std::vector<double> &f64_args, const std::vector<int64_t> &i64_args,
      const std::string &schedule, int64_t maxDrives, int64_t maxReadouts,
      int64_t readoutLatency, int64_t switchPenalty,
      const std::vector<int64_t> &carriers,
      std::optional<nb::ndarray<const double, nb::ndim<1>, nb::c_contig>>
          frequencies,
      const std::string &device,
      std::optional<nb::ndarray<const double, nb::ndim<1>, nb::c_contig>>
          readoutFrequencies = std::nullopt) {
    Slots slots{carriers};
    if (!carriers.empty()) {
      if (!frequencies)
        throw std::invalid_argument(
            "specialize: carriers need the device's frequencies");
      slots.frequencies = frequencies->data();
      slots.count = frequencies->shape(0);
      if (readoutFrequencies) {
        slots.readoutFrequencies = readoutFrequencies->data();
        slots.readoutCount = readoutFrequencies->shape(0);
      }
    }
    // By bits, so that -0.0 and 0.0, or two NaNs, are told apart. A device
    // stands for the frequencies its carriers are bound with.
    llvm::SmallString<192> key;
    auto put = [&key](const void *data, size_t size) {
      key.append(static_cast<const char *>(data),
                 static_cast<const char *>(data) + size);
    };
    size_t counts[3] = {f64_args.size(), i64_args.size(), carriers.size()};
    int64_t options[4] = {maxDrives, maxReadouts, readoutLatency,
                          switchPenalty};
    put(counts, sizeof counts);
    put(f64_args.data(), 8 * f64_args.size());
    put(i64_args.data(), 8 * i64_args.size());
    put(carriers.data(), 8 * carriers.size());
    put(options, sizeof options);
    key += schedule;
    if (!carriers.empty())
      key += device;

    uint64_t version = version_.load();
    // Released outside the lock, before building, so that their memory can
    // be reused.
    std::vector<nb::object> dropped;
    bool keep = false;
    {
      nb::ft_lock_guard guard(specialized_.mutex);
      auto &entries = specialized_.entries;
      for (auto it = entries.begin(); it != entries.end(); ++it) {
        if (llvm::StringRef(it->key) != key.str())
          continue;
        if (it->version == version &&
            nb::inst_ptr<PulseModule>(it->module)->version_.load() ==
                it->moduleVersion) {
          std::rotate(it, it + 1, entries.end());
          return entries.back().module;
        }
        dropped.push_back(std::move(it->module));
        entries.erase(it);
        keep = true;
        break;
      }
      if (!keep) {
        size_t hash = llvm::hash_value(key.str()) | 1;
        auto &missed = specialized_.missed;
        auto seen = std::find(missed.begin(), missed.end(), hash);
        if (seen != missed.end()) {
          *seen = 0;
          keep = true;
        } else {
          missed[specialized_.nextMissed] = hash;
          specialized_.nextMissed =
              (specialized_.nextMissed + 1) % missed.size();
        }
      }
      size_t room = specialized_.maxEntries;
      int64_t ops = specialized_.ops.load();
      if (ops > 0)
        room = std::min<size_t>(
            room, std::max<int64_t>(specialized_.maxOps / ops, 1));
      if (keep && entries.size() >= room) {
        size_t drop = entries.size() + 1 - std::max<size_t>(room, 1);
        for (size_t i = 0; i < drop; ++i)
          dropped.push_back(std::move(entries[i].module));
        entries.erase(entries.begin(), entries.begin() + drop);
      }
    }
    dropped.clear();

    PulseModule built = [&] {
      nb::gil_scoped_release release;
      return specialize(f64_args, i64_args, schedule, maxDrives, maxReadouts,
                        readoutLatency, switchPenalty, slots);
    }();
    nb::object result = nb::cast(std::move(built), nb::rv_policy::move);
    if (!keep || specialized_.maxEntries == 0 || specialized_.maxOps == 0)
      return result;
    nb::ft_lock_guard guard(specialized_.mutex);
    auto &entries = specialized_.entries;
    for (auto &entry : entries)
      if (llvm::StringRef(entry.key) == key.str())
        return entry.module;
    entries.push_back({key.str().str(), version,
                       nb::inst_ptr<PulseModule>(result)->version_.load(),
                       result});
    return result;
  }

  /// (specialized modules kept, at most how many, at most how many ops in
  /// all); a non-negative argument sets that bound.
  std::tuple<size_t, size_t, int64_t> specialized_cache(int64_t entries,
                                                        int64_t ops) {
    std::vector<nb::object> dropped;
    nb::ft_lock_guard guard(specialized_.mutex);
    if (entries >= 0)
      specialized_.maxEntries = static_cast<size_t>(entries);
    if (ops >= 0)
      specialized_.maxOps = ops;
    if (entries >= 0 || ops >= 0) {
      for (auto &entry : specialized_.entries)
        dropped.push_back(std::move(entry.module));
      specialized_.entries.clear();
      specialized_.missed.fill(0);
    }
    return {specialized_.entries.size(), specialized_.maxEntries,
            specialized_.maxOps};
  }

  /// Bind each qref argument of *module* to its slot's carrier, as a static
  /// qudit is allocated, and give its lines the carrier's frequencies.
  static void bindSlots(mlir::ModuleOp module, const Slots &slots) {
    auto funcOp = module.lookupSymbol<mlir::func::FuncOp>("main");
    if (!funcOp)
      throw std::runtime_error("specialize: no 'main' func in module");
    auto &entryBlock = funcOp.getBody().front();
    mlir::OpBuilder builder(module.getContext());
    builder.setInsertionPointToStart(&entryBlock);
    int64_t n = 0;
    if (auto attr =
            module->getAttrOfType<mlir::IntegerAttr>("pulse.qudit_count"))
      n = attr.getInt();
    for (unsigned i = 0; i < entryBlock.getNumArguments(); ++i) {
      auto slot = funcOp.getArgAttrOfType<mlir::IntegerAttr>(i, "pulse.slot");
      if (!slot)
        continue;
      int64_t k = slot.getInt();
      if (k < 0 || k >= static_cast<int64_t>(slots.carriers.size()))
        throw std::invalid_argument("specialize: no carrier for slot " +
                                    std::to_string(k));
      int64_t carrier = slots.carriers[k];
      if (carrier < 0 || carrier >= static_cast<int64_t>(slots.count))
        throw std::invalid_argument("specialize: carrier " +
                                    std::to_string(carrier) +
                                    " has no frequency");
      n = std::max(n, carrier + 1);
      auto arg = entryBlock.getArgument(i);
      if (arg.use_empty())
        continue;
      auto qubit = builder.getI64IntegerAttr(carrier);
      auto frequency = builder.getF64FloatAttr(slots.frequencies[carrier]);
      auto readoutFrequency =
          static_cast<size_t>(carrier) < slots.readoutCount
              ? builder.getF64FloatAttr(slots.readoutFrequencies[carrier])
              : frequency;
      for (mlir::Operation *user : arg.getUsers())
        if (mlir::isa<pulse::GetDriveLineOp, pulse::GetReadoutLineOp>(user)) {
          user->setAttr("frequency_hz", mlir::isa<pulse::GetReadoutLineOp>(user)
                                            ? readoutFrequency
                                            : frequency);
          user->setAttr("pulse.slot", slot);
        }
      auto alloc = pulse::QuditAllocOp::create(
          builder, funcOp.getLoc(), pulse::QuditType::get(module.getContext()),
          qubit);
      arg.replaceAllUsesWith(alloc.getResult());
    }
    module->setAttr("pulse.qudit_count", builder.getI64IntegerAttr(n));
    module->setAttr(
        pulse::kQuantumSystemAttr,
        pulse::quantumSystemAt(
            module.getContext(),
            llvm::ArrayRef<double>(slots.frequencies, static_cast<size_t>(n))));
  }

  /// Make `main` take no arguments, once every use of them is replaced.
  static void dropArguments(mlir::ModuleOp module) {
    dropArguments(module.lookupSymbol<mlir::func::FuncOp>("main"));
  }

  static void dropArguments(mlir::func::FuncOp funcOp) {
    auto &entryBlock = funcOp.getBody().front();
    while (entryBlock.getNumArguments() > 0)
      entryBlock.eraseArgument(entryBlock.getNumArguments() - 1);
    funcOp.setType(mlir::FunctionType::get(funcOp.getContext(), {}, {}));
    if (funcOp.getArgAttrsAttr())
      funcOp.removeArgAttrsAttr();
  }

  static void dropQrefArguments(mlir::ModuleOp module) {
    auto funcOp = module.lookupSymbol<mlir::func::FuncOp>("main");
    llvm::BitVector qrefs(funcOp.getNumArguments());
    for (auto [i, type] : llvm::enumerate(funcOp.getArgumentTypes()))
      if (mlir::isa<pulse::QuditType>(type))
        qrefs.set(i);
    (void)funcOp.eraseArguments(qrefs);
  }

  /// This module with its runtime qudits bound to *carriers*, and its float
  /// arguments kept: a simulation evaluates it at each call's values.
  PulseModule place_carriers(
      const std::vector<int64_t> &carriers,
      nb::ndarray<const double, nb::ndim<1>, nb::c_contig> frequencies,
      std::optional<nb::ndarray<const double, nb::ndim<1>, nb::c_contig>>
          readoutFrequencies) {
    std::lock_guard<std::recursive_mutex> guard(*lock_);
    Slots slots{carriers};
    slots.frequencies = frequencies.data();
    slots.count = frequencies.shape(0);
    if (readoutFrequencies) {
      slots.readoutFrequencies = readoutFrequencies->data();
      slots.readoutCount = readoutFrequencies->shape(0);
    }
    auto owning = mlir::OwningOpRef<mlir::ModuleOp>(module_->clone());
    bindSlots(*owning, slots);
    dropQrefArguments(*owning);
    PulseModule result(ctx_, std::move(owning));
    result.loops_ = loops_;
    if (source_) {
      result.source_ = mlir::OwningOpRef<mlir::ModuleOp>(source_->clone());
      bindSlots(*result.source_, slots);
      dropQrefArguments(*result.source_);
    }
    return result;
  }

  PulseModule specialize(const std::vector<double> &f64_args,
                         const std::vector<int64_t> &i64_args,
                         const std::string &schedule = "alap",
                         int64_t maxDrives = 4, int64_t maxReadouts = 2,
                         int64_t readoutLatency = 0, int64_t switchPenalty = 0,
                         const Slots &slots = {}) {
    std::lock_guard<std::recursive_mutex> guard(*lock_);
    if (specialized_.ops.load() < 0) {
      int64_t ops = 0;
      source().walk([&](mlir::Operation *) { ++ops; });
      specialized_.frameChanges = pulse::hasFrameChangeWork(source());
      specialized_.ops.store(ops);
    }
    bool placed = !slots.carriers.empty();
    bool numeric = false;
    if (placed)
      if (auto main = module_->lookupSymbol<mlir::func::FuncOp>("main"))
        for (mlir::Type type : main.getArgumentTypes())
          numeric |= !mlir::isa<pulse::QuditType>(type);
    if (placed && !numeric) {
      // Carriers move no op: the schedule stands, and nothing folds.
      auto owning = mlir::OwningOpRef<mlir::ModuleOp>(module_->clone());
      bindSlots(*owning, slots);
      dropArguments(*owning);
      PulseModule result(ctx_, std::move(owning));
      result.loops_ = loops_;
      if (source_) {
        result.source_ = mlir::OwningOpRef<mlir::ModuleOp>(source_->clone());
        bindSlots(*result.source_, slots);
        dropArguments(*result.source_);
      }
      return result;
    }
    auto clonedModule = source().clone();
    auto funcOp = clonedModule.lookupSymbol<mlir::func::FuncOp>("main");
    if (!funcOp)
      throw std::runtime_error("specialize: no 'main' func in module");
    if (placed)
      bindSlots(clonedModule, slots);

    auto &entryBlock = funcOp.getBody().front();
    auto loc = funcOp.getLoc();
    mlir::OpBuilder builder(ctx_.get());
    builder.setInsertionPointToStart(&entryBlock);

    size_t f64Idx = 0, i64Idx = 0;
    auto f64Ty = builder.getF64Type();
    llvm::SmallVector<mlir::Value> bound;

    for (unsigned i = 0; i < entryBlock.getNumArguments(); ++i) {
      auto arg = entryBlock.getArgument(i);
      mlir::Value replacement;
      if (mlir::isa<pulse::QuditType>(arg.getType())) {
        if (!arg.use_empty())
          throw std::invalid_argument(
              "specialize: runtime qudits need carriers");
        continue;
      }
      if (arg.getType().isInteger(64)) {
        if (i64Idx >= i64_args.size())
          throw std::runtime_error("specialize: not enough i64 arguments");
        replacement = mlir::arith::ConstantIntOp::create(
            builder, loc, i64_args[i64Idx++], 64);
      } else {
        if (f64Idx >= f64_args.size())
          throw std::runtime_error("specialize: not enough f64 arguments");
        replacement = mlir::arith::ConstantFloatOp::create(
            builder, loc, f64Ty, llvm::APFloat(f64_args[f64Idx++]));
        bound.push_back(replacement);
      }
      arg.replaceAllUsesWith(replacement);
    }

    dropArguments(funcOp);

    // Fold parameter expressions before deriving concrete durations.
    foldBinding(clonedModule, bound, placed);
    auto owning = mlir::OwningOpRef<mlir::ModuleOp>(clonedModule);
    PulseModule result(ctx_, std::move(owning));
    result.loops_ = loops_;
    result.runTime_ = runTime_;
    result.keepSource();
    try {
      runSchedule(*result.module_, true, schedule, maxDrives, maxReadouts,
                  readoutLatency, switchPenalty,
                  "specialize: pass pipeline failed", specialized_.frameChanges,
                  verifyEverything());
    } catch (const std::runtime_error &error) {
      // Values that take the schedule past a controller's longest program
      // are what is wrong: an error of the values.
      pulse::rethrowPastProgramLimit(error);
    }
    // Verified, or trusted as the verified module it is rescheduled from.
    result.verified();
    return result;
  }

private:
  struct SymbolicParameter {
    size_t slot;
    size_t argument;
    double coefficient;
  };
  struct TermTemplate {
    Term term;
    std::vector<SymbolicParameter> symbolic;
  };
  struct ProgramTemplate {
    std::vector<int64_t> extents;
    std::vector<TermTemplate> hamiltonian, collapse;
    double tStart, tEnd;
    int64_t steps;
    std::string integrator;
    size_t arguments = 0;
  };
  static constexpr size_t kMaxPrograms = 64;

  ProgramTemplate lowerOperatorProgram() {
    auto lowered = mlir::OwningOpRef<mlir::ModuleOp>(module_->clone());
    mlir::PassManager pm(ctx_.get());
    pm.addPass(pulse::createPulseToQOpPass());
    runPipeline(pm, *lowered, "pulse-to-qop failed");

    size_t arguments = 0;
    auto constantI64 = [](mlir::Value value) -> int64_t {
      mlir::Attribute attr;
      if (mlir::matchPattern(value, mlir::m_Constant(&attr)))
        if (auto integer = mlir::dyn_cast<mlir::IntegerAttr>(attr))
          return integer.getInt();
      throw std::runtime_error("qop operator target is not a constant");
    };
    auto spinData = [](llvm::StringRef kind) -> std::vector<double> {
      if (kind == "spin_x")
        return {0, 0, 1, 0, 1, 0, 0, 0};
      if (kind == "spin_y")
        return {0, 0, 0, 1, 0, -1, 0, 0};
      if (kind == "spin_z")
        return {1, 0, 0, 0, 0, 0, -1, 0};
      if (kind == "spin_i")
        return {1, 0, 0, 0, 0, 0, 1, 0};
      if (kind == "spin_lowering")
        return {0, 0, 0, 0, 1, 0, 0, 0};
      if (kind == "spin_raising")
        return {0, 0, 1, 0, 0, 0, 0, 0};
      throw std::runtime_error("unsupported qop spin kind " + kind.str());
    };
    auto term = [&](mlir::Value productValue) -> TermTemplate {
      mlir::Operation *product = productValue.getDefiningOp();
      if (!product || product->getName().getStringRef() != "qop.make_product")
        throw std::runtime_error("qop sum term is not a qop.make_product");
      TermTemplate result{{1.0, 0.0, 0, {}, {}}, {}};
      Term &data = result.term;
      mlir::Operation *scalar = product->getOperand(0).getDefiningOp();
      if (scalar->getName().getStringRef() == "qop.const_scalar") {
        std::get<0>(data) = mlir::cast<mlir::FloatAttr>(scalar->getAttr("real"))
                                .getValueAsDouble();
        std::get<1>(data) = mlir::cast<mlir::FloatAttr>(scalar->getAttr("imag"))
                                .getValueAsDouble();
      } else {
        std::get<2>(data) =
            mlir::cast<mlir::IntegerAttr>(scalar->getAttr("cudm.callback_kind"))
                .getInt();
        auto params = mlir::cast<mlir::DenseF64ArrayAttr>(
            scalar->getAttr("cudm.callback_params"));
        std::get<3>(data).assign(params.asArrayRef().begin(),
                                 params.asArrayRef().end());
        if (auto slots = scalar->getAttrOfType<mlir::DenseI64ArrayAttr>(
                "cudm.callback_param_arguments")) {
          auto coefficients = scalar->getAttrOfType<mlir::DenseF64ArrayAttr>(
              "cudm.callback_param_coefficients");
          for (size_t i = 0; i < coefficients.size(); ++i) {
            auto argument = static_cast<size_t>(slots[2 * i + 1]);
            result.symbolic.push_back(
                {static_cast<size_t>(slots[2 * i]), argument, coefficients[i]});
            arguments = std::max(arguments, argument + 1);
          }
        }
      }
      for (mlir::Value factor : product->getOperands().drop_front()) {
        mlir::Operation *leaf = factor.getDefiningOp();
        if (leaf->getName().getStringRef() == "qop.spin") {
          std::string kind;
          llvm::raw_string_ostream os(kind);
          leaf->getAttr("kind").print(os);
          llvm::StringRef name(kind);
          name = name.substr(name.rfind(' ') + 1).rtrim('>');
          std::get<4>(data).emplace_back(constantI64(leaf->getOperand(0)), 2,
                                         spinData(name));
        } else {
          auto targets =
              mlir::cast<mlir::DenseI64ArrayAttr>(leaf->getAttr("targets"));
          auto dims =
              mlir::cast<mlir::DenseI64ArrayAttr>(leaf->getAttr("dimensions"));
          mlir::Attribute matrix;
          if (!mlir::matchPattern(leaf->getOperand(0),
                                  mlir::m_Constant(&matrix)))
            throw std::runtime_error("qop matrix leaf data is not a constant");
          auto values = mlir::cast<mlir::DenseFPElementsAttr>(matrix);
          std::get<4>(data).emplace_back(
              targets[0], dims[0],
              std::vector<double>(values.getValues<double>().begin(),
                                  values.getValues<double>().end()));
        }
      }
      return result;
    };

    mlir::Operation *lindblad = nullptr;
    lowered->walk([&](mlir::Operation *op) {
      if (op->getName().getStringRef() == "qop.evolve")
        lindblad = op->getOperand(0).getDefiningOp();
    });
    if (!lindblad || lindblad->getName().getStringRef() != "qop.lindblad")
      throw std::runtime_error("pulse-to-qop produced no qop.evolve of a "
                               "qop.lindblad");
    auto sumTerms = [&](mlir::Value sum, std::vector<TermTemplate> &terms) {
      for (mlir::Value product : sum.getDefiningOp()->getOperands())
        terms.push_back(term(product));
    };
    ProgramTemplate program;
    sumTerms(lindblad->getOperand(0), program.hamiltonian);
    for (mlir::Value op : lindblad->getOperands().drop_front())
      sumTerms(op, program.collapse);

    mlir::Operation *top = lowered->getOperation();
    if (auto attr =
            top->getAttrOfType<mlir::DenseI64ArrayAttr>("qop.mode_extents"))
      program.extents.assign(attr.asArrayRef().begin(),
                             attr.asArrayRef().end());
    else {
      int64_t modes = 1;
      if (auto attr = top->getAttrOfType<mlir::IntegerAttr>("qop.qudit_count"))
        modes = attr.getInt();
      else if (auto attr =
                   top->getAttrOfType<mlir::IntegerAttr>("pulse.qudit_count"))
        modes = attr.getInt();
      for (const TermTemplate &t : program.hamiltonian)
        for (const Factor &f : std::get<4>(t.term))
          modes = std::max(modes, std::get<0>(f) + 1);
      program.extents.assign(static_cast<size_t>(modes), 2);
    }
    auto number = [&](llvm::StringRef name, double fallback) {
      if (auto attr = top->getAttrOfType<mlir::FloatAttr>(name))
        return attr.getValueAsDouble();
      return fallback;
    };
    program.tStart = number("qop.t_start", 0.0);
    program.tEnd = number("qop.t_end", 0.0);
    program.steps = 100;
    if (auto attr = top->getAttrOfType<mlir::IntegerAttr>("qop.step_count"))
      program.steps = attr.getInt();
    program.integrator = "rk4";
    if (auto attr = top->getAttrOfType<mlir::StringAttr>("qop.integrator"))
      program.integrator = attr.getValue().str();
    program.arguments = arguments;
    return program;
  }

  /// Schedule `module` with the schedule pass of `policy` and these
  /// resource limits, check its classical calls if `checkCalls` and it
  /// declares any, then run the passes that run on a scheduled module,
  /// raising with `what` on failure. `frameChanges`, when known, says
  /// whether the module has frame-change work; otherwise a walk finds out.
  /// The module is verified at the end, unless not `verify`.
  void runSchedule(mlir::ModuleOp module, bool checkCalls,
                   const std::string &policy, int64_t maxDrives,
                   int64_t maxReadouts, int64_t readoutLatency,
                   int64_t switchPenalty, const std::string &what,
                   std::optional<bool> frameChanges = std::nullopt,
                   bool verify = true) {
    bool rcp = policy == "rcp" || policy == "alap_rcp";
    ContextStore::Options options{
        policy,
        rcp ? maxDrives : 0,
        rcp ? maxReadouts : 0,
        readoutLatency,
        rcp ? switchPenalty : 0,
        checkCalls && classical::declaresClassicalCalls(module),
        frameChanges ? *frameChanges : pulse::hasFrameChangeWork(module)};
    auto passes = ContextStore::take(ctx_.get(), options);
    if (!passes) {
      passes = std::make_unique<mlir::PassManager>(ctx_.get());
      addSchedule(*passes, options);
    }
    runPipelineVerifyingOnce(*passes, module, what, verify);
    ContextStore::keep(ctx_.get(), std::move(options), std::move(passes));
  }

  /// Add to `pm` the passes runSchedule runs with `options`.
  static void addSchedule(mlir::PassManager &pm,
                          const ContextStore::Options &options) {
    auto &[policy, maxDrives, maxReadouts, readoutLatency, switchPenalty,
           checkCalls, frameChanges] = options;
    if (policy == "asap")
      pm.addNestedPass<mlir::func::FuncOp>(
          pulse::createPulseScheduleAsapPass(readoutLatency));
    else if (policy == "alap")
      pm.addNestedPass<mlir::func::FuncOp>(
          pulse::createPulseScheduleAlapPass(readoutLatency));
    else if (policy == "rcp" || policy == "alap_rcp")
      pm.addNestedPass<mlir::func::FuncOp>(pulse::createPulseScheduleRcpPass(
          maxDrives, maxReadouts, readoutLatency, switchPenalty,
          policy == "alap_rcp"));
    else
      throw std::invalid_argument("unknown schedule policy: " + policy);
    if (checkCalls)
      pm.addPass(pulse::createPulseClassicalCheckPass(readoutLatency));
    addAfterSchedulePasses(pm, frameChanges);
  }

  /// Passes that run on every module once it is scheduled and checked, by
  /// schedule(), for a call's values (specialize) and for simulation: a
  /// pass that needs the times of the ops, or that must see each binding,
  /// goes here. A pass that would leave the module as it is is not added:
  /// a pass manager's run of one costs more than the walk that finds out.
  static void addAfterSchedulePasses(mlir::PassManager &pm, bool frameChanges) {
    if (frameChanges)
      pm.addPass(pulse::createPulseFrameChangesPass());
  }

  using ToneSeeds = llvm::DenseMap<mlir::Operation *, std::array<double, 4>>;

  /// bind_call() along `path` on a simulation module or window: the window
  /// that starts after the latest run-time wait the path is past and ends
  /// at the first whose length is not bound, or the whole program where no
  /// such wait is. None for a module the program has no windows of.
  std::optional<PulseModule>
  bindInWindow(int64_t index, const std::vector<double> &values,
               const std::vector<std::vector<int64_t>> &path,
               const std::string &policy, int64_t maxDrives,
               int64_t maxReadouts, int64_t readoutLatency,
               int64_t switchPenalty) {
    if (!program_) {
      program_ = simulationProgram(*module_);
      if (program_->eligible) {
        first_ = 0;
        last_ = static_cast<int64_t>(program_->ops.size()) - 1;
        for (mlir::Operation &op :
             getFuncOp().getBody().front().without_terminator())
          windowOps_.push_back(&op);
        for (auto [own, theirs] : llvm::zip(windowOps_, program_->ops))
          if (program_->rootOps.contains(theirs))
            windowRoots_[own] = theirs;
      }
    }
    // Without a run-time wait the window is the whole program.
    if (!program_->eligible || (!window_ && program_->barriers.empty()))
      return std::nullopt;
    const SimulationProgram &program = *program_;
    const int64_t count = static_cast<int64_t>(program.ops.size());
    const int64_t base = program.submitsBefore[first_];
    const int64_t call = base + index;
    if (index < 0 || call >= program.submitsBefore[last_ + 1])
      throw std::invalid_argument("bind_call: no call execution " +
                                  std::to_string(index));
    std::map<int64_t, std::vector<double>> answers = answers_;
    answers[call] = values;
    auto bound = [&](int64_t c) { return c < base || answers.count(c) != 0; };
    auto unbound = [&](int64_t at) {
      return !llvm::all_of(program.barriers.at(at), bound);
    };
    auto whole = [&]() -> std::optional<PulseModule> {
      std::map<int64_t, std::vector<double>> all = answers;
      for (const RetiredAnswer *r = retired_.get(); r; r = r->next.get())
        all.try_emplace(r->call, r->values);
      auto result = makeWindow(0, count - 1, 0, std::move(all), nullptr,
                               ToneSeeds(), policy, maxDrives, maxReadouts,
                               readoutLatency, switchPenalty);
      if (!result && window_)
        throw std::runtime_error(
            "bind_call: the program a window was cut from does not schedule "
            "whole");
      return result;
    };

    // The latest run-time wait that holds every line, is bound, ends by the
    // time the call starts and that only lines, constants and answers
    // cross: the window starts after it.
    const int64_t held = last_ - first_ + 1;
    mlir::func::FuncOp function = getFuncOp();
    std::vector<int64_t> lowest(held, std::numeric_limits<int64_t>::max()),
        highest(held, std::numeric_limits<int64_t>::min());
    classical::SubmitOp submit;
    int64_t seenCalls = 0;
    for (int64_t i = 0; i < held; ++i)
      windowOps_[i]->walk([&](mlir::Operation *op) {
        if (auto found = mlir::dyn_cast<classical::SubmitOp>(op);
            found && seenCalls++ == index)
          submit = found;
        if (!mlir::isa<pulse::DriveOp, pulse::ReadoutOp, pulse::WaitOp,
                       classical::SubmitOp, classical::WaitOp>(op))
          return;
        auto start = pulse::startCycles(op);
        auto duration = pulse::scheduledDuration(op);
        if (!start || !duration)
          return;
        int64_t end = *start + *duration;
        for (auto [loop, period] :
             llvm::zip(pulse::enclosingLoops(op, function),
                       pulse::periodCycles(op))) {
          int64_t trips = pulse::constantTripCount(loop).value_or(1);
          end += std::max<int64_t>(0, trips - 1) * std::max<int64_t>(0, period);
        }
        lowest[i] = std::min(lowest[i], *start);
        highest[i] = std::max(highest[i], end);
      });
    if (!submit)
      throw std::invalid_argument("bind_call: no call execution " +
                                  std::to_string(index));
    for (classical::WaitOp wait : classical::waitsOf(submit))
      if (wait.getValues().size() != values.size())
        throw std::invalid_argument(
            "bind_call: " + submit.getCallee().str() + " answers " +
            std::to_string(wait.getValues().size()) + " values, got " +
            std::to_string(values.size()));
    std::optional<int64_t> callStart = pulse::startCycles(submit);
    if (!callStart)
      return whole();
    std::vector<int64_t> before(held + 1, std::numeric_limits<int64_t>::min()),
        after(held + 1, std::numeric_limits<int64_t>::max());
    for (int64_t i = 0; i < held; ++i)
      before[i + 1] = std::max(before[i], highest[i]);
    for (int64_t i = held - 1; i >= 0; --i)
      after[i] = std::min(after[i + 1], lowest[i]);

    int64_t head = first_, offset = offsetCycles_;
    ToneSeeds seeds = *seedsOf(nullptr);
    const int64_t callAt = program.submitAt[call];
    auto from = program.barriers.lower_bound(first_);
    for (auto it = program.barriers.upper_bound(callAt - 1);
         it != from && head == first_;) {
      --it;
      int64_t at = it->first, local = at - first_;
      if (at >= callAt || program.crossed.contains(at) ||
          program.crossedByCall.contains(at) || unbound(at))
        continue;
      auto start = pulse::startCycles(windowOps_[local]);
      auto duration = pulse::scheduledDuration(windowOps_[local]);
      if (!start || !duration)
        continue;
      int64_t end = *start + *duration;
      if (end > *callStart || before[local] > end || after[local + 1] < end)
        continue;
      std::optional<ToneSeeds> found = seedsOf(windowOps_[local + 1], &path);
      if (!found)
        continue;
      head = at + 1;
      offset = end;
      seeds = std::move(*found);
    }

    // The first wait after it whose length is not bound yet, that no call
    // is in flight across: the window ends with it.
    int64_t tail = count - 1;
    for (auto it = program.barriers.lower_bound(std::max(head, callAt));
         it != program.barriers.end(); ++it)
      if (!program.crossedByCall.contains(it->first) && unbound(it->first)) {
        tail = it->first;
        break;
      }

    std::shared_ptr<const RetiredAnswer> retired = retired_;
    const int64_t newBase = program.submitsBefore[head];
    while (!answers.empty() && answers.begin()->first < newBase) {
      auto node = std::make_shared<RetiredAnswer>();
      node->call = answers.begin()->first;
      node->values = std::move(answers.begin()->second);
      node->next = std::move(retired);
      retired = std::move(node);
      answers.erase(answers.begin());
    }
    if (auto window = makeWindow(head, tail, offset, std::move(answers),
                                 std::move(retired), seeds, policy, maxDrives,
                                 maxReadouts, readoutLatency, switchPenalty))
      return window;
    return whole();
  }

  /// The frame of each tone this module's own starting frame or its ops
  /// before `stop` leave it in, by the program's op the tone descends from:
  /// none if a measurement before `stop` has no outcome on `path`, a call
  /// before it no answer, or a frame is not a number. Without `stop`, the
  /// frames the module starts with.
  std::optional<ToneSeeds>
  seedsOf(mlir::Operation *stop,
          const std::vector<std::vector<int64_t>> *path = nullptr) {
    ToneSeeds seeds;
    for (auto [own, theirs] : windowRoots_)
      if (auto seed = own->getAttrOfType<mlir::DenseF64ArrayAttr>(
              pulse::kToneStateAttr))
        seeds[theirs] = {seed[0], seed[1], seed[2], seed[3]};
    if (!stop)
      return seeds;
    pulse::Path chosen = toPath(*path, readoutOps());
    pulse::ExecutionWalker walker(clock(), chosen);
    walker.stopBefore = stop;
    bool unanswered = false;
    walker.onSubmit = [&](const pulse::Occurrence &occurrence) {
      unanswered |= !occurrence.op->hasAttr("classical.bound");
      return false;
    };
    if (!walker.run(getFuncOp()) || unanswered || !walker.undecided.empty())
      return std::nullopt;
    for (auto &[root, tone] : walker.toneStates()) {
      auto theirs = windowRoots_.find(root);
      if (theirs == windowRoots_.end() || !tone.frequencyHz.isConstant() ||
          !tone.frameFrequencyHz.isConstant() || !tone.phase.isConstant())
        return std::nullopt;
      seeds[theirs->second] = {tone.frequencyHz.constant,
                               tone.frameFrequencyHz.constant,
                               tone.phase.constant, tone.timeNs};
    }
    return seeds;
  }

  /// The window of top-level ops [head, tail] of the program, after its
  /// allocations, lines and links and what the window reads from before
  /// it, with `answers` bound (and `retired`, for what it reads), the tones
  /// starting in `seeds`, scheduled and moved `offset` cycles later. None
  /// if it reads a value from before it that it cannot be given, or if its
  /// lines do not share one timeline.
  std::optional<PulseModule>
  makeWindow(int64_t head, int64_t tail, int64_t offset,
             std::map<int64_t, std::vector<double>> answers,
             std::shared_ptr<const RetiredAnswer> retired,
             const ToneSeeds &seeds, const std::string &policy,
             int64_t maxDrives, int64_t maxReadouts, int64_t readoutLatency,
             int64_t switchPenalty) {
    const SimulationProgram &program = *program_;
    mlir::ModuleOp source = *program.module;
    auto sourceMain = source.lookupSymbol<mlir::func::FuncOp>("main");
    mlir::Block &sourceBody = sourceMain.getBody().front();
    mlir::OpBuilder b(ctx_.get());
    auto module = mlir::OwningOpRef<mlir::ModuleOp>(
        mlir::ModuleOp::create(source.getLoc()));
    (*module)->setAttrs(source->getAttrDictionary());
    b.setInsertionPointToEnd(module->getBody());
    mlir::func::FuncOp main;
    for (mlir::Operation &op : *source.getBody()) {
      if (&op != sourceMain.getOperation()) {
        b.clone(op);
        continue;
      }
      main = mlir::cast<mlir::func::FuncOp>(b.cloneWithoutRegions(op));
    }
    mlir::Block *body = main.addEntryBlock();
    b.setInsertionPointToEnd(body);

    mlir::IRMapping map;
    llvm::DenseMap<mlir::Operation *, mlir::Operation *> roots;
    auto answerOf = [&](int64_t call) -> const std::vector<double> * {
      auto found = answers.find(call);
      if (found != answers.end())
        return &found->second;
      for (const RetiredAnswer *r = retired.get(); r; r = r->next.get())
        if (r->call == call)
          return &r->values;
      return nullptr;
    };
    std::function<bool(mlir::Operation *)> importOp;
    std::function<bool(mlir::Value)> importValue = [&](mlir::Value value) {
      if (map.contains(value))
        return true;
      mlir::Operation *def = value.getDefiningOp();
      if (!def)
        return false;
      if (isThreadedType(value.getType())) {
        mlir::Operation *root = program.rootOf.lookup(value);
        if (!root || !importOp(root))
          return false;
        for (mlir::Value result : root->getResults())
          if (result.getType() == value.getType()) {
            map.map(value, map.lookup(result));
            return true;
          }
        return false;
      }
      if (auto wait = mlir::dyn_cast<classical::WaitOp>(def)) {
        const std::vector<double> *values =
            answerOf(program.callOfWait.lookup(def));
        unsigned n = mlir::cast<mlir::OpResult>(value).getResultNumber();
        if (!values || n >= values->size())
          return false;
        mlir::Type type = value.getType();
        mlir::TypedAttr attr;
        if (type.isF64())
          attr = b.getF64FloatAttr((*values)[n]);
        else if (type.isSignlessInteger(64))
          attr = b.getI64IntegerAttr(
              static_cast<int64_t>(std::round((*values)[n])));
        else
          return false;
        map.map(value, mlir::arith::ConstantOp::create(b, wait.getLoc(), attr));
        return true;
      }
      return importOp(def);
    };
    importOp = [&](mlir::Operation *op) {
      if (op->getNumResults() && map.contains(op->getResult(0)))
        return true;
      if (op->getNumRegions() != 0)
        return false;
      for (mlir::Value operand : op->getOperands())
        if (!importValue(operand))
          return false;
      mlir::Operation *copy = b.clone(*op, map);
      if (program.rootOps.contains(op)) {
        roots[copy] = op;
        if (auto seed = seeds.find(op); seed != seeds.end())
          copy->setAttr(pulse::kToneStateAttr,
                        mlir::DenseF64ArrayAttr::get(ctx_.get(), seed->second));
      }
      return true;
    };
    for (int64_t root : program.roots)
      if (root < head && !importOp(program.ops[root]))
        return std::nullopt;
    for (int64_t i = head; i <= tail; ++i) {
      mlir::WalkResult result = program.ops[i]->walk([&](mlir::Operation *op) {
        for (mlir::Value operand : op->getOperands()) {
          mlir::Operation *def = operand.getDefiningOp();
          bool outside =
              def ? def->getBlock() == &sourceBody &&
                        program.position.lookup(def) < head
                  : mlir::cast<mlir::BlockArgument>(operand).getOwner() ==
                        &sourceBody;
          if (outside && !importValue(operand))
            return mlir::WalkResult::interrupt();
        }
        return mlir::WalkResult::advance();
      });
      if (result.wasInterrupted())
        return std::nullopt;
    }
    llvm::SmallVector<mlir::Operation *> ops;
    for (int64_t i = head; i <= tail; ++i) {
      mlir::Operation *copy = b.clone(*program.ops[i], map);
      ops.push_back(copy);
      if (program.rootOps.contains(program.ops[i]))
        roots[copy] = program.ops[i];
    }
    b.clone(*sourceBody.getTerminator(), map);

    // The answers bound so far, on the calls in the window.
    int64_t call = program.submitsBefore[head];
    bool fits = true;
    main.walk([&](classical::SubmitOp submit) {
      auto found = answers.find(call++);
      if (found == answers.end())
        return;
      submit->setAttr("classical.bound", mlir::UnitAttr::get(ctx_.get()));
      for (classical::WaitOp wait : classical::waitsOf(submit)) {
        if (wait.getValues().size() != found->second.size())
          fits = false;
        else
          bindConstants(wait, found->second);
      }
    });
    if (!fits)
      throw std::invalid_argument("bind_call: an answer does not fit its call");
    // A program built from a kernel numbered its readouts; one that was not
    // is numbered here as in the program, before scheduling the window can
    // copy a readout (pulse::numberReadouts does nothing to numbered ones).
    pulse::numberReadouts(*module, program.readoutsBefore[head]);

    PulseModule result(ctx_, std::move(module));
    result.loops_ = loops_;
    result.runTime_ = runTime_;
    result.schedule(policy, maxDrives, maxReadouts, readoutLatency,
                    switchPenalty);
    mlir::WalkResult split = result.module_->walk([](mlir::Operation *op) {
      return op->hasAttr(classical::kTimelineAttr)
                 ? mlir::WalkResult::interrupt()
                 : mlir::WalkResult::advance();
    });
    if (split.wasInterrupted())
      return std::nullopt;
    if (offset != 0)
      result.module_->walk([&](mlir::Operation *op) {
        if (auto start = pulse::startCycles(op))
          pulse::setStartCycles(op, *start + offset);
      });
    result.program_ = program_;
    result.window_ = true;
    result.first_ = head;
    result.last_ = tail;
    result.offsetCycles_ = offset;
    result.answers_ = std::move(answers);
    result.retired_ = std::move(retired);
    result.windowOps_ = std::move(ops);
    result.windowRoots_ = std::move(roots);
    return result;
  }

  /// Bind `values` to the results of `wait`: the wait stays, so what it
  /// feeds keeps waiting for it, and its results read as the values.
  void bindConstants(classical::WaitOp wait,
                     const std::vector<double> &values) {
    std::vector<double> rounded(values);
    for (auto [value, type] : llvm::zip(rounded, wait.getValues().getTypes()))
      if (!mlir::isa<mlir::FloatType>(type))
        value = std::round(value);
    wait->setAttr("classical.answers",
                  mlir::DenseF64ArrayAttr::get(ctx_.get(), rounded));
  }

  llvm::SmallVector<mlir::Operation *> readoutOps() {
    llvm::SmallVector<mlir::Operation *> readouts;
    module_->walk([&](mlir::Operation *op) {
      if (mlir::isa<pulse::ReadoutOp>(op) ||
          op->getName().getStringRef() == "pulse.iq_acquire")
        readouts.push_back(op);
    });
    return readouts;
  }

  static pulse::Path toPath(const std::vector<std::vector<int64_t>> &entries,
                            llvm::ArrayRef<mlir::Operation *> readouts) {
    pulse::Path path;
    for (const auto &entry : entries) {
      if (entry.size() < 2 || entry.front() < 0 ||
          entry.front() >= static_cast<int64_t>(readouts.size()))
        throw std::invalid_argument("path entries are [readout index, "
                                    "iteration..., outcome]");
      path[{readouts[entry.front()],
            std::vector<int64_t>(entry.begin() + 1, entry.end() - 1)}] =
          entry.back();
    }
    return path;
  }

  double clock() { return pulse::clockGHz(*module_); }

  mlir::func::FuncOp getFuncOp() {
    return module_->lookupSymbol<mlir::func::FuncOp>("main");
  }

  /// Keep a copy of the module before it is scheduled if it has schedule
  /// blocks: scheduling inlines them, and a module scheduled again (with
  /// values or answers bound) must place each block whole again.
  void keepSource() {
    bool blocks = false;
    module_->walk([&](pulse::ScheduleBlockOp) {
      blocks = true;
      return mlir::WalkResult::interrupt();
    });
    if (blocks)
      source_ = mlir::OwningOpRef<mlir::ModuleOp>(module_->clone());
  }

  /// What a module scheduled again starts from: the module before it was
  /// scheduled, if it had blocks, else the module.
  mlir::ModuleOp source() { return source_ ? *source_ : *module_; }

  /// Whether the module keeps a select; notes (in runTime_) whether it
  /// computes something at run time or changes a frame in a branch.
  bool keepsSelects() {
    bool kept = false, runTime = false;
    module_->walk([&](mlir::Operation *op) {
      if (mlir::isa<mlir::arith::SelectOp>(op)) {
        kept = true;
        return mlir::WalkResult::interrupt();
      }
      if (mlir::isa<pulse::ShiftPhaseOp, pulse::SetPhaseOp,
                    pulse::ShiftFrequencyOp, pulse::SetFrequencyOp>(op))
        runTime |= op->getParentOfType<mlir::scf::IfOp>() != nullptr;
      else if (mlir::isa<classical::SubmitOp>(op))
        runTime = true;
      else if (mlir::isa<mlir::scf::IfOp>(op))
        runTime |= llvm::any_of(op->getResultTypes(), yieldsValue);
      return mlir::WalkResult::advance();
    });
    runTime_ = kept || runTime;
    return kept;
  }

  friend class PulseModuleBuilder;

  /// The module changed in place: a specialized module kept from before, or
  /// this one kept as another's, is not reused.
  void changed() {
    ++version_;
    specialized_.ops.store(-1);
    // The canonicalization patterns were chosen for the ops the module held.
    canonicalPatterns_[0].reset();
    canonicalPatterns_[1].reset();
  }

  /// Note that the module, as it is now, has been verified.
  void verified() { verified_ = version_.load(); }

  struct Specialized {
    std::string key;
    uint64_t version;       // of this module when it was built
    uint64_t moduleVersion; // of the module built
    nb::object module;
  };
  /// Used with the GIL held (attached, when free-threaded), so its mutex
  /// locks only when free-threaded.
  struct SpecializedCache {
    nb::ft_mutex mutex;
    std::vector<Specialized> entries; // least recently used first
    size_t maxEntries = 16;
    int64_t maxOps = int64_t(1) << 16;
    std::atomic<int64_t> ops{-1}; // of this module, once counted
    // Whether the frame-changes pass has work in this module, counted with
    // `ops`: binding values removes frame changes at most.
    bool frameChanges = true;
    // Hashes of the keys last built and not kept: a module is kept only
    // once its key recurs, so that values seen once hold no memory.
    std::array<size_t, 32> missed{};
    size_t nextMissed = 0;
    SpecializedCache() = default;
    SpecializedCache(SpecializedCache &&other) noexcept
        : entries(std::move(other.entries)), maxEntries(other.maxEntries),
          maxOps(other.maxOps), ops(other.ops.load()),
          frameChanges(other.frameChanges), missed(other.missed),
          nextMissed(other.nextMissed) {}
  };
  struct Version {
    std::atomic<uint64_t> value{0};
    Version() = default;
    Version(Version &&other) noexcept : value(other.value.load()) {}
    uint64_t load() const { return value.load(); }
    void operator++() { ++value; }
  };

  std::shared_ptr<mlir::MLIRContext> ctx_;
  mlir::OwningOpRef<mlir::ModuleOp> module_;
  // The version_ at which the module was last verified (verified()): none
  // yet for a module just built or copied.
  uint64_t verified_ = UINT64_MAX;
  // Whether the program can compute something at run time or change a frame
  // in a branch: false only where the builder saw neither, so that
  // controller_program() looks for them only where there can be any.
  bool runTime_ = true;
  // False only for a module built from a program with no loop, whose
  // event_tree() is its events().
  bool loops_ = true;
  /// For a module with waits whose length floats set: what it lasts op after
  /// op with each at its lowest length, the most it may last, whether that
  /// is the controller's, and how often each such wait runs.
  struct Length {
    int64_t cycles = 0, most = 0, controller = 0;
    std::vector<int64_t> repeats, excess;
  };
  Length length_;
  // The module before scheduling, while it has schedule blocks (see
  // keepSource).
  mlir::OwningOpRef<mlir::ModuleOp> source_;
  /// Fold `module`, whose float arguments were replaced by the constants
  /// `values`: as the canonicalizer does, then with the pattern passes this
  /// module ran, on the only ops the values can open to them, frame changes
  /// by a constant phase and drives of a square pulse, if the values reach
  /// one. Both pattern sets are gathered once per module, the canonical one
  /// for its bindings placed on slots (which allocate qudits) and once for
  /// the others.
  void foldBinding(mlir::ModuleOp module, llvm::ArrayRef<mlir::Value> values,
                   bool placed) {
    bool phases = false, amplitudes = false;
    if (!rewrites_.empty()) {
      // What the values reach, as mlir::getForwardSlice gathers it (the
      // users of a value, the ops in a user's regions, and the users of
      // all their results), with each op looked at once and without
      // recursion: the ops of one line follow one another, as many deep
      // as the line has ops, and the slice is taken a call deep for each.
      // A user with regions is in `reached` once it and all it holds have
      // been walked, so a loop that many of the values reach is walked once.
      llvm::SmallVector<mlir::Value, 16> open(values.begin(), values.end());
      llvm::DenseSet<mlir::Operation *> reached;
      auto reach = [&](mlir::Operation *op) {
        if (!reached.insert(op).second)
          return;
        phases |= mlir::isa<pulse::ShiftPhaseOp>(op);
        amplitudes |= mlir::isa<pulse::SquarePulseOp>(op);
        for (mlir::Value result : op->getResults())
          open.push_back(result);
      };
      while (!open.empty() && !(phases && amplitudes)) {
        mlir::Value value = open.pop_back_val();
        for (mlir::Operation *user : value.getUsers()) {
          if (user->getNumRegions() == 0)
            reach(user);
          else if (!reached.contains(user))
            user->walk(reach);
        }
      }
    }

    std::shared_ptr<const mlir::FrozenRewritePatternSet> &canonical =
        canonicalPatterns_[placed];
    if (!canonical)
      canonical = canonicalPatternsFor(ctx_.get(), module);
    // The canonicalizer pass's defaults.
    mlir::GreedyRewriteConfig config;
    config.setRegionSimplificationLevel(
        mlir::GreedySimplifyRegionLevel::Normal);
    (void)mlir::applyPatternsGreedily(module, *canonical, config);
    if (!phases && !amplitudes)
      return;

    llvm::SmallVector<mlir::Operation *> ops;
    module.walk([&](mlir::Operation *op) {
      if (auto shift = mlir::dyn_cast<pulse::ShiftPhaseOp>(op); shift && phases)
        if (mlir::Operation *phase = shift.getPhaseRad().getDefiningOp();
            phase && mlir::isa<mlir::arith::ConstantFloatOp>(phase)) {
          ops.push_back(phase);
          ops.push_back(op);
        }
      if (auto drive = mlir::dyn_cast<pulse::DriveOp>(op); drive && amplitudes)
        if (drive.getPulse().getDefiningOp<pulse::SquarePulseOp>())
          ops.push_back(op);
    });
    if (ops.empty())
      return;
    if (!rewritePatterns_) {
      // One frozen set per context for each list of stages.
      std::string key = "rewrite:";
      for (const pulse::RewriteStage *stage : rewrites_)
        key += stage->argument.str() + ",";
      rewritePatterns_ = ContextStore::frozen(ctx_.get(), key, [&] {
        mlir::RewritePatternSet set(ctx_.get());
        for (const pulse::RewriteStage *stage : rewrites_)
          stage->populate(set);
        return set;
      });
    }
    mlir::GreedyRewriteConfig opConfig;
    opConfig.enableFolding(false).enableConstantCSE(false);
    opConfig.setStrictness(mlir::GreedyRewriteStrictness::ExistingOps);
    (void)mlir::applyOpPatternsGreedily(ops, *rewritePatterns_, opConfig);
  }

  // The pattern passes run_passes ran that rerun on a binding
  // (pulse-fold-phases, pulse-fuse-pulses), in the order first run.
  llvm::SmallVector<const pulse::RewriteStage *, 4> rewrites_;
  // What foldBinding rewrites with, once gathered: the canonical patterns
  // of the bindings not placed on slots, then of those placed.
  std::shared_ptr<const mlir::FrozenRewritePatternSet> canonicalPatterns_[2];
  std::shared_ptr<const mlir::FrozenRewritePatternSet> rewritePatterns_;
  // Lowered operator programs by module attribute dictionary.
  std::map<const void *, ProgramTemplate> programs_;
  Version version_;
  // Its specialized modules by their arguments (specialize_shared).
  SpecializedCache specialized_;
  // Made by simulation_module(): bind_call() along a path cuts windows of
  // it (see SimulationProgram).
  bool simulation_ = false;
  // A window: top-level ops [first_, last_] of program_, the first moved
  // offsetCycles_ later, with answers_ of its calls bound and retired_ of
  // calls before it; windowOps_ are its ops for those, windowRoots_ the
  // program's op each of its allocations, lines, links and tones is.
  bool window_ = false;
  std::shared_ptr<const SimulationProgram> program_;
  int64_t first_ = 0, last_ = -1, offsetCycles_ = 0;
  std::map<int64_t, std::vector<double>> answers_;
  std::shared_ptr<const RetiredAnswer> retired_;
  llvm::SmallVector<mlir::Operation *> windowOps_;
  llvm::DenseMap<mlir::Operation *, mlir::Operation *> windowRoots_;
  // Python threads may call into one module concurrently once the GIL is
  // released; MLIR does not allow concurrent mutation of one module.
  std::shared_ptr<std::recursive_mutex> lock_;
};

// -----------------------------------------------------------------------
// PulseModuleBuilder: decode a packed int64 buffer into in-memory Pulse IR
// -----------------------------------------------------------------------
/// Loading the dialects into a new context takes a fraction of a millisecond,
/// so builders share a context, made with the dialects its modules can hold
/// (ContextDialects.h). A context keeps every attribute it ever uniqued (each
/// compile with new qubit frequencies, for instance, adds a few hundred bytes
/// per qubit), so it is retired after a number of builders or once the
/// programs built in it reach a number of packed words, whichever comes
/// first, and freed with its last module. Both limits can be set with
/// environment variables, read whenever a builder takes a context.
class SharedContextPool {
public:
  struct Budget {
    int64_t modules;
    int64_t words;
  };

  static constexpr const char *kModulesVariable =
      "CUDAQ_PULSE_CONTEXT_MAX_MODULES";
  static constexpr const char *kWordsVariable = "CUDAQ_PULSE_CONTEXT_MAX_WORDS";

  /// The limits in effect; throws std::invalid_argument for a malformed
  /// environment variable.
  static Budget budget() {
    return {limit(kModulesVariable, 256), limit(kWordsVariable, 1 << 20)};
  }

  static std::shared_ptr<mlir::MLIRContext> acquire() {
    const Budget limits = budget();
    std::lock_guard<std::mutex> guard(lock());
    auto &pool = state();
    if (!pool.current || ++pool.uses > limits.modules ||
        pool.words >= limits.words) {
      pool.current = makeContext(cudaq_pulse::sharedContextDialects());
      pool.current->loadAllAvailableDialects();
      pool.uses = 1;
      pool.words = 0;
      ++pool.created;
    }
    return pool.current;
  }

  /// Count a program of `words` packed words against `context`'s budget.
  static void charge(const mlir::MLIRContext *context, int64_t words) {
    std::lock_guard<std::mutex> guard(lock());
    auto &pool = state();
    if (pool.current.get() == context)
      pool.words += words;
  }

  /// (contexts created, builders on the current one, its packed words).
  static std::tuple<int64_t, int64_t, int64_t> stats() {
    std::lock_guard<std::mutex> guard(lock());
    auto &pool = state();
    return {pool.created, pool.current ? pool.uses : 0, pool.words};
  }

  /// A forked child has only the forking thread. The pool's lock is held
  /// across fork so the child inherits it unlocked, and the child starts a
  /// new context: the current one's thread pool has no workers there. That
  /// context is abandoned rather than destroyed, since destroying it would
  /// wait for those workers.
  static void handleForks() {
    pthread_atfork([] { lock().lock(); }, [] { lock().unlock(); },
                   [] {
                     auto &pool = state();
                     if (pool.current)
                       new std::shared_ptr<mlir::MLIRContext>(
                           std::move(pool.current));
                     pool.uses = 0;
                     pool.words = 0;
                     lock().unlock();
                   });
  }

private:
  struct State {
    std::shared_ptr<mlir::MLIRContext> current;
    int64_t uses = 0;
    int64_t words = 0;
    int64_t created = 0;
  };

  static int64_t limit(const char *variable, int64_t fallback) {
    constexpr int64_t kMaximum = int64_t{1} << 40;
    const char *text = std::getenv(variable);
    if (!text || !*text)
      return fallback;
    int64_t value = 0;
    const char *end = text + std::strlen(text);
    auto [stop, error] = std::from_chars(text, end, value);
    if (error != std::errc() || stop != end || value < 1 || value > kMaximum)
      throw std::invalid_argument(
          std::string(variable) + "=" + text +
          " is not valid; it must be a whole number from 1 to " +
          std::to_string(kMaximum) + " (default " + std::to_string(fallback) +
          ")");
    return value;
  }

  static std::mutex &lock() {
    static std::mutex mutex;
    return mutex;
  }
  // Never destroyed: the last context must not be torn down during static
  // destruction, after LLVM's own statics.
  static State &state() {
    static auto *pool = new State();
    return *pool;
  }
};

bool isTimedPulseType(mlir::Type type) {
  return mlir::isa<pulse::DriveLineType, pulse::ReadoutLineType,
                   pulse::ToneType, pulse::MeasurementType>(type) ||
         pulse::isClassicalType(type);
}

/// Whether `value` is defined in `body` or in a region nested in it.
bool definedIn(mlir::Value value, mlir::Block *body) {
  for (mlir::Block *block = value.getParentBlock(); block;) {
    if (block == body)
      return true;
    mlir::Operation *parent = block->getParentOp();
    if (!parent)
      return false;
    block = parent->getBlock();
  }
  return false;
}

/// Wrap the ops traced into `body` in a `pulse.schedule_block` at the
/// builder's insertion point. The lines, tones and measurements the body
/// uses from outside become its operands; the latest version of every line
/// and tone it holds, and every measurement it takes, become its results.
pulse::ScheduleBlockOp buildScheduleBlock(mlir::OpBuilder &builder,
                                          mlir::Location loc,
                                          std::unique_ptr<mlir::Block> body,
                                          llvm::StringRef policy,
                                          llvm::StringRef callee) {
  // Untimed values (waveforms, constants) may be used after the block, so
  // they are defined before it.
  mlir::Block *parent = builder.getInsertionBlock();
  for (mlir::Operation &op : llvm::make_early_inc_range(*body)) {
    bool untimed = op.getNumRegions() == 0 &&
                   llvm::none_of(op.getOperandTypes(), isTimedPulseType) &&
                   llvm::none_of(op.getResultTypes(), isTimedPulseType) &&
                   llvm::none_of(op.getOperands(), [&](mlir::Value operand) {
                     return operand.getParentBlock() == body.get();
                   });
    if (untimed)
      op.moveBefore(parent, parent->end());
  }

  llvm::SmallVector<mlir::Value> inputs;
  llvm::DenseSet<mlir::Value> seen;
  body->walk([&](mlir::Operation *op) {
    for (mlir::Value operand : op->getOperands())
      if (isTimedPulseType(operand.getType()) &&
          !definedIn(operand, body.get()) && seen.insert(operand).second)
        inputs.push_back(operand);
  });
  llvm::SmallVector<mlir::Value> outputs;
  for (mlir::Operation &op : *body)
    for (mlir::Value result : op.getResults())
      if (mlir::isa<pulse::MeasurementType>(result.getType()) ||
          (pulse::isLinearPulseType(result.getType()) && result.use_empty()))
        outputs.push_back(result);

  llvm::SmallVector<mlir::Type> types;
  for (mlir::Value output : outputs)
    types.push_back(output.getType());
  auto block = pulse::ScheduleBlockOp::create(
      builder, loc, types,
      policy == "alap" ? pulse::SchedulePolicy::ALAP
                       : pulse::SchedulePolicy::ASAP,
      inputs,
      callee.empty() ? mlir::StringAttr() : builder.getStringAttr(callee),
      /*start_cycles=*/{}, /*duration_cycles=*/{}, /*period_cycles=*/{});
  block.getBody().push_back(body.release());
  mlir::Block *inner = &block.getBody().front();
  for (mlir::Value input : inputs) {
    mlir::BlockArgument arg = inner->addArgument(input.getType(), loc);
    input.replaceUsesWithIf(arg, [&](mlir::OpOperand &use) {
      return block->isProperAncestor(use.getOwner());
    });
  }
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToEnd(inner);
  pulse::YieldOp::create(builder, loc, outputs);
  return block;
}

/// The cycles a waveform lasts, from constants; 0 when they do not say.
static int64_t waveformCycles(mlir::Value waveform) {
  mlir::Operation *def = waveform.getDefiningOp();
  if (!def)
    return 0;
  if (auto op = mlir::dyn_cast<pulse::CustomSamplesOp>(def))
    return op.getDurationCycles();
  if (mlir::isa<pulse::PulseAddOp, pulse::PulseSubOp, pulse::PulseMulOp,
                pulse::PulseScaleOp, pulse::PulseNegOp>(def))
    return waveformCycles(def->getOperand(0));
  mlir::Value duration;
  if (auto op = mlir::dyn_cast<pulse::GaussianPulseOp>(def))
    duration = op.getDuration();
  else if (auto op = mlir::dyn_cast<pulse::SquarePulseOp>(def))
    duration = op.getDuration();
  else if (auto op = mlir::dyn_cast<pulse::DRAGPulseOp>(def))
    duration = op.getDuration();
  else if (auto op = mlir::dyn_cast<pulse::CosinePulseOp>(def))
    duration = op.getDuration();
  else if (auto op = mlir::dyn_cast<pulse::TanhRampOp>(def))
    duration = op.getDuration();
  else if (auto op = mlir::dyn_cast<pulse::GaussianSquarePulseOp>(def))
    duration = op.getDuration();
  else if (auto op = mlir::dyn_cast<pulse::CustomOp>(def))
    duration = op.getDuration();
  else
    return 0;
  return pulse::constantIndex(duration).value_or(0);
}

class PulseModuleBuilder {
public:
  PulseModuleBuilder() : ctx_(SharedContextPool::acquire()) {
    builder_ = std::make_unique<mlir::OpBuilder>(ctx_.get());
  }

  /// The ids of the lines and tones that the failing operation of a trace
  /// was given: build_from_packed then rejects one that is a result of an
  /// if whose arms leave different lines there, where the if is.
  void expectLost(std::vector<int64_t> results) {
    std::lock_guard<std::mutex> guard(lock_);
    lostLines_ = std::move(results);
  }

  // ─── Packed-buffer decoder (zero-copy path) ──────────────────────────
  PulseModule build_from_packed(
      nb::ndarray<int64_t, nb::ndim<1>, nb::c_contig> stream, double clock_ghz,
      int64_t qudit_count,
      nb::ndarray<double, nb::ndim<1>, nb::c_contig> qubit_freqs,
      std::vector<std::string> param_names = {},
      std::vector<std::string> param_types = {},
      const std::string &name = "main", int64_t slot_count = 0,
      std::optional<nb::ndarray<const double, nb::ndim<1>, nb::c_contig>>
          readout_freqs = std::nullopt) {
    std::lock_guard<std::mutex> guard(lock_);

    const int64_t *d = stream.data();
    const size_t len = stream.shape(0);
    SharedContextPool::charge(ctx_.get(), static_cast<int64_t>(len));
    auto loc = builder_->getUnknownLoc();

    if (clock_ghz <= 0.0)
      throw std::invalid_argument("packed: clock_ghz must be positive");
    if (qudit_count < 0)
      throw std::invalid_argument("packed: qudit_count cannot be negative");
    if (slot_count < 0)
      throw std::invalid_argument("packed: slot_count cannot be negative");
    if (qubit_freqs.shape(0) < static_cast<size_t>(qudit_count))
      throw std::invalid_argument(
          "packed: fewer qubit frequencies than qudit_count");
    if (param_names.size() != param_types.size())
      throw std::invalid_argument(
          "packed: parameter names and types have different lengths");

    auto module = mlir::ModuleOp::create(loc, mlir::StringRef(name));
    // Owned from the start, so a record that fails frees what was built.
    mlir::OwningOpRef<mlir::ModuleOp> owningModule(module);
    module->setAttr("pulse.qudit_count",
                    builder_->getI64IntegerAttr(qudit_count));
    auto controlSystem =
        pulse::controlSystemAt(builder_->getContext(), clock_ghz);
    if (!controlSystem)
      throw std::invalid_argument("packed: clock_ghz " +
                                  std::to_string(clock_ghz) +
                                  " has no exact period in femtoseconds");
    module->setAttr(pulse::kControlSystemAttr, controlSystem);

    // The frequencies of qudits 0 to qudit_count - 1, in the module's
    // quantum system.
    const double *fdata = qubit_freqs.data();
    size_t flen = static_cast<size_t>(qudit_count);
    llvm::SmallVector<double> freqVec(fdata, fdata + flen);
    module->setAttr(pulse::kQuantumSystemAttr,
                    pulse::quantumSystemAt(builder_->getContext(), freqVec));
    // A readout line's tone, by qudit: the readout frequency where given.
    const double *readoutData = readout_freqs ? readout_freqs->data() : fdata;
    size_t readoutLen = readout_freqs ? readout_freqs->shape(0) : flen;

    // Build func.func type: parameters become block arguments
    llvm::SmallVector<mlir::Type> argTypes;
    auto f64Ty = builder_->getF64Type();
    auto i64Ty = builder_->getIntegerType(64);
    for (auto &pt : param_types) {
      if (pt == "i64")
        argTypes.push_back(i64Ty);
      else if (pt == "f64")
        argTypes.push_back(f64Ty);
      else
        throw std::invalid_argument("packed: unsupported parameter type '" +
                                    pt + "'");
    }

    // Runtime qudits follow the numeric parameters: qref arguments with
    // their slot, bound to carriers by specialize().
    auto qrefTy = pulse::QuditType::get(ctx_.get());
    const unsigned firstSlot = argTypes.size();
    argTypes.append(static_cast<size_t>(slot_count), qrefTy);
    auto funcType = builder_->getFunctionType(argTypes, {});
    auto funcOp = mlir::func::FuncOp::create(loc, "main", funcType);
    for (int64_t k = 0; k < slot_count; ++k)
      funcOp.setArgAttr(firstSlot + k, "pulse.slot",
                        builder_->getI64IntegerAttr(k));
    module.push_back(funcOp);
    auto *entryBlock = funcOp.addEntryBlock();
    builder_->setInsertionPointToStart(entryBlock);

    // Store parameter name metadata on the module for later use
    if (!param_names.empty()) {
      llvm::SmallVector<mlir::Attribute> nameAttrs;
      for (auto &n : param_names)
        nameAttrs.push_back(builder_->getStringAttr(n));
      module->setAttr("pulse.param_names", builder_->getArrayAttr(nameAttrs));
    }

    auto driveTy = pulse::DriveLineType::get(ctx_.get());
    auto readoutTy = pulse::ReadoutLineType::get(ctx_.get());
    auto toneTy = pulse::ToneType::get(ctx_.get());
    auto wfTy = pulse::WaveformType::get(ctx_.get());
    auto durTy = pulse::DurationType::get(ctx_.get());
    auto measTy = pulse::MeasurementType::get(ctx_.get());

    llvm::DenseMap<int64_t, mlir::Value> ssa;
    llvm::DenseMap<int64_t, mlir::Value> qubitSSA;
    // The open schedule blocks as (policy, callee), innermost last, and the
    // ids of the values bound inside them, so that a block's results can
    // take them over when it closes.
    // (policy, callee, max_duration_cycles or -1) of each open block.
    std::vector<std::tuple<std::string, std::string, int64_t>> blocks;
    llvm::DenseMap<mlir::Value, llvm::SmallVector<int64_t, 1>> blockVids;
    // Whether the program computes something at run time (a select, a
    // classical call, a value an if yields) or changes a frame in a branch;
    // `openIfs` counts the ifs being built.
    bool runTime = false;
    int64_t openIfs = 0;
    // The readouts in the kernel's order, whether an if or a select has
    // begun, and whether a pass can copy a readout: one in a loop (peeled
    // or unrolled) or after an if or a select begins (run in both arms of
    // an if on a selected value). Only then are the readouts numbered.
    llvm::SmallVector<mlir::Operation *> kernelReadouts;
    bool branched = false;
    bool copyable = false;

    auto bind = [&](int64_t vid, mlir::Value v) {
      ssa[vid] = v;
      if (!blocks.empty())
        blockVids[v].push_back(vid);
    };
    // The results of ifs that are no value: one whose arms leave versions
    // of different lines (or of different lines' tones) there, as two names
    // swapped in one arm do. By result id, the word of the if's record and
    // the result's place. Such a result is not made, so the if yields each
    // line once; what reads it is the error, at its if.
    llvm::DenseMap<int64_t, MixedResult> mixed;
    // For the results of an if in an arm of another: what entered it.
    llvm::DenseMap<mlir::Value, mlir::Value> entered;
    auto look = [&](int64_t vid) -> mlir::Value {
      auto it = ssa.find(vid);
      if (it == ssa.end()) [[unlikely]]
        unknownValue(mixed, vid);
      return it->second;
    };
    // Region bodies are built in detached blocks and moved into their scf op
    // when the region closes; values they use from outside stay in scope.
    struct Region {
      bool isLoop;
      int64_t lb = 0, ub = 0, step = 1;
      mlir::Value condition;
      mlir::Block *parent;
      std::unique_ptr<mlir::Block> body;
      std::unique_ptr<mlir::Block> elseBody;
      // The links of classical calls as the region began, and as its then
      // arm ended: a link a region sends on is carried out of it.
      std::map<int64_t, mlir::Value> linksBefore = {}, linksThen = {};
      // What the region lasts serially (its else arm in `elseCycles`), and
      // the word of its first record.
      int64_t cycles = 0, elseCycles = 0;
      size_t word = 0;
      // Where its record in `excess` begins, once a wait whose length floats
      // set is in it.
      size_t excessAt = 0;
    };
    std::vector<Region> regions;
    // A record that fails leaves regions open, whose blocks use values of
    // the blocks around them, and a vector frees its elements in no set
    // order: every open block drops its uses before any is freed.
    auto dropOpenRegions = llvm::make_scope_exit([&regions] {
      for (Region &region : regions)
        for (mlir::Block *block : {region.body.get(), region.elseBody.get()})
          if (block)
            for (mlir::Operation &op : *block)
              op.dropAllReferences();
    });
    // A schedule block is a region with neither loop bounds nor a condition.
    auto isBlock = [](const Region &region) {
      return !region.isLoop && !region.condition;
    };

    // A negative index -1 - k is slot k.
    auto getQubit = [&](int64_t q) -> mlir::Value {
      if (q < 0 && -1 - q < slot_count)
        return entryBlock->getArgument(firstSlot + (-1 - q));
      if (q < 0 || q >= qudit_count)
        throw std::invalid_argument("packed: qubit index out of range: " +
                                    std::to_string(q));
      auto it = qubitSSA.find(q);
      if (it != qubitSSA.end())
        return it->second;
      // Allocate outside any region, so every later use is dominated.
      mlir::OpBuilder::InsertionGuard guard(*builder_);
      if (!regions.empty())
        builder_->setInsertionPointToEnd(regions.front().parent);
      auto alloc = pulse::QuditAllocOp::create(*builder_, loc, qrefTy,
                                               builder_->getI64IntegerAttr(q));
      auto v = alloc.getResult();
      qubitSSA[q] = v;
      return v;
    };

    // Classical calls: the declared functions by index, and the current
    // version of each link, created at function entry when first used.
    std::vector<classical::CallableOp> callables;
    std::map<int64_t, mlir::Value> links;
    auto linkOf = [&](int64_t device) -> mlir::Value {
      auto it = links.find(device);
      if (it != links.end())
        return it->second;
      mlir::OpBuilder::InsertionGuard guard(*builder_);
      if (!regions.empty())
        builder_->setInsertionPointToEnd(regions.front().parent);
      auto get = classical::GetLinkOp::create(
          *builder_, loc, classical::LinkType::get(ctx_.get()),
          builder_->getI64IntegerAttr(device));
      for (Region &open : regions)
        open.linksBefore.try_emplace(device, get.getLink());
      links[device] = get.getLink();
      return get.getLink();
    };
    auto callableAt = [&](int64_t index) -> classical::CallableOp {
      if (index < 0 || static_cast<size_t>(index) >= callables.size())
        throw std::invalid_argument("packed: unknown classical function " +
                                    std::to_string(index));
      return callables[static_cast<size_t>(index)];
    };
    // The links one arm or body changed, as (device, before, after).
    auto changedLinks = [](const std::map<int64_t, mlir::Value> &before,
                           const std::map<int64_t, mlir::Value> &a,
                           const std::map<int64_t, mlir::Value> &b) {
      llvm::SmallVector<std::tuple<int64_t, mlir::Value, mlir::Value>> out;
      for (auto &[device, value] : before) {
        auto ia = a.find(device), ib = b.find(device);
        mlir::Value va = ia == a.end() ? value : ia->second;
        mlir::Value vb = ib == b.end() ? value : ib->second;
        if (va != value || vb != value)
          out.push_back({device, va, vb});
      }
      return out;
    };

    size_t cur = 0;
    // What the program lasts played op after op, in cycles, which no
    // schedule of it exceeds: a loop's body times its trip count, the longer
    // arm of an if. Past `maxCycles` (the controller's longest program, else
    // 2**60) it is an error at the record that crosses it.
    int64_t maxCycles = int64_t(1) << 60;
    bool controllerLimit = false;
    bool loops = false;
    int64_t serialCycles = 0;
    std::vector<int64_t> waitRepeats;
    // The loops and ifs around those waits, from the first: wait k is
    // [0, k], a loop [1, trips] ... [3], an if [2, then cycles, else
    // cycles] ... [4] ... [3] (its arms at their waits' lowest lengths).
    // Kept only when such a wait is in an if, whose longer arm can depend on
    // the lengths.
    std::vector<int64_t> excess;
    bool waitInIf = false;
    auto tooLong = [&](size_t word) {
      throw PackedValueError(
          controllerLimit
              ? "the program lasts more than " + std::to_string(maxCycles) +
                    " cycles, the longest program of the controller "
                    "(Controller.max_program_duration)"
              : std::string("the program lasts more than 2**60 cycles, more "
                            "than a schedule counts"),
          word);
    };
    auto lengthen = [&](int64_t cycles, size_t word) {
      int64_t &into = regions.empty()           ? serialCycles
                      : regions.back().elseBody ? regions.back().elseCycles
                                                : regions.back().cycles;
      std::optional<int64_t> sum = checkedAdd(into, cycles);
      if (!sum || *sum > maxCycles) [[unlikely]]
        tooLong(word);
      into = *sum;
    };
    auto trips = [](const Region &loop) -> uint64_t {
      return loop.ub > loop.lb
                 ? (static_cast<uint64_t>(loop.ub) -
                    static_cast<uint64_t>(loop.lb) + loop.step - 1) /
                       static_cast<uint64_t>(loop.step)
                 : 0;
    };
    while (cur < len) {
      int64_t hdr = d[cur];
      int opcode = static_cast<int>(hdr & 0xFF);
      int64_t plen = (hdr >> 8) & 0xFF;
      int pmask = static_cast<int>((hdr >> 16) & 0xFFFF);
      size_t headerWords = 1;
      if (plen == kExtendedPayload) {
        if (cur + 1 >= len || d[cur + 1] < 0)
          throw std::invalid_argument("packed: truncated record at word " +
                                      std::to_string(cur));
        plen = d[cur + 1];
        headerWords = 2;
      }
      if (cur + headerWords + static_cast<size_t>(plen) > len)
        throw std::invalid_argument("packed: truncated record at word " +
                                    std::to_string(cur));
      const int64_t *p = d + cur + headerWords;

      auto requirePayload = [&](int64_t expected) {
        if (plen != expected)
          throw std::invalid_argument(
              "packed: opcode " + std::to_string(opcode) + " expects " +
              std::to_string(expected) + " payload words, got " +
              std::to_string(plen));
      };

      // Helper: get SSA value for a waveform arg slot. If the param_mask
      // bit is set, the slot holds a vid (block arg ref); otherwise it's
      // a literal that needs an arith.constant.
      //
      // Each of these helpers may create a constant or reject a literal, and
      // C++ leaves the order in which a call's arguments are evaluated
      // unspecified: the Linux x86_64 build took them right to left, the
      // others left to right. A case that uses more than one of them calls
      // each in a statement of its own, in the order of the operation's
      // operands, so that every build creates the same constants in the same
      // order and names the same literal when two are rejected.
      auto getI64Arg = [&](int slot) -> mlir::Value {
        if (pmask & (1 << slot))
          return look(p[slot]);
        return mlir::arith::ConstantIntOp::create(*builder_, loc, p[slot], 64)
            .getResult();
      };
      auto getF64Arg = [&](int slot) -> mlir::Value {
        if (pmask & (1 << slot))
          return look(p[slot]);
        return mlir::arith::ConstantFloatOp::create(*builder_, loc, f64Ty,
                                                    llvm::APFloat(i2f(p[slot])))
            .getResult();
      };
      // The literals an op cannot take, named as the kernel wrote them.
      auto literal = [&](int slot) { return !(pmask & (1 << slot)); };
      auto finite = [&](int slot, const char *what) {
        if (literal(slot) && !std::isfinite(i2f(p[slot]))) [[unlikely]]
          rejectFloat(what, "must be finite", i2f(p[slot]), cur);
        return getF64Arg(slot);
      };
      auto positive = [&](int slot, const char *what) {
        if (literal(slot) && !(i2f(p[slot]) > 0.0)) [[unlikely]]
          rejectFloat(what, "must be positive", i2f(p[slot]), cur);
        return finite(slot, what);
      };

      switch (opcode) {

      case kOpForBegin: {
        requirePayload(3);
        Region region{true,
                      p[0],
                      p[1],
                      p[2],
                      {},
                      builder_->getBlock(),
                      std::make_unique<mlir::Block>(),
                      nullptr};
        if (region.step <= 0)
          throw std::invalid_argument("packed: loop step must be positive");
        region.word = cur;
        region.linksBefore = links;
        builder_->setInsertionPointToEnd(region.body.get());
        regions.push_back(std::move(region));
        loops = true;
        break;
      }
      case kOpForEnd: {
        if (regions.empty() || !regions.back().isLoop || plen < 1 || p[0] < 0 ||
            plen != 1 + 3 * p[0])
          throw std::invalid_argument("packed: malformed loop end");
        Region region = std::move(regions.back());
        regions.pop_back();
        {
          std::optional<uint64_t> total =
              checkedMul(trips(region), static_cast<uint64_t>(region.cycles));
          if (!total || *total > static_cast<uint64_t>(maxCycles)) [[unlikely]]
            tooLong(region.word);
          lengthen(static_cast<int64_t>(*total), region.word);
        }
        if (region.excessAt)
          excess.push_back(3);
        builder_->setInsertionPointToEnd(region.parent);
        const int64_t n = p[0];
        // A value carried under two names is one iteration argument.
        llvm::SmallVector<mlir::Value> inits, yields;
        llvm::SmallVector<size_t> argOf;
        llvm::DenseMap<int64_t, size_t> argOfInit;
        // A name an if in the body leaves as no one line (the if binds it
        // to different lines) is no value to carry. One the body itself
        // made is bound in every iteration before it is read, as it was in
        // the first, so it is left out, and is no value after the loop.
        // One from before the loop may be read by the next iteration, which
        // the records do not tell; so may one that was no one line when the
        // loop began.
        const size_t notCarried = std::numeric_limits<size_t>::max();
        for (int64_t i = 0; i < n; ++i) {
          int64_t init = p[1 + i];
          if (!mixed.empty()) [[unlikely]] {
            auto lost = mixed.find(p[1 + n + i]);
            if (lost != mixed.end()) {
              MixedResult from = lost->second;
              if (init >= 0)
                rejectArms(from, /*carried=*/true);
              mixed.insert_or_assign(p[1 + 2 * n + i], from);
              if (llvm::is_contained(lostLines_, p[1 + 2 * n + i]))
                rejectArms(from);
              argOf.push_back(notCarried);
              continue;
            }
            if (init >= 0) {
              auto began = mixed.find(init);
              if (began != mixed.end())
                rejectArms(began->second, /*carried=*/true);
            }
          }
          mlir::Value next = look(p[1 + n + i]);
          if (init >= 0) {
            auto [it, inserted] = argOfInit.try_emplace(init, inits.size());
            if (!inserted) {
              if (yields[it->second] != next)
                throw std::invalid_argument(
                    "packed: one loop value carried to two different values");
              argOf.push_back(it->second);
              continue;
            }
          }
          argOf.push_back(inits.size());
          inits.push_back(init >= 0 ? look(init)
                                    : mlir::ub::PoisonOp::create(
                                          *builder_, loc, next.getType()));
          yields.push_back(next);
        }
        // Links the body sent on are carried by the loop.
        llvm::SmallVector<std::pair<int64_t, size_t>> carriedLinks;
        for (auto [device, after, unused] :
             changedLinks(region.linksBefore, links, links)) {
          (void)unused;
          carriedLinks.push_back({device, inits.size()});
          inits.push_back(region.linksBefore.at(device));
          yields.push_back(after);
        }
        auto index = [&](int64_t v) -> mlir::Value {
          return mlir::arith::ConstantIndexOp::create(*builder_, loc, v);
        };
        // One statement for each bound: the order in which a call's
        // arguments are evaluated is unspecified, and each creates a constant.
        mlir::Value lowerBound = index(region.lb);
        mlir::Value upperBound = index(region.ub);
        mlir::Value step = index(region.step);
        auto loop = mlir::scf::ForOp::create(*builder_, loc, lowerBound,
                                             upperBound, step, inits);
        mlir::Block *body = loop.getBody();
        if (!body->empty())
          body->getTerminator()->erase();
        body->getOperations().splice(body->end(), region.body->getOperations());
        {
          mlir::OpBuilder::InsertionGuard guard(*builder_);
          builder_->setInsertionPointToEnd(body);
          mlir::scf::YieldOp::create(*builder_, loc, yields);
        }
        for (auto [init, arg] : llvm::zip(inits, loop.getRegionIterArgs()))
          init.replaceUsesWithIf(arg, [&](mlir::OpOperand &use) {
            return loop->isProperAncestor(use.getOwner());
          });
        for (int64_t i = 0; i < n; ++i)
          if (argOf[i] != notCarried)
            bind(p[1 + 2 * n + i], loop.getResult(argOf[i]));
        for (auto [device, index] : carriedLinks)
          links[device] = loop.getResult(index);
        break;
      }
      case kOpIfBegin: {
        requirePayload(1);
        Region region{false,
                      0,
                      0,
                      1,
                      look(p[0]),
                      builder_->getBlock(),
                      std::make_unique<mlir::Block>(),
                      nullptr};
        region.word = cur;
        region.linksBefore = links;
        builder_->setInsertionPointToEnd(region.body.get());
        regions.push_back(std::move(region));
        ++openIfs;
        branched = true;
        break;
      }
      case kOpElse: {
        requirePayload(0);
        if (regions.empty() || regions.back().isLoop ||
            isBlock(regions.back()) || regions.back().elseBody)
          throw std::invalid_argument("packed: else outside an if");
        regions.back().elseBody = std::make_unique<mlir::Block>();
        if (regions.back().excessAt)
          excess.push_back(4);
        regions.back().linksThen = links;
        links = regions.back().linksBefore;
        builder_->setInsertionPointToEnd(regions.back().elseBody.get());
        break;
      }
      case kOpIfEnd: {
        if (regions.empty() || regions.back().isLoop ||
            isBlock(regions.back()) || plen < 1 || p[0] < 0 ||
            plen != 1 + 3 * p[0])
          throw std::invalid_argument("packed: malformed if end");
        Region region = std::move(regions.back());
        regions.pop_back();
        lengthen(std::max(region.cycles, region.elseCycles), region.word);
        if (region.excessAt) {
          excess[region.excessAt + 1] = region.cycles;
          excess[region.excessAt + 2] = region.elseCycles;
          excess.push_back(3);
        }
        --openIfs;
        builder_->setInsertionPointToEnd(region.parent);
        const int64_t n = p[0];
        llvm::SmallVector<mlir::Value> thenValues, elseValues;
        llvm::SmallVector<mlir::Type> types;
        // Whether a tone's op is still in view at this if: made in a block
        // that is open around it. A read made there is the one a later
        // read of the same tone returns, in the arms of this if too.
        auto kept = [&](mlir::Operation *op) {
          mlir::Block *block = op->getBlock();
          return block == region.parent ||
                 llvm::any_of(regions, [&](const Region &around) {
                   return block == around.parent;
                 });
        };
        // The record's results that the if has, by place in the record,
        // and, in an arm of another if, what entered this one for each.
        // `made` is filled only once a result is left out: until then the
        // results are the record's, place for place.
        llvm::SmallVector<int64_t> made;
        bool dropped = false;
        auto drop = [&](int64_t place) {
          if (!dropped)
            for (int64_t before = 0; before < place; ++before)
              made.push_back(before);
          dropped = true;
        };
        llvm::SmallVector<std::pair<size_t, mlir::Value>, 4> began;
        // The arms' values were made before this record: whether any is no
        // value is known before its results are looked at.
        const bool someMixed = !mixed.empty();
        for (int64_t i = 0; i < n; ++i) {
          // A result either arm takes from an if that left different lines
          // there is none either.
          if (someMixed) [[unlikely]] {
            auto lost = mixed.find(p[1 + i]);
            if (lost == mixed.end())
              lost = mixed.find(p[1 + n + i]);
            if (lost != mixed.end()) {
              MixedResult from = lost->second;
              drop(i);
              mixed.insert_or_assign(p[1 + 2 * n + i], from);
              if (llvm::is_contained(lostLines_, p[1 + 2 * n + i]))
                rejectArms(from);
              continue;
            }
          }
          mlir::Value then = look(p[1 + i]), other = look(p[1 + n + i]);
          // A result is one line, or one tone, on either path: an arm that
          // leaves another line there (two names swapped in one arm) would
          // make every later op on it play one path's line on both. An arm
          // that operated once on what the other arm left alone began with
          // it: one line or tone, with no walk and no look at its type.
          mlir::Value entry;
          if (madeFrom(then, other)) [[likely]] {
            entry = other;
          } else if (madeFrom(other, then)) {
            entry = then;
          } else if (mlir::isa<pulse::DriveLineType, pulse::ReadoutLineType,
                               pulse::ToneType>(then.getType())) {
            mlir::Value a = enteringArm(then, region.body.get(), entered);
            mlir::Value b = enteringArm(other, region.elseBody.get(), entered);
            if (a != b && differentLines(then, region.body.get(), a, other,
                                         region.elseBody.get(), kept))
                [[unlikely]] {
              MixedResult from{cur, i,
                               mlir::isa<pulse::ToneType>(then.getType())};
              drop(i);
              mixed.insert_or_assign(p[1 + 2 * n + i], from);
              if (llvm::is_contained(lostLines_, p[1 + 2 * n + i]))
                rejectArms(from);
              continue;
            }
            if (a == b)
              entry = a;
          }
          if (openIfs > 0 && entry)
            began.push_back({thenValues.size(), entry});
          if (dropped) [[unlikely]]
            made.push_back(i);
          thenValues.push_back(then);
          elseValues.push_back(other);
          types.push_back(then.getType());
          runTime |= yieldsValue(types.back());
        }
        // A link either arm sent on leaves the branch as a result.
        const auto &thenLinks = region.elseBody ? region.linksThen : links;
        const auto &elseLinks = region.elseBody ? links : region.linksBefore;
        llvm::SmallVector<std::pair<int64_t, unsigned>> branchLinks;
        for (auto [device, a, b] :
             changedLinks(region.linksBefore, thenLinks, elseLinks)) {
          branchLinks.push_back({device, static_cast<unsigned>(types.size())});
          thenValues.push_back(a);
          elseValues.push_back(b);
          types.push_back(a.getType());
        }
        // A measurement is tested with pulse.is_set; an answer of a
        // classical call is already a boolean.
        mlir::Value condition = region.condition;
        if (mlir::isa<pulse::MeasurementType>(condition.getType()))
          condition = pulse::IsSetOp::create(*builder_, loc,
                                             builder_->getI1Type(), condition);
        auto branch = mlir::scf::IfOp::create(*builder_, loc, types, condition,
                                              /*withElseRegion=*/true);
        auto fill = [&](mlir::Block *arm, mlir::Block *source,
                        llvm::ArrayRef<mlir::Value> values) {
          if (!arm->empty())
            arm->getTerminator()->erase();
          if (source)
            arm->getOperations().splice(arm->end(), source->getOperations());
          mlir::OpBuilder::InsertionGuard guard(*builder_);
          builder_->setInsertionPointToEnd(arm);
          mlir::scf::YieldOp::create(*builder_, loc, values);
        };
        fill(branch.thenBlock(), region.body.get(), thenValues);
        fill(branch.elseBlock(), region.elseBody.get(), elseValues);
        if (!dropped) [[likely]] {
          for (int64_t i = 0; i < n; ++i)
            bind(p[1 + 2 * n + i], branch.getResult(i));
        } else {
          for (size_t k = 0; k < made.size(); ++k)
            bind(p[1 + 2 * n + made[k]], branch.getResult(k));
        }
        for (auto [k, value] : began)
          entered[branch.getResult(k)] = value;
        for (auto [device, index] : branchLinks)
          links[device] = branch.getResult(index);
        break;
      }
      case kOpBlockBegin: {
        // [policy (0 = asap, 1 = alap), callee name bytes, name words...,
        //  max_duration_cycles (optional)]
        if (plen < 2 || p[0] < 0 || p[0] > 1 || p[1] < 0)
          throw std::invalid_argument("packed: malformed schedule block");
        int64_t nameWords = (p[1] + 7) / 8;
        if (plen != 2 + nameWords && plen != 3 + nameWords)
          throw std::invalid_argument("packed: malformed schedule block");
        int64_t budget = plen == 3 + nameWords ? p[2 + nameWords] : -1;
        if (plen == 3 + nameWords && budget <= 0)
          throw std::invalid_argument("packed: malformed schedule block");
        Region region{false,
                      0,
                      0,
                      1,
                      {},
                      builder_->getBlock(),
                      std::make_unique<mlir::Block>(),
                      nullptr};
        blocks.emplace_back(p[0] == 0 ? "asap" : "alap",
                            std::string(reinterpret_cast<const char *>(p + 2),
                                        static_cast<size_t>(p[1])),
                            budget);
        region.linksBefore = links;
        builder_->setInsertionPointToEnd(region.body.get());
        regions.push_back(std::move(region));
        break;
      }
      case kOpBlockEnd: {
        requirePayload(0);
        if (regions.empty() || !isBlock(regions.back()))
          throw std::invalid_argument(
              "packed: schedule block end without a begin");
        Region region = std::move(regions.back());
        regions.pop_back();
        lengthen(region.cycles, cur);
        auto [policy, callee, budget] = std::move(blocks.back());
        blocks.pop_back();
        builder_->setInsertionPointToEnd(region.parent);
        auto block = buildScheduleBlock(*builder_, loc, std::move(region.body),
                                        policy, callee);
        if (budget > 0)
          block->setAttr("max_duration_cycles",
                         builder_->getI64IntegerAttr(budget));
        auto yielded = block.getBody().front().getTerminator()->getOperands();
        // The last version of a link the block sent on is a block result.
        for (auto &[device, value] : links)
          for (auto [inner, result] : llvm::zip(yielded, block.getResults()))
            if (inner == value) {
              value = result;
              break;
            }
        for (auto [inner, result] : llvm::zip(yielded, block.getResults())) {
          auto it = blockVids.find(inner);
          if (it == blockVids.end())
            continue;
          for (int64_t vid : it->second)
            bind(vid, result);
        }
        if (blocks.empty())
          blockVids.clear();
        break;
      }

      case kOpAllocDrive: {
        requirePayload(3);
        auto qref = getQubit(p[0]);
        auto gdl = pulse::GetDriveLineOp::create(*builder_, loc, driveTy,
                                                 toneTy, qref);
        if (p[0] >= 0)
          gdl->setAttr("frequency_hz",
                       builder_->getF64FloatAttr(freqVec[p[0]]));
        bind(p[1], gdl.getLine());
        bind(p[2], gdl.getTone());
        break;
      }

      case kOpAllocReadout: {
        requirePayload(3);
        auto qref = getQubit(p[0]);
        auto grl = pulse::GetReadoutLineOp::create(*builder_, loc, readoutTy,
                                                   toneTy, qref);
        if (p[0] >= 0) {
          grl->setAttr(
              "frequency_hz",
              builder_->getF64FloatAttr(static_cast<size_t>(p[0]) < readoutLen
                                            ? readoutData[p[0]]
                                            : freqVec[p[0]]));
        }
        bind(p[1], grl.getLine());
        bind(p[2], grl.getTone());
        break;
      }

      case kOpAllocTone: {
        requirePayload(3);
        double freq = i2f(p[1]);
        double phase = i2f(p[2]);
        auto fc = mlir::arith::ConstantFloatOp::create(
            *builder_, loc, builder_->getF64Type(), llvm::APFloat(freq));
        auto pc = mlir::arith::ConstantFloatOp::create(
            *builder_, loc, builder_->getF64Type(), llvm::APFloat(phase));
        auto t = pulse::ToneOp::create(*builder_, loc, toneTy, fc.getResult(),
                                       pc.getResult());
        bind(p[0], t.getResult());
        break;
      }

      case kOpWfGaussian: {
        requirePayload(4);
        mlir::Value duration = getI64Arg(1);
        mlir::Value amplitude = finite(2, "gaussian amplitude");
        mlir::Value sigma = positive(3, "gaussian sigma");
        auto op = pulse::GaussianPulseOp::create(*builder_, loc, wfTy, duration,
                                                 amplitude, sigma);
        bind(p[0], op.getResult());
        break;
      }

      case kOpWfSquare: {
        requirePayload(4);
        mlir::Value duration = getI64Arg(1);
        mlir::Value ampReal = finite(2, "square amplitude");
        mlir::Value ampImag = finite(3, "square amplitude's imaginary part");
        auto op = pulse::SquarePulseOp::create(*builder_, loc, wfTy, duration,
                                               ampReal, ampImag);
        bind(p[0], op.getResult());
        break;
      }

      case kOpWfDrag: {
        requirePayload(5);
        mlir::Value duration = getI64Arg(1);
        mlir::Value amplitude = finite(2, "drag amplitude");
        mlir::Value sigma = positive(3, "drag sigma");
        mlir::Value beta = finite(4, "drag beta");
        auto op = pulse::DRAGPulseOp::create(*builder_, loc, wfTy, duration,
                                             amplitude, sigma, beta);
        bind(p[0], op.getResult());
        break;
      }

      case kOpWfCosine: {
        requirePayload(3);
        mlir::Value duration = getI64Arg(1);
        mlir::Value amplitude = finite(2, "cosine amplitude");
        auto op = pulse::CosinePulseOp::create(*builder_, loc, wfTy, duration,
                                               amplitude);
        bind(p[0], op.getResult());
        break;
      }

      case kOpWfTanhRamp: {
        requirePayload(4);
        mlir::Value duration = getI64Arg(1);
        mlir::Value amplitude = finite(2, "tanh_ramp amplitude");
        mlir::Value sigma = positive(3, "tanh_ramp sigma");
        auto op = pulse::TanhRampOp::create(*builder_, loc, wfTy, duration,
                                            amplitude, sigma);
        bind(p[0], op.getResult());
        break;
      }

      case kOpWfGaussSquare: {
        requirePayload(5);
        mlir::Value duration = getI64Arg(1);
        mlir::Value amplitude = finite(2, "gaussian_square amplitude");
        mlir::Value sigma = positive(3, "gaussian_square sigma");
        mlir::Value risefall = getI64Arg(4);
        auto op = pulse::GaussianSquarePulseOp::create(
            *builder_, loc, wfTy, duration, amplitude, sigma, risefall);
        bind(p[0], op.getResult());
        break;
      }

      case kOpWfCustom: {
        if (plen < 4)
          throw std::invalid_argument(
              "packed: custom waveform record is too short");
        int64_t byteCount = p[2];
        if (byteCount <= 0)
          throw std::invalid_argument(
              "packed: custom waveform callback name is empty");
        size_t wordCount = (static_cast<size_t>(byteCount) + 7) / 8;
        if (3 + wordCount != static_cast<size_t>(plen))
          throw std::invalid_argument(
              "packed: custom waveform callback payload is malformed");
        std::string callbackName(reinterpret_cast<const char *>(p + 3),
                                 static_cast<size_t>(byteCount));
        auto callee = mlir::FlatSymbolRefAttr::get(ctx_.get(), callbackName);
        auto op =
            pulse::CustomOp::create(*builder_, loc, wfTy, callee, getI64Arg(1));
        bind(p[0], op.getResult());
        break;
      }

      case kOpWfCustomSamples: {
        // (result, count, samples...[, n, d]): n samples every d cycles.
        if (plen < 2 || p[1] < 0 || (p[1] + 2 != plen && p[1] + 4 != plen))
          throw std::invalid_argument(
              "packed: custom sample payload is malformed");
        llvm::SmallVector<double> samples;
        samples.reserve(static_cast<size_t>(p[1]));
        for (int64_t i = 0; i < p[1]; ++i) {
          samples.push_back(i2f(p[2 + i]));
          if (!std::isfinite(samples.back())) [[unlikely]]
            rejectSample(i, samples.back(), cur);
        }
        mlir::DenseI64ArrayAttr rate;
        if (p[1] + 4 == plen)
          rate = builder_->getDenseI64ArrayAttr({p[2 + p[1]], p[3 + p[1]]});
        auto op = pulse::CustomSamplesOp::create(
            *builder_, loc, wfTy,
            builder_->getF64ArrayAttr(llvm::ArrayRef<double>(samples)), rate);
        bind(p[0], op.getResult());
        break;
      }

      case kOpParam: {
        requirePayload(2);
        // Parameter reference: p[0] = vid, p[1] = param_index
        // The value comes from the func.func block argument
        int64_t paramIdx = p[1];
        if (paramIdx < 0 || paramIdx >= static_cast<int64_t>(firstSlot))
          throw std::invalid_argument("packed: parameter index out of range: " +
                                      std::to_string(paramIdx));
        bind(p[0], entryBlock->getArgument(paramIdx));
        break;
      }

      case kOpSelect: {
        // [result id, condition (a measurement or i1), then id, else id]
        requirePayload(4);
        mlir::Value condition = look(p[1]);
        if (mlir::isa<pulse::MeasurementType>(condition.getType()))
          condition = pulse::IsSetOp::create(*builder_, loc,
                                             builder_->getI1Type(), condition);
        else if (!condition.getType().isInteger(1))
          throw std::invalid_argument(
              "packed: a select's condition must be a measurement or i1");
        mlir::Value then = look(p[2]), other = look(p[3]);
        if (then.getType() != other.getType())
          throw std::invalid_argument(
              "packed: a select's values must have one type");
        bind(p[0], mlir::arith::SelectOp::create(*builder_, loc, condition,
                                                 then, other));
        runTime = true;
        branched = true;
        break;
      }

      case kOpNumConst: {
        requirePayload(3);
        if (p[1] == 0) {
          bind(p[0],
               mlir::arith::ConstantIntOp::create(*builder_, loc, p[2], 64));
        } else if (p[1] == 1) {
          bind(p[0], mlir::arith::ConstantFloatOp::create(
                         *builder_, loc, f64Ty, llvm::APFloat(i2f(p[2]))));
        } else {
          throw std::invalid_argument("packed: invalid numeric constant type");
        }
        break;
      }

      case kOpNumBinary: {
        requirePayload(5);
        auto lhs = look(p[3]);
        auto rhs = look(p[4]);
        mlir::Value result;
        if (p[1] == 0) {
          switch (p[2]) {
          case 0:
            result = mlir::arith::AddIOp::create(*builder_, loc, lhs, rhs);
            break;
          case 1:
            result = mlir::arith::SubIOp::create(*builder_, loc, lhs, rhs);
            break;
          case 2:
            result = mlir::arith::MulIOp::create(*builder_, loc, lhs, rhs);
            break;
          case 3:
            result = mlir::arith::DivSIOp::create(*builder_, loc, lhs, rhs);
            break;
          case 4:
            result =
                mlir::arith::FloorDivSIOp::create(*builder_, loc, lhs, rhs);
            break;
          case 5:
            result = mlir::arith::RemSIOp::create(*builder_, loc, lhs, rhs);
            break;
          default:
            throw std::invalid_argument("packed: invalid integer binary op");
          }
        } else if (p[1] == 1) {
          switch (p[2]) {
          case 0:
            result = mlir::arith::AddFOp::create(*builder_, loc, lhs, rhs);
            break;
          case 1:
            result = mlir::arith::SubFOp::create(*builder_, loc, lhs, rhs);
            break;
          case 2:
            result = mlir::arith::MulFOp::create(*builder_, loc, lhs, rhs);
            break;
          case 3:
            result = mlir::arith::DivFOp::create(*builder_, loc, lhs, rhs);
            break;
          default:
            throw std::invalid_argument("packed: invalid float binary op");
          }
        } else {
          throw std::invalid_argument("packed: invalid binary numeric type");
        }
        bind(p[0], result);
        break;
      }

      case kOpNumNeg: {
        requirePayload(3);
        auto operand = look(p[2]);
        if (p[1] == 0) {
          auto zero = mlir::arith::ConstantIntOp::create(*builder_, loc, 0, 64);
          bind(p[0],
               mlir::arith::SubIOp::create(*builder_, loc, zero, operand));
        } else if (p[1] == 1) {
          bind(p[0], mlir::arith::NegFOp::create(*builder_, loc, operand));
        } else {
          throw std::invalid_argument("packed: invalid negation type");
        }
        break;
      }

      case kOpNumCast: {
        requirePayload(4);
        auto operand = look(p[3]);
        if (p[1] == 0 && p[2] == 1)
          bind(p[0],
               mlir::arith::SIToFPOp::create(*builder_, loc, f64Ty, operand));
        else if (p[1] == 1 && p[2] == 0)
          bind(p[0],
               mlir::arith::FPToSIOp::create(*builder_, loc, i64Ty, operand));
        else
          throw std::invalid_argument("packed: invalid numeric cast");
        break;
      }

      case kOpWfAdd:
      case kOpWfSub:
      case kOpWfMul: {
        requirePayload(3);
        mlir::Value result;
        if (opcode == kOpWfAdd)
          result = pulse::PulseAddOp::create(*builder_, loc, wfTy, look(p[1]),
                                             look(p[2]));
        else if (opcode == kOpWfSub)
          result = pulse::PulseSubOp::create(*builder_, loc, wfTy, look(p[1]),
                                             look(p[2]));
        else
          result = pulse::PulseMulOp::create(*builder_, loc, wfTy, look(p[1]),
                                             look(p[2]));
        bind(p[0], result);
        break;
      }

      case kOpWfScale: {
        requirePayload(3);
        auto op = pulse::PulseScaleOp::create(*builder_, loc, wfTy, look(p[1]),
                                              look(p[2]));
        bind(p[0], op.getResult());
        break;
      }

      case kOpWfNeg: {
        requirePayload(2);
        auto op = pulse::PulseNegOp::create(*builder_, loc, wfTy, look(p[1]));
        bind(p[0], op.getResult());
        break;
      }

      case kOpDrive: {
        requirePayload(7);
        auto driveOp = pulse::DriveOp::create(
            *builder_, loc, driveTy, toneTy, look(p[0]), look(p[1]), look(p[2]),
            p[5] != -1 ? std::optional<int64_t>(p[5]) : std::nullopt,
            p[6] != -1 ? std::optional<int64_t>(p[6]) : std::nullopt,
            /*period_cycles=*/{});
        bind(p[3], driveOp.getUpdatedLine());
        bind(p[4], driveOp.getUpdatedTone());
        lengthen(waveformCycles(driveOp.getPulse()), cur);
        break;
      }

      case kOpReadout: {
        requirePayload(6);
        auto roOp = pulse::ReadoutOp::create(
            *builder_, loc, readoutTy, toneTy, measTy, look(p[0]), look(p[1]),
            look(p[2]), pulse::ReadoutMode::IQ, /*start_cycles=*/{},
            /*duration_cycles=*/{}, /*period_cycles=*/{});
        bind(p[3], roOp.getUpdatedLine());
        bind(p[4], roOp.getUpdatedTone());
        bind(p[5], roOp.getResult());
        lengthen(waveformCycles(roOp.getPulse()), cur);
        kernelReadouts.push_back(roOp);
        copyable =
            copyable || branched ||
            llvm::any_of(regions, [](const Region &r) { return r.isLoop; });
        break;
      }

      case kOpSync: {
        if (plen < 1 || p[0] < 0 || 1 + 3 * p[0] != plen)
          throw std::invalid_argument("packed: malformed sync payload");
        int64_t n = p[0];
        llvm::SmallVector<mlir::Value> inVals;
        llvm::SmallVector<mlir::Type> outTypes;
        for (int64_t j = 0; j < n; ++j) {
          inVals.push_back(look(p[1 + 3 * j]));
          int64_t vtype = p[3 + 3 * j];
          outTypes.push_back(vtype == 1 ? mlir::Type(readoutTy)
                                        : mlir::Type(driveTy));
        }
        if (n == 2) {
          if (inVals[0] == inVals[1]) [[unlikely]]
            rejectRepeatedLine(1, 2, cur);
        } else if (n > 2) {
          llvm::SmallDenseMap<mlir::Value, int64_t, 16> places;
          for (int64_t j = 0; j < n; ++j) {
            auto [seen, added] = places.try_emplace(inVals[j], j);
            if (!added) [[unlikely]]
              rejectRepeatedLine(seen->second + 1, j + 1, cur);
          }
        }
        auto syncOp = pulse::SyncOp::create(*builder_, loc, outTypes, inVals);
        for (int64_t j = 0; j < n; ++j)
          bind(p[2 + 3 * j], syncOp.getResult(j));
        break;
      }

      case kOpWait: {
        requirePayload(3);
        mlir::Value lineVal = look(p[0]);
        mlir::Value durCycles;
        if (pmask & (1 << 2))
          durCycles = look(p[2]);
        else
          durCycles =
              mlir::arith::ConstantIntOp::create(*builder_, loc, p[2], 64)
                  .getResult();
        auto durOp =
            pulse::DurationFromIntOp::create(*builder_, loc, durTy, durCycles);
        auto waitOp = pulse::WaitOp::create(
            *builder_, loc, lineVal.getType(), lineVal, durOp.getResult(),
            /*start_cycles=*/{}, /*duration_cycles=*/{}, /*period_cycles=*/{});
        bind(p[1], waitOp.getResult());
        lengthen((pmask & (1 << 2))
                     ? pulse::constantIndex(durCycles).value_or(0)
                     : p[2],
                 cur);
        break;
      }

      case kOpVariableWait: {
        // [line, result line, cycles id, lowest, highest (-1: none),
        //  1 if the length is an answer of a classical call]
        requirePayload(6);
        auto dur =
            pulse::DurationFromIntOp::create(*builder_, loc, durTy, look(p[2]));
        mlir::Value line = look(p[0]);
        auto waitOp = pulse::WaitOp::create(
            *builder_, loc, line.getType(), line, dur.getResult(),
            /*start_cycles=*/{}, /*duration_cycles=*/{}, /*period_cycles=*/{});
        waitOp->setAttr("pulse.variable_wait", builder_->getUnitAttr());
        waitOp->setAttr("pulse.min_cycles", builder_->getI64IntegerAttr(p[3]));
        if (p[4] >= 0)
          waitOp->setAttr("pulse.max_cycles",
                          builder_->getI64IntegerAttr(p[4]));
        if (p[5])
          waitOp->setAttr("classical.variable", builder_->getUnitAttr());
        bind(p[1], waitOp.getResult());
        // A length from floats is checked again with each call's values; one
        // that waits for an answer is clamped to its highest length.
        if (p[5]) {
          lengthen(p[4] >= 0 ? p[4] : p[3], cur);
        } else {
          uint64_t repeats = 1;
          for (const Region &region : regions)
            if (region.isLoop)
              repeats = checkedMul(repeats, trips(region))
                            .value_or(std::numeric_limits<int64_t>::max());
          if (excess.empty())
            excess.push_back(0);
          for (Region &region : regions) {
            if (region.excessAt || isBlock(region))
              continue;
            region.excessAt = excess.size();
            if (region.isLoop) {
              excess.insert(excess.end(),
                            {1, static_cast<int64_t>(std::min<uint64_t>(
                                    trips(region),
                                    std::numeric_limits<int64_t>::max()))});
            } else {
              excess.insert(excess.end(), {2, 0, 0});
              if (region.elseBody)
                excess.push_back(4);
            }
          }
          waitInIf |= openIfs > 0;
          excess.insert(excess.end(),
                        {0, static_cast<int64_t>(waitRepeats.size())});
          waitRepeats.push_back(static_cast<int64_t>(std::min<uint64_t>(
              repeats, std::numeric_limits<int64_t>::max())));
          lengthen(p[3], cur);
        }
        break;
      }

      case kOpShiftPhase: {
        requirePayload(3);
        mlir::Value phaseVal = finite(2, "shift_phase phase");
        auto spOp = pulse::ShiftPhaseOp::create(
            *builder_, loc, toneTy, look(p[0]), phaseVal, /*start_cycles=*/{},
            /*duration_cycles=*/{}, /*period_cycles=*/{});
        bind(p[1], spOp.getResult());
        runTime |= openIfs > 0;
        break;
      }

      case kOpSetPhase: {
        requirePayload(3);
        mlir::Value phaseVal = finite(2, "set_phase phase");
        auto spOp = pulse::SetPhaseOp::create(
            *builder_, loc, toneTy, look(p[0]), phaseVal, /*start_cycles=*/{},
            /*duration_cycles=*/{}, /*period_cycles=*/{});
        bind(p[1], spOp.getResult());
        runTime |= openIfs > 0;
        break;
      }

      case kOpShiftFreq: {
        requirePayload(3);
        mlir::Value fVal = finite(2, "shift_frequency frequency");
        auto sfOp = pulse::ShiftFrequencyOp::create(
            *builder_, loc, toneTy, look(p[0]), fVal, /*start_cycles=*/{},
            /*duration_cycles=*/{}, /*period_cycles=*/{});
        bind(p[1], sfOp.getResult());
        runTime |= openIfs > 0;
        break;
      }

      case kOpSetFreq: {
        requirePayload(3);
        mlir::Value fVal = finite(2, "set_frequency frequency");
        auto sfOp = pulse::SetFrequencyOp::create(
            *builder_, loc, toneTy, look(p[0]), fVal, /*start_cycles=*/{},
            /*duration_cycles=*/{}, /*period_cycles=*/{});
        bind(p[1], sfOp.getResult());
        runTime |= openIfs > 0;
        break;
      }

      case kOpClassicalDeclare: {
        // [index, id, link, flags, min, nominal, max, timeout, deadline,
        //  name length, args length, results length, text words...,
        //  reaction (with flag 64)]
        if (plen < 12)
          throw std::invalid_argument("packed: malformed classical function");
        int64_t lengths = p[9] + p[10] + p[11];
        int64_t textWords = (lengths + 7) / 8;
        bool reaction = p[3] & 64;
        if (p[0] != static_cast<int64_t>(callables.size()) || p[9] <= 0 ||
            p[10] < 0 || p[11] < 0 || plen != 12 + textWords + reaction)
          throw std::invalid_argument("packed: malformed classical function");
        std::string text(reinterpret_cast<const char *>(p + 12),
                         static_cast<size_t>(lengths));
        std::string symbol = text.substr(0, static_cast<size_t>(p[9]));
        auto split = [&](llvm::StringRef list) {
          llvm::SmallVector<llvm::StringRef> items;
          if (!list.empty())
            list.split(items, ',');
          return builder_->getStrArrayAttr(items);
        };
        llvm::StringRef args = llvm::StringRef(text).substr(
            static_cast<size_t>(p[9]), static_cast<size_t>(p[10]));
        llvm::StringRef results = llvm::StringRef(text).substr(
            static_cast<size_t>(p[9] + p[10]), static_cast<size_t>(p[11]));
        int64_t flags = p[3];
        static const char *modes[] = {"bounded", "timeout", "blocking"};
        if ((flags & 3) > 2)
          throw std::invalid_argument("packed: unknown latency mode");
        mlir::OpBuilder::InsertionGuard guard(*builder_);
        builder_->setInsertionPoint(funcOp);
        auto callable = classical::CallableOp::create(
            *builder_, loc, builder_->getStringAttr(symbol),
            builder_->getI64IntegerAttr(p[1]),
            builder_->getI64IntegerAttr(p[2]), split(args), split(results),
            builder_->getF64FloatAttr(i2f(p[4])),
            builder_->getF64FloatAttr(i2f(p[5])),
            builder_->getF64FloatAttr(i2f(p[6])),
            builder_->getStringAttr(modes[flags & 3]),
            builder_->getStringAttr(flags & 4 ? "lazy" : "eager"),
            flags & 16 ? builder_->getF64FloatAttr(i2f(p[7]))
                       : mlir::FloatAttr(),
            flags & 32 ? builder_->getF64FloatAttr(i2f(p[8]))
                       : mlir::FloatAttr(),
            flags & 8 ? builder_->getUnitAttr() : mlir::UnitAttr(),
            reaction ? builder_->getF64FloatAttr(i2f(p[12 + textWords]))
                     : mlir::FloatAttr());
        callables.push_back(callable);
        break;
      }
      case kOpClassicalSubmit: {
        // [function index, future id, n, argument ids...]
        if (plen < 3 || p[2] < 0 || plen != 3 + p[2])
          throw std::invalid_argument("packed: malformed classical submit");
        auto callable = callableAt(p[0]);
        llvm::SmallVector<mlir::Value> args;
        for (int64_t i = 0; i < p[2]; ++i)
          args.push_back(look(p[3 + i]));
        int64_t device = callable.getDevice();
        auto submit = classical::SubmitOp::create(
            *builder_, loc, classical::LinkType::get(ctx_.get()),
            classical::FutureType::get(ctx_.get()), linkOf(device),
            mlir::FlatSymbolRefAttr::get(callable.getSymNameAttr()), args,
            /*start_cycles=*/{}, /*duration_cycles=*/{}, /*period_cycles=*/{});
        links[device] = submit.getUpdatedLink();
        bind(p[1], submit.getFuture());
        runTime = true;
        break;
      }
      case kOpClassicalWait: {
        // [function index, future id, has timeout, timeout, n, result ids...]
        if (plen < 5 || p[4] < 0 || plen != 5 + p[4])
          throw std::invalid_argument("packed: malformed classical wait");
        auto callable = callableAt(p[0]);
        llvm::SmallVector<mlir::Type> types =
            classical::answerTypes(callable, p[2] != 0);
        if (static_cast<int64_t>(types.size()) != p[4])
          throw std::invalid_argument("packed: classical wait expects " +
                                      std::to_string(types.size()) +
                                      " results");
        auto wait = classical::WaitOp::create(
            *builder_, loc, types, look(p[1]),
            mlir::FlatSymbolRefAttr::get(callable.getSymNameAttr()),
            p[2] ? builder_->getF64FloatAttr(i2f(p[3])) : mlir::FloatAttr(),
            /*start_cycles=*/{}, /*duration_cycles=*/{}, /*period_cycles=*/{});
        for (int64_t i = 0; i < p[4]; ++i)
          bind(p[5 + i], wait.getValues()[i]);
        break;
      }
      case kOpClassicalCompare: {
        // [result id, predicate (eq ne lt le gt ge, and or of two i1),
        //  type (0 i64, 1 f64), lhs, rhs]
        requirePayload(5);
        if (p[1] < 0 || p[1] > 7 || p[2] < 0 || p[2] > 1 ||
            (p[1] > 5 && p[2] != 0))
          throw std::invalid_argument("packed: malformed classical compare");
        mlir::Value lhs = look(p[3]), rhs = look(p[4]);
        mlir::Value result;
        if (p[1] > 5) {
          if (!lhs.getType().isInteger(1) || !rhs.getType().isInteger(1))
            throw std::invalid_argument(
                "packed: and/or of values that are not i1");
          if (p[1] == 6)
            result = mlir::arith::AndIOp::create(*builder_, loc, lhs, rhs);
          else
            result = mlir::arith::OrIOp::create(*builder_, loc, lhs, rhs);
        } else if (p[2] == 1) {
          static const mlir::arith::CmpFPredicate fp[] = {
              mlir::arith::CmpFPredicate::OEQ, mlir::arith::CmpFPredicate::UNE,
              mlir::arith::CmpFPredicate::OLT, mlir::arith::CmpFPredicate::OLE,
              mlir::arith::CmpFPredicate::OGT, mlir::arith::CmpFPredicate::OGE};
          result =
              mlir::arith::CmpFOp::create(*builder_, loc, fp[p[1]], lhs, rhs);
        } else {
          static const mlir::arith::CmpIPredicate ip[] = {
              mlir::arith::CmpIPredicate::eq,  mlir::arith::CmpIPredicate::ne,
              mlir::arith::CmpIPredicate::slt, mlir::arith::CmpIPredicate::sle,
              mlir::arith::CmpIPredicate::sgt, mlir::arith::CmpIPredicate::sge};
          result =
              mlir::arith::CmpIOp::create(*builder_, loc, ip[p[1]], lhs, rhs);
        }
        bind(p[0], result);
        break;
      }

      case kOpMaxCycles: {
        // [the controller's longest program, in cycles]
        requirePayload(1);
        if (p[0] < 1)
          throw std::invalid_argument(
              "packed: longest program must be positive");
        controllerLimit = p[0] < maxCycles;
        maxCycles = std::min(maxCycles, p[0]);
        break;
      }

      case kOpSite: {
        // [site number, kind: 0 drive, 1 readout, 2 wait, 3 sync, 4 frame
        // change] of the op just built, written by idle replacement's trace
        // only: the op's location is pulse.site:<number>, and an op written
        // into an idle window (a negative number) is marked pulse.written.
        requirePayload(2);
        mlir::Block *block = builder_->getInsertionBlock();
        auto at = builder_->getInsertionPoint();
        if (!block || at == block->begin())
          throw std::invalid_argument("packed: a site record follows no op");
        mlir::Operation *op = &*std::prev(at);
        int64_t kind =
            mlir::isa<pulse::DriveOp>(op)     ? 0
            : mlir::isa<pulse::ReadoutOp>(op) ? 1
            : mlir::isa<pulse::WaitOp>(op)    ? 2
            : mlir::isa<pulse::SyncOp>(op)    ? 3
            : mlir::isa<pulse::ShiftPhaseOp, pulse::SetPhaseOp,
                        pulse::ShiftFrequencyOp, pulse::SetFrequencyOp>(op)
                ? 4
                : -1;
        if (kind != p[1])
          throw std::invalid_argument(
              "packed: a site record names another kind of op");
        int64_t site = p[0];
        if (site < INT32_MIN || site > INT32_MAX)
          throw std::invalid_argument("packed: a site is out of range");
        op->setLoc(mlir::FileLineColLoc::get(
            builder_->getContext(), "pulse.site",
            static_cast<unsigned>(site < 0 ? -site - 1 : site),
            site < 0 ? 1 : 0));
        if (site < 0)
          op->setAttr("pulse.written",
                      mlir::UnitAttr::get(builder_->getContext()));
        break;
      }

      default:
        throw std::invalid_argument("packed: unknown opcode " +
                                    std::to_string(opcode));
      }

      cur += headerWords + plen;
    }
    if (!regions.empty())
      throw std::invalid_argument("packed: region left open");

    mlir::func::ReturnOp::create(*builder_, loc);
    // The kernel's numbers of its readouts, before any pass can copy one:
    // a copy (in both arms of an if, out of a peeled or unrolled loop)
    // reports as the readout the kernel made. Where no pass can copy one, a
    // readout's place among them is its number (pulse::kernelReadoutKey).
    if (copyable) {
      auto name = builder_->getStringAttr(pulse::kReadoutNumberAttr);
      for (auto [number, op] : llvm::enumerate(kernelReadouts))
        op->setAttr(name, builder_->getI64IntegerAttr(number));
    }
    PulseModule result(ctx_, std::move(owningModule));
    result.runTime_ = runTime;
    result.loops_ = loops;
    if (!waitRepeats.empty())
      result.length_ = {serialCycles, maxCycles, controllerLimit ? 1 : 0,
                        std::move(waitRepeats),
                        waitInIf ? std::move(excess) : std::vector<int64_t>{}};
    return result;
  }

  /// A module PulseModule::serialize() wrote, read into this builder's
  /// context; counted against its budget as bytes / 8 packed words.
  PulseModule load(const std::string &module, const std::string &source,
                   bool runTime, const std::vector<std::string> &rewrites) {
    std::lock_guard<std::mutex> guard(lock_);
    SharedContextPool::charge(
        ctx_.get(), static_cast<int64_t>((module.size() + source.size()) / 8));
    PulseModule result(
        ctx_, parseModule(module, ctx_.get(), "cannot read a pickled module"));
    if (!source.empty())
      result.source_ =
          parseModule(source, ctx_.get(), "cannot read a pickled module");
    result.runTime_ = runTime;
    for (const std::string &name : rewrites) {
      const pulse::RewriteStage *stage = pulse::findRewriteStage(name);
      if (!stage)
        throw std::runtime_error("a pickled module names an unknown rewrite: " +
                                 name);
      result.rewrites_.push_back(stage);
    }
    return result;
  }

  /// What load() reads from another compiler's module and the module it
  /// writes from it (ModuleLoader.h), on this builder's context. The
  /// arguments go on as Python gives them: what reads them is built apart
  /// from the code that compiles a kernel.
  static nb::object loaded(PyObject *made) {
    if (!made)
      throw nb::python_error();
    return nb::steal(made);
  }
  nb::object load_callables(nb::handle module) {
    return loaded(cudaq_pulse::loader::callables(ctx_.get(), module.ptr()));
  }
  nb::object load_link_names(nb::handle module) {
    return loaded(cudaq_pulse::loader::linkNames(ctx_.get(), module.ptr()));
  }
  nb::object load_submissions(nb::handle module) {
    return loaded(cudaq_pulse::loader::submissions(ctx_.get(), module.ptr()));
  }
  PulseModule load_function(nb::handle module, nb::handle entry,
                            double clock_ghz, int64_t qudits,
                            nb::handle latencies, nb::handle keys,
                            nb::handle devices) {
    Py_ssize_t size =
        PyBytes_Check(module.ptr()) ? PyBytes_Size(module.ptr()) : 0;
    SharedContextPool::charge(ctx_.get(), static_cast<int64_t>(size / 8));
    return PulseModule(ctx_,
                       cudaq_pulse::loader::function(
                           ctx_.get(), module.ptr(), entry.ptr(), clock_ghz,
                           qudits, latencies.ptr(), keys.ptr(), devices.ptr()));
  }

private:
  std::shared_ptr<mlir::MLIRContext> ctx_;
  std::unique_ptr<mlir::OpBuilder> builder_;
  std::mutex lock_;
  // See expectLost.
  std::vector<int64_t> lostLines_;
};

// -----------------------------------------------------------------------
// JITModule: an LLVM-dialect module compiled in-process by ORC
// -----------------------------------------------------------------------
// Shared libraries a JIT module calls into, opened RTLD_LOCAL. MLIR's
// sharedLibPaths opens them through llvm::sys::DynamicLibrary, which uses
// RTLD_GLOBAL and so would add the GPU runtime and its CUDA dependencies to
// the process's global symbol scope.
class RuntimeLibraries {
public:
  explicit RuntimeLibraries(const std::vector<std::string> &paths) {
    for (const auto &path : paths) {
      void *handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
      if (!handle) {
        const char *reason = dlerror();
        close();
        throw std::runtime_error("JIT could not load " + path + ": " +
                                 (reason ? reason : "unknown error"));
      }
      handles_.push_back(handle);
    }
  }
  ~RuntimeLibraries() { close(); }
  RuntimeLibraries(const RuntimeLibraries &) = delete;
  RuntimeLibraries &operator=(const RuntimeLibraries &) = delete;

  void *lookup(const std::string &name) const {
    for (void *handle : handles_)
      if (void *symbol = dlsym(handle, name.c_str()))
        return symbol;
    return nullptr;
  }

private:
  void close() {
    for (void *handle : handles_)
      dlclose(handle);
    handles_.clear();
  }

  std::vector<void *> handles_;
};

class JITModule {
public:
  JITModule(std::unique_ptr<RuntimeLibraries> libraries,
            std::unique_ptr<mlir::ExecutionEngine> engine,
            std::vector<std::string> unresolved)
      : libraries_(std::move(libraries)), engine_(std::move(engine)),
        unresolved_(std::move(unresolved)) {}

  uintptr_t lookup(const std::string &name) {
    auto symbol = engine_->lookup(name);
    if (!symbol) {
      std::string message;
      llvm::raw_string_ostream os(message);
      os << symbol.takeError();
      // The missing symbols themselves go to the session's error reporter;
      // name the external functions no given library defines.
      if (!unresolved_.empty()) {
        os << "; external functions in none of the given libraries:";
        for (const std::string &external : unresolved_)
          os << " " << external;
      }
      throw std::runtime_error("JIT symbol lookup failed for '" + name +
                               "': " + message);
    }
    return reinterpret_cast<uintptr_t>(*symbol);
  }

private:
  // Declared first so the engine, whose code points into them, goes first.
  std::unique_ptr<RuntimeLibraries> libraries_;
  std::unique_ptr<mlir::ExecutionEngine> engine_;
  std::vector<std::string> unresolved_;
};

void addPulseLoweringPasses(mlir::PassManager &pm) {
  pm.addPass(pulse::createPulseToQOpPass());
  pm.addPass(qop::createQOpToCuDensityMatPass());
  pm.addPass(cudm::createCuDensityMatToLLVMPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createArithToLLVMConversionPass());
  pm.addPass(mlir::createConvertFuncToLLVMPass());
  pm.addPass(mlir::createReconcileUnrealizedCastsPass());
}

JITModule createJITModule(mlir::ModuleOp module,
                          const std::vector<std::string> &shared_libs) {
  static const bool nativeTargetReady = [] {
    return !llvm::InitializeNativeTarget() &&
           !llvm::InitializeNativeTargetAsmPrinter();
  }();
  if (!nativeTargetReady)
    throw std::runtime_error("LLVM native target initialization failed");

  auto libraries = std::make_unique<RuntimeLibraries>(shared_libs);
  std::vector<std::pair<std::string, void *>> resolved;
  std::vector<std::string> unresolved;
  module.walk([&](mlir::LLVM::LLVMFuncOp function) {
    if (!function.isExternal())
      return;
    std::string name = function.getName().str();
    if (void *address = libraries->lookup(name))
      resolved.emplace_back(std::move(name), address);
    else
      unresolved.push_back(std::move(name));
  });

  // ExecutionEngineOptions::transformer is a function_ref: the
  // std::function it refers to must outlive ExecutionEngine::create.
  auto transformer = mlir::makeOptimizingTransformer(
      /*optLevel=*/2, /*sizeLevel=*/0, /*targetMachine=*/nullptr);
  mlir::ExecutionEngineOptions options;
  options.transformer = transformer;
  auto engine = mlir::ExecutionEngine::create(module, options);
  if (!engine) {
    std::string message;
    llvm::raw_string_ostream os(message);
    os << engine.takeError();
    throw std::runtime_error("JIT compilation failed: " + message);
  }
  (*engine)->registerSymbols([&](llvm::orc::MangleAndInterner interner) {
    llvm::orc::SymbolMap symbols;
    for (const auto &[name, address] : resolved)
      symbols[interner(name)] = {llvm::orc::ExecutorAddr::fromPtr(address),
                                 llvm::JITSymbolFlags::Exported};
    return symbols;
  });
  return JITModule(std::move(libraries), std::move(*engine),
                   std::move(unresolved));
}

// -----------------------------------------------------------------------
// MLIRPipeline: the text-in, text-out lowerings and the JIT entry points of
// the Python runtime
// -----------------------------------------------------------------------
class MLIRPipeline {
public:
  MLIRPipeline() : registry_(makeRegistry()) {}

  JITModule jit_pulse(const std::string &pulse_mlir,
                      const std::vector<std::string> &shared_libs) {
    mlir::MLIRContext ctx;
    ctx.appendDialectRegistry(registry_);
    ctx.loadAllAvailableDialects();
    mlir::registerBuiltinDialectTranslation(ctx);
    mlir::registerLLVMDialectTranslation(ctx);
    auto module =
        parseModule(pulse_mlir, &ctx, "Failed to parse pulse MLIR text");
    mlir::PassManager pm(&ctx);
    addPulseLoweringPasses(pm);
    runPipeline(pm, *module, "MLIR pass pipeline failed");
    return createJITModule(*module, shared_libs);
  }

  JITModule jit_llvm_dialect(const std::string &mlir_text,
                             const std::vector<std::string> &shared_libs) {
    mlir::MLIRContext ctx;
    ctx.appendDialectRegistry(registry_);
    ctx.loadAllAvailableDialects();
    mlir::registerBuiltinDialectTranslation(ctx);
    mlir::registerLLVMDialectTranslation(ctx);
    auto module =
        parseModule(mlir_text, &ctx, "Failed to parse LLVM-dialect MLIR text");
    return createJITModule(*module, shared_libs);
  }

  std::string run_pulse_to_qop(const std::string &pulse_mlir) {
    mlir::MLIRContext ctx;
    ctx.appendDialectRegistry(registry_);
    ctx.loadAllAvailableDialects();
    auto module =
        parseModule(pulse_mlir, &ctx, "Failed to parse pulse MLIR text");
    mlir::PassManager pm(&ctx);
    pm.addPass(pulse::createPulseToQOpPass());
    runPipeline(pm, *module, "pulse-to-qop pass failed");
    std::string result;
    llvm::raw_string_ostream os(result);
    module->print(os);
    return result;
  }

  std::string run_qop_to_cudm(const std::string &qop_mlir) {
    mlir::MLIRContext ctx;
    ctx.appendDialectRegistry(registry_);
    ctx.loadAllAvailableDialects();
    auto module = parseModule(qop_mlir, &ctx, "Failed to parse qop MLIR text");
    mlir::PassManager pm(&ctx);
    pm.addPass(qop::createQOpToCuDensityMatPass());
    runPipeline(pm, *module, "qop-to-cudm pass failed");
    std::string result;
    llvm::raw_string_ostream os(result);
    module->print(os);
    return result;
  }

  PulseModule parse_to_module(const std::string &mlir_text) {
    auto ctx = makeContext(registry_);
    ctx->loadAllAvailableDialects();
    auto module =
        parseModule(mlir_text, ctx.get(), "Failed to parse MLIR text");
    return PulseModule(ctx, std::move(module));
  }

private:
  mlir::DialectRegistry registry_;
};

} // namespace

namespace {

/// Make `cls` immutable, as built-in types are: setting or deleting an
/// attribute of the class raises TypeError. Its bases must be immutable
/// already; its subclasses are not made immutable.
void freezeType(nb::handle cls) {
  if (!PyType_Check(cls.ptr()))
    throw nb::type_error("freeze_type takes a class");
  auto *type = reinterpret_cast<PyTypeObject *>(cls.ptr());
#if !defined(Py_LIMITED_API) && PY_VERSION_HEX >= 0x030E0000
  if (PyType_Freeze(type) < 0)
    throw nb::python_error();
#else
#if defined(Py_LIMITED_API)
  // Built against an older stable ABI: PyType_Freeze is the running
  // Python's own where it has one (3.14).
  using Freeze = int (*)(PyTypeObject *);
  static const Freeze freeze =
      Py_Version >= 0x030E0000
          ? reinterpret_cast<Freeze>(dlsym(RTLD_DEFAULT, "PyType_Freeze"))
          : nullptr;
  if (freeze) {
    if (freeze(type) < 0)
      throw nb::python_error();
    return;
  }
#endif
  // What PyType_Freeze does, before 3.14.
  nb::tuple mro = nb::borrow<nb::tuple>(cls.attr("__mro__"));
  for (size_t i = 1; i < mro.size(); ++i)
    if (!(PyType_GetFlags(reinterpret_cast<PyTypeObject *>(mro[i].ptr())) &
          Py_TPFLAGS_IMMUTABLETYPE))
      throw nb::type_error("freeze_type: a base of the class is mutable");
#if defined(Py_LIMITED_API)
  // The stable ABI hides PyTypeObject. tp_flags follows the object header
  // and 18 pointer-sized slots in every CPython 3.12 and 3.13 (no stable
  // ABI build runs free-threaded); checked against PyType_GetFlags.
  auto *flags = reinterpret_cast<unsigned long *>(
      reinterpret_cast<char *>(type) + 21 * sizeof(void *));
  if (*flags != PyType_GetFlags(type))
    throw std::runtime_error("freeze_type: unknown PyTypeObject layout");
  *flags |= Py_TPFLAGS_IMMUTABLETYPE;
#else
  type->tp_flags |= Py_TPFLAGS_IMMUTABLETYPE;
#endif
  // No PyType_Modified: what the interpreter has cached of the class holds
  // whether or not it is mutable, and keeping its version tag keeps the
  // method cache as it was. Invalidating it would retag the class from the
  // built-in types' counter.
#endif
}

// The span CUDA-Q passes a std::vector<bool> to a device function as.
struct DecoderBits {
  std::uint8_t *buffer;
  std::uint64_t length;
};

template <typename Call>
void guardedDecoderCall(Call &&call) {
  std::string message;
  try {
    call();
    return;
  } catch (const std::exception &error) {
    message = error.what();
  } catch (...) {
    message = "the decoder raised an unknown C++ exception";
  }
  throw std::runtime_error(message);
}

} // namespace

NB_MODULE(_cudaq_pulse_native, m) {
  m.doc() = "cudaq-pulse native MLIR pipeline bindings";
  // Fork takes the locks in the reverse order of these: the pool's is held
  // while a context, and its ContextStore, is made or destroyed.
  ContextStore::handleForks();
  SharedContextPool::handleForks();

  // Kept for the life of the process, like the module that holds it.
  static const std::string packedValueErrorName =
      nb::cast<std::string>(m.attr("__name__")) + ".PackedValueError";
  static PyObject *packedValueError = PyErr_NewExceptionWithDoc(
      packedValueErrorName.c_str(),
      "A literal of a packed program that its op cannot take; args are the "
      "message and the word offset of the record that holds it.",
      PyExc_ValueError, nullptr);
  m.attr("PackedValueError") = nb::handle(packedValueError);
  nb::register_exception_translator(
      [](const std::exception_ptr &error, void *) {
        try {
          std::rethrow_exception(error);
        } catch (const PackedValueError &e) {
          nb::object args = nb::make_tuple(e.what(), e.word);
          PyErr_SetObject(packedValueError, args.ptr());
        } catch (const NotSupportedYet &e) {
          PyErr_SetString(PyExc_NotImplementedError, e.what());
        }
      });

  nb::class_<PulseModule>(m, "PulseModule")
      .def("print", &PulseModule::print,
           nb::call_guard<nb::gil_scoped_release>(),
           "Print MLIR text representation")
      .def("run_passes", &PulseModule::run_passes,
           nb::call_guard<nb::gil_scoped_release>(),
           "Run named MLIR passes on this module")
      .def("lower_selects", &PulseModule::lower_selects,
           nb::call_guard<nb::gil_scoped_release>(), nb::arg("runtime"),
           "Keep the selects of the run-time parameter kinds given; run "
           "every other op that reads one in both arms of an if")
      .def("resolve_selects", &PulseModule::resolve_selects,
           nb::call_guard<nb::gil_scoped_release>(),
           "This module with every op that reads a run-time select in both "
           "arms of an if on its condition, for simulation")
      .def("run_full_lowering", &PulseModule::run_full_lowering,
           nb::call_guard<nb::gil_scoped_release>(),
           "Run full lowering: pulse -> qop -> cudm -> llvm")
      .def("schedule", &PulseModule::schedule,
           nb::call_guard<nb::gil_scoped_release>(),
           "Schedule pulse operations", nb::arg("policy"),
           nb::arg("max_drives") = 4, nb::arg("max_readouts") = 2,
           nb::arg("readout_latency") = 0, nb::arg("switch_penalty") = 0)
      .def(
          "events", &PulseModule::events,
          nb::call_guard<nb::gil_scoped_release>(),
          "(kind, line, qubit, start_cycles, duration_cycles) of each timed op")
      .def("event_tree", &PulseModule::event_tree, nb::arg("limit"),
           "events(), or past `limit` events the ops once each with the "
           "loops they repeat in")
      .def("strip_sites", &PulseModule::strip_sites,
           "Remove the site numbers the packed site records gave")
      .def("site_tree", &PulseModule::site_tree,
           "The lines and the scheduled ops with their numbers, times and "
           "the loops and branches they lie in")
      .def("controller_dependencies", &PulseModule::controller_dependencies,
           "For each event of controller_program(), not rolled: the "
           "arguments its tone frequency, frame phase and waveform depend on")
      .def("place_carriers", &PulseModule::place_carriers, nb::arg("carriers"),
           nb::arg("frequencies"),
           nb::arg("readout_frequencies").none() = nb::none(),
           "This module with its runtime qudits bound to carriers and its "
           "float arguments kept")
      .def("controller_native_tones", &PulseModule::controller_native_tones,
           nb::arg("summarize") = false, nb::arg("unsynced") = false,
           "For each event of controller_program(), not rolled: the qref "
           "argument whose native frequency its tone plays, or -1; with "
           "summarize, of controller_events()'s summarizing walk, with its "
           "repeats and line periods")
      .def("controller_rolled_dependencies",
           &PulseModule::controller_rolled_dependencies,
           "For each event of controller_program() rolled: its kind, its "
           "trips for a loop (else -1), and what it depends on")
      .def("controller_events", &PulseModule::controller_events,
           nb::arg("parameters"), nb::arg("amplitudes"),
           nb::arg("slots").none() = nb::none(),
           nb::arg("repeats").none() = nb::none(),
           nb::arg("periods").none() = nb::none(),
           nb::arg("segments").none() = nb::none(),
           "The lines and events of the export for a controller, as "
           "controller.export() gives them; a list `slots` gets each line's "
           "runtime-qudit slot")
      .def("controller_events_on", &PulseModule::controller_events_on,
           nb::arg("parameters"), nb::arg("amplitudes"), nb::arg("num"),
           nb::arg("den"), nb::arg("grid"),
           nb::arg("slots").none() = nb::none(),
           nb::arg("repeats").none() = nb::none(),
           nb::arg("periods").none() = nb::none(),
           "controller_events() in a controller's own cycles, num of them "
           "every den program cycles, with the line events off its grid of "
           "grid program cycles")
      .def("program_length", &PulseModule::program_length,
           "For waits whose length floats set: (serial cycles at their "
           "lowest lengths, the longest program, whether that is the "
           "controller's, the runs of each such wait)")
      .def("controller_program", &PulseModule::controller_program,
           nb::arg("parameters"), nb::arg("amplitudes"), nb::arg("segments"),
           nb::arg("rolled") = false, nb::arg("waveform").none() = nb::none(),
           nb::arg("own") = nb::none(), nb::arg("slots").none() = nb::none(),
           nb::arg("exportable") = true, nb::arg("ids").none() = nb::none(),
           "The lines and the events of the program, for a controller; a "
           "list `slots` gets each line's runtime-qudit slot, a list `ids` "
           "each event's op number (with_holds)")
      .def("controller_check", &PulseModule::controller_check,
           nb::call_guard<nb::gil_scoped_release>(), nb::arg("issue"),
           nb::arg("gap"), nb::arg("samples_num"), nb::arg("samples_den"),
           nb::arg("memory"), nb::arg("max_frequencies_per_line"),
           nb::arg("max_pulses"), nb::arg("detectors"),
           nb::arg("detector_recovery"), nb::arg("feedback_latency"),
           nb::arg("supports_timeouts"), nb::arg("timeout_functions"),
           nb::arg("resolve"), "The controller limits the schedule exceeds")
      .def("parameter_roles", &PulseModule::parameter_roles,
           "What each float argument of main reaches")
      .def("with_line_delays", &PulseModule::with_line_delays,
           nb::call_guard<nb::gil_scoped_release>(),
           "This module with other line delays, scheduled again",
           nb::arg("lines"), nb::arg("policy"), nb::arg("max_drives") = 4,
           nb::arg("max_readouts") = 2, nb::arg("readout_latency") = 0,
           nb::arg("switch_penalty") = 0)
      .def("with_holds", &PulseModule::with_holds,
           nb::call_guard<nb::gil_scoped_release>(),
           "This module with its ops held back as `holds` gives by op "
           "number, scheduled again",
           nb::arg("holds"), nb::arg("policy"), nb::arg("max_drives") = 4,
           nb::arg("max_readouts") = 2, nb::arg("readout_latency") = 0,
           nb::arg("switch_penalty") = 0)
      .def("simulation_module", &PulseModule::simulation_module,
           "This module with loops holding calls unrolled, for simulation")
      .def("next_call", &PulseModule::next_call, nb::arg("path"),
           "The first call on a path whose answer is not bound yet")
      .def("call_due_ns", &PulseModule::call_due_ns,
           "Per call, by its submit index, the ns from its start until its "
           "answer is due")
      .def("bind_call", &PulseModule::bind_call,
           "This module with one call's answer bound, scheduled again; "
           "along a path, the window of the program it plays next",
           nb::arg("index"), nb::arg("values"), nb::arg("policy"),
           nb::arg("max_drives"), nb::arg("max_readouts"),
           nb::arg("readout_latency"), nb::arg("switch_penalty"),
           nb::arg("path") = nb::none())
      .def("window_bases", &PulseModule::window_bases,
           "The readouts and calls of the program before this window")
      .def("readout_keys", &PulseModule::readout_keys, nb::arg("entries"),
           "The kernel's number and iteration of each readout execution")
      .def("placeholders", &PulseModule::placeholders,
           "This module with unbound answers set to zero")
      .def("segment_boundaries", &PulseModule::segment_boundaries,
           "The variable and blocking waits that start new segments")
      .def("event_segments", &PulseModule::event_segments, nb::arg("syncs"),
           "The segment of each event, in the order the events are listed")
      .def("classical_calls", &PulseModule::classical_calls,
           "Every execution of a classical call, with its timing")
      .def("readout_drives", &PulseModule::readout_drives, nb::arg("path"),
           nb::call_guard<nb::gil_scoped_release>(),
           "Every readout on a path: (index, iteration, frequency Hz, "
           "amplitude)")
      .def("drive_frames", &PulseModule::drive_frames, nb::arg("path"),
           nb::arg("outcome") = -1, nb::call_guard<nb::gil_scoped_release>(),
           "Every drive on a path with the frame it plays with, and the "
           "frame each line's tone ends in")
      .def("next_measurement", &PulseModule::next_measurement,
           nb::call_guard<nb::gil_scoped_release>(),
           "The next measurement without an outcome on a path", nb::arg("path"))
      .def("operator_program", &PulseModule::operator_program,
           nb::call_guard<nb::gil_scoped_release>(),
           "The lowered Hamiltonian and collapse operators as data, with the "
           "drives evaluated at the given float arguments",
           nb::arg("arguments") = std::vector<double>{})
      .def("set_attributes", &PulseModule::set_attributes,
           nb::call_guard<nb::gil_scoped_release>(),
           "Merge an MLIR dictionary attribute into the module attributes, "
           "then another unless it is empty",
           nb::arg("dictionary"), nb::arg("then") = "")
      .def("param_names", &PulseModule::param_names,
           nb::call_guard<nb::gil_scoped_release>(), "Get parameter names")
      .def(
          "specialize",
          [](PulseModule &self, const std::vector<double> &f64,
             const std::vector<int64_t> &i64, const std::string &schedule,
             int64_t maxDrives, int64_t maxReadouts, int64_t readoutLatency,
             int64_t switchPenalty) {
            return self.specialize_shared(f64, i64, schedule, maxDrives,
                                          maxReadouts, readoutLatency,
                                          switchPenalty, {}, std::nullopt, "");
          },
          "Clone module, substitute parameters with constants, fold, "
          "schedule; shared once its arguments recur, while neither module "
          "changes in place",
          nb::arg("f64_args"), nb::arg("i64_args"),
          nb::arg("schedule") = "alap", nb::arg("max_drives") = 4,
          nb::arg("max_readouts") = 2, nb::arg("readout_latency") = 0,
          nb::arg("switch_penalty") = 0)
      // Separate from specialize, whose calls do not convert these.
      .def("specialize_placed", &PulseModule::specialize_shared,
           "specialize, with each runtime qudit (slot k) bound to carrier "
           "carriers[k] of a device with the given 0-1 frequency of every "
           "carrier (and readout frequency, else the 0-1 one); device (its "
           "fingerprint) is part of the key",
           nb::arg("f64_args"), nb::arg("i64_args"), nb::arg("schedule"),
           nb::arg("max_drives"), nb::arg("max_readouts"),
           nb::arg("readout_latency"), nb::arg("switch_penalty"),
           nb::arg("carriers"), nb::arg("frequencies").none(),
           nb::arg("device"),
           nb::arg("readout_frequencies").none() = nb::none())
      .def(
          "serialize",
          [](PulseModule &self) {
            std::tuple<std::string, std::string, bool, std::vector<std::string>>
                state;
            {
              nb::gil_scoped_release release;
              state = self.serialize();
            }
            auto &[module, source, runTime, rewrites] = state;
            return nb::make_tuple(nb::bytes(module.data(), module.size()),
                                  nb::bytes(source.data(), source.size()),
                                  runTime, rewrites);
          },
          "(module bytecode, source bytecode or b'', computes at run time, "
          "rewrites rerun on a binding): what PulseModuleBuilder.load takes")
      .def("specialized_cache", &PulseModule::specialized_cache,
           "(specialized modules kept, at most how many, at most how many "
           "ops in all); a non-negative argument sets that bound and empties "
           "the cache",
           nb::arg("entries") = -1, nb::arg("ops") = -1);

  nb::class_<PulseModuleBuilder>(m, "PulseModuleBuilder")
      .def(nb::init<>())
      .def("build_from_packed", &PulseModuleBuilder::build_from_packed,
           nb::call_guard<nb::gil_scoped_release>(),
           "Build an in-memory PulseModule from a packed int64 numpy buffer; "
           "qubit_freqs and readout_freqs give each qudit's 0-1 frequency "
           "and readout tone by index (the first qudit_count are recorded)",
           nb::arg("stream"), nb::arg("clock_ghz"), nb::arg("qudit_count"),
           nb::arg("qubit_freqs"),
           nb::arg("param_names") = std::vector<std::string>{},
           nb::arg("param_types") = std::vector<std::string>{},
           nb::arg("name") = "main", nb::arg("slot_count") = 0,
           nb::arg("readout_freqs").none() = nb::none())
      .def(
          "build_from_packed",
          [](PulseModuleBuilder &self,
             nb::ndarray<int64_t, nb::ndim<1>, nb::c_contig> stream,
             double clock_ghz, int64_t qudit_count,
             nb::ndarray<double, nb::ndim<1>, nb::c_contig> qubit_freqs,
             std::vector<std::string> param_names,
             std::vector<std::string> param_types, const std::string &name,
             int64_t slot_count,
             std::optional<nb::ndarray<const double, nb::ndim<1>, nb::c_contig>>
                 readout_freqs,
             std::vector<int64_t> lost_lines) {
            self.expectLost(std::move(lost_lines));
            nb::gil_scoped_release release;
            return self.build_from_packed(stream, clock_ghz, qudit_count,
                                          qubit_freqs, std::move(param_names),
                                          std::move(param_types), name,
                                          slot_count, readout_freqs);
          },
          "The same for a trace that failed: lost_lines are the ids of the "
          "lines and tones its failing operation was given. One that is an "
          "if's result whose arms leave different lines there is an error "
          "at that if, also when no record reads it",
          nb::arg("stream"), nb::arg("clock_ghz"), nb::arg("qudit_count"),
          nb::arg("qubit_freqs"), nb::arg("param_names"),
          nb::arg("param_types"), nb::arg("name"), nb::arg("slot_count"),
          nb::arg("readout_freqs").none(), nb::arg("lost_lines"))
      .def(
          "load",
          [](PulseModuleBuilder &self, nb::bytes module, nb::bytes source,
             bool runTime, const std::vector<std::string> &rewrites) {
            std::string moduleBytes(module.c_str(), module.size());
            std::string sourceBytes(source.c_str(), source.size());
            nb::gil_scoped_release release;
            return self.load(moduleBytes, sourceBytes, runTime, rewrites);
          },
          "Read a PulseModule from what PulseModule.serialize returned",
          nb::arg("module"), nb::arg("source"), nb::arg("run_time"),
          nb::arg("rewrites"))
      .def("load_link_names", &PulseModuleBuilder::load_link_names,
           "The names a module (assembly or bytecode) gives its links, by id",
           nb::arg("module"))
      .def("load_submissions", &PulseModuleBuilder::load_submissions,
           "Every submission of a module (assembly or bytecode): its function, "
           "the cycles its schedule gives it and the cycles its link takes",
           nb::arg("module"))
      .def("load_callables", &PulseModuleBuilder::load_callables,
           "The classical functions a module (assembly or bytecode) declares",
           nb::arg("module"))
      .def("load_function", &PulseModuleBuilder::load_function,
           "The scheduled function `entry` of a module (assembly or bytecode) "
           "as a module whose main takes the function's qudits as slots",
           nb::arg("module"), nb::arg("entry"), nb::arg("clock_ghz"),
           nb::arg("qudits"), nb::arg("latencies"), nb::arg("keys"),
           nb::arg("devices"));

  m.def(
      "build_stamp", [] { return std::string(cudaqPulseBuildStamp); },
      "A hash of the sources and LLVM this extension was built from");

  m.def(
      "shared_context_budget",
      [] {
        auto limits = SharedContextPool::budget();
        return std::make_tuple(limits.modules, limits.words);
      },
      "(modules, packed words) a shared MLIR context serves before it is "
      "retired, from CUDAQ_PULSE_CONTEXT_MAX_MODULES and "
      "CUDAQ_PULSE_CONTEXT_MAX_WORDS");
  m.def("shared_context_stats", &SharedContextPool::stats,
        "(contexts created, builders on the current context, its packed "
        "words)");
  m.def(
      "_canonical_pattern_audit",
      [](std::vector<std::string> held) {
        // Every dialect of the registry loaded, so that every pattern a
        // module can meet is in the set.
        mlir::MLIRContext context(makeRegistry());
        context.loadAllAvailableDialects();
        mlir::RewritePatternSet all = allCanonicalPatterns(&context);
        llvm::StringSet<> keys;
        for (const std::unique_ptr<mlir::RewritePattern> &pattern :
             all.getNativePatterns())
          keys.insert(patternKey(*pattern));
        std::vector<std::string> unused;
        for (const auto &entry : undeclaredMakes())
          if (!keys.contains(entry.getKey()))
            unused.push_back(entry.getKey().str());
        std::sort(unused.begin(), unused.end());
        nb::dict audit;
        audit["patterns"] = all.getNativePatterns().size();
        audit["unlisted"] = unlistedPatterns(all);
        audit["unused_entries"] = unused;
        llvm::StringSet<> names;
        for (const std::string &name : held)
          names.insert(name);
        mlir::RewritePatternSet reached =
            reachablePatterns(&context, std::move(all), names);
        std::vector<std::string> kept;
        for (const std::unique_ptr<mlir::RewritePattern> &pattern :
             reached.getNativePatterns())
          kept.push_back(patternKey(*pattern));
        audit["kept"] = kept;
        return audit;
      },
      nb::arg("held") = std::vector<std::string>(),
      "For tests: the canonicalization patterns of every dialect (count), "
      "those that declare no ops they make and are not listed in "
      "bindings.cpp's undeclaredMakes() (unlisted), the entries there that "
      "match no pattern (unused_entries), and how many patterns a module "
      "holding the ops `held` keeps (kept, as root|name).");
  m.def("verifies_everything", &verifyEverything,
        "Whether CUDAQ_PULSE_VERIFY=1 asks for the verifications that only "
        "catch a bug of the compiler itself");

  // CUDA-Q QEC's simulation decoder functions (those a CUDA-Q kernel on a
  // simulator calls, in cudaq-qec 0.6.0 to 0.8.1) throw C++ exceptions for an
  // unconfigured decoder or a wrong size, which must not unwind through
  // ctypes: they are called here, by address, and raised as RuntimeError.
  m.def(
      "decoder_reset",
      [](std::uintptr_t address, std::uint64_t decoder_id) {
        auto fn = reinterpret_cast<void (*)(std::uint64_t)>(address);
        guardedDecoderCall([&] { fn(decoder_id); });
      },
      nb::arg("address"), nb::arg("decoder_id"),
      "Call cudaq::qec::decoding::simulation::reset_decoder at address");
  m.def(
      "decoder_enqueue",
      [](std::uintptr_t address, std::uint64_t decoder_id,
         const nb::bytes &syndromes, std::uint64_t tag) {
        auto fn = reinterpret_cast<void (*)(std::uint64_t, DecoderBits,
                                            std::uint64_t)>(address);
        std::vector<std::uint8_t> bits(
            reinterpret_cast<const std::uint8_t *>(syndromes.c_str()),
            reinterpret_cast<const std::uint8_t *>(syndromes.c_str()) +
                syndromes.size());
        guardedDecoderCall(
            [&] { fn(decoder_id, {bits.data(), bits.size()}, tag); });
      },
      nb::arg("address"), nb::arg("decoder_id"), nb::arg("syndromes"),
      nb::arg("tag"),
      "Call simulation_enqueue_syndromes at address, one byte per syndrome "
      "bit");
  m.def(
      "decoder_get",
      [](std::uintptr_t address, std::uint64_t decoder_id, std::uint64_t count,
         bool reset) {
        auto fn = reinterpret_cast<void (*)(std::uint64_t, DecoderBits, bool)>(
            address);
        std::vector<std::uint8_t> bits(count, 0);
        guardedDecoderCall(
            [&] { fn(decoder_id, {bits.data(), bits.size()}, reset); });
        return nb::bytes(reinterpret_cast<const char *>(bits.data()),
                         bits.size());
      },
      nb::arg("address"), nb::arg("decoder_id"), nb::arg("count"),
      nb::arg("reset"),
      "Call simulation_get_corrections at address; the corrections, one byte "
      "per bit");

  cudaq_pulse::loadLengthInference();
  m.def(
      "infer_list_lengths",
      [](nb::handle function, nb::handle names, nb::handle lengths,
         nb::handle positions, Py_ssize_t result_count) {
        PyObject *found = cudaq_pulse::inferListLengths(
            function.ptr(), names.ptr(), lengths.ptr(), positions.ptr(),
            result_count);
        if (!found)
          throw nb::python_error();
        return nb::steal(found);
      },
      nb::arg("function"), nb::arg("names"), nb::arg("lengths"),
      nb::arg("positions"), nb::arg("result_count"),
      nb::sig(
          "def infer_list_lengths(function: object, names: tuple[str, ...], "
          "lengths: dict[str, int], positions: tuple[int, ...], "
          "result_count: int) -> tuple[int, ...] | None"),
      "The lengths of the list results at positions, for list inputs of the "
      "given lengths, read from the body's bytecode; None for a body outside "
      "what this follows (classical._infer then decides)");

  m.def(
      "evaluate_callback",
      [](int32_t kind, const std::vector<double> &params,
         const std::vector<double> &times) {
        std::vector<double> values;
        values.reserve(times.size());
        for (double time : times) {
          double value = 0.0;
          if (cudaq_pulse::evaluateCallback(kind, params.data(), params.size(),
                                            time, value))
            throw std::invalid_argument("invalid waveform callback");
          values.push_back(value);
        }
        return values;
      },
      nb::call_guard<nb::gil_scoped_release>(),
      "Evaluate a drive-envelope callback at the given times (ns)",
      nb::arg("kind"), nb::arg("params"), nb::arg("times"));

  m.def(
      "sample_envelope",
      [](int32_t kind, const std::vector<double> &params, size_t count,
         double rate) {
        if (params.size() < cudaq_pulse::kWaveformHeader)
          throw std::invalid_argument("too few envelope parameters");
        std::pair<std::vector<double>, std::vector<double>> parts;
        auto &[real, imaginary] = parts;
        real.resize(count);
        imaginary.resize(count);
        for (size_t k = 0; k < count; ++k)
          if (cudaq_pulse::evaluateEnvelope(
                  kind, params.data(), cudaq_pulse::kWaveformHeader,
                  params.size(), static_cast<double>(k) / rate, real[k],
                  imaginary[k]))
            throw std::invalid_argument("unknown envelope kind");
        return parts;
      },
      nb::call_guard<nb::gil_scoped_release>(),
      "The real and imaginary parts of an envelope (a callback's kind and "
      "parameters) at count times k / rate from its start",
      nb::arg("kind"), nb::arg("params"), nb::arg("count"), nb::arg("rate"),
      nb::sig("def sample_envelope(kind: int, params: Sequence[float], count: "
              "int, rate: float) -> tuple[list[float], list[float]]"));

  m.def(
      "identity_keys",
      [](nb::dict mapping, nb::handle first, nb::handle second) {
        PyObject *key = nullptr;
        PyObject *value = nullptr;
        PyObject *firstKey = nullptr;
        PyObject *secondKey = nullptr;
        Py_ssize_t pos = 0;
        while (PyDict_Next(mapping.ptr(), &pos, &key, &value)) {
          if (!firstKey && value == first.ptr()) {
            firstKey = key;
            if (secondKey)
              break;
          } else if (!secondKey && value == second.ptr()) {
            secondKey = key;
            if (firstKey)
              break;
          }
        }
        auto keyOrNone = [](PyObject *found) -> nb::object {
          return found ? nb::borrow(found) : nb::none();
        };
        return nb::make_tuple(keyOrNone(firstKey), keyOrNone(secondKey));
      },
      "(the first key whose value is first, the first other key whose value "
      "is second), by identity, None where there is none; the tracer's scan "
      "of a kernel's locals",
      nb::arg("mapping"), nb::arg("first"), nb::arg("second"));

  m.def("freeze_type", &freezeType,
        nb::sig("def freeze_type(cls: type) -> None"),
        "Make cls immutable, as built-in types are: setting or deleting its "
        "attributes raises TypeError. Its bases must be immutable already",
        nb::arg("cls"));

  nb::class_<JITModule>(m, "JITModule")
      .def("lookup", &JITModule::lookup,
           nb::call_guard<nb::gil_scoped_release>(),
           "Return the address of a JIT-compiled symbol", nb::arg("name"));

  nb::class_<MLIRPipeline>(m, "MLIRPipeline")
      .def(nb::init<>())
      .def("jit_pulse", &MLIRPipeline::jit_pulse,
           nb::call_guard<nb::gil_scoped_release>(),
           "Lower pulse MLIR and JIT-compile it in-process",
           nb::arg("pulse_mlir"), nb::arg("shared_libs"))
      .def("jit_llvm_dialect", &MLIRPipeline::jit_llvm_dialect,
           nb::call_guard<nb::gil_scoped_release>(),
           "JIT-compile an LLVM-dialect MLIR module in-process",
           nb::arg("mlir_text"), nb::arg("shared_libs"))
      .def("run_pulse_to_qop", &MLIRPipeline::run_pulse_to_qop,
           nb::call_guard<nb::gil_scoped_release>(),
           "Run pulse-to-qop pass only")
      .def("run_qop_to_cudm", &MLIRPipeline::run_qop_to_cudm,
           nb::call_guard<nb::gil_scoped_release>(),
           "Run qop-to-cudm pass only")
      .def("parse_to_module", &MLIRPipeline::parse_to_module,
           nb::call_guard<nb::gil_scoped_release>(),
           "Parse MLIR text into a PulseModule");
}
