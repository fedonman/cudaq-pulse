// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// QOpToCuDensityMat conversion pass: lower qop dialect ops to cudm dialect ops
// for GPU-accelerated quantum state evolution.
//
// Contents, in file order:
//   Matrices                  Matrix, Banded, kron, embed
//   Leaves and products       what the pass reads of the qop operators
//   The pass                  runOnOperation, in these steps:
//     Setup and checks                      options, @main, modes
//     1 to 4. Runtime objects and the evolve
//     5. Leaves                             matrices, elementary operators
//     6. Products                           each product and its term
//     The generator's products; appending a dissipator
//     7 and 8. Operator terms               one term each, or fused:
//       Fused: two-sided pieces
//       Fused: the dissipators' anticommutators
//       Fused: time-dependent terms of one operator
//       Fused: one-sided sums and the terms left, per side
//     9. Evolve                             cudm.evolve
//     10. Cleanup                           destroy in reverse order

#include "llvm/ADT/MapVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "cudaq-pulse/Dialect/QOp/QOpDialect.h.inc"
#include "cudaq-pulse/Dialect/QOp/QOpEnums.h.inc"
#define GET_ATTRDEF_CLASSES
#include "cudaq-pulse/Dialect/QOp/QOpAttrs.h.inc"
#define GET_TYPEDEF_CLASSES
#include "cudaq-pulse/Dialect/QOp/QOpTypes.h.inc"
#define GET_OP_CLASSES
#include "cudaq-pulse/Dialect/QOp/QOpOps.h.inc"

#include "cudaq-pulse/Dialect/CuDensityMat/CuDensityMatDialect.h.inc"
#include "cudaq-pulse/Dialect/CuDensityMat/CuDensityMatEnums.h.inc"
#define GET_ATTRDEF_CLASSES
#include "cudaq-pulse/Dialect/CuDensityMat/CuDensityMatAttrs.h.inc"
#define GET_TYPEDEF_CLASSES
#include "cudaq-pulse/Dialect/CuDensityMat/CuDensityMatTypes.h.inc"
#define GET_OP_CLASSES
#include "cudaq-pulse/Dialect/CuDensityMat/CuDensityMatOps.h.inc"

#include "cudaq-pulse/Runtime/WaveformCallback.h"

#include <complex>
#include <cstdlib>

using namespace mlir;

namespace {

using Complex = std::complex<double>;

/// The default `max-fused-dimension`. A fused block that acts on both sides
/// of the density matrix is a superoperator of the block's dimension squared;
/// larger ones cost cuDensityMat seconds of one-time setup per operator shape
/// on some GPUs.
constexpr int64_t kDefaultMaxFusedDimension = 16;
/// A single-mode operator of at least this many levels with at most
/// kMaxDiagonals nonzero diagonals is stored by its diagonals, as CUDA-Q
/// does: the ladder and number operators and their sums.
constexpr int64_t kMinMultidiagonalExtent = 4;
constexpr size_t kMaxDiagonals = 1;

// ── Matrices ────────────────────────────────────────────────────────────────

/// A square matrix on an ordered list of modes: column-major, the first
/// mode's index varying fastest, as cuDensityMat stores a dense elementary
/// operator.
struct Matrix {
  int64_t dimension = 0;
  SmallVector<Complex> values;

  static Matrix zero(int64_t dimension) {
    return {dimension, SmallVector<Complex>(dimension * dimension)};
  }
  static Matrix identity(int64_t dimension) {
    Matrix result = zero(dimension);
    for (int64_t i = 0; i < dimension; ++i)
      result.at(i, i) = 1.0;
    return result;
  }
  /// From column-major interleaved real and imaginary parts.
  static Matrix interleaved(int64_t dimension, ArrayRef<double> data) {
    Matrix result = zero(dimension);
    for (size_t i = 0; i < result.values.size(); ++i)
      result.values[i] = Complex(data[2 * i], data[2 * i + 1]);
    return result;
  }

  Complex &at(int64_t row, int64_t column) {
    return values[column * dimension + row];
  }
  Complex at(int64_t row, int64_t column) const {
    return values[column * dimension + row];
  }
  SmallVector<double> data() const {
    SmallVector<double> result;
    result.reserve(2 * values.size());
    for (Complex value : values) {
      result.push_back(value.real());
      result.push_back(value.imag());
    }
    return result;
  }
  Matrix adjoint() const {
    Matrix result = zero(dimension);
    for (int64_t row = 0; row < dimension; ++row)
      for (int64_t column = 0; column < dimension; ++column)
        result.at(column, row) = std::conj(at(row, column));
    return result;
  }
  Matrix operator*(const Matrix &right) const {
    Matrix result = zero(dimension);
    for (int64_t column = 0; column < dimension; ++column)
      for (int64_t k = 0; k < dimension; ++k) {
        Complex factor = right.at(k, column);
        if (factor == 0.0)
          continue;
        for (int64_t row = 0; row < dimension; ++row)
          result.at(row, column) += at(row, k) * factor;
      }
    return result;
  }
  Matrix operator*(Complex scale) const {
    Matrix result = *this;
    for (Complex &value : result.values)
      value *= scale;
    return result;
  }
  Matrix &operator+=(const Matrix &other) {
    for (size_t i = 0; i < values.size(); ++i)
      values[i] += other.values[i];
    return *this;
  }
  bool operator==(const Matrix &other) const {
    return dimension == other.dimension && values == other.values;
  }
};

/// A single-mode matrix by its nonzero diagonals, as cuDensityMat stores a
/// multi-diagonal operator: offsets ascending (positive above the main
/// diagonal), one vector of the dimension per offset, element
/// (r, r + offset) at index min(r, r + offset).
struct Banded {
  int64_t dimension = 0;
  SmallVector<int32_t> offsets;
  SmallVector<Complex> values;

  /// `matrix` by its diagonals, if it has one to `maxDiagonals` nonzero ones.
  static std::optional<Banded> of(const Matrix &matrix, size_t maxDiagonals) {
    const int64_t d = matrix.dimension;
    SmallVector<bool> nonzero(2 * d - 1, false);
    for (int64_t column = 0; column < d; ++column)
      for (int64_t row = 0; row < d; ++row)
        if (matrix.at(row, column) != 0.0)
          nonzero[column - row + d - 1] = true;
    Banded result{d, {}, {}};
    for (int64_t offset = 1 - d; offset < d; ++offset)
      if (nonzero[offset + d - 1])
        result.offsets.push_back(static_cast<int32_t>(offset));
    if (result.offsets.empty() || result.offsets.size() > maxDiagonals)
      return std::nullopt;
    result.values.assign(result.offsets.size() * d, 0.0);
    for (auto [i, offset] : llvm::enumerate(result.offsets))
      for (int64_t row = std::max<int64_t>(0, -offset);
           row < std::min<int64_t>(d, d - offset); ++row)
        result.values[i * d + std::min<int64_t>(row, row + offset)] =
            matrix.at(row, row + offset);
    return result;
  }

  /// Row `row`'s element of a single-diagonal matrix (0 off the matrix).
  Complex onRow(int64_t row) const {
    const int64_t column = row + offsets[0];
    if (row < 0 || row >= dimension || column < 0 || column >= dimension)
      return 0.0;
    return values[std::min(row, column)];
  }
  /// The product of single-diagonal matrices, if it is not zero.
  std::optional<Banded> times(const Banded &right) const {
    const int64_t offset = offsets[0] + right.offsets[0];
    if (offset <= -dimension || offset >= dimension)
      return std::nullopt;
    Banded result{dimension,
                  {static_cast<int32_t>(offset)},
                  SmallVector<Complex>(dimension, 0.0)};
    for (int64_t row = std::max<int64_t>(0, -offset);
         row < std::min<int64_t>(dimension, dimension - offset); ++row)
      result.values[std::min(row, row + offset)] =
          onRow(row) * right.onRow(row + offsets[0]);
    return result;
  }
  /// The adjoint of a single-diagonal matrix: the same storage index holds
  /// the transposed element.
  Banded adjoint() const {
    Banded result{dimension, {static_cast<int32_t>(-offsets[0])}, values};
    for (Complex &value : result.values)
      value = std::conj(value);
    return result;
  }
};

/// The operator acting as `fast` on the fast index and as `slow` on the slow
/// one (numpy.kron(slow, fast)).
Matrix kron(const Matrix &slow, const Matrix &fast) {
  const int64_t d = fast.dimension;
  Matrix result = Matrix::zero(slow.dimension * d);
  for (int64_t slowColumn = 0; slowColumn < slow.dimension; ++slowColumn)
    for (int64_t slowRow = 0; slowRow < slow.dimension; ++slowRow) {
      Complex factor = slow.at(slowRow, slowColumn);
      if (factor == 0.0)
        continue;
      for (int64_t column = 0; column < d; ++column)
        for (int64_t row = 0; row < d; ++row)
          result.at(slowRow * d + row, slowColumn * d + column) =
              factor * fast.at(row, column);
    }
  return result;
}

using Modes = SmallVector<int32_t, 4>;

int64_t dimensionOf(ArrayRef<int32_t> modes, ArrayRef<int64_t> extents) {
  int64_t dimension = 1;
  for (int32_t mode : modes)
    dimension *= extents[mode];
  return dimension;
}

/// `matrix`, an operator on `sub`, as one on `modes` (both ascending, `sub`
/// within `modes`): the identity on the other modes.
Matrix embed(const Matrix &matrix, ArrayRef<int32_t> sub,
             ArrayRef<int32_t> modes, ArrayRef<int64_t> extents) {
  if (sub == modes)
    return matrix;
  const int64_t dimension = dimensionOf(modes, extents);
  // Per mode of `modes`: its stride in `sub`'s index, or 0 if not in `sub`.
  SmallVector<int64_t> subStride(modes.size(), 0);
  int64_t stride = 1;
  for (int32_t mode : sub) {
    subStride[llvm::find(modes, mode) - modes.begin()] = stride;
    stride *= extents[mode];
  }
  auto split = [&](int64_t index, int64_t &subIndex, int64_t &rest) {
    subIndex = 0;
    rest = 0;
    int64_t restStride = 1;
    for (auto [i, mode] : llvm::enumerate(modes)) {
      int64_t digit = index % extents[mode];
      index /= extents[mode];
      if (subStride[i]) {
        subIndex += digit * subStride[i];
      } else {
        rest += digit * restStride;
        restStride *= extents[mode];
      }
    }
  };
  Matrix result = Matrix::zero(dimension);
  for (int64_t column = 0; column < dimension; ++column) {
    int64_t subColumn, restColumn;
    split(column, subColumn, restColumn);
    for (int64_t row = 0; row < dimension; ++row) {
      int64_t subRow, restRow;
      split(row, subRow, restRow);
      if (restRow == restColumn)
        result.at(row, column) = matrix.at(subRow, subColumn);
    }
  }
  return result;
}

// ── Leaves and products ─────────────────────────────────────────────────────

/// A qop leaf operator: the mode it acts on, its matrix and its adjoint's.
struct Leaf {
  int32_t mode = 0;
  Matrix matrix;
  Matrix adjoint;
};

/// A qop product: leaves in application order and a constant or callback
/// coefficient.
struct Product {
  SmallVector<Operation *> leaves;
  Modes modes;
  Complex coefficient = 1.0;
  qop::CallbackScalarOp callback;

  /// The modes it acts on, ascending, if each factor acts on its own mode.
  std::optional<Modes> distinctModes() const {
    Modes sorted(modes);
    llvm::sort(sorted);
    if (std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end())
      return std::nullopt;
    return sorted;
  }
};

// ── The pass ────────────────────────────────────────────────────────────────

struct QOpToCuDensityMatPass
    : public PassWrapper<QOpToCuDensityMatPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(QOpToCuDensityMatPass)

  QOpToCuDensityMatPass() {
    const char *fromEnvironment = std::getenv("CUDAQ_PULSE_GPU_OPERATORS");
    if (fromEnvironment && *fromEnvironment)
      operators = std::string(fromEnvironment);
  }
  QOpToCuDensityMatPass(const QOpToCuDensityMatPass &other)
      : PassWrapper(other) {}

  StringRef getArgument() const final { return "qop-to-cudm"; }
  StringRef getDescription() const final {
    return "Lower qop dialect Hamiltonian/Lindblad ops to cudm operator "
           "construction and evolve";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<cudm::CuDensityMatDialect>();
    registry.insert<arith::ArithDialect>();
    registry.insert<func::FuncDialect>();
  }

  Option<std::string> operators{
      *this, "operators",
      llvm::cl::desc(
          "How the Lindbladian becomes cuDensityMat terms: 'fused' sums the "
          "constant Hamiltonian terms and dissipators on small groups of "
          "modes into dense blocks, makes the time-dependent terms of one "
          "operator one term whose callback sums theirs, and stores the "
          "operators of one diagonal on modes of four or more levels by "
          "their diagonal; 'banded' only does the latter; 'dense' makes "
          "every product its own term of dense operators "
          "(CUDAQ_PULSE_GPU_OPERATORS sets the default)"),
      llvm::cl::init("fused")};
  Option<int64_t> maxFusedDimension{
      *this, "max-fused-dimension",
      llvm::cl::desc("Largest dimension of a fused block on one side of the "
                     "density matrix; a block on both sides needs its "
                     "dimension squared within it. 0 fuses nothing"),
      llvm::cl::init(kDefaultMaxFusedDimension)};

  void runOnOperation() override {
    // ── Setup and checks ────────────────────────────────────────────────────
    auto module = getOperation();
    OpBuilder b(module.getContext());

    if (operators != "fused" && operators != "banded" && operators != "dense") {
      module.emitError("qop-to-cudm operators must be 'fused', 'banded' or "
                       "'dense' (CUDAQ_PULSE_GPU_OPERATORS), not '")
          << operators << "'";
      return signalPassFailure();
    }
    const bool fuse = operators == "fused" && maxFusedDimension > 0;
    const bool band = operators != "dense";

    // Find @main
    func::FuncOp mainFunc;
    module.walk([&](func::FuncOp fn) {
      if (fn.getName() == "main")
        mainFunc = fn;
    });
    if (!mainFunc) {
      module.emitError("no @main function found");
      return signalPassFailure();
    }
    // The runtime takes constant callback parameters.
    auto symbolic = module.walk([&](qop::CallbackScalarOp callback) {
      if (!callback->hasAttr("cudm.callback_param_arguments"))
        return WalkResult::advance();
      callback.emitError("drive callback depends on the program's float "
                         "arguments; bind them before lowering to "
                         "cuDensityMat");
      return WalkResult::interrupt();
    });
    if (symbolic.wasInterrupted())
      return signalPassFailure();

    auto &block = mainFunc.getBody().front();
    Operation *returnOp = block.getTerminator();
    b.setInsertionPoint(returnOp);
    auto loc = returnOp->getLoc();

    // The qudit count, from the module's attributes or from the qop.spin ops.
    int64_t quditCount = 1;
    auto nqAttr = module->getAttrOfType<IntegerAttr>("qop.qudit_count");
    if (!nqAttr)
      nqAttr = module->getAttrOfType<IntegerAttr>("qudit_count");
    if (nqAttr)
      quditCount = nqAttr.getInt();
    else {
      // Count unique qubit targets from qop.spin ops
      DenseSet<int64_t> targets;
      module.walk([&](qop::SpinOp spin) {
        if (auto cst = spin.getTarget().getDefiningOp<arith::ConstantOp>()) {
          if (auto ia = dyn_cast<IntegerAttr>(cst.getValue()))
            targets.insert(ia.getInt());
        }
      });
      if (!targets.empty())
        quditCount = *std::max_element(targets.begin(), targets.end()) + 1;
    }

    SmallVector<int64_t> modeExtents(quditCount, 2);
    if (auto extents =
            module->getAttrOfType<DenseI64ArrayAttr>("qop.mode_extents")) {
      if (static_cast<int64_t>(extents.size()) < quditCount) {
        module.emitError("qop.mode_extents has fewer entries than modes");
        return signalPassFailure();
      }
      modeExtents.assign(extents.asArrayRef().begin(),
                         extents.asArrayRef().end());
      quditCount = static_cast<int64_t>(modeExtents.size());
    }

    auto handleTy = cudm::HandleType::get(b.getContext());
    auto stateTy = cudm::StateType::get(b.getContext());
    auto wsTy = cudm::WorkspaceType::get(b.getContext());
    auto elemOpTy = cudm::ElementaryOpType::get(b.getContext());
    auto opTermTy = cudm::OpTermType::get(b.getContext());
    auto operatorTy = cudm::OperatorType::get(b.getContext());

    // ── 1 to 4. Runtime objects and the evolve ──────────────────────────────
    // 1. cudm.init_handle
    auto handle = cudm::InitHandleOp::create(b, loc, handleTy);

    SmallVector<qop::EvolveOp> evolveOps;
    module.walk([&](qop::EvolveOp op) { evolveOps.push_back(op); });
    if (evolveOps.size() != 1) {
      module.emitError("qop-to-cudm requires exactly one qop.evolve op");
      return signalPassFailure();
    }
    auto lindblad =
        evolveOps.front().getGenerator().getDefiningOp<qop::LindbladOp>();
    if (!lindblad) {
      evolveOps.front().emitError("generator must be a qop.lindblad");
      return signalPassFailure();
    }
    const bool hasDissipation = !lindblad.getCollapseOps().empty() ||
                                module->hasAttr("qop.mixed_state");

    // 2. cudm.create_state (|0...0>). Open-system evolution requires a
    // density matrix; unitary evolution can retain the cheaper state vector
    // unless qop.mixed_state asks to evolve one (a mixed initial state).
    auto purityAttr = cudm::StatePurityAttr::get(
        b.getContext(),
        hasDissipation ? cudm::StatePurity::Mixed : cudm::StatePurity::Pure);
    auto dtypeAttr =
        cudm::ComputeTypeAttr::get(b.getContext(), cudm::ComputeType::F64);
    auto modeExtentsAttr = b.getDenseI64ArrayAttr(modeExtents);
    auto stateIn = cudm::CreateStateOp::create(
        b, loc, stateTy, handle, purityAttr, dtypeAttr, modeExtentsAttr,
        b.getI64IntegerAttr(0), b.getBoolAttr(false));
    auto stateOut = cudm::CreateStateOp::create(
        b, loc, stateTy, handle, purityAttr, dtypeAttr, modeExtentsAttr,
        b.getI64IntegerAttr(0), b.getBoolAttr(false));

    // 3. cudm.create_workspace
    auto workspace = cudm::CreateWorkspaceOp::create(b, loc, wsTy, handle);

    // 4. cudm.create_operator (the composite Hamiltonian operator)
    auto compositeOp = cudm::CreateOperatorOp::create(b, loc, operatorTy,
                                                      handle, modeExtentsAttr);

    // ── 5. Leaves ───────────────────────────────────────────────────────────
    // The matrix of each leaf: a Pauli matrix for qop.spin, the data of
    // a qop.matrix_leaf (dense, column-major, for qudits of more levels).
    llvm::MapVector<Operation *, Leaf> leaves;
    module.walk([&](qop::SpinOp spin) {
      // Column-major interleaved real and imaginary parts.
      SmallVector<double, 8> pauliData;
      switch (spin.getKind()) {
      case qop::HandlerKind::SpinX:
        pauliData = {0, 0, 1, 0, 1, 0, 0, 0}; // [[0,1],[1,0]]
        break;
      case qop::HandlerKind::SpinY:
        // cuDensityMat uses column-major storage: [[0,-i],[i,0]].
        pauliData = {0, 0, 0, 1, 0, -1, 0, 0};
        break;
      case qop::HandlerKind::SpinZ:
        pauliData = {1, 0, 0, 0, 0, 0, -1, 0}; // [[1,0],[0,-1]]
        break;
      case qop::HandlerKind::SpinI:
        pauliData = {1, 0, 0, 0, 0, 0, 1, 0}; // [[1,0],[0,1]]
        break;
      case qop::HandlerKind::SpinLowering:
        pauliData = {0, 0, 0, 0, 1, 0, 0, 0}; // [[0,1],[0,0]]
        break;
      case qop::HandlerKind::SpinRaising:
        pauliData = {0, 0, 1, 0, 0, 0, 0, 0}; // [[0,0],[1,0]]
        break;
      default:
        spin.emitError("unsupported spin kind for cudm lowering");
        return;
      }
      int32_t mode = -1;
      if (auto target = spin.getTarget().getDefiningOp<arith::ConstantOp>())
        if (auto attr = dyn_cast<IntegerAttr>(target.getValue()))
          mode = static_cast<int32_t>(attr.getInt());
      // Pauli operators are self-adjoint. Raising and lowering are each
      // other's adjoints.
      SmallVector<double, 8> adjointData = pauliData;
      if (spin.getKind() == qop::HandlerKind::SpinLowering)
        adjointData = {0, 0, 1, 0, 0, 0, 0, 0};
      else if (spin.getKind() == qop::HandlerKind::SpinRaising)
        adjointData = {0, 0, 0, 0, 1, 0, 0, 0};
      leaves[spin.getOperation()] = {mode, Matrix::interleaved(2, pauliData),
                                     Matrix::interleaved(2, adjointData)};
    });
    module.walk([&](qop::MatrixLeafOp leaf) {
      auto constant = leaf.getData().getDefiningOp<arith::ConstantOp>();
      auto values = constant
                        ? dyn_cast<DenseFPElementsAttr>(constant.getValue())
                        : DenseFPElementsAttr();
      if (!values || leaf.getTargets().size() != 1) {
        leaf.emitError("cudm lowering supports single-mode matrix leaves with "
                       "constant data");
        return;
      }
      const int64_t d = leaf.getDimensions()[0];
      SmallVector<double> data(values.getValues<double>().begin(),
                               values.getValues<double>().end());
      if (static_cast<int64_t>(data.size()) != 2 * d * d) {
        leaf.emitError("matrix leaf data does not match its dimension");
        return;
      }
      // Column-major interleaved; the adjoint swaps rows and columns and
      // conjugates.
      SmallVector<double> adjoint(data.size());
      for (int64_t row = 0; row < d; ++row)
        for (int64_t column = 0; column < d; ++column) {
          adjoint[2 * (column * d + row)] = data[2 * (row * d + column)];
          adjoint[2 * (column * d + row) + 1] =
              -data[2 * (row * d + column) + 1];
        }
      leaves[leaf.getOperation()] = {static_cast<int32_t>(leaf.getTargets()[0]),
                                     Matrix::interleaved(d, data),
                                     Matrix::interleaved(d, adjoint)};
    });

    SmallVector<Value> allElementaryOps;
    auto createBanded = [&](const Banded &banded) -> Value {
      SmallVector<double> data;
      for (Complex value : banded.values) {
        data.push_back(value.real());
        data.push_back(value.imag());
      }
      auto type = RankedTensorType::get(
          {static_cast<int64_t>(banded.offsets.size()), banded.dimension, 2},
          b.getF64Type());
      auto value = arith::ConstantOp::create(
          b, loc, DenseFPElementsAttr::get(type, ArrayRef<double>(data)));
      Value elementary = cudm::CreateElementaryOpOp::create(
          b, loc, elemOpTy, handle, value,
          cudm::SparsityAttr::get(b.getContext(),
                                  cudm::Sparsity::Multidiagonal),
          dtypeAttr, b.getDenseI64ArrayAttr({banded.dimension}),
          FlatSymbolRefAttr(), b.getDenseI32ArrayAttr(banded.offsets));
      allElementaryOps.push_back(elementary);
      return elementary;
    };
    auto createElementary = [&](const Matrix &matrix,
                                ArrayRef<int64_t> extents) -> Value {
      const int64_t d = matrix.dimension;
      if (band && extents.size() == 1 && d >= kMinMultidiagonalExtent)
        if (auto banded = Banded::of(matrix, kMaxDiagonals))
          return createBanded(*banded);
      auto type = RankedTensorType::get({d, d, 2}, b.getF64Type());
      SmallVector<double> data = matrix.data();
      auto value = arith::ConstantOp::create(
          b, loc, DenseFPElementsAttr::get(type, ArrayRef<double>(data)));
      Value elementary = cudm::CreateElementaryOpOp::create(
          b, loc, elemOpTy, handle, value,
          cudm::SparsityAttr::get(b.getContext(), cudm::Sparsity::None),
          dtypeAttr, b.getDenseI64ArrayAttr(extents), FlatSymbolRefAttr(),
          DenseI32ArrayAttr());
      allElementaryOps.push_back(elementary);
      return elementary;
    };
    // A leaf's elementary operator and that of its adjoint (the same one
    // for a Hermitian leaf), made on first use.
    DenseMap<Operation *, Value> leafElementary, leafAdjointElementary;
    auto elementaryOf = [&](Operation *op) -> Value {
      Value &elementary = leafElementary[op];
      if (!elementary) {
        const Leaf &leaf = leaves.find(op)->second;
        elementary = createElementary(leaf.matrix, {leaf.matrix.dimension});
      }
      return elementary;
    };
    auto adjointElementaryOf = [&](Operation *op) -> Value {
      Value &adjoint = leafAdjointElementary[op];
      if (!adjoint) {
        const Leaf &leaf = leaves.find(op)->second;
        adjoint =
            leaf.adjoint == leaf.matrix
                ? elementaryOf(op)
                : createElementary(leaf.adjoint, {leaf.adjoint.dimension});
      }
      return adjoint;
    };

    // ── 6. Products ─────────────────────────────────────────────────────────
    llvm::MapVector<Operation *, Product> products;
    bool conversionFailed = false;
    module.walk([&](qop::MakeProductOp op) {
      Product product;
      if (auto scalar =
              op.getCoefficient().getDefiningOp<qop::ConstScalarOp>()) {
        product.coefficient = Complex(scalar.getReal().convertToDouble(),
                                      scalar.getImag().convertToDouble());
      } else if (auto scalar = op.getCoefficient()
                                   .getDefiningOp<qop::CallbackScalarOp>()) {
        product.callback = scalar;
      } else {
        op.emitError("coefficient must be a qop constant or callback");
        conversionFailed = true;
        return;
      }

      for (Value factor : op.getFactors()) {
        Operation *leaf = factor.getDefiningOp();
        if (!isa_and_nonnull<qop::SpinOp, qop::MatrixLeafOp>(leaf)) {
          op.emitError(
              "only qop.spin and qop.matrix_leaf factors are supported");
          conversionFailed = true;
          return;
        }
        auto found = leaves.find(leaf);
        if (found == leaves.end() || found->second.mode < 0 ||
            found->second.mode >= quditCount) {
          op.emitError("operator target must be a valid constant mode "
                       "index");
          conversionFailed = true;
          return;
        }
        product.leaves.push_back(leaf);
        product.modes.push_back(found->second.mode);
      }
      if (product.leaves.empty()) {
        op.emitError("operator product must contain at least one factor");
        conversionFailed = true;
        return;
      }
      products[op.getOperation()] = std::move(product);
    });
    if (conversionFailed)
      return signalPassFailure();

    SmallVector<Value> allOperatorTerms;
    auto createTerm =
        [&](ArrayRef<Value> elementaryOps, ArrayRef<int32_t> modes,
            ArrayRef<int32_t> dualities, Complex coefficient,
            qop::CallbackScalarOp callback = {},
            ArrayRef<qop::CallbackScalarOp> summed = {}) -> Value {
      auto term = cudm::CreateOpTermOp::create(b, loc, opTermTy, handle,
                                               modeExtentsAttr);
      FlatSymbolRefAttr callbackAttr;
      if (callback)
        callbackAttr = callback.getCallbackAttr();
      auto append = cudm::AppendElementaryProductOp::create(
          b, loc, handle, term, elementaryOps, b.getDenseI32ArrayAttr(modes),
          b.getDenseI32ArrayAttr(dualities),
          b.getF64FloatAttr(coefficient.real()),
          b.getF64FloatAttr(coefficient.imag()), callbackAttr);
      if (callback) {
        if (auto kind = callback->getAttr("cudm.callback_kind"))
          append->setAttr("cudm.callback_kind", kind);
        if (auto parameters = callback->getAttr("cudm.callback_params"))
          append->setAttr("cudm.callback_params", parameters);
      }
      if (!summed.empty()) {
        SmallVector<double> parameters = {static_cast<double>(summed.size())};
        for (qop::CallbackScalarOp member : summed) {
          auto memberParameters =
              member->getAttrOfType<DenseF64ArrayAttr>("cudm.callback_params")
                  .asArrayRef();
          parameters.push_back(static_cast<double>(
              member->getAttrOfType<IntegerAttr>("cudm.callback_kind")
                  .getInt()));
          parameters.push_back(static_cast<double>(memberParameters.size()));
          parameters.append(memberParameters.begin(), memberParameters.end());
        }
        append->setAttr("cudm.callback_kind",
                        b.getI32IntegerAttr(cudaq_pulse::kSumKind));
        append->setAttr("cudm.callback_params",
                        b.getDenseF64ArrayAttr(parameters));
      }
      allOperatorTerms.push_back(term);
      return term;
    };
    auto appendTerm = [&](Value term, int32_t duality, Complex coefficient) {
      cudm::OperatorAppendTermOp::create(
          b, loc, handle, compositeOp, term, b.getI32IntegerAttr(duality),
          b.getF64FloatAttr(coefficient.real()),
          b.getF64FloatAttr(coefficient.imag()), FlatSymbolRefAttr());
    };
    // A product as a ket-side term of its leaves, made on first use.
    DenseMap<Operation *, Value> productTerm;
    auto termOf = [&](Operation *op) -> Value {
      Value &term = productTerm[op];
      if (!term) {
        const Product &product = products.find(op)->second;
        SmallVector<Value> elementaryOps;
        for (Operation *leaf : product.leaves)
          elementaryOps.push_back(elementaryOf(leaf));
        SmallVector<int32_t> ketDualities(elementaryOps.size(), 0);
        term = createTerm(elementaryOps, product.modes, ketDualities,
                          product.coefficient, product.callback);
      }
      return term;
    };
    // The matrix of a product with distinct modes on `modes` (ascending),
    // its factors applied in order.
    auto matrixOf = [&](const Product &product, ArrayRef<int32_t> modes) {
      Matrix result = Matrix::identity(dimensionOf(modes, modeExtents));
      for (Operation *op : product.leaves) {
        const Leaf &leaf = leaves.find(op)->second;
        result = embed(leaf.matrix, {leaf.mode}, modes, modeExtents) * result;
      }
      return result * product.coefficient;
    };

    if (!fuse) {
      // Every leaf and product in program order, as one term each.
      for (auto &[op, leaf] : leaves) {
        elementaryOf(op);
        adjointElementaryOf(op);
      }
      for (auto &entry : products)
        termOf(entry.first);
    }

    // ── The generator's products; appending a dissipator ────────────────────
    auto hamiltonian =
        lindblad.getHamiltonian().getDefiningOp<qop::MakeSumOp>();
    if (!hamiltonian) {
      lindblad.emitError("Hamiltonian must be represented by qop.make_sum");
      return signalPassFailure();
    }
    SmallVector<Operation *> hamiltonianProducts;
    for (Value product : hamiltonian.getTerms()) {
      if (!products.count(product.getDefiningOp())) {
        hamiltonian.emitError("contains an unsupported product term");
        return signalPassFailure();
      }
      hamiltonianProducts.push_back(product.getDefiningOp());
    }
    // Pulse-generated collapse operators contain one product each. Reject a
    // sum here rather than silently dropping the required cross terms.
    SmallVector<Operation *> collapseProducts;
    for (Value collapse : lindblad.getCollapseOps()) {
      auto sum = collapse.getDefiningOp<qop::MakeSumOp>();
      if (!sum || sum.getTerms().size() != 1) {
        lindblad.emitError(
            "each collapse operator must contain exactly one product");
        return signalPassFailure();
      }
      auto found = products.find(sum.getTerms().front().getDefiningOp());
      if (found == products.end() || found->second.callback) {
        lindblad.emitError("collapse products must have constant coefficients");
        return signalPassFailure();
      }
      collapseProducts.push_back(found->first);
    }

    // D[C](rho) = C rho C^dagger - 1/2 {C^dagger C, rho}, as the two-sided
    // C rho C^dagger term and, unless `withNorm` is false, the
    // anticommutator.
    auto appendDissipator = [&](Operation *op, bool withNorm) {
      const Product &product = products.find(op)->second;
      const double coefficientNorm = std::norm(product.coefficient);
      SmallVector<Value> jumpOps;
      for (Operation *leaf : product.leaves)
        jumpOps.push_back(elementaryOf(leaf));
      SmallVector<int32_t> jumpModes(product.modes.begin(),
                                     product.modes.end());
      // C rho C^dagger: the bra-side product is the reversed sequence of
      // adjoint factors, not a second copy of C. This distinction is
      // essential for non-Hermitian collapse operators such as sigma-minus.
      for (auto [leaf, mode] :
           llvm::reverse(llvm::zip(product.leaves, product.modes))) {
        jumpOps.push_back(adjointElementaryOf(leaf));
        jumpModes.push_back(mode);
      }
      SmallVector<int32_t> jumpDualities(product.leaves.size(), 0);
      jumpDualities.append(product.leaves.size(), 1);
      appendTerm(createTerm(jumpOps, jumpModes, jumpDualities, 1.0), 0,
                 coefficientNorm);
      if (!withNorm)
        return;
      // On one mode of enough levels, C^dagger C of single-diagonal factors
      // is diagonal: one multi-diagonal operator.
      std::optional<Banded> norm;
      const int32_t mode = product.modes[0];
      if (band && modeExtents[mode] >= kMinMultidiagonalExtent &&
          llvm::all_equal(product.modes)) {
        for (Operation *leaf : product.leaves) {
          auto factor = Banded::of(leaves.find(leaf)->second.matrix, 1);
          norm = factor && norm ? factor->times(*norm) : factor;
          if (!norm)
            break;
        }
        if (norm)
          norm = norm->adjoint().times(*norm);
      }
      if (norm) {
        auto normTerm = createTerm({createBanded(*norm)}, {mode}, {0}, 1.0);
        for (int32_t duality : {0, 1})
          appendTerm(normTerm, duality, -0.5 * coefficientNorm);
        return;
      }
      // cuDensityMat composes an elementary product in application order.
      // Its documented C * C.dag() construction is therefore represented by
      // the same factor sequence as the two-sided term, with every factor
      // acting on the ket side.
      SmallVector<int32_t> normDualities(jumpOps.size(), 0);
      auto normTerm = createTerm(jumpOps, jumpModes, normDualities, 1.0);
      for (int32_t duality : {0, 1})
        appendTerm(normTerm, duality, -0.5 * coefficientNorm);
    };

    // ── 7 and 8. Operator terms ─────────────────────────────────────────────
    if (!fuse) {
      // 7. -i[H,rho]. A pure-state evolution only needs the ket action; a
      // mixed-state evolution also needs the opposite-sign bra action.
      for (Operation *op : hamiltonianProducts) {
        appendTerm(termOf(op), 0, Complex(0.0, -1.0));
        if (hasDissipation)
          appendTerm(termOf(op), 1, Complex(0.0, 1.0));
      }
      // 8. The dissipators.
      for (Operation *op : collapseProducts)
        appendDissipator(op, true);
    } else {
      // 7. The constant Hamiltonian terms and the dissipators that act on
      // few enough levels become a few dense blocks: on both sides of the
      // density matrix (superoperators) where their dimension squared fits
      // maxFusedDimension, otherwise one-sided sums per group of modes.
      // Time-dependent terms stay out of the blocks; those of one operator
      // become one term (below).
      const int64_t cap = maxFusedDimension;
      auto fits = [&](ArrayRef<int32_t> modes, bool bothSides) {
        const int64_t d = dimensionOf(modes, modeExtents);
        return !modes.empty() && d <= cap && (!bothSides || d * d <= cap);
      };
      struct Group {
        Modes modes;
        SmallVector<size_t> members;
      };
      // The groups of `pieces` (each with its modes) by modes, a group
      // whose modes are a strict subset of another's folded into it.
      auto groupByModes = [](ArrayRef<Modes> pieces) {
        SmallVector<Group> groups;
        for (auto piece : llvm::enumerate(pieces)) {
          const size_t i = piece.index();
          const Modes &modes = piece.value();
          auto group = llvm::find_if(
              groups, [&](const Group &g) { return g.modes == modes; });
          if (group != groups.end()) {
            group->members.push_back(i);
            continue;
          }
          groups.push_back({modes, {i}});
        }
        auto strictSubset = [](const Modes &sub, const Modes &super) {
          return sub.size() < super.size() &&
                 std::includes(super.begin(), super.end(), sub.begin(),
                               sub.end());
        };
        SmallVector<size_t> targets;
        for (auto candidate : llvm::enumerate(groups)) {
          const Group &group = candidate.value();
          if (llvm::none_of(groups, [&](const Group &other) {
                return strictSubset(group.modes, other.modes);
              }))
            targets.push_back(candidate.index());
        }
        for (Group &group : groups)
          for (size_t target : targets)
            if (strictSubset(group.modes, groups[target].modes)) {
              groups[target].members.append(group.members);
              group.members.clear();
              break;
            }
        SmallVector<Group> result;
        for (size_t target : targets)
          result.push_back(std::move(groups[target]));
        return result;
      };

      // ── Fused: two-sided pieces ───────────────────────────────────────────
      // rho -> ket rho, rho bra, or ket rho bra.
      struct SidedPiece {
        Modes modes;
        std::optional<Matrix> ket, bra;
      };
      SmallVector<SidedPiece> sided;
      SmallVector<Operation *> remainingHamiltonian, remainingCollapse;
      for (Operation *op : hamiltonianProducts) {
        const Product &product = products.find(op)->second;
        auto modes = product.distinctModes();
        if (product.callback || !modes || !hasDissipation ||
            !fits(*modes, true)) {
          remainingHamiltonian.push_back(op);
          continue;
        }
        Matrix matrix = matrixOf(product, *modes);
        sided.push_back({*modes, matrix * Complex(0.0, -1.0), std::nullopt});
        sided.push_back({*modes, std::nullopt, matrix * Complex(0.0, 1.0)});
      }
      for (Operation *op : collapseProducts) {
        const Product &product = products.find(op)->second;
        auto modes = product.distinctModes();
        if (!modes || !fits(*modes, true)) {
          remainingCollapse.push_back(op);
          continue;
        }
        Matrix jump = matrixOf(product, *modes);
        Matrix adjoint = jump.adjoint();
        Matrix norm = adjoint * jump * Complex(-0.5);
        sided.push_back({*modes, jump, adjoint});
        sided.push_back({*modes, norm, std::nullopt});
        sided.push_back({*modes, std::nullopt, norm});
      }
      if (!sided.empty()) {
        SmallVector<Modes> pieceModes;
        for (const SidedPiece &piece : sided)
          pieceModes.push_back(piece.modes);
        SmallVector<Group> groups = groupByModes(pieceModes);
        llvm::sort(groups, [](const Group &left, const Group &right) {
          return left.modes < right.modes;
        });
        // Windows: groups in mode order, each merged into the one before it
        // when the two share a mode and the union still fits.
        SmallVector<Group> windows;
        for (Group &group : groups) {
          if (!windows.empty()) {
            Group &last = windows.back();
            Modes merged(last.modes);
            merged.append(group.modes);
            llvm::sort(merged);
            merged.erase(std::unique(merged.begin(), merged.end()),
                         merged.end());
            if (merged.size() < last.modes.size() + group.modes.size() &&
                fits(merged, true)) {
              last.modes = std::move(merged);
              last.members.append(group.members);
              continue;
            }
          }
          windows.push_back(std::move(group));
        }
        // S = I (x) K + B (x) I + sum B_k (x) K_k for
        // rho -> K rho + rho B + sum K_k rho B_k, the ket index fastest.
        for (const Group &window : windows) {
          const int64_t d = dimensionOf(window.modes, modeExtents);
          Matrix ket = Matrix::zero(d), bra = Matrix::zero(d);
          Matrix superoperator = Matrix::zero(d * d);
          for (size_t member : window.members) {
            const SidedPiece &piece = sided[member];
            auto widen = [&](const Matrix &matrix) {
              return embed(matrix, piece.modes, window.modes, modeExtents);
            };
            if (piece.ket && piece.bra)
              superoperator += kron(widen(*piece.bra), widen(*piece.ket));
            else if (piece.ket)
              ket += widen(*piece.ket);
            else
              bra += widen(*piece.bra);
          }
          superoperator += kron(Matrix::identity(d), ket);
          superoperator += kron(bra, Matrix::identity(d));
          SmallVector<int64_t> extents;
          Modes modes;
          SmallVector<int32_t> dualities;
          for (int32_t duality : {0, 1})
            for (int32_t mode : window.modes) {
              extents.push_back(modeExtents[mode]);
              modes.push_back(mode);
              dualities.push_back(duality);
            }
          Value elementary = createElementary(superoperator, extents);
          appendTerm(createTerm({elementary}, modes, dualities, 1.0), 0, 1.0);
        }
      }

      // ── Fused: the dissipators' anticommutators ───────────────────────────
      // One-sided pieces: the remaining constant terms on few enough levels,
      // and the anticommutators of the remaining dissipators, summed per
      // group of modes on each side.
      struct OneSidedPiece {
        Modes modes;
        Matrix matrix;
        Operation *product = nullptr; // a lone single-factor term
      };
      SmallVector<OneSidedPiece> norms;
      DenseSet<Operation *> foldedNorms;
      for (Operation *op : remainingCollapse) {
        const Product &product = products.find(op)->second;
        auto modes = product.distinctModes();
        if (!modes || !fits(*modes, false))
          continue;
        Matrix jump = matrixOf(product, *modes);
        norms.push_back({*modes, jump.adjoint() * jump * Complex(-0.5)});
        foldedNorms.insert(op);
      }

      // ── Fused: time-dependent terms of one operator ───────────────────────
      // Such terms, for example the pulses one drive plays, become one term
      // whose callback sums theirs. A group is named by its first product; a
      // product the Hamiltonian lists twice is a member twice.
      DenseMap<Operation *, Operation *> groupOf;
      DenseMap<Operation *, SmallVector<Operation *>> membersOf;
      {
        // A callback whose kind and parameters a sum can list. Others, such
        // as one with parameters of another type, stay terms of their own.
        auto summable = [&](const Product &product) {
          if (!product.callback || product.coefficient != Complex(1.0))
            return false;
          auto kind = product.callback->getAttrOfType<IntegerAttr>(
              "cudm.callback_kind");
          return kind && kind.getInt() != cudaq_pulse::kSumKind &&
                 product.callback->getAttrOfType<DenseF64ArrayAttr>(
                     "cudm.callback_params");
        };
        auto sameOperator = [&](const Product &left, const Product &right) {
          return left.modes == right.modes &&
                 llvm::equal(left.leaves, right.leaves,
                             [&](Operation *x, Operation *y) {
                               return leaves.find(x)->second.matrix ==
                                      leaves.find(y)->second.matrix;
                             });
        };
        SmallVector<Operation *> firsts;
        for (Operation *op : remainingHamiltonian) {
          const Product &product = products.find(op)->second;
          if (!summable(product))
            continue;
          auto first = llvm::find_if(firsts, [&](Operation *other) {
            return sameOperator(products.find(other)->second, product);
          });
          if (first == firsts.end()) {
            firsts.push_back(op);
            first = firsts.end() - 1;
          }
          groupOf[op] = *first;
          membersOf[*first].push_back(op);
        }
      }
      // The term of a group of more than one member, whose callback sums
      // the members' callbacks, made on first use.
      DenseMap<Operation *, Value> summedTerm;
      auto summedTermOf = [&](Operation *group) -> Value {
        Value &term = summedTerm[group];
        if (!term) {
          const Product &product = products.find(group)->second;
          SmallVector<Value> elementaryOps;
          for (Operation *leaf : product.leaves)
            elementaryOps.push_back(elementaryOf(leaf));
          SmallVector<int32_t> ketDualities(elementaryOps.size(), 0);
          SmallVector<qop::CallbackScalarOp> callbacks;
          for (Operation *member : membersOf.find(group)->second)
            callbacks.push_back(products.find(member)->second.callback);
          term = createTerm(elementaryOps, product.modes, ketDualities, 1.0, {},
                            callbacks);
        }
        return term;
      };

      // ── Fused: one-sided sums and the terms left, per side ────────────────
      for (int32_t side : {0, 1}) {
        if (side == 1 && !hasDissipation)
          break;
        const Complex sign(0.0, side == 0 ? -1.0 : 1.0);
        SmallVector<OneSidedPiece> pieces(norms);
        SmallVector<Operation *> separate;
        for (Operation *op : remainingHamiltonian) {
          const Product &product = products.find(op)->second;
          auto modes = product.distinctModes();
          if (product.callback || !modes || !fits(*modes, false)) {
            separate.push_back(op);
            continue;
          }
          pieces.push_back({*modes, matrixOf(product, *modes) * sign,
                            product.leaves.size() == 1 ? op : nullptr});
        }
        SmallVector<Modes> pieceModes;
        for (const OneSidedPiece &piece : pieces)
          pieceModes.push_back(piece.modes);
        for (const Group &group : groupByModes(pieceModes)) {
          if (group.members.size() == 1 && pieces[group.members[0]].product) {
            separate.push_back(pieces[group.members[0]].product);
            continue;
          }
          Matrix sum = Matrix::zero(dimensionOf(group.modes, modeExtents));
          for (size_t member : group.members)
            sum += embed(pieces[member].matrix, pieces[member].modes,
                         group.modes, modeExtents);
          SmallVector<int64_t> extents;
          for (int32_t mode : group.modes)
            extents.push_back(modeExtents[mode]);
          Value elementary = createElementary(sum, extents);
          SmallVector<int32_t> ketDualities(group.modes.size(), 0);
          appendTerm(createTerm({elementary}, group.modes, ketDualities, 1.0),
                     side, 1.0);
        }
        // Each listing of a product appends its term, except that a group of
        // more than one member appends its summed term once per side, where
        // its first member is listed.
        DenseSet<Operation *> summedOnSide;
        for (Operation *op : separate) {
          auto found = groupOf.find(op);
          if (found == groupOf.end() ||
              membersOf.find(found->second)->second.size() == 1)
            appendTerm(termOf(op), side, sign);
          else if (summedOnSide.insert(found->second).second)
            appendTerm(summedTermOf(found->second), side, sign);
        }
      }
      for (Operation *op : remainingCollapse)
        appendDissipator(op, !foldedNorms.contains(op));
    }

    // ── 9. Evolve ───────────────────────────────────────────────────────────
    double tStart = 0.0, tEnd = 100.0;
    int64_t numSteps = 100;
    auto tsAttr = module->getAttrOfType<FloatAttr>("qop.t_start");
    if (!tsAttr)
      tsAttr = module->getAttrOfType<FloatAttr>("t_start");
    if (tsAttr)
      tStart = tsAttr.getValueAsDouble();
    auto teAttr = module->getAttrOfType<FloatAttr>("qop.t_end");
    if (!teAttr)
      teAttr = module->getAttrOfType<FloatAttr>("t_end");
    if (teAttr)
      tEnd = teAttr.getValueAsDouble();
    auto nsAttr = module->getAttrOfType<IntegerAttr>("qop.step_count");
    if (!nsAttr)
      nsAttr = module->getAttrOfType<IntegerAttr>("step_count");
    if (nsAttr)
      numSteps = nsAttr.getInt();

    auto integrator = cudm::IntegratorKind::RungeKutta4;
    if (auto value = module->getAttrOfType<StringAttr>("qop.integrator")) {
      auto parsed = cudm::symbolizeIntegratorKind(value.getValue());
      if (!parsed) {
        module.emitError("unsupported qop.integrator: ") << value.getValue();
        return signalPassFailure();
      }
      integrator = *parsed;
    }
    auto integratorAttr =
        cudm::IntegratorKindAttr::get(b.getContext(), integrator);

    cudm::EvolveOp::create(
        b, loc, stateTy, handle, compositeOp, stateIn, stateOut, workspace,
        integratorAttr, b.getF64FloatAttr(tStart), b.getF64FloatAttr(tEnd),
        b.getI64IntegerAttr(numSteps), cudm::ComputeTypeAttr());

    // ── 10. Cleanup ─────────────────────────────────────────────────────────
    cudm::DestroyOperatorOp::create(b, loc, compositeOp);
    for (Value term : llvm::reverse(allOperatorTerms))
      cudm::DestroyOpTermOp::create(b, loc, term);
    for (Value elementaryOp : llvm::reverse(allElementaryOps))
      cudm::DestroyElementaryOpOp::create(b, loc, elementaryOp);
    cudm::DestroyWorkspaceOp::create(b, loc, workspace);
    cudm::DestroyStateOp::create(b, loc, stateOut);
    cudm::DestroyStateOp::create(b, loc, stateIn);
    cudm::DestroyHandleOp::create(b, loc, handle);

    // The cuDensityMat graph no longer depends on the source QOp graph.
    SmallVector<Operation *> qopOps;
    module.walk([&](Operation *op) {
      if (op->getName().getDialectNamespace() == "qop")
        qopOps.push_back(op);
    });
    for (auto iter = qopOps.rbegin(); iter != qopOps.rend(); ++iter)
      (*iter)->erase();
  }
};

} // namespace

namespace qop {

std::unique_ptr<mlir::Pass> createQOpToCuDensityMatPass() {
  return std::make_unique<QOpToCuDensityMatPass>();
}

} // namespace qop
