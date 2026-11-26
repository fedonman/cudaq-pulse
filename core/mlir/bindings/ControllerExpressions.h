// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// The expressions of what a controller program computes at run time, in
// the export's closed vocabulary: the walks of the controller export and
// of the check (bindings.cpp) record their run-time values with them.
// Included by bindings.cpp inside its anonymous namespace.

#pragma once

#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

struct ControllerExpressions {
  static bool same(double a, double b) {
    return a == b || (std::isnan(a) && std::isnan(b));
  }

  static constexpr double kUnknown = std::numeric_limits<double>::quiet_NaN();

  /// A number computed from constants alone, else NaN.
  static double numberOf(mlir::Value value) {
    if (auto constant = pulse::constantNumber(value))
      return *constant;
    if (auto affine = pulse::affineNumber(value);
        affine && affine->isConstant())
      return affine->evaluate({});
    return kUnknown;
  }

  /// numberOf(), once per value and per operation it follows: the walk
  /// asks again at every execution and at every node of an expression.
  double number(mlir::Value value) {
    auto [found, inserted] = numbers.try_emplace(value);
    if (inserted) {
      found->second = kUnknown;
      if (auto constant = pulse::constantNumber(value))
        found->second = *constant;
      else if (auto affine = pulse::affineNumber(value, affines);
               affine && affine->isConstant())
        found->second = affine->evaluate({});
    }
    return found->second;
  }
  // What the arithmetic of the program computes of its arguments.
  pulse::AffineNumbers affines;
  llvm::DenseMap<mlir::Value, double> numbers;

  // The record of the latest execution of each readout and classical wait,
  // which an expression of its result refers to.
  llvm::DenseMap<mlir::Operation *, int64_t> latest;

  /// How a classical wait takes its answer: "blocking", "timeout" with its
  /// own timeout or, without one, its function's in timeout mode, else
  /// "bounded"; and the timeout in ns (NaN for none).
  std::pair<const char *, double> waitMode(mlir::Operation *op) {
    if (op->hasAttr("classical.blocking"))
      return {"blocking", kUnknown};
    if (auto timeout = op->getAttrOfType<mlir::FloatAttr>("timeout_ns"))
      return {"timeout", timeout.getValueAsDouble()};
    auto callee = op->getAttrOfType<mlir::FlatSymbolRefAttr>("callee");
    if (!callee)
      return {"bounded", kUnknown};
    auto [found, inserted] = callables.try_emplace(callee.getAttr());
    if (inserted)
      found->second =
          mlir::SymbolTable::lookupNearestSymbolFrom<classical::CallableOp>(
              op, callee);
    classical::CallableOp callable = found->second;
    if (callable && callable.getMode() == "timeout" && callable.getTimeoutNs())
      return {"timeout", callable.getTimeoutNs()->convertToDouble()};
    return {"bounded", kUnknown};
  }
  llvm::DenseMap<mlir::Attribute, classical::CallableOp> callables;

  /// How a value is computed at run time, children before their parent: a
  /// number (a frame change's value), a constant, the result of a readout
  /// or answer record, an op of the export's vocabulary (`name`) over the
  /// expressions `of` of one `type`, or what the vocabulary has no term
  /// for (`name`: an op's MLIR name, else what it is), which no export
  /// gives.
  struct Expression {
    enum Kind : uint8_t { Number, Constant, Unknown, Readout, Answer, Op };
    enum Type : uint8_t { Int, Float, Bool, Types };
    // The terms of the export's expression vocabulary, in `names` order.
    enum Term : uint8_t {
      Add,
      Sub,
      Mul,
      Div,
      FloorDiv,
      Rem,
      Neg,
      Max,
      Min,
      Eq,
      Ne,
      Lt,
      Le,
      Gt,
      Ge,
      Select,
      ToFloat,
      ToInt,
      Terms
    };
    static constexpr const char *names[Terms] = {
        "add", "sub", "mul", "div",    "floor_div", "rem",
        "neg", "max", "min", "eq",     "ne",        "lt",
        "le",  "gt",  "ge",  "select", "to_float",  "to_int"};
    static constexpr const char *typeNames[Types] = {"int", "float", "bool"};
    Kind kind = Number;
    Type type = Float;
    Term term = Add;
    double number = 0.0;
    int64_t event = 0, result = 0;
    llvm::StringRef name; // what an Unknown is computed from
    // Whether several expressions may have this one in their `of`: a walk
    // that changes expressions, or compares them, enters it once.
    bool tabled = false;
    // Whether no term of this expression is used twice or is many terms
    // deep: the export writes it in place, whole.
    bool plain = false;
    std::vector<int32_t> of;
  };
  std::vector<Expression> expressions;

  // While one expression is made, the expression of each value with several
  // uses, so that the value's operation is a term once however many terms
  // use it, and of each value made on its own (see expression()).
  llvm::DenseMap<mlir::Value, int32_t> made;
  // In one attempt: the values `made` got, those of them made whole (over
  // no operand not yet made), and the operands not yet made.
  llvm::SmallVector<mlir::Value> entered, whole, missing;
  // How many terms deep an expression is made by calls of its terms.
  static constexpr int kInline = 48;
  // How many terms deep the text of an expression nests at most: a term
  // that deep is named. It is the deepest an expression was ever written.
  static constexpr int kNested = 65;
  // Expressions by their place in `expressions`: the one map of the walks
  // over them.
  using Places = llvm::SmallDenseMap<int32_t, int32_t, 8>;

  int32_t add(Expression expression) {
    expressions.push_back(std::move(expression));
    return static_cast<int32_t>(expressions.size()) - 1;
  }

  int32_t add(Expression::Kind kind, double number, int64_t event = 0,
              int64_t result = 0, llvm::StringRef name = {}) {
    Expression expression;
    expression.kind = kind;
    expression.number = number;
    expression.event = event;
    expression.result = result;
    expression.name = name;
    return add(std::move(expression));
  }

  int32_t constant(double number, Expression::Type type) {
    int32_t index = add(Expression::Constant, number);
    expressions[index].type = type;
    return index;
  }

  int32_t op(Expression::Term term, Expression::Type type,
             std::vector<int32_t> of) {
    Expression expression;
    expression.kind = Expression::Op;
    expression.type = type;
    expression.term = term;
    expression.of = std::move(of);
    return add(std::move(expression));
  }

  static std::optional<Expression::Type> valueType(mlir::Type type) {
    if (type.isInteger(1))
      return Expression::Bool;
    if (type.isInteger(64) || type.isIndex())
      return Expression::Int;
    if (type.isF64())
      return Expression::Float;
    return std::nullopt;
  }

  static std::optional<Expression::Term>
  comparison(mlir::arith::CmpIPredicate predicate) {
    using P = mlir::arith::CmpIPredicate;
    switch (predicate) {
    case P::eq:
      return Expression::Eq;
    case P::ne:
      return Expression::Ne;
    case P::slt:
      return Expression::Lt;
    case P::sle:
      return Expression::Le;
    case P::sgt:
      return Expression::Gt;
    case P::sge:
      return Expression::Ge;
    default:
      return std::nullopt;
    }
  }

  // Ordered, but for ne: a NaN compares unequal and in no order.
  static std::optional<Expression::Term>
  comparison(mlir::arith::CmpFPredicate predicate) {
    using P = mlir::arith::CmpFPredicate;
    switch (predicate) {
    case P::OEQ:
      return Expression::Eq;
    case P::UNE:
      return Expression::Ne;
    case P::OLT:
      return Expression::Lt;
    case P::OLE:
      return Expression::Le;
    case P::OGT:
      return Expression::Gt;
    case P::OGE:
      return Expression::Ge;
    default:
      return std::nullopt;
    }
  }

  /// How `value` is computed at run time: a constant, a result of a readout
  /// or answer record, or a term of the export's vocabulary over such
  /// expressions (see term()), however many terms deep. The terms are made
  /// by calls kInline deep; an operand deeper is made on its own first, and
  /// the term that asked for it again.
  int32_t expression(mlir::Value value) {
    // The table is of one expression: the records its leaves refer to are
    // the latest when it is made.
    if (!made.empty())
      made.clear();
    entered.clear();
    whole.clear();
    missing.clear();
    size_t mark = expressions.size();
    int32_t index = make(value, 0);
    if (missing.empty()) {
      if (made.empty())
        expressions[index].plain = true;
      return index;
    }
    return deep(value, mark);
  }

  /// expression() of a value with operands not yet made: the attempt that
  /// began at `mark` is undone, and each operand is made on its own first.
  LLVM_ATTRIBUTE_NOINLINE int32_t deep(mlir::Value value, size_t mark) {
    llvm::SmallVector<mlir::Value> todo{value};
    for (;;) {
      // What the attempt made is not kept: it is over operands not made.
      // What it made whole is made on its own next, and kept, so that no
      // later attempt makes it again.
      expressions.resize(mark);
      for (mlir::Value attempted : entered)
        made.erase(attempted);
      todo.append(missing.begin(), missing.end());
      todo.append(whole.begin(), whole.end());
      for (;;) {
        mlir::Value next = todo.back();
        if (todo.size() > 1 && made.count(next)) {
          todo.pop_back();
          continue;
        }
        entered.clear();
        whole.clear();
        missing.clear();
        mark = expressions.size();
        int32_t index = make(next, 0);
        if (!missing.empty())
          break;
        if (todo.size() == 1)
          return index;
        // The term that asked may be made more than once.
        expressions[index].tabled = true;
        made[next] = index;
        todo.pop_back();
      }
    }
  }

  /// expression(), `depth` calls of terms deep.
  int32_t make(mlir::Value value, int depth) {
    double number = this->number(value);
    if (!std::isnan(number)) {
      if (value.getType().isInteger(1))
        return constant(number != 0.0, Expression::Bool);
      return constant(number, Expression::Float);
    }
    mlir::Operation *def = value.getDefiningOp();
    if (!def)
      return add(Expression::Unknown, 0.0, 0, 0, "a value the kernel is given");
    if (mlir::isa<pulse::ReadoutOp, classical::WaitOp>(def)) {
      auto found = latest.find(def);
      if (found == latest.end())
        return add(Expression::Unknown, 0.0, 0, 0,
                   "a readout or answer the program has not reached");
      if (mlir::isa<pulse::ReadoutOp>(def))
        return add(Expression::Readout, 0.0, found->second);
      return add(Expression::Answer, 0.0, found->second,
                 static_cast<int64_t>(
                     mlir::cast<mlir::OpResult>(value).getResultNumber()));
    }
    if (def->getNumRegions() != 0 || def->getNumResults() != 1)
      return add(Expression::Unknown, 0.0, 0, 0, def->getName().getStringRef());
    bool several = !value.hasOneUse();
    if (several || depth >= kInline) {
      if (auto found = made.find(value); found != made.end())
        return found->second;
      if (depth >= kInline) {
        missing.push_back(value);
        return add(Expression::Unknown, 0.0, 0, 0, "an operand not yet made");
      }
    }
    size_t missed = missing.size();
    int32_t index = term(def, depth);
    if (several) {
      expressions[index].tabled = true;
      made[value] = index;
      entered.push_back(value);
      if (missing.size() == missed)
        whole.push_back(value);
    }
    return index;
  }

  /// Calls `leaf` on each readout and answer leaf of expression `root`,
  /// once: a term several others use is entered once.
  template <typename Leaf>
  void eachRecord(int32_t root, Leaf leaf) {
    llvm::SmallVector<int32_t, 16> todo{root};
    Places entered;
    while (!todo.empty()) {
      int32_t index = todo.pop_back_val();
      Expression &e = expressions[index];
      if (e.tabled && !entered.try_emplace(index, 0).second)
        continue;
      if (e.kind == Expression::Readout || e.kind == Expression::Answer)
        leaf(e);
      todo.append(e.of.begin(), e.of.end());
    }
  }

  /// The term of the export's vocabulary `def` computes, over its operands'
  /// expressions; forms that compute the same value give the same term.
  int32_t term(mlir::Operation *def, int depth) {
    namespace arith = mlir::arith;
    auto of = [&](mlir::Value value) { return make(value, depth + 1); };
    auto unknown = [&] {
      return add(Expression::Unknown, 0.0, 0, 0, def->getName().getStringRef());
    };
    // The export counts times in cycles, which a duration holds.
    if (mlir::isa<pulse::DurationFromIntOp>(def))
      return of(def->getOperand(0));
    // A readout's leaf is the level it observed.
    if (mlir::isa<pulse::OutcomeOp>(def))
      return of(def->getOperand(0));
    // A readout's outcome is set when it observed a level other than 0.
    if (mlir::isa<pulse::IsSetOp>(def)) {
      int32_t outcome = of(def->getOperand(0));
      return op(Expression::Ne, Expression::Int,
                {outcome, constant(0.0, Expression::Int)});
    }
    std::optional<Expression::Type> result =
        valueType(def->getResult(0).getType());
    if (!result || def->getNumOperands() == 0)
      return unknown();
    std::optional<Expression::Type> type =
        valueType(def->getOperand(0).getType());
    if (!type)
      return unknown();
    using T = Expression::Type;
    struct Simple {
      mlir::TypeID op;
      Expression::Term term;
      T type, result;
    };
    static const Simple simple[] = {
        {mlir::TypeID::get<arith::AddFOp>(), Expression::Add, T::Float,
         T::Float},
        {mlir::TypeID::get<arith::MulFOp>(), Expression::Mul, T::Float,
         T::Float},
        {mlir::TypeID::get<arith::FPToSIOp>(), Expression::ToInt, T::Float,
         T::Int},
        {mlir::TypeID::get<arith::SIToFPOp>(), Expression::ToFloat, T::Int,
         T::Float},
        {mlir::TypeID::get<arith::SubFOp>(), Expression::Sub, T::Float,
         T::Float},
        {mlir::TypeID::get<arith::DivFOp>(), Expression::Div, T::Float,
         T::Float},
        {mlir::TypeID::get<arith::NegFOp>(), Expression::Neg, T::Float,
         T::Float},
        {mlir::TypeID::get<arith::AddIOp>(), Expression::Add, T::Int, T::Int},
        {mlir::TypeID::get<arith::MulIOp>(), Expression::Mul, T::Int, T::Int},
        {mlir::TypeID::get<arith::DivSIOp>(), Expression::Div, T::Int, T::Int},
        {mlir::TypeID::get<arith::FloorDivSIOp>(), Expression::FloorDiv, T::Int,
         T::Int},
        {mlir::TypeID::get<arith::RemSIOp>(), Expression::Rem, T::Int, T::Int},
        {mlir::TypeID::get<arith::MaxSIOp>(), Expression::Max, T::Int, T::Int},
        {mlir::TypeID::get<arith::MinSIOp>(), Expression::Min, T::Int, T::Int},
    };
    mlir::TypeID id = def->getName().getTypeID();
    for (const Simple &s : simple) {
      if (s.op != id)
        continue;
      if (*type != s.type || *result != s.result)
        return unknown();
      std::vector<int32_t> operands;
      for (mlir::Value operand : def->getOperands()) {
        if (valueType(operand.getType()) != s.type)
          return unknown();
        operands.push_back(of(operand));
      }
      return op(s.term, s.type, std::move(operands));
    }
    // Integer subtraction from 0 is negation.
    if (mlir::isa<arith::SubIOp>(def)) {
      if (*type != T::Int || *result != T::Int)
        return unknown();
      if (this->number(def->getOperand(0)) == 0.0)
        return op(Expression::Neg, T::Int, {of(def->getOperand(1))});
      int32_t lhs = of(def->getOperand(0));
      return op(Expression::Sub, T::Int, {lhs, of(def->getOperand(1))});
    }
    if (auto select = mlir::dyn_cast<arith::SelectOp>(def)) {
      int32_t extremum = minMax(select, *result, depth);
      if (extremum >= 0)
        return extremum;
      // A choice of bool constants is the condition or its negation.
      double then = number(select.getTrueValue()),
             other = number(select.getFalseValue());
      if (*result == T::Bool && !std::isnan(then) && !std::isnan(other) &&
          (then != 0.0) != (other != 0.0))
        return equality(true, select.getCondition(), select.getTrueValue(),
                        depth);
      int32_t condition = of(select.getCondition());
      int32_t chosen = of(select.getTrueValue());
      return op(Expression::Select, *result,
                {condition, chosen, of(select.getFalseValue())});
    }
    if (mlir::isa<arith::AndIOp, arith::OrIOp, arith::XOrIOp>(def)) {
      if (*type != T::Bool || *result != T::Bool)
        return unknown();
      return logic(def, depth);
    }
    if (auto cmp = mlir::dyn_cast<arith::CmpIOp>(def)) {
      using P = arith::CmpIPredicate;
      if (*type == T::Bool &&
          (cmp.getPredicate() == P::eq || cmp.getPredicate() == P::ne))
        return equality(cmp.getPredicate() == P::eq, cmp.getLhs(), cmp.getRhs(),
                        depth);
      std::optional<Expression::Term> term = comparison(cmp.getPredicate());
      if (!term || *type == T::Bool)
        return unknown();
      return compared(*term, *type, cmp.getLhs(), cmp.getRhs(), depth);
    }
    if (auto cmp = mlir::dyn_cast<arith::CmpFOp>(def)) {
      std::optional<Expression::Term> term = comparison(cmp.getPredicate());
      if (!term)
        return unknown();
      return compared(*term, *type, cmp.getLhs(), cmp.getRhs(), depth);
    }
    // A bool as an integer: 1, or -1 sign-extended, when true, else 0.
    if (mlir::isa<arith::ExtUIOp, arith::ExtSIOp>(def)) {
      if (*type != T::Bool || *result != T::Int)
        return unknown();
      int32_t condition = of(def->getOperand(0));
      double set = mlir::isa<arith::ExtUIOp>(def) ? 1.0 : -1.0;
      int32_t then = constant(set, T::Int);
      return op(Expression::Select, T::Int,
                {condition, then, constant(0.0, T::Int)});
    }
    return unknown();
  }

  /// Bool logic `def` computes: a and b is select(a, b, false), a or b is
  /// select(a, true, b), and exclusive or is ne.
  LLVM_ATTRIBUTE_NOINLINE int32_t logic(mlir::Operation *def, int depth) {
    using T = Expression::Type;
    mlir::Value x = def->getOperand(0), y = def->getOperand(1);
    if (mlir::isa<mlir::arith::XOrIOp>(def))
      return equality(false, x, y, depth);
    bool both = mlir::isa<mlir::arith::AndIOp>(def);
    if (!std::isnan(number(x)))
      std::swap(x, y);
    // With a constant: a and true is a, a or false is a, and the others
    // the constant.
    if (double c = number(y); !std::isnan(c))
      return (c != 0.0) == both ? make(x, depth + 1)
                                : constant(c != 0.0, T::Bool);
    int32_t a = make(x, depth + 1), b = make(y, depth + 1);
    if (both)
      return op(Expression::Select, T::Bool, {a, b, constant(0.0, T::Bool)});
    return op(Expression::Select, T::Bool, {a, constant(1.0, T::Bool), b});
  }

  /// Comparison `term` of `x` with `y` in `type`, a constant second: c < x
  /// is x > c, for integers and, as neither holds of NaN, for floats.
  LLVM_ATTRIBUTE_NOINLINE int32_t compared(Expression::Term term,
                                           Expression::Type type, mlir::Value x,
                                           mlir::Value y, int depth) {
    if (!std::isnan(number(x)) && std::isnan(number(y))) {
      std::swap(x, y);
      switch (term) {
      case Expression::Lt:
        term = Expression::Gt;
        break;
      case Expression::Le:
        term = Expression::Ge;
        break;
      case Expression::Gt:
        term = Expression::Lt;
        break;
      case Expression::Ge:
        term = Expression::Le;
        break;
      default:
        break;
      }
    }
    int32_t lhs = make(x, depth + 1);
    return op(term, type, {lhs, make(y, depth + 1)});
  }

  /// Makes comparison `e` its negation where that is a comparison too, and
  /// says whether it did: not (a < b) is a >= b for integers, and for
  /// floats, where NaN is in no order, only eq and ne negate each other.
  static bool negate(Expression &e) {
    if (e.kind != Expression::Op)
      return false;
    using X = Expression;
    switch (e.term) {
    case X::Eq:
      e.term = X::Ne;
      return true;
    case X::Ne:
      e.term = X::Eq;
      return true;
    case X::Lt:
    case X::Le:
    case X::Gt:
    case X::Ge:
      if (e.type != X::Int)
        return false;
      e.term = e.term == X::Lt   ? X::Ge
               : e.term == X::Le ? X::Gt
               : e.term == X::Gt ? X::Le
                                 : X::Lt;
      return true;
    default:
      return false;
    }
  }

  /// Whether bools `x` and `y` are equal (`equal`), else whether they
  /// differ. Against a constant that is the other bool itself or its
  /// negation, ne(it, true).
  LLVM_ATTRIBUTE_NOINLINE int32_t equality(bool equal, mlir::Value x,
                                           mlir::Value y, int depth) {
    double first = number(x), second = number(y);
    if (!std::isnan(first)) {
      std::swap(x, y);
      std::swap(first, second);
    }
    int32_t lhs = make(x, depth + 1);
    if (!std::isnan(second)) {
      if ((second != 0.0) == equal)
        return lhs;
      // The negation of a term others use is a term of its own, over the
      // same operands.
      if (expressions[lhs].tabled) {
        Expression negated = expressions[lhs];
        negated.tabled = false;
        if (negate(negated)) {
          for (int32_t operand : negated.of)
            expressions[operand].tabled = true;
          return add(std::move(negated));
        }
      } else if (negate(expressions[lhs]))
        return lhs;
      return op(Expression::Ne, Expression::Bool,
                {lhs, constant(1.0, Expression::Bool)});
    }
    return op(equal ? Expression::Eq : Expression::Ne, Expression::Bool,
              {lhs, make(y, depth + 1)});
  }

  /// `select` as the max or min of the values its condition compares,
  /// when it is that for every value they take, else -1: max(a, b) is a
  /// when a > b, else b, so select(a > b, a, b) is max(a, b) and
  /// select(a < b, b, a) is max(b, a), and integers also for >= and <=.
  /// The max and min of integers do not depend on the order of their
  /// operands, which keep the comparison's.
  int32_t minMax(mlir::arith::SelectOp select, Expression::Type type,
                 int depth) {
    namespace arith = mlir::arith;
    mlir::Value a, b;
    bool greater;
    if (auto cmp = select.getCondition().getDefiningOp<arith::CmpIOp>()) {
      using P = arith::CmpIPredicate;
      P predicate = cmp.getPredicate();
      if (type != Expression::Int ||
          (predicate != P::sgt && predicate != P::sge && predicate != P::slt &&
           predicate != P::sle))
        return -1;
      greater = predicate == P::sgt || predicate == P::sge;
      a = cmp.getLhs();
      b = cmp.getRhs();
    } else if (auto cmp =
                   select.getCondition().getDefiningOp<arith::CmpFOp>()) {
      using P = arith::CmpFPredicate;
      P predicate = cmp.getPredicate();
      if (type != Expression::Float ||
          (predicate != P::OGT && predicate != P::OLT))
        return -1;
      greater = predicate == P::OGT;
      a = cmp.getLhs();
      b = cmp.getRhs();
    } else {
      return -1;
    }
    mlir::Value then = select.getTrueValue(), other = select.getFalseValue();
    if (then == b && other == a) {
      greater = !greater;
      if (type != Expression::Int)
        std::swap(a, b);
    } else if (then != a || other != b)
      return -1;
    int32_t first = make(a, depth + 1);
    return op(greater ? Expression::Max : Expression::Min, type,
              {first, make(b, depth + 1)});
  }

  /// Whether expression `y` is `x` one iteration on: records [a, b) moved
  /// to [b, 2b - a), the rest the same.
  bool sameExpression(int32_t x, int32_t y, int64_t a, int64_t b) const {
    return sameExpression(x, y, a, b, b - a);
  }

  /// Whether expression `y` is `x` with the records it refers to in
  /// [low, high) `shift` later, the rest the same.
  bool sameExpression(int32_t x, int32_t y, int64_t low, int64_t high,
                      int64_t shift) const {
    // The pairs still to compare, and of the terms others use that were
    // entered, the one each was compared with. The two expressions are made
    // from the same values in two iterations, so they share alike: a term
    // meets one partner.
    llvm::SmallVector<std::pair<int32_t, int32_t>, 16> todo{{x, y}};
    Places entered;
    while (!todo.empty()) {
      auto pair = todo.pop_back_val();
      const Expression &p = expressions[pair.first],
                       &q = expressions[pair.second];
      if (p.tabled) {
        auto [found, fresh] = entered.try_emplace(pair.first, pair.second);
        if (!fresh && found->second == pair.second)
          continue;
      }
      if (p.kind != q.kind || p.term != q.term || p.name != q.name ||
          p.type != q.type || p.result != q.result ||
          p.of.size() != q.of.size() || !same(p.number, q.number))
        return false;
      if (p.kind == Expression::Readout || p.kind == Expression::Answer) {
        int64_t moved =
            p.event >= low && p.event < high ? p.event + shift : p.event;
        if (q.event != moved)
          return false;
      }
      for (size_t i = 0; i < p.of.size(); ++i)
        todo.emplace_back(p.of[i], q.of[i]);
    }
    return true;
  }

  /// The terms of expression `root` the export names, in the order they
  /// are made, which is after the terms they are over: each term that more
  /// than one term of the expression uses, and each kNested terms deep in
  /// what is written in place, so that no text nests deeper.
  void namedTerms(int32_t root, llvm::SmallVectorImpl<int32_t> &named) const {
    if (expressions[root].plain)
      return;
    // How many terms use each term others may use, then how deep each term
    // reached is written.
    llvm::SmallVector<int32_t, 32> todo{root}, reached;
    Places count;
    while (!todo.empty()) {
      int32_t index = todo.pop_back_val();
      const Expression &e = expressions[index];
      if (e.kind != Expression::Op)
        continue;
      if (e.tabled && ++count[index] > 1)
        continue;
      reached.push_back(index);
      todo.append(e.of.begin(), e.of.end());
    }
    // A term's operands are made before it.
    llvm::sort(reached);
    Places deep;
    for (int32_t index : reached) {
      const Expression &e = expressions[index];
      int32_t depth = 0;
      for (int32_t operand : e.of)
        depth = std::max(depth, deep.lookup(operand));
      ++depth;
      bool name = index != root &&
                  ((e.tabled && count.lookup(index) > 1) || depth >= kNested);
      if (name)
        named.push_back(index);
      deep[index] = name ? 0 : depth;
    }
  }

  /// Expression `root` as the export gives it, without a call per term:
  /// `leaf(index)` writes what is no term, `reference(n)` a use of named
  /// term n, and `node(e, of, table)` term `e` over the written operands
  /// `of`, the root with the `table` of the named terms when there are any.
  LLVM_ATTRIBUTE_NOINLINE nb::object written(
      int32_t root, llvm::function_ref<nb::object(int32_t)> leaf,
      llvm::function_ref<nb::object(int)> reference,
      llvm::function_ref<nb::object(const Expression &, nb::handle, nb::handle)>
          node) const {
    if (expressions[root].kind != Expression::Op)
      return leaf(root);
    llvm::SmallVector<int32_t, 8> named;
    namedTerms(root, named);
    Places names;
    for (size_t n = 0; n < named.size(); ++n)
      names[named[n]] = static_cast<int32_t>(n);
    auto list = [](size_t size) {
      PyObject *made = PyList_New(static_cast<Py_ssize_t>(size));
      if (!made)
        throw nb::python_error();
      return nb::steal(made);
    };
    auto set = [](nb::handle to, size_t at, nb::object value) {
      if (PyList_SetItem(to.ptr(), static_cast<Py_ssize_t>(at),
                         value.release().ptr()) != 0)
        throw nb::python_error();
    };
    // The terms being written, innermost last: each with its operands
    // written so far.
    struct Open {
      int32_t index;
      nb::object of;
      size_t next;
    };
    auto tree = [&](int32_t top, nb::handle table) -> nb::object {
      llvm::SmallVector<Open, 8> open;
      open.push_back({top, list(expressions[top].of.size()), 0});
      for (;;) {
        Open &last = open.back();
        const Expression &e = expressions[last.index];
        if (last.next < e.of.size()) {
          int32_t operand = e.of[last.next];
          if (expressions[operand].kind != Expression::Op)
            set(last.of, last.next++, leaf(operand));
          else if (auto found = names.find(operand); found != names.end())
            set(last.of, last.next++, reference(found->second));
          else
            open.push_back({operand, list(expressions[operand].of.size()), 0});
          continue;
        }
        nb::object out =
            node(e, last.of, open.size() == 1 ? table : nb::handle());
        open.pop_back();
        if (open.empty())
          return out;
        Open &over = open.back();
        set(over.of, over.next++, std::move(out));
      }
    };
    if (named.empty())
      return tree(root, nb::handle());
    nb::object table = list(named.size());
    for (size_t n = 0; n < named.size(); ++n)
      set(table, n, tree(named[n], nb::handle()));
    return tree(root, table);
  }

  /// Whether `e` is whether a readout is set: ne(readout, 0).
  bool isSet(const Expression &e) const {
    if (e.kind != Expression::Op || e.term != Expression::Ne ||
        e.type != Expression::Int || e.of.size() != 2)
      return false;
    const Expression &level = expressions[e.of[0]],
                     &zero = expressions[e.of[1]];
    return level.kind == Expression::Readout &&
           zero.kind == Expression::Constant && zero.number == 0.0;
  }
};
