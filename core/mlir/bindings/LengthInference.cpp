//===- LengthInference.cpp - List result lengths of a classical body ------===//
//
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
//
// Runs the body on values that know their lengths: every list parameter has
// its version's length, every number is unknown. Tests the lengths decide
// take one path; a test of values is followed only when what it skips can't
// change a name or a length. A `for` loop is followed when each pass leaves
// every length the same but a comprehension's own list, which grows by the
// same amount. Anything else (methods, module-level names, other calls,
// exceptions, while loops, a test of values around stores) gives None, and
// classical._infer decides: it gives no answer this code does not.
//
// Only the limited API is used, and the opcodes are read from the running
// interpreter's `opcode` module, so one build serves CPython 3.12 and later.
//
// Contents, in file order:
//   Tables              what the running interpreter's bytecode means
//   Values              Value, State, checked arithmetic
//   The code            Instruction and Run, which follows one body:
//     Stack, locals and jump targets
//     Constants and globals
//     Operators and builtin calls
//     Tests the values decide
//     The instructions  go, the loop over instructions
//     For loops
//   Decoding and entry  decode, lengthsOf, inferListLengths
//
//===----------------------------------------------------------------------===//

#include "LengthInference.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string_view>
#include <utility>
#include <vector>

namespace cudaq_pulse {
namespace {

// ── Tables ──────────────────────────────────────────────────────────────────

enum class Op : uint8_t {
  Unknown,
  Cache,
  ExtendedArg,
  Nop,
  LoadFast,
  LoadFastPair,
  LoadFastAndClear,
  StoreFast,
  StoreFastPair,
  StoreFastLoadFast,
  LoadConst,
  LoadSmallInt,
  LoadCommonConstant,
  ReturnConst,
  ReturnValue,
  LoadGlobal,
  PushNull,
  BuildList,
  BuildTuple,
  BuildSlice,
  ListAppend,
  ListExtend,
  BinaryOp,
  BinarySubscr,
  BinarySlice,
  StoreSubscr,
  CompareOp,
  IsOp,
  ContainsOp,
  ToBool,
  UnaryNot,
  UnaryNegative,
  UnaryInvert,
  GetIter,
  ForIter,
  EndFor,
  PopIter,
  JumpForward,
  JumpBackward,
  PopJumpIfFalse,
  PopJumpIfTrue,
  PopJumpIfNone,
  PopJumpIfNotNone,
  Call,
  Copy,
  Swap,
  PopTop,
  UnpackSequence,
  PushExcInfo,
};

struct OpName {
  const char *name;
  Op op;
};

constexpr OpName kOpNames[] = {
    {"CACHE", Op::Cache},
    {"EXTENDED_ARG", Op::ExtendedArg},
    {"RESUME", Op::Nop},
    {"NOP", Op::Nop},
    {"NOT_TAKEN", Op::Nop},
    {"LOAD_FAST", Op::LoadFast},
    {"LOAD_FAST_CHECK", Op::LoadFast},
    {"LOAD_FAST_BORROW", Op::LoadFast},
    {"LOAD_FAST_LOAD_FAST", Op::LoadFastPair},
    {"LOAD_FAST_BORROW_LOAD_FAST_BORROW", Op::LoadFastPair},
    {"LOAD_FAST_AND_CLEAR", Op::LoadFastAndClear},
    {"STORE_FAST", Op::StoreFast},
    {"STORE_FAST_STORE_FAST", Op::StoreFastPair},
    {"STORE_FAST_LOAD_FAST", Op::StoreFastLoadFast},
    {"LOAD_CONST", Op::LoadConst},
    {"LOAD_SMALL_INT", Op::LoadSmallInt},
    {"LOAD_COMMON_CONSTANT", Op::LoadCommonConstant},
    {"RETURN_CONST", Op::ReturnConst},
    {"RETURN_VALUE", Op::ReturnValue},
    {"LOAD_GLOBAL", Op::LoadGlobal},
    {"PUSH_NULL", Op::PushNull},
    {"BUILD_LIST", Op::BuildList},
    {"BUILD_TUPLE", Op::BuildTuple},
    {"BUILD_SLICE", Op::BuildSlice},
    {"LIST_APPEND", Op::ListAppend},
    {"LIST_EXTEND", Op::ListExtend},
    {"BINARY_OP", Op::BinaryOp},
    {"BINARY_SUBSCR", Op::BinarySubscr},
    {"BINARY_SLICE", Op::BinarySlice},
    {"STORE_SUBSCR", Op::StoreSubscr},
    {"COMPARE_OP", Op::CompareOp},
    {"IS_OP", Op::IsOp},
    {"CONTAINS_OP", Op::ContainsOp},
    {"TO_BOOL", Op::ToBool},
    {"UNARY_NOT", Op::UnaryNot},
    {"UNARY_NEGATIVE", Op::UnaryNegative},
    {"UNARY_INVERT", Op::UnaryInvert},
    {"GET_ITER", Op::GetIter},
    {"FOR_ITER", Op::ForIter},
    {"END_FOR", Op::EndFor},
    {"POP_ITER", Op::PopIter},
    {"JUMP_FORWARD", Op::JumpForward},
    {"JUMP_BACKWARD", Op::JumpBackward},
    {"POP_JUMP_IF_FALSE", Op::PopJumpIfFalse},
    {"POP_JUMP_IF_TRUE", Op::PopJumpIfTrue},
    {"POP_JUMP_IF_NONE", Op::PopJumpIfNone},
    {"POP_JUMP_IF_NOT_NONE", Op::PopJumpIfNotNone},
    {"CALL", Op::Call},
    {"COPY", Op::Copy},
    {"SWAP", Op::Swap},
    {"POP_TOP", Op::PopTop},
    {"UNPACK_SEQUENCE", Op::UnpackSequence},
    {"PUSH_EXC_INFO", Op::PushExcInfo},
};

// BINARY_OP's operators, by their names in opcode._nb_ops.
enum class Arith : uint8_t {
  Unknown,
  Add,
  And,
  FloorDivide,
  LeftShift,
  Multiply,
  Remainder,
  Or,
  Power,
  RightShift,
  Subtract,
  TrueDivide,
  Xor,
  Subscript,
};

struct ArithName {
  const char *name;
  Arith arith;
};

constexpr ArithName kArithNames[] = {
    {"ADD", Arith::Add},
    {"AND", Arith::And},
    {"FLOOR_DIVIDE", Arith::FloorDivide},
    {"LSHIFT", Arith::LeftShift},
    {"MULTIPLY", Arith::Multiply},
    {"REMAINDER", Arith::Remainder},
    {"OR", Arith::Or},
    {"POWER", Arith::Power},
    {"RSHIFT", Arith::RightShift},
    {"SUBTRACT", Arith::Subtract},
    {"TRUE_DIVIDE", Arith::TrueDivide},
    {"XOR", Arith::Xor},
};

enum class Builtin : uint8_t {
  Len,
  Range,
  Zip,
  List,
  Bool,
  Int,
  Float,
  Abs,
  Min,
  Max,
  Sum,
  Count,
};

constexpr const char *kBuiltinNames[] = {"len",  "range", "zip",   "list",
                                         "bool", "int",   "float", "abs",
                                         "min",  "max",   "sum"};

struct BinaryOperator {
  Arith arith = Arith::Unknown;
  bool inPlace = false;
};

/// What the running interpreter's bytecode means, read once.
struct Tables {
  bool ready = false;
  bool usable = false;
  unsigned long version = 0;
  Op ops[256] = {};
  std::vector<BinaryOperator> binary;
  PyObject *builtins[static_cast<int>(Builtin::Count)] = {};
  PyObject *commonConstants = nullptr; // a tuple, 3.14 and later
};

Tables &tables() {
  static Tables t;
  return t;
}

bool loadTables(Tables &t) {
  t.ready = true;
  t.version = Py_Version;
  PyObject *opcode = PyImport_ImportModule("opcode");
  if (!opcode)
    return false;
  bool ok = false;
  PyObject *opmap = PyObject_GetAttrString(opcode, "opmap");
  PyObject *nbOps = PyObject_GetAttrString(opcode, "_nb_ops");
  PyObject *builtins = PyImport_ImportModule("builtins");
  // opmap is a dict, and a frozendict from 3.15.
  if (opmap && nbOps && builtins && PyMapping_Check(opmap)) {
    ok = true;
    for (const OpName &entry : kOpNames) {
      PyObject *number = PyMapping_GetItemString(opmap, entry.name);
      if (!number) {
        PyErr_Clear();
        continue;
      }
      long n = PyLong_AsLong(number);
      Py_DECREF(number);
      if (n >= 0 && n < 256)
        t.ops[n] = entry.op;
    }
    // CACHE is 0 in every version: a deoptimized cache unit reads 0, 0.
    if (t.ops[0] != Op::Cache)
      ok = false;
    Py_ssize_t count = PySequence_Size(nbOps);
    for (Py_ssize_t i = 0; ok && i < count; ++i) {
      BinaryOperator entry;
      PyObject *pair = PySequence_GetItem(nbOps, i);
      PyObject *name = pair ? PySequence_GetItem(pair, 0) : nullptr;
      const char *text =
          name ? PyUnicode_AsUTF8AndSize(name, nullptr) : nullptr;
      if (text) {
        std::string_view view(text);
        if (view.substr(0, 3) == "NB_")
          view.remove_prefix(3);
        if (view.substr(0, 8) == "INPLACE_") {
          view.remove_prefix(8);
          entry.inPlace = true;
        }
        if (view == "SUBSCR")
          entry.arith = Arith::Subscript;
        for (const ArithName &a : kArithNames)
          if (view == a.name)
            entry.arith = a.arith;
      } else {
        ok = false;
      }
      Py_XDECREF(name);
      Py_XDECREF(pair);
      t.binary.push_back(entry);
    }
    for (int i = 0; ok && i < static_cast<int>(Builtin::Count); ++i) {
      t.builtins[i] = PyObject_GetAttrString(builtins, kBuiltinNames[i]);
      if (!t.builtins[i])
        ok = false;
    }
    if (ok && t.version >= 0x030E0000) {
      t.commonConstants = PyObject_GetAttrString(opcode, "_common_constants");
      if (!t.commonConstants)
        PyErr_Clear();
    }
  }
  Py_XDECREF(builtins);
  Py_XDECREF(nbOps);
  Py_XDECREF(opmap);
  Py_DECREF(opcode);
  if (!ok)
    PyErr_Clear();
  return ok;
}

// ── Values ──────────────────────────────────────────────────────────────────

enum class Kind : uint8_t {
  Unbound, // a local not bound yet, or the value LOAD_FAST_AND_CLEAR saves
  Null,
  None,
  Int,     // a known integer (n)
  Scalar,  // a value known only when the function runs
  List,    // a list of n items
  Tuple,   // its items
  Builtin, // one of Builtin (n)
  Range,   // a range of n items
  Zip,     // zip of n items, each a tuple of `arity` values
  Iter,    // an iterator of n items, each a value or a tuple of `arity`
  Slice,   // items: start, stop, step
};

struct Value {
  Kind kind = Kind::Unbound;
  int64_t n = 0;
  int arity = 0;
  // Which list this is: a copy of the value keeps it, a new list gets its
  // own. classical._infer gives up where one name may be either of two lists.
  uint64_t list = 0;
  std::shared_ptr<const std::vector<Value>> items;

  static Value of(Kind kind, int64_t n = 0) {
    thread_local uint64_t lists = 0;
    Value v;
    v.kind = kind;
    v.n = n;
    if (kind == Kind::List)
      v.list = ++lists;
    return v;
  }
  static Value tuple(std::vector<Value> items) {
    Value v;
    v.kind = Kind::Tuple;
    v.n = static_cast<int64_t>(items.size());
    v.items = std::make_shared<const std::vector<Value>>(std::move(items));
    return v;
  }

  bool operator==(const Value &other) const {
    if (kind != other.kind || n != other.n || arity != other.arity ||
        list != other.list)
      return false;
    if (kind != Kind::Tuple && kind != Kind::Slice)
      return true;
    return items == other.items || *items == *other.items;
  }
  bool operator!=(const Value &other) const { return !(*this == other); }

  bool isNumber() const { return kind == Kind::Int || kind == Kind::Scalar; }
  /// How many items iterating it gives, or -1.
  int64_t count() const {
    switch (kind) {
    case Kind::List:
    case Kind::Tuple:
    case Kind::Range:
    case Kind::Zip:
    case Kind::Iter:
      return n;
    default:
      return -1;
    }
  }
};

struct State {
  std::vector<Value> locals;
  std::vector<Value> stack;

  bool operator==(const State &other) const {
    return locals == other.locals && stack == other.stack;
  }
};

/// Thrown out of the run when the body is not one this follows.
struct Declined {};

[[noreturn]] void decline() { throw Declined{}; }

int64_t add(int64_t a, int64_t b) {
  int64_t r = 0;
  if (__builtin_add_overflow(a, b, &r))
    decline();
  return r;
}

int64_t subtract(int64_t a, int64_t b) {
  int64_t r = 0;
  if (__builtin_sub_overflow(a, b, &r))
    decline();
  return r;
}

int64_t multiply(int64_t a, int64_t b) {
  int64_t r = 0;
  if (__builtin_mul_overflow(a, b, &r))
    decline();
  return r;
}

int64_t floorDivide(int64_t a, int64_t b) {
  if (b == 0 || (a == INT64_MIN && b == -1))
    decline();
  int64_t q = a / b;
  if ((a % b != 0) && ((a < 0) != (b < 0)))
    --q;
  return q;
}

int64_t remainder(int64_t a, int64_t b) {
  return subtract(a, multiply(floorDivide(a, b), b));
}

/// Python's slice length of a sequence of `length` items.
int64_t sliceLength(int64_t length, const Value &startValue,
                    const Value &stopValue, const Value &stepValue) {
  for (const Value *part : {&startValue, &stopValue, &stepValue})
    if (part->kind != Kind::None && part->kind != Kind::Int)
      decline();
  int64_t step = stepValue.kind == Kind::Int ? stepValue.n : 1;
  if (step == 0 || step == INT64_MIN)
    decline();
  int64_t lower = step > 0 ? 0 : -1;
  int64_t upper = step > 0 ? length : length - 1;
  auto bound = [&](const Value &part, int64_t whenNone) {
    if (part.kind == Kind::None)
      return whenNone;
    int64_t i = part.n;
    if (i < 0) {
      i += length;
      if (i < lower)
        i = lower;
    } else if (i > upper) {
      i = upper;
    }
    return i;
  };
  int64_t start = bound(startValue, step < 0 ? upper : lower);
  int64_t stop = bound(stopValue, step < 0 ? lower : upper);
  if (step < 0)
    return stop < start ? (start - stop - 1) / (-step) + 1 : 0;
  return start < stop ? (stop - start - 1) / step + 1 : 0;
}

// ── The code ────────────────────────────────────────────────────────────────

struct Instruction {
  Op op = Op::Unknown;
  uint8_t opcode = 0;
  uint32_t arg = 0;
  int next = 0; // the unit after this instruction and its caches
};

constexpr int kCoVarargs = 0x04;
constexpr int kCoVarkeywords = 0x08;
constexpr int kCoGenerator = 0x20;
constexpr int kCoCoroutine = 0x80;
constexpr int kCoIterableCoroutine = 0x100;
constexpr int kCoAsyncGenerator = 0x200;
constexpr int64_t kStepBudget = 100000;
constexpr int kDepthBudget = 200;

class Run {
public:
  Run(const Tables &t, std::vector<Instruction> code, PyObject *consts,
      PyObject *names, PyObject *globals, PyObject *builtins)
      : t(t), code(std::move(code)), consts(consts), names(names),
        globals(globals), builtins(builtins) {}

  std::vector<Value> returns;
  int stopAt = -1;
  State stopped;

  void run(State state) { go(std::move(state), 0, nullptr, 0); }

private:
  struct Loop {
    int header;
    int exit;
    std::vector<State> ends;
  };

  const Tables &t;
  std::vector<Instruction> code;
  PyObject *consts, *names, *globals, *builtins;
  int64_t steps = 0;

  bool atLeast(unsigned long hex) const { return t.version >= hex; }
  int iteratorSlots() const { return atLeast(0x030F0000) ? 2 : 1; }

  // ── Stack, locals and jump targets ────────────────────────────────────────

  static Value pop(State &s) {
    if (s.stack.empty())
      decline();
    Value v = std::move(s.stack.back());
    s.stack.pop_back();
    return v;
  }
  static Value &peek(State &s, uint32_t depth) {
    if (depth == 0 || depth > s.stack.size())
      decline();
    return s.stack[s.stack.size() - depth];
  }
  static const Value &local(const State &s, uint32_t i) {
    if (i >= s.locals.size() || s.locals[i].kind == Kind::Unbound)
      decline();
    return s.locals[i];
  }
  static void store(State &s, uint32_t i, Value v) {
    if (i >= s.locals.size())
      decline();
    s.locals[i] = std::move(v);
  }

  int target(int at, bool backward, const Loop *loop = nullptr) const {
    const Instruction &ins = code[at];
    int to = backward ? ins.next - static_cast<int>(ins.arg)
                      : ins.next + static_cast<int>(ins.arg);
    if (to < 0 || to >= static_cast<int>(code.size()))
      decline();
    // A jump out of a loop's body (break) ends the loop early.
    if (loop && !backward && to >= loop->exit)
      decline();
    return to;
  }

  // ── Constants and globals ─────────────────────────────────────────────────

  Value constant(PyObject *object, int depth = 0) const {
    if (object == Py_None)
      return Value::of(Kind::None);
    if (PyBool_Check(object))
      return Value::of(Kind::Int, object == Py_True ? 1 : 0);
    if (PyLong_CheckExact(object)) {
      int overflow = 0;
      long long v = PyLong_AsLongLongAndOverflow(object, &overflow);
      if (overflow || (v == -1 && PyErr_Occurred())) {
        PyErr_Clear();
        decline();
      }
      return Value::of(Kind::Int, v);
    }
    if (PyFloat_CheckExact(object) || PyComplex_CheckExact(object))
      return Value::of(Kind::Scalar);
    if (PyTuple_CheckExact(object) && depth < 4) {
      std::vector<Value> items;
      Py_ssize_t size = PyTuple_Size(object);
      for (Py_ssize_t i = 0; i < size; ++i)
        items.push_back(constant(PyTuple_GetItem(object, i), depth + 1));
      return Value::tuple(std::move(items));
    }
    // From 3.14 a slice of constants is one constant.
    if (PySlice_Check(object) && depth < 4) {
      std::vector<Value> parts;
      for (const char *name : {"start", "stop", "step"}) {
        PyObject *part = PyObject_GetAttrString(object, name);
        if (!part) {
          PyErr_Clear();
          decline();
        }
        Value v;
        try {
          v = constant(part, depth + 1);
        } catch (const Declined &) {
          Py_DECREF(part);
          throw;
        }
        Py_DECREF(part);
        parts.push_back(std::move(v));
      }
      Value v = Value::tuple(std::move(parts));
      v.kind = Kind::Slice;
      return v;
    }
    for (int i = 0; i < static_cast<int>(Builtin::Count); ++i)
      if (object == t.builtins[i])
        return Value::of(Kind::Builtin, i);
    decline();
  }

  Value global(uint32_t index) const {
    PyObject *name = PyTuple_GetItem(names, index);
    if (!name) {
      PyErr_Clear();
      decline();
    }
    // A module-level name is read by classical._infer, which tracks it.
    PyObject *found = PyDict_GetItemWithError(globals, name);
    if (found || PyErr_Occurred()) {
      PyErr_Clear();
      decline();
    }
    found = PyDict_Check(builtins) ? PyDict_GetItemWithError(builtins, name)
                                   : nullptr;
    if (!found) {
      PyErr_Clear();
      decline();
    }
    for (int i = 0; i < static_cast<int>(Builtin::Count); ++i)
      if (found == t.builtins[i])
        return Value::of(Kind::Builtin, i);
    decline();
  }

  // ── Operators and builtin calls ───────────────────────────────────────────

  static int truth(const Value &v) {
    switch (v.kind) {
    case Kind::Int:
      return v.n != 0;
    case Kind::List:
    case Kind::Tuple:
    case Kind::Range:
      return v.n > 0;
    case Kind::None:
      return 0;
    case Kind::Builtin:
    case Kind::Zip:
    case Kind::Iter:
      return 1;
    case Kind::Scalar:
      return -1;
    default:
      decline();
    }
  }

  Value subscript(const Value &container, const Value &index) const {
    if (index.kind == Kind::Slice) {
      if (container.kind != Kind::List)
        decline();
      const std::vector<Value> &parts = *index.items;
      return Value::of(Kind::List,
                       sliceLength(container.n, parts[0], parts[1], parts[2]));
    }
    switch (container.kind) {
    case Kind::List:
      if (index.kind == Kind::Int &&
          (index.n < -container.n || index.n >= container.n))
        decline(); // raises IndexError
      if (!index.isNumber())
        decline();
      return Value::of(Kind::Scalar);
    case Kind::Tuple:
      if (index.kind != Kind::Int || index.n < -container.n ||
          index.n >= container.n)
        decline();
      return (*container.items)[index.n < 0 ? index.n + container.n : index.n];
    case Kind::Scalar:
      if (!index.isNumber())
        decline();
      return Value::of(Kind::Scalar);
    default:
      decline();
    }
  }

  Value arithmetic(Arith arith, bool inPlace, const Value &a,
                   const Value &b) const {
    if (arith == Arith::Subscript)
      return subscript(a, b);
    if (a.kind == Kind::Int && b.kind == Kind::Int) {
      int64_t x = a.n, y = b.n;
      switch (arith) {
      case Arith::Add:
        return Value::of(Kind::Int, add(x, y));
      case Arith::Subtract:
        return Value::of(Kind::Int, subtract(x, y));
      case Arith::Multiply:
        return Value::of(Kind::Int, multiply(x, y));
      case Arith::FloorDivide:
        return Value::of(Kind::Int, floorDivide(x, y));
      case Arith::Remainder:
        return Value::of(Kind::Int, remainder(x, y));
      case Arith::And:
        return Value::of(Kind::Int, x & y);
      case Arith::Or:
        return Value::of(Kind::Int, x | y);
      case Arith::Xor:
        return Value::of(Kind::Int, x ^ y);
      case Arith::LeftShift:
        if (y < 0 || y >= 62 ||
            (x != 0 && (x > (INT64_MAX >> y) || x < (INT64_MIN >> y))))
          decline();
        return Value::of(Kind::Int, x * (int64_t(1) << y));
      case Arith::RightShift:
        if (y < 0)
          decline();
        return Value::of(Kind::Int, y >= 63 ? (x < 0 ? -1 : 0) : x >> y);
      case Arith::Power: {
        if (y < 0 || y > 62)
          decline();
        int64_t p = 1;
        for (int64_t i = 0; i < y; ++i)
          p = multiply(p, x);
        return Value::of(Kind::Int, p);
      }
      case Arith::TrueDivide:
        if (y == 0)
          decline();
        return Value::of(Kind::Scalar);
      default:
        decline();
      }
    }
    if (a.isNumber() && b.isNumber()) {
      bool byZero = b.kind == Kind::Int && b.n == 0;
      if (byZero && (arith == Arith::FloorDivide || arith == Arith::Remainder ||
                     arith == Arith::TrueDivide))
        decline();
      if (arith == Arith::Unknown)
        decline();
      return Value::of(Kind::Scalar);
    }
    // A list changed in place may have other names: classical._infer.
    if (inPlace)
      decline();
    if (arith == Arith::Add && a.kind == Kind::List && b.kind == Kind::List)
      return Value::of(Kind::List, add(a.n, b.n));
    if (arith == Arith::Add && a.kind == Kind::Tuple && b.kind == Kind::Tuple) {
      std::vector<Value> items(*a.items);
      items.insert(items.end(), b.items->begin(), b.items->end());
      return Value::tuple(std::move(items));
    }
    if (arith == Arith::Multiply) {
      const Value *list = a.kind == Kind::List ? &a : &b;
      const Value *times = a.kind == Kind::List ? &b : &a;
      if (list->kind == Kind::List && times->kind == Kind::Int)
        return Value::of(Kind::List,
                         multiply(list->n, std::max<int64_t>(0, times->n)));
    }
    decline();
  }

  Value compare(uint32_t arg, const Value &a, const Value &b) const {
    uint32_t which = arg >> (atLeast(0x030D0000) ? 5 : 4);
    if (a.kind == Kind::Int && b.kind == Kind::Int) {
      bool r = false;
      switch (which) {
      case 0:
        r = a.n < b.n;
        break;
      case 1:
        r = a.n <= b.n;
        break;
      case 2:
        r = a.n == b.n;
        break;
      case 3:
        r = a.n != b.n;
        break;
      case 4:
        r = a.n > b.n;
        break;
      case 5:
        r = a.n >= b.n;
        break;
      default:
        decline();
      }
      return Value::of(Kind::Int, r);
    }
    if (a.isNumber() && b.isNumber() && which <= 5)
      return Value::of(Kind::Scalar);
    decline();
  }

  Value call(Builtin which, std::vector<Value> &args) const {
    size_t argc = args.size();
    auto one = [&]() -> const Value & {
      if (argc != 1)
        decline();
      return args[0];
    };
    switch (which) {
    case Builtin::Len: {
      const Value &x = one();
      if (x.kind != Kind::List && x.kind != Kind::Tuple &&
          x.kind != Kind::Range)
        decline();
      return Value::of(Kind::Int, x.n);
    }
    case Builtin::Range: {
      if (argc < 1 || argc > 3)
        decline();
      for (const Value &a : args)
        if (a.kind != Kind::Int)
          decline();
      int64_t start = argc == 1 ? 0 : args[0].n;
      int64_t stop = argc == 1 ? args[0].n : args[1].n;
      int64_t step = argc == 3 ? args[2].n : 1;
      if (step == 0 || step == INT64_MIN)
        decline();
      int64_t span = step > 0 ? subtract(stop, start) : subtract(start, stop);
      int64_t width = step > 0 ? step : -step;
      return Value::of(Kind::Range, span <= 0 ? 0 : (span - 1) / width + 1);
    }
    case Builtin::Zip: {
      if (argc == 0)
        decline();
      int64_t n = INT64_MAX;
      for (const Value &a : args) {
        if (a.count() < 0 || a.kind == Kind::Tuple)
          decline();
        n = std::min(n, a.count());
      }
      Value v = Value::of(Kind::Zip, n);
      v.arity = static_cast<int>(argc);
      return v;
    }
    case Builtin::List: {
      if (argc == 0)
        return Value::of(Kind::List, 0);
      const Value &x = one();
      if (x.count() < 0)
        decline();
      return Value::of(Kind::List, x.count());
    }
    case Builtin::Bool:
      // classical._infer doesn't fold bool(), not even of a list.
      if (argc > 1)
        decline();
      return Value::of(Kind::Scalar);
    case Builtin::Int: {
      if (argc == 0)
        return Value::of(Kind::Int, 0);
      const Value &x = one();
      if (!x.isNumber())
        decline();
      return x;
    }
    case Builtin::Float:
      if (argc != 0 && !one().isNumber())
        decline();
      return Value::of(Kind::Scalar);
    case Builtin::Abs: {
      const Value &x = one();
      if (x.kind == Kind::Int && x.n != INT64_MIN)
        return Value::of(Kind::Int, x.n < 0 ? -x.n : x.n);
      if (x.kind != Kind::Scalar)
        decline();
      return x;
    }
    case Builtin::Min:
    case Builtin::Max: {
      if (argc == 1) {
        const Value &x = args[0];
        if (x.kind != Kind::List && x.kind != Kind::Range)
          decline();
        if (x.n <= 0)
          decline(); // raises ValueError
        return Value::of(Kind::Scalar);
      }
      if (argc == 0)
        decline();
      bool known = true;
      int64_t best = 0;
      for (size_t i = 0; i < argc; ++i) {
        if (!args[i].isNumber())
          decline();
        if (args[i].kind != Kind::Int) {
          known = false;
          continue;
        }
        if (i == 0 ||
            (which == Builtin::Min ? args[i].n < best : args[i].n > best))
          best = args[i].n;
      }
      return known ? Value::of(Kind::Int, best) : Value::of(Kind::Scalar);
    }
    case Builtin::Sum: {
      const Value &x = one();
      if (x.kind != Kind::List && x.kind != Kind::Range)
        decline();
      return Value::of(Kind::Scalar);
    }
    default:
      decline();
    }
  }

  // ── Tests the values decide ───────────────────────────────────────────────

  static bool straight(Op op) {
    switch (op) {
    case Op::Cache:
    case Op::ExtendedArg:
    case Op::Nop:
    case Op::LoadFast:
    case Op::LoadFastPair:
    case Op::LoadConst:
    case Op::LoadSmallInt:
    case Op::LoadCommonConstant:
    case Op::LoadGlobal:
    case Op::PushNull:
    case Op::BuildList:
    case Op::BuildTuple:
    case Op::BuildSlice:
    case Op::BinaryOp:
    case Op::BinarySubscr:
    case Op::BinarySlice:
    case Op::StoreSubscr:
    case Op::CompareOp:
    case Op::IsOp:
    case Op::ContainsOp:
    case Op::ToBool:
    case Op::UnaryNot:
    case Op::UnaryNegative:
    case Op::UnaryInvert:
    case Op::Call:
    case Op::Copy:
    case Op::Swap:
    case Op::PopTop:
    case Op::UnpackSequence:
      return true;
    default:
      return false;
    }
  }

  /// Run the straight code from `pc` up to `stop` or the first instruction
  /// that isn't straight; the pc reached, with `s` the state there.
  int straightRun(State &s, int pc, int stop, Loop *loop, int depth) {
    const int size = static_cast<int>(code.size());
    int end = pc;
    while (end != stop && end < size && straight(code[end].op))
      end = code[end].next;
    if (end == pc)
      return pc;
    int saved = stopAt;
    stopAt = end;
    go(std::move(s), pc, loop, depth + 1);
    stopAt = saved;
    s = std::move(stopped);
    return end;
  }

  /// Where the path at `at` goes next: a jump's target, or `at`.
  int meeting(int at) const {
    const Instruction &ins = code[at];
    if (ins.op == Op::JumpBackward)
      return -2 - target(at, true);
    if (ins.op == Op::JumpForward)
      return target(at, false);
    return at;
  }

  /// A test only the values decide, jumping from `at` to `to`: followed when
  /// both ways run straight code to one place in one state, so neither can
  /// change a name, a length or the stack. Any other such test is
  /// classical._infer's to explain. The pc to go on from, with `s` the state.
  int skippable(State &s, int at, int to, Loop *loop, int depth) {
    if (to <= at)
      decline();
    State fell = s;
    int end = straightRun(fell, code[at].next, to, loop, depth);
    if (end == to) {
      if (!(fell == s))
        decline();
      return to;
    }
    int other = straightRun(s, to, -1, loop, depth);
    if (meeting(end) != meeting(other) || !(fell == s))
      decline();
    return other;
  }

  // ── The instructions ──────────────────────────────────────────────────────

  /// Run from `pc` until the path returns, ends, comes back to `loop`'s
  /// header, or reaches `stopAt` (left in `stopped`).
  void go(State s, int pc, Loop *loop, int depth) {
    if (depth > kDepthBudget)
      decline();
    const int size = static_cast<int>(code.size());
    while (true) {
      if (pc == stopAt) {
        stopped = std::move(s);
        return;
      }
      if (pc < 0 || pc >= size || ++steps > kStepBudget)
        decline();
      const Instruction &ins = code[pc];
      uint32_t arg = ins.arg;
      switch (ins.op) {
      case Op::Cache:
      case Op::ExtendedArg:
      case Op::Nop:
        break;
      case Op::LoadFast:
        s.stack.push_back(local(s, arg));
        break;
      case Op::LoadFastPair:
        s.stack.push_back(local(s, arg >> 4));
        s.stack.push_back(local(s, arg & 15));
        break;
      case Op::LoadFastAndClear:
        if (arg >= s.locals.size())
          decline();
        s.stack.push_back(s.locals[arg]);
        s.locals[arg] = Value();
        break;
      case Op::StoreFast:
        store(s, arg, pop(s));
        break;
      case Op::StoreFastPair:
        store(s, arg >> 4, pop(s));
        store(s, arg & 15, pop(s));
        break;
      case Op::StoreFastLoadFast:
        store(s, arg >> 4, pop(s));
        s.stack.push_back(local(s, arg & 15));
        break;
      case Op::LoadConst:
      case Op::ReturnConst: {
        PyObject *object = PyTuple_GetItem(consts, arg);
        if (!object) {
          PyErr_Clear();
          decline();
        }
        Value v = constant(object);
        if (ins.op == Op::ReturnConst) {
          returns.push_back(std::move(v));
          return;
        }
        s.stack.push_back(std::move(v));
        break;
      }
      case Op::LoadSmallInt:
        s.stack.push_back(Value::of(Kind::Int, arg));
        break;
      case Op::LoadCommonConstant: {
        PyObject *object = t.commonConstants
                               ? PySequence_GetItem(t.commonConstants, arg)
                               : nullptr;
        if (!object) {
          PyErr_Clear();
          decline();
        }
        Value v;
        try {
          v = constant(object);
        } catch (const Declined &) {
          Py_DECREF(object);
          throw;
        }
        Py_DECREF(object);
        s.stack.push_back(std::move(v));
        break;
      }
      case Op::ReturnValue:
        returns.push_back(pop(s));
        return;
      case Op::LoadGlobal: {
        Value v = global(arg >> 1);
        bool null = arg & 1;
        if (null && !atLeast(0x030D0000))
          s.stack.push_back(Value::of(Kind::Null));
        s.stack.push_back(std::move(v));
        if (null && atLeast(0x030D0000))
          s.stack.push_back(Value::of(Kind::Null));
        break;
      }
      case Op::PushNull:
        s.stack.push_back(Value::of(Kind::Null));
        break;
      case Op::BuildList:
      case Op::BuildTuple: {
        if (arg > s.stack.size())
          decline();
        std::vector<Value> items(s.stack.end() - arg, s.stack.end());
        s.stack.resize(s.stack.size() - arg);
        s.stack.push_back(ins.op == Op::BuildList
                              ? Value::of(Kind::List, arg)
                              : Value::tuple(std::move(items)));
        break;
      }
      case Op::BuildSlice: {
        if (arg != 2 && arg != 3)
          decline();
        Value step = arg == 3 ? pop(s) : Value::of(Kind::None);
        Value stop = pop(s);
        Value start = pop(s);
        Value v = Value::tuple({start, stop, step});
        v.kind = Kind::Slice;
        s.stack.push_back(std::move(v));
        break;
      }
      case Op::ListAppend: {
        pop(s);
        Value &list = peek(s, arg);
        if (list.kind != Kind::List)
          decline();
        ++list.n;
        break;
      }
      case Op::ListExtend: {
        Value items = pop(s);
        Value &list = peek(s, arg);
        if (list.kind != Kind::List || items.count() < 0)
          decline();
        list.n += items.count();
        break;
      }
      case Op::BinaryOp: {
        if (arg >= t.binary.size())
          decline();
        Value b = pop(s);
        Value a = pop(s);
        s.stack.push_back(
            arithmetic(t.binary[arg].arith, t.binary[arg].inPlace, a, b));
        break;
      }
      case Op::BinarySubscr: {
        Value index = pop(s);
        Value container = pop(s);
        s.stack.push_back(subscript(container, index));
        break;
      }
      case Op::BinarySlice: {
        Value stop = pop(s);
        Value start = pop(s);
        Value container = pop(s);
        Value slice = Value::tuple({start, stop, Value::of(Kind::None)});
        slice.kind = Kind::Slice;
        s.stack.push_back(subscript(container, slice));
        break;
      }
      case Op::StoreSubscr: {
        Value index = pop(s);
        Value container = pop(s);
        pop(s);
        if (container.kind != Kind::List || !index.isNumber())
          decline();
        if (index.kind == Kind::Int &&
            (index.n < -container.n || index.n >= container.n))
          decline();
        break;
      }
      case Op::CompareOp: {
        Value b = pop(s);
        Value a = pop(s);
        s.stack.push_back(compare(arg, a, b));
        break;
      }
      case Op::IsOp: {
        Value b = pop(s);
        Value a = pop(s);
        auto known = [](const Value &v) {
          return v.kind != Kind::Scalar && v.kind != Kind::Unbound;
        };
        if (!known(a) || !known(b) ||
            (a.kind != Kind::None && b.kind != Kind::None)) {
          if (a.kind != Kind::None && b.kind != Kind::None)
            decline();
          s.stack.push_back(Value::of(Kind::Scalar));
        } else {
          bool same = a.kind == Kind::None && b.kind == Kind::None;
          s.stack.push_back(Value::of(Kind::Int, same != (arg == 1)));
        }
        break;
      }
      case Op::ContainsOp: {
        Value b = pop(s);
        pop(s);
        if (b.count() < 0)
          decline();
        s.stack.push_back(Value::of(Kind::Scalar));
        break;
      }
      case Op::ToBool:
      case Op::UnaryNot: {
        int known = truth(pop(s));
        if (known < 0)
          s.stack.push_back(Value::of(Kind::Scalar));
        else
          s.stack.push_back(
              Value::of(Kind::Int, ins.op == Op::UnaryNot ? !known : known));
        break;
      }
      case Op::UnaryNegative:
      case Op::UnaryInvert: {
        Value v = pop(s);
        if (v.kind == Kind::Int && v.n != INT64_MIN)
          s.stack.push_back(
              Value::of(Kind::Int, ins.op == Op::UnaryNegative ? -v.n : ~v.n));
        else if (v.kind == Kind::Scalar)
          s.stack.push_back(v);
        else
          decline();
        break;
      }
      case Op::GetIter: {
        Value v = pop(s);
        Value it;
        if (v.kind == Kind::Iter) {
          it = v;
        } else if (v.kind == Kind::Zip) {
          it = Value::of(Kind::Iter, v.n);
          it.arity = v.arity;
        } else if (v.kind == Kind::List || v.kind == Kind::Range) {
          it = Value::of(Kind::Iter, v.n);
        } else {
          decline();
        }
        s.stack.push_back(std::move(it));
        if (iteratorSlots() == 2)
          s.stack.push_back(Value::of(Kind::Null));
        break;
      }
      case Op::ForIter:
        pc = forLoop(s, pc, depth);
        if (pc < 0)
          return; // no pass reaches the end of the body
        continue;
      case Op::JumpForward:
        pc = target(pc, false, loop);
        continue;
      case Op::JumpBackward: {
        int to = target(pc, true);
        if (!loop || to != loop->header)
          decline();
        loop->ends.push_back(std::move(s));
        return;
      }
      case Op::PopJumpIfFalse:
      case Op::PopJumpIfTrue:
      case Op::PopJumpIfNone:
      case Op::PopJumpIfNotNone: {
        Value v = pop(s);
        int jumps; // 1, 0, or -1 when only the values decide
        if (ins.op == Op::PopJumpIfNone || ins.op == Op::PopJumpIfNotNone) {
          int none = v.kind == Kind::None ? 1 : v.kind == Kind::Scalar ? -1 : 0;
          jumps = none < 0 ? -1 : (none == 1) == (ins.op == Op::PopJumpIfNone);
        } else {
          int known = truth(v);
          jumps =
              known < 0 ? -1 : (known == 1) == (ins.op == Op::PopJumpIfTrue);
        }
        int to = target(pc, false, loop);
        if (jumps < 0) {
          pc = skippable(s, pc, to, loop, depth);
        } else {
          pc = jumps ? to : ins.next;
        }
        continue;
      }
      case Op::Call: {
        if (arg + 2 > s.stack.size())
          decline();
        std::vector<Value> args(s.stack.end() - arg, s.stack.end());
        s.stack.resize(s.stack.size() - arg);
        Value second = pop(s);
        Value first = pop(s);
        const Value &callable = atLeast(0x030D0000) ? first : second;
        const Value &other = atLeast(0x030D0000) ? second : first;
        if (other.kind != Kind::Null || callable.kind != Kind::Builtin)
          decline();
        s.stack.push_back(call(static_cast<Builtin>(callable.n), args));
        break;
      }
      case Op::Copy:
        s.stack.push_back(peek(s, arg));
        break;
      case Op::Swap:
        std::swap(peek(s, 1), peek(s, arg));
        break;
      case Op::PopTop:
        pop(s);
        break;
      case Op::UnpackSequence: {
        Value v = pop(s);
        if (v.kind == Kind::Tuple && v.n == arg) {
          for (auto it = v.items->rbegin(); it != v.items->rend(); ++it)
            s.stack.push_back(*it);
        } else if ((v.kind == Kind::List && v.n == arg) ||
                   v.kind == Kind::Scalar) {
          for (uint32_t i = 0; i < arg; ++i)
            s.stack.push_back(Value::of(Kind::Scalar));
        } else {
          decline();
        }
        break;
      }
      default:
        decline();
      }
      pc = ins.next;
    }
  }

  // ── For loops ─────────────────────────────────────────────────────────────

  /// One pass of the loop at `header` from `s`; the state every path of the
  /// pass brings back to the header, or false when none does.
  bool pass(const State &s, int header, int exit, int depth, State &out) {
    Loop loop{header, exit, {}};
    State body = s;
    const Value &it = body.stack[body.stack.size() - iteratorSlots()];
    if (it.arity == 0) {
      body.stack.push_back(Value::of(Kind::Scalar));
    } else {
      std::vector<Value> items(it.arity, Value::of(Kind::Scalar));
      body.stack.push_back(Value::tuple(std::move(items)));
    }
    go(std::move(body), code[header].next, &loop, depth + 1);
    if (loop.ends.empty())
      return false;
    for (const State &end : loop.ends)
      if (!(end == loop.ends[0]))
        decline(); // its paths change lengths differently
    out = std::move(loop.ends[0]);
    return true;
  }

  /// The `for` loop at `header`: where it continues, with `s` the state
  /// after it; -1 when no pass completes.
  int forLoop(State &s, int header, int depth) {
    size_t slots = iteratorSlots();
    if (s.stack.size() < slots)
      decline();
    size_t at = s.stack.size() - slots;
    const Value it = s.stack[at];
    if (it.kind != Kind::Iter)
      decline();
    // Where the loop goes on: after END_FOR, and the POP_TOP (3.13) or
    // POP_ITER (3.14 and later) that drops the iterator.
    int exit = target(header, false);
    if (code[exit].op != Op::EndFor)
      decline();
    int after = code[exit].next;
    if (after >= static_cast<int>(code.size()))
      decline();
    if (code[after].op == Op::PopIter ||
        (t.version >> 16 == 0x030D && code[after].op == Op::PopTop))
      after = code[after].next;

    int64_t n = it.n;
    if (n > 0) {
      State first;
      if (!pass(s, header, exit, depth, first))
        return -1;
      // Each pass must leave the locals as the first leaves them, and only a
      // list under the iterator (a comprehension's) may grow.
      if (first.stack.size() != s.stack.size())
        decline();
      std::vector<int64_t> growth(s.stack.size(), 0);
      for (size_t i = 0; i < s.locals.size(); ++i)
        if (s.locals[i].kind != Kind::Unbound && s.locals[i] != first.locals[i])
          decline();
      for (size_t i = 0; i < s.stack.size(); ++i) {
        if (s.stack[i] == first.stack[i])
          continue;
        if (i >= at || s.stack[i].kind != Kind::List ||
            first.stack[i].kind != Kind::List ||
            first.stack[i].n < s.stack[i].n)
          decline();
        growth[i] = first.stack[i].n - s.stack[i].n;
      }
      State result = first;
      if (n > 1) {
        State second;
        if (!pass(first, header, exit, depth, second))
          return -1;
        if (second.locals != first.locals ||
            second.stack.size() != first.stack.size())
          decline();
        for (size_t i = 0; i < first.stack.size(); ++i) {
          Value expected = first.stack[i];
          expected.n += growth[i];
          if (second.stack[i] != expected)
            decline();
        }
        for (size_t i = 0; i < result.stack.size(); ++i) {
          if (!growth[i])
            continue;
          result.stack[i].n = add(s.stack[i].n, multiply(growth[i], n));
        }
      }
      s = std::move(result);
    }
    s.stack.resize(at);
    return after;
  }
};

// ── Decoding and entry ──────────────────────────────────────────────────────

/// The code of `function` decoded, or false when it is not one to follow.
bool decode(const Tables &t, PyObject *code, std::vector<Instruction> &out) {
  PyObject *bytes = PyObject_GetAttrString(code, "co_code");
  if (!bytes)
    return false;
  char *data = nullptr;
  Py_ssize_t length = 0;
  bool ok =
      PyBytes_AsStringAndSize(bytes, &data, &length) == 0 && length % 2 == 0;
  if (ok) {
    int units = static_cast<int>(length / 2);
    out.assign(units, Instruction{});
    uint32_t extended = 0;
    for (int i = 0; i < units; ++i) {
      uint8_t opcode = static_cast<uint8_t>(data[2 * i]);
      uint8_t raw = static_cast<uint8_t>(data[2 * i + 1]);
      Instruction &ins = out[i];
      ins.opcode = opcode;
      ins.op = t.ops[opcode];
      ins.arg = (extended << 8) | raw;
      extended = ins.op == Op::ExtendedArg ? ins.arg : 0;
      int next = i + 1;
      if (ins.op != Op::Cache)
        while (next < units && data[2 * next] == 0 && data[2 * next + 1] == 0)
          ++next;
      ins.next = next;
      if (ins.op == Op::PushExcInfo)
        ok = false; // a handler: classical._infer follows exceptions
    }
  }
  Py_DECREF(bytes);
  if (!ok)
    PyErr_Clear();
  return ok;
}

long intAttribute(PyObject *object, const char *name) {
  PyObject *value = PyObject_GetAttrString(object, name);
  long n = value ? PyLong_AsLong(value) : -1;
  Py_XDECREF(value);
  if (n == -1)
    PyErr_Clear();
  return n;
}

Py_ssize_t tupleSize(PyObject *object, const char *name) {
  PyObject *value = PyObject_GetAttrString(object, name);
  Py_ssize_t n = value && PyTuple_Check(value) ? PyTuple_Size(value) : -1;
  Py_XDECREF(value);
  if (n == -1)
    PyErr_Clear();
  return n;
}

/// The lengths, or an empty vector when the body is not followed.
std::vector<int64_t> lengthsOf(const Tables &t, PyObject *function,
                               PyObject *parameterNames, PyObject *lengths,
                               PyObject *positions, Py_ssize_t resultCount) {
  std::vector<int64_t> none;
  PyObject *code = PyObject_GetAttrString(function, "__code__");
  if (!code) {
    PyErr_Clear();
    return none;
  }
  struct Holder {
    std::vector<PyObject *> held;
    ~Holder() {
      for (PyObject *o : held)
        Py_XDECREF(o);
    }
  } holder;
  holder.held.push_back(code);
  long flags = intAttribute(code, "co_flags");
  if (flags < 0 ||
      (flags & (kCoVarargs | kCoVarkeywords | kCoGenerator | kCoCoroutine |
                kCoIterableCoroutine | kCoAsyncGenerator)))
    return none;
  // A closure's cells are classical._infer's.
  if (tupleSize(code, "co_freevars") != 0 ||
      tupleSize(code, "co_cellvars") != 0)
    return none;
  auto attribute = [&](PyObject *of, const char *name) {
    PyObject *v = PyObject_GetAttrString(of, name);
    if (!v)
      PyErr_Clear();
    holder.held.push_back(v);
    return v;
  };
  PyObject *consts = attribute(code, "co_consts");
  PyObject *names = attribute(code, "co_names");
  PyObject *varnames = attribute(code, "co_varnames");
  PyObject *globals = attribute(function, "__globals__");
  PyObject *builtins = attribute(function, "__builtins__");
  if (!consts || !names || !varnames || !globals || !builtins ||
      !PyTuple_Check(consts) || !PyTuple_Check(names) ||
      !PyTuple_Check(varnames) || !PyDict_Check(globals) ||
      !PyDict_Check(lengths))
    return none;

  std::vector<Instruction> instructions;
  if (!decode(t, code, instructions))
    return none;

  State state;
  state.locals.assign(PyTuple_Size(varnames), Value());
  Py_ssize_t parameters = PySequence_Size(parameterNames);
  if (parameters < 0) {
    PyErr_Clear();
    return none;
  }
  for (Py_ssize_t i = 0; i < parameters; ++i) {
    PyObject *name = PySequence_GetItem(parameterNames, i);
    if (!name) {
      PyErr_Clear();
      return none;
    }
    holder.held.push_back(name);
    Py_ssize_t slot = -1;
    for (Py_ssize_t j = 0; j < PyTuple_Size(varnames); ++j) {
      int same =
          PyObject_RichCompareBool(PyTuple_GetItem(varnames, j), name, Py_EQ);
      if (same < 0) {
        PyErr_Clear();
        return none;
      }
      if (same) {
        slot = j;
        break;
      }
    }
    if (slot < 0)
      return none;
    PyObject *length = PyDict_GetItemWithError(lengths, name);
    if (length) {
      long long n = PyLong_AsLongLong(length);
      if (n < 0) {
        PyErr_Clear();
        return none;
      }
      state.locals[slot] = Value::of(Kind::List, n);
    } else {
      if (PyErr_Occurred()) {
        PyErr_Clear();
        return none;
      }
      state.locals[slot] = Value::of(Kind::Scalar);
    }
  }

  Run run(t, std::move(instructions), consts, names, globals, builtins);
  try {
    run.run(std::move(state));
  } catch (const Declined &) {
    PyErr_Clear();
    return none;
  }
  if (run.returns.empty())
    return none;

  Py_ssize_t wanted = PySequence_Size(positions);
  if (wanted <= 0) {
    PyErr_Clear();
    return none;
  }
  std::vector<int64_t> out;
  for (Py_ssize_t k = 0; k < wanted; ++k) {
    PyObject *item = PySequence_GetItem(positions, k);
    long position = item ? PyLong_AsLong(item) : -1;
    Py_XDECREF(item);
    if (position < 0 || position >= resultCount) {
      PyErr_Clear();
      return none;
    }
    int64_t found = -1;
    for (const Value &returned : run.returns) {
      const Value *value = &returned;
      if (resultCount > 1) {
        if (returned.kind != Kind::Tuple || returned.n != resultCount)
          return none;
        value = &(*returned.items)[position];
      }
      if (value->kind != Kind::List && value->kind != Kind::Tuple)
        return none;
      if (found >= 0 && value->n != found)
        return none; // its returns differ
      found = value->n;
    }
    out.push_back(found);
  }
  return out;
}

} // namespace

void loadLengthInference() {
  Tables &t = tables();
  if (!t.ready)
    t.usable = loadTables(t);
}

PyObject *inferListLengths(PyObject *function, PyObject *names,
                           PyObject *lengths, PyObject *positions,
                           Py_ssize_t resultCount) {
  const Tables &t = tables();
  std::vector<int64_t> found;
  if (t.usable)
    found = lengthsOf(t, function, names, lengths, positions, resultCount);
  if (found.empty())
    Py_RETURN_NONE;
  PyObject *out = PyTuple_New(static_cast<Py_ssize_t>(found.size()));
  if (!out)
    return nullptr;
  for (size_t i = 0; i < found.size(); ++i) {
    PyObject *n = PyLong_FromLongLong(found[i]);
    if (!n) {
      Py_DECREF(out);
      return nullptr;
    }
    PyTuple_SetItem(out, static_cast<Py_ssize_t>(i), n);
  }
  return out;
}

} // namespace cudaq_pulse
