// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// The argument and result descriptors of `classical.callable` ("bits:4",
// "f64", "i32[8]", ...): how many SSA values each one takes or gives, and
// its size on the wire.

#pragma once

#include "cudaq-pulse/Runtime/WireFormat.h"

#include "llvm/ADT/StringRef.h"

#include <cmath>
#include <cstdint>
#include <optional>

namespace classical {

enum class Kind : uint8_t {
  U8,
  I32,
  I64,
  F32,
  F64,
  Bits,
  Levels,
  IQ,
  ArrayU8,
  ArrayI32,
  ArrayF32,
  ArrayF64,
};

struct Descriptor {
  Kind kind;
  int64_t count = 1;

  /// Its SSA values are `!pulse.measurement`s.
  bool takesMeasurements() const {
    return kind == Kind::Bits || kind == Kind::Levels || kind == Kind::IQ;
  }
  bool isFloat() const {
    return kind == Kind::F32 || kind == Kind::F64 || kind == Kind::ArrayF32 ||
           kind == Kind::ArrayF64 || kind == Kind::IQ;
  }
  /// SSA values it takes as an argument or gives as a result.
  int64_t values() const { return count; }

  int64_t wireBytes() const {
    switch (kind) {
    case Kind::U8:
    case Kind::Levels:
    case Kind::ArrayU8:
      return count;
    case Kind::I32:
    case Kind::F32:
    case Kind::ArrayI32:
    case Kind::ArrayF32:
      return 4 * count;
    case Kind::I64:
    case Kind::F64:
    case Kind::ArrayF64:
      return 8 * count;
    case Kind::Bits:
      return (count + 7) / 8;
    case Kind::IQ:
      return 8 * count;
    }
    return 0;
  }
};

/// Parse a descriptor; nullopt if it is malformed or its count is not
/// positive.
inline std::optional<Descriptor> parseDescriptor(llvm::StringRef text) {
  struct Scalar {
    llvm::StringRef name;
    Kind kind;
  };
  static const Scalar scalars[] = {{"u8", Kind::U8},
                                   {"i32", Kind::I32},
                                   {"i64", Kind::I64},
                                   {"f32", Kind::F32},
                                   {"f64", Kind::F64}};
  for (const Scalar &s : scalars)
    if (text == s.name)
      return Descriptor{s.kind, 1};
  int64_t count = 0;
  auto [head, tail] = text.split(':');
  if (!tail.empty()) {
    if (tail.getAsInteger(10, count) || count <= 0)
      return std::nullopt;
    if (head == "bits")
      return Descriptor{Kind::Bits, count};
    if (head == "levels")
      return Descriptor{Kind::Levels, count};
    if (head == "iq")
      return Descriptor{Kind::IQ, count};
    return std::nullopt;
  }
  if (!text.ends_with("]"))
    return std::nullopt;
  auto [element, rest] = text.drop_back().split('[');
  if (rest.getAsInteger(10, count) || count <= 0)
    return std::nullopt;
  if (element == "u8")
    return Descriptor{Kind::ArrayU8, count};
  if (element == "i32")
    return Descriptor{Kind::ArrayI32, count};
  if (element == "f32")
    return Descriptor{Kind::ArrayF32, count};
  if (element == "f64")
    return Descriptor{Kind::ArrayF64, count};
  return std::nullopt;
}

/// Clock cycles covering `ns` nanoseconds at `clockGHz`, rounded up so
/// a budget is never optimistic.
inline int64_t ceilCycles(double ns, double clockGHz) {
  double cycles = ns * clockGHz;
  double rounded = std::ceil(cycles - 1e-9);
  return rounded < 0 ? 0 : static_cast<int64_t>(rounded);
}

} // namespace classical
