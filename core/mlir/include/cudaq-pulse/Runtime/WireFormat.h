// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// The wire format of a call to a classical function: the request and
// response headers, the payload type ids and the handler schema of the
// CUDA-Q Realtime messaging protocol, declared here so that cudaq-pulse needs
// no CUDA-Q header. tests/classical/test_wire_format.py checks the layout
// against the protocol. Payloads are little-endian and tightly packed in
// schema order.

#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string_view>

namespace cudaq_pulse::wire {

constexpr uint32_t kRequestMagic = 0x43555152;  // 'CUQR'
constexpr uint32_t kResponseMagic = 0x43555153; // 'CUQS'

struct RequestHeader {
  uint32_t magic;
  uint32_t functionId;
  uint32_t argLen;
  uint32_t requestId;
  uint64_t ptpTimestamp;
};

struct ResponseHeader {
  uint32_t magic;
  int32_t status;
  uint32_t resultLen;
  uint32_t requestId;
  uint64_t ptpTimestamp;
};

static_assert(sizeof(RequestHeader) == 24 && alignof(RequestHeader) == 8);
static_assert(sizeof(ResponseHeader) == 24 && alignof(ResponseHeader) == 8);
static_assert(offsetof(RequestHeader, ptpTimestamp) == 16);
static_assert(offsetof(ResponseHeader, ptpTimestamp) == 16);

enum PayloadType : uint8_t {
  kUInt8 = 0x10,
  kInt32 = 0x11,
  kInt64 = 0x12,
  kFloat32 = 0x13,
  kFloat64 = 0x14,
  kArrayUInt8 = 0x20,
  kArrayInt32 = 0x21,
  kArrayFloat32 = 0x22,
  kArrayFloat64 = 0x23,
  kBitPacked = 0x30,
};

struct TypeDesc {
  uint8_t typeId;
  uint8_t reserved[3];
  uint32_t sizeBytes;
  /// 1 for scalars, the element count of an array, the bit count of
  /// bit-packed data.
  uint32_t numElements;
};

constexpr size_t kMaxArgs = 8;
constexpr size_t kMaxResults = 4;

struct HandlerSchema {
  uint8_t numArgs;
  uint8_t numResults;
  uint16_t reserved;
  TypeDesc args[kMaxArgs];
  TypeDesc results[kMaxResults];
};

static_assert(sizeof(TypeDesc) == 12);
static_assert(sizeof(HandlerSchema) == 4 + 12 * (kMaxArgs + kMaxResults));

/// A host handler: reads the request (header and arguments) from `request`
/// and writes the response (header and results) to `response`; each buffer
/// holds `slotSize` bytes.
using HostHandler = void (*)(const void *request, void *response,
                             size_t slotSize);

/// The function id of a handler named `name`: 32-bit FNV-1a.
constexpr uint32_t fnv1a32(std::string_view name) {
  uint32_t hash = 2166136261u;
  for (char c : name) {
    hash ^= static_cast<uint8_t>(c);
    hash *= 16777619u;
  }
  return hash;
}

static_assert(fnv1a32("enqueue_syndromes") == 0x7ED8BE82u);

} // namespace cudaq_pulse::wire
