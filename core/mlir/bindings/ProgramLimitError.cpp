// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// The error of a call whose values take the schedule past a controller's
// longest program (pulse::rethrowPastProgramLimit). Apart from the
// extension's other code: only the handler of a schedule that failed runs
// it.

#include "cudaq-pulse/Transforms/Passes.h"

#include <stdexcept>
#include <string_view>

void pulse::rethrowPastProgramLimit(const std::exception &error) {
  // The schedule's check says "kernel 'k' lasts [up to] n cycles ..., more
  // than <controller>'s max_program_duration of ...", on one line: the
  // sentence compile() reports for a kernel without values. With a call's
  // values the values are what is wrong. (A wait with no highest length or
  // a blocking call under the limit "runs under" it: the kernel's error.)
  std::string_view text = error.what();
  size_t limit = text.find("'s max_program_duration of ");
  size_t kernel = limit == text.npos ? limit : text.rfind("kernel '", limit);
  if (kernel != text.npos) {
    std::string_view sentence = text.substr(kernel);
    sentence = sentence.substr(0, sentence.find('\n'));
    if (sentence.find("' lasts ") < limit - kernel)
      throw std::invalid_argument(std::string(sentence));
  }
  throw;
}
