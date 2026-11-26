//===- ModuleLoader.h - A scheduled function of another compiler's module -===//
//
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
//
// What load() reads from a module that another compiler wrote, and the module
// it writes from it. The module is untrusted input: what is checked of it is
// said at each function. These are in a translation unit of their own so
// that the code compiling a kernel is built as it is without them.
//
//===----------------------------------------------------------------------===//

#ifndef CUDAQ_PULSE_BINDINGS_MODULELOADER_H
#define CUDAQ_PULSE_BINDINGS_MODULELOADER_H

#include <Python.h>

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"
#include <cstdint>

namespace cudaq_pulse::loader {

// Each function takes the module as a Python bytes object (assembly or
// bytecode) with the GIL held, and releases the GIL while it reads the
// module. One that returns a PyObject gives a new reference, or null with a
// Python exception set when Python fails. What the module does not allow
// (or an argument of another kind than said here) is thrown as
// std::invalid_argument, and a module that does not parse or
// verify as std::runtime_error, with the reasons.

/// A list with one tuple per classical function the module declares, in its
/// order: name, dispatch key, link, argument and result descriptors (lists
/// of str), (min, nominal, max) latency in ns, mode, submit policy, timeout,
/// deadline and reaction deadline (each a float or None).
PyObject *callables(mlir::MLIRContext *ctx, PyObject *module);

/// A dict from link id to the name the module gives that link.
PyObject *linkNames(mlir::MLIRContext *ctx, PyObject *module);

/// A list with one tuple per submission of the module: the function it
/// calls, the cycles its schedule gives it, and the cycles its link takes
/// as the classical check wrote them on it (-1 when it wrote none).
PyObject *submissions(mlir::MLIRContext *ctx, PyObject *module);

/// The scheduled function `entry` (a str) of the module, as a module of its
/// own whose `main` takes the function's qudits as slots, in argument order.
/// `latencies`, `keys` and `devices` are dicts that replace what the module
/// states for its classical functions: (min, nominal, max) latency in ns
/// and dispatch key by function name, and link id by the module's link id.
mlir::OwningOpRef<mlir::ModuleOp> function(mlir::MLIRContext *ctx,
                                           PyObject *module, PyObject *entry,
                                           double clockGHz, int64_t qudits,
                                           PyObject *latencies, PyObject *keys,
                                           PyObject *devices);

} // namespace cudaq_pulse::loader

#endif // CUDAQ_PULSE_BINDINGS_MODULELOADER_H
