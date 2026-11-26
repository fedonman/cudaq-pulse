// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// The timing of classical calls: what a link costs and how long an answer is
// budgeted, in clock cycles.
//
// A link's parameters come from the module attribute `classical.links`, a
// dictionary from the link id (as a string) to a dictionary with any of
// `fixed_ns` (fixed cost of sending a request), `bytes_per_ns` (bandwidth;
// 0 for unbounded), `feedback_ns` (from an answer arriving to a pulse being
// able to use it), `start_uncertainty_ns` (on a link that joins controllers,
// how far apart the starts of the two it joins known least well may be),
// `slots` (requests a link holds at once) and `slot_bytes` (the
// largest request or answer, header included). On several controllers, a
// link with `per_controller` (and its `name`) is one instance per
// controller, each with its own slots and its own order of submissions.
//
// An answer's budget adds the start uncertainty between the controller that
// sends the request and the one that takes the answer: none when one
// controller does both, the link's `start_uncertainty_ns` when the module's
// the control system (see Controllers.h) does not say which they are.
//
// `annotateClassicalCalls` writes on every `classical.submit` the attributes
// scheduling reads: `classical.submit_cycles` (how long the submission holds
// the link), `classical.budget_cycles` (from its end until the answer is due),
// `classical.device`, `classical.slots` and `classical.eager`; and on
// every `classical.wait` `classical.device` and, in blocking mode,
// `classical.blocking`. A call over a per-controller link gets, on its
// submission and its waits, `classical.instance`: the controller that sends
// its measurements and takes its answer, whose instance runs it.
//
// Lines that no run-time wait ties together wait independently: the
// scheduler numbers each such group of lines a timeline, `classical.timeline`
// on their allocations and on the calls made for them. A link used by
// several timelines shares out its slots between them and sends for each in
// turn, so each timeline's requests are timed as if the link were its own.
//
// Include after the Pulse and Classical op headers.

#pragma once

#include "cudaq-pulse/Analysis/Controllers.h"
#include "cudaq-pulse/Dialect/Classical/Schema.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"

#include <algorithm>
#include <cstdint>
#include <map>
#include <optional>
#include <tuple>
#include <vector>

namespace classical {

struct LinkModel {
  double fixedNs = 0.0;
  double bytesPerNs = 0.0;
  double feedbackNs = 0.0;
  double uncertaintyNs = 0.0;
  int64_t slots = 64;
  int64_t slotBytes = 1024;
  bool perController = false;
  llvm::StringRef name;
};

inline LinkModel linkModel(mlir::ModuleOp module, int64_t device) {
  LinkModel link;
  auto links = module->getAttrOfType<mlir::DictionaryAttr>("classical.links");
  if (!links)
    return link;
  auto entry = links.getAs<mlir::DictionaryAttr>(std::to_string(device));
  if (!entry)
    return link;
  auto number = [&](llvm::StringRef name, double fallback) {
    mlir::Attribute attr = entry.get(name);
    if (auto f = mlir::dyn_cast_or_null<mlir::FloatAttr>(attr))
      return f.getValueAsDouble();
    if (auto i = mlir::dyn_cast_or_null<mlir::IntegerAttr>(attr))
      return static_cast<double>(i.getInt());
    return fallback;
  };
  link.fixedNs = number("fixed_ns", link.fixedNs);
  link.bytesPerNs = number("bytes_per_ns", link.bytesPerNs);
  link.feedbackNs = number("feedback_ns", link.feedbackNs);
  link.uncertaintyNs = number("start_uncertainty_ns", link.uncertaintyNs);
  link.slots =
      static_cast<int64_t>(number("slots", static_cast<double>(link.slots)));
  link.slotBytes = static_cast<int64_t>(
      number("slot_bytes", static_cast<double>(link.slotBytes)));
  link.perController = static_cast<bool>(entry.get("per_controller"));
  if (auto name = entry.getAs<mlir::StringAttr>("name"))
    link.name = name.getValue();
  return link;
}

/// Wire bytes of the descriptors in `list`.
inline int64_t payloadBytes(mlir::ArrayAttr list) {
  int64_t bytes = 0;
  for (mlir::Attribute entry : list)
    if (auto desc = parseDescriptor(mlir::cast<mlir::StringAttr>(entry)))
      bytes += desc->wireBytes();
  return bytes;
}

constexpr int64_t kHeaderBytes = sizeof(cudaq_pulse::wire::RequestHeader);

inline int64_t requestBytes(CallableOp callable) {
  return kHeaderBytes + payloadBytes(callable.getArgs());
}

inline int64_t answerBytes(CallableOp callable) {
  return kHeaderBytes + payloadBytes(callable.getResults());
}

/// How long a submission holds the link: at least one cycle.
inline int64_t submitCycles(CallableOp callable, const LinkModel &link,
                            double clockGHz) {
  double ns = link.fixedNs;
  if (link.bytesPerNs > 0.0)
    ns += static_cast<double>(requestBytes(callable)) / link.bytesPerNs;
  return std::max<int64_t>(1, ceilCycles(ns, clockGHz));
}

/// From the end of a submission until its answer is due: the maximum
/// latency (the timeout in timeout mode, or the timeout of the wait that
/// takes the answer), the link's feedback latency and the start uncertainty
/// between sender and taker (`uncertaintyNs`, else the link's).
inline int64_t
budgetCycles(CallableOp callable, const LinkModel &link, double clockGHz,
             std::optional<double> waitTimeoutNs = std::nullopt,
             std::optional<double> uncertaintyNs = std::nullopt) {
  double latency = waitTimeoutNs ? *waitTimeoutNs
                   : callable.getMode() == "timeout"
                       ? callable.getTimeoutNs()->convertToDouble()
                       : callable.getLatencyMaxNs().convertToDouble();
  return ceilCycles(latency, clockGHz) +
         ceilCycles(link.feedbackNs +
                        uncertaintyNs.value_or(link.uncertaintyNs),
                    clockGHz);
}

/// The waits that take `future`, following it into and out of
/// `pulse.schedule_block`s.
inline void waitsOf(mlir::Value future, llvm::SmallVectorImpl<WaitOp> &found) {
  for (mlir::OpOperand &use : future.getUses()) {
    mlir::Operation *user = use.getOwner();
    if (auto wait = mlir::dyn_cast<WaitOp>(user))
      found.push_back(wait);
    else if (auto block = mlir::dyn_cast<pulse::ScheduleBlockOp>(user))
      waitsOf(block.getBody().getArgument(use.getOperandNumber()), found);
    else if (mlir::isa<pulse::YieldOp>(user))
      if (auto block =
              mlir::dyn_cast<pulse::ScheduleBlockOp>(user->getParentOp()))
        waitsOf(block->getResult(use.getOperandNumber()), found);
  }
}

inline llvm::SmallVector<WaitOp, 1> waitsOf(SubmitOp submit) {
  llvm::SmallVector<WaitOp, 1> found;
  waitsOf(submit.getFuture(), found);
  return found;
}

/// Adds to `found` the controllers of the readouts `value` is computed from,
/// each once; false if one is on no known controller.
inline bool readers(mlir::Value value, const pulse::Controllers &controllers,
                    pulse::AllocationTracer &trace,
                    llvm::SmallVectorImpl<int64_t> &found,
                    llvm::DenseSet<mlir::Value> &seen) {
  // The values still to follow, the next one last: an operation's operands
  // in their order, each with all it is computed from before the next.
  llvm::SmallVector<mlir::Value, 16> todo{value};
  while (!todo.empty()) {
    mlir::Value next = todo.pop_back_val();
    if (!seen.insert(next).second)
      continue;
    if (auto readout = next.getDefiningOp<pulse::ReadoutOp>()) {
      if (next != readout.getResult())
        continue;
      int64_t reader = controllers.of(readout->getOperand(0), trace);
      if (!llvm::is_contained(found, reader))
        found.push_back(reader);
      if (reader < 0)
        return false;
      continue;
    }
    if (mlir::isa<pulse::DriveLineType, pulse::ReadoutLineType, pulse::ToneType,
                  LinkType>(next.getType()))
      continue;
    mlir::Operation *def = next.getDefiningOp();
    if (!def || mlir::isa<WaitOp, SubmitOp>(def) || def->getNumRegions())
      continue;
    for (mlir::Value operand : llvm::reverse(def->getOperands()))
      todo.push_back(operand);
  }
  return true;
}

/// The controllers of the pulses that use a value, or of the branches on
/// it, each once; not `known` if one of them is unknown, or the value
/// leaves what can be followed (the controllers then say nothing).
struct Takers {
  llvm::SmallVector<int64_t, 2> controllers;
  bool known = true;
  /// Being followed: its uses are not all done.
  bool open = false;

  void add(int64_t controller) {
    if (!llvm::is_contained(controllers, controller))
      controllers.push_back(controller);
  }
};

/// The takers of the values followed so far in a function that does not
/// change meanwhile.
using TakersByValue = llvm::DenseMap<mlir::Value, Takers>;

/// Adds to `found` the controllers of the pulses that use `value`, or of
/// the branches on it, each once; false if one of them is unknown, or
/// `value` leaves what can be followed.
///
/// A value is followed through the results of the ops that use it: a
/// branch on an answer gives the lines its arms touch, and a line goes on
/// through every sync and branch after it. That can be as long as the
/// program, so the walk keeps its own stack: per value its uses, and for
/// the use at hand the results still to follow, in the order a recursion
/// would visit them. And the calls of a program share most of it, so each
/// value's takers are kept in `takers` and followed once for all calls.
inline bool takers(mlir::Value value, const pulse::Controllers &controllers,
                   pulse::AllocationTracer &trace,
                   llvm::SmallVectorImpl<int64_t> &found,
                   TakersByValue &takers) {
  struct Frame {
    mlir::Value value;
    mlir::Value::use_iterator use, end;
    llvm::SmallVector<mlir::Value, 2> results;
    unsigned next = 0;
    Takers taken;
  };
  llvm::SmallVector<Frame, 8> frames;
  // What `next` reaches goes to the value above it: at once if it was
  // followed before, else when its own uses are done.
  auto follow = [&](mlir::Value next, Takers &into) {
    auto old = takers.find(next);
    if (old == takers.end()) {
      takers[next].open = true;
      frames.push_back({next, next.use_begin(), next.use_end(), {}, 0, {}});
      return true;
    }
    // A value that uses itself, which a block nothing reaches may hold:
    // not followed.
    if (old->second.open)
      return false;
    for (int64_t controller : old->second.controllers)
      into.add(controller);
    return old->second.known;
  };
  // Every value on the stack reaches the use that cannot be followed.
  auto unknown = [&] {
    for (Frame &frame : frames)
      takers[frame.value] = {{}, false, false};
    return false;
  };
  Takers all;
  if (!follow(value, all))
    return false;
  while (!frames.empty()) {
    Frame &frame = frames.back();
    if (frame.next < frame.results.size()) {
      // Not `frame` after this: following a value may move the frames.
      mlir::Value result = frame.results[frame.next++];
      if (!follow(result, frame.taken))
        return unknown();
      continue;
    }
    if (frame.use == frame.end) {
      Takers taken = std::move(frame.taken);
      mlir::Value done = frame.value;
      frames.pop_back();
      Takers &above = frames.empty() ? all : frames.back().taken;
      for (int64_t controller : taken.controllers)
        above.add(controller);
      takers[done] = std::move(taken);
      continue;
    }
    mlir::OpOperand &use = *frame.use++;
    frame.results.clear();
    frame.next = 0;
    mlir::Operation *user = use.getOwner();
    auto pulse = [&](mlir::Operation *op) {
      int64_t taker = controllers.of(op->getOperand(0), trace);
      frame.taken.add(taker);
      return taker >= 0;
    };
    if (mlir::isa<pulse::DriveOp, pulse::ReadoutOp, pulse::WaitOp>(user)) {
      if (!pulse(user))
        return unknown();
    } else if (auto branch = mlir::dyn_cast<mlir::scf::IfOp>(user)) {
      bool known = true;
      branch->walk([&](mlir::Operation *inner) {
        if (known &&
            mlir::isa<pulse::DriveOp, pulse::ReadoutOp, pulse::WaitOp>(inner))
          known = pulse(inner);
      });
      if (!known)
        return unknown();
      frame.results.append(branch.getResults().begin(),
                           branch.getResults().end());
    } else if (mlir::isa<mlir::scf::YieldOp>(user)) {
      auto branch = mlir::dyn_cast<mlir::scf::IfOp>(user->getParentOp());
      if (!branch)
        return unknown();
      frame.results.push_back(branch->getResult(use.getOperandNumber()));
    } else if (user->getNumRegions() || user->getNumResults() == 0 ||
               mlir::isa<SubmitOp>(user)) {
      return unknown();
    } else {
      frame.results.append(user->getResults().begin(),
                           user->getResults().end());
    }
  }
  for (int64_t controller : all.controllers)
    if (!llvm::is_contained(found, controller))
      found.push_back(controller);
  return true;
}

/// The controllers that send the request of `submit` (`senders`) and take
/// its answer (`users`); false if one of them is not known. `takers`: one
/// for all the calls of a function, while it does not change.
inline bool callEnds(SubmitOp submit, const pulse::Controllers &controllers,
                     pulse::AllocationTracer &trace,
                     llvm::SmallVectorImpl<int64_t> &senders,
                     llvm::SmallVectorImpl<int64_t> &users,
                     TakersByValue &takers) {
  llvm::DenseSet<mlir::Value> seen;
  for (mlir::Value operand : submit->getOperands())
    if (!readers(operand, controllers, trace, senders, seen))
      return false;
  for (WaitOp wait : waitsOf(submit))
    for (mlir::Value value : wait.getValues())
      if (!classical::takers(value, controllers, trace, users, takers))
        return false;
  return true;
}

/// On several controllers, the start uncertainty between the controllers
/// that send the request of `submit` and take its answer, in ns: none when
/// one controller does both. Empty when they are not known. `takers`: as
/// for callEnds.
inline std::optional<double>
callUncertaintyNs(SubmitOp submit, const pulse::Controllers &controllers,
                  pulse::AllocationTracer &trace, TakersByValue &takers) {
  if (!controllers.enabled())
    return std::nullopt;
  llvm::SmallVector<int64_t, 2> senders, users;
  if (!callEnds(submit, controllers, trace, senders, users, takers) ||
      senders.empty() || users.empty())
    return std::nullopt;
  double most = 0.0;
  for (int64_t sender : senders)
    for (int64_t user : users)
      if (sender != user)
        most = std::max(most, controllers.uncertainty(sender) +
                                  controllers.uncertainty(user));
  return most;
}

/// The timeout of the wait that takes the answer of `submit`, if it has one.
inline std::optional<double> waitTimeoutNs(SubmitOp submit) {
  for (WaitOp wait : waitsOf(submit))
    if (auto timeout = wait.getTimeoutNs())
      return timeout->convertToDouble();
  return std::nullopt;
}

inline double moduleClockGHz(mlir::ModuleOp module) {
  return pulse::clockGHz(module);
}

/// Whether `module` declares any classical function: only then can it hold
/// classical calls. Looks at top-level ops only.
inline bool declaresClassicalCalls(mlir::ModuleOp module) {
  for (mlir::Operation &op : *module.getBody())
    if (mlir::isa<CallableOp>(op))
      return true;
  return false;
}

inline constexpr const char *kTimelineAttr = "classical.timeline";
inline constexpr const char *kInstanceAttr = "classical.instance";

/// On several controllers, the controller whose instance of a per-controller
/// link runs each call of `function` over one: the controller that sends its
/// measurements and takes its answer. A link with a call whose controllers
/// are not known, or that has neither, keeps one set of slots for all its
/// calls (they get no entry). Fails, with an error on the call, when a call
/// uses several controllers.
inline mlir::LogicalResult
callInstances(mlir::func::FuncOp function, mlir::ModuleOp module,
              mlir::SymbolTable &symbols, const pulse::Controllers &controllers,
              pulse::AllocationTracer &trace, TakersByValue &takers,
              llvm::DenseMap<mlir::Operation *, int64_t> &found) {
  auto links = module->getAttrOfType<mlir::DictionaryAttr>("classical.links");
  if (!controllers.enabled() || !links ||
      llvm::none_of(links, [](mlir::NamedAttribute entry) {
        auto link = mlir::dyn_cast<mlir::DictionaryAttr>(entry.getValue());
        return link && link.get("per_controller");
      }))
    return mlir::success();
  std::map<int64_t, bool> pooled;
  std::vector<std::tuple<mlir::Operation *, int64_t, int64_t>> calls;
  bool failed = false;
  function.walk([&](SubmitOp submit) {
    auto callable =
        symbols.lookup<CallableOp>(submit.getCalleeAttr().getAttr());
    if (!callable)
      return;
    int64_t device = callable.getDevice();
    LinkModel link = linkModel(module, device);
    if (!link.perController)
      return;
    llvm::SmallVector<int64_t, 2> ends, users;
    bool known = callEnds(submit, controllers, trace, ends, users, takers);
    ends.append(users.begin(), users.end());
    llvm::sort(ends);
    ends.erase(std::unique(ends.begin(), ends.end()), ends.end());
    bool &pool = pooled[device];
    if (!known || ends.empty()) {
      pool = true;
      return;
    }
    if (ends.size() > 1) {
      auto diag = submit.emitError("the call of '")
                  << submit.getCallee() << "' over link '" << link.name
                  << "', which has one instance per controller, uses ";
      for (auto [k, c] : llvm::enumerate(ends))
        diag << (k == 0                 ? ""
                 : k + 1 == ends.size() ? " and "
                                        : ", ")
             << "controllers[" << c << "]";
      diag << "; a call over it runs on one controller: play its "
              "measurements and the pulses that use its answer on one "
              "controller, or make the link shared (drop per_controller)";
      failed = true;
      return;
    }
    calls.emplace_back(submit, device, ends.front());
  });
  if (failed)
    return mlir::failure();
  for (auto [op, device, instance] : calls)
    if (!pooled[device])
      found[op] = instance;
  return mlir::success();
}

/// Write the timing attributes of every call in `function` (see the file
/// comment). Returns false if `function` makes no call; fails when a call
/// over a per-controller link uses several controllers.
inline mlir::FailureOr<bool>
annotateClassicalCalls(mlir::func::FuncOp function) {
  auto module = function->getParentOfType<mlir::ModuleOp>();
  if (!module || !declaresClassicalCalls(module))
    return false;
  double clock = moduleClockGHz(module);
  mlir::SymbolTable symbols(module);
  mlir::Builder b(function.getContext());
  pulse::Controllers controllers(module);
  pulse::AllocationTracer trace;
  llvm::DenseMap<mlir::Operation *, int64_t> instances;
  // Setting attributes below changes no use, so one table serves all calls.
  TakersByValue takers;
  if (mlir::failed(callInstances(function, module, symbols, controllers, trace,
                                 takers, instances)))
    return mlir::failure();
  bool any = false;
  // Every name interned and every link read once, not once per call: each
  // interning is a lookup in the context's uniquer.
  mlir::MLIRContext *context = function.getContext();
  auto name = [&](llvm::StringRef text) {
    return mlir::StringAttr::get(context, text);
  };
  mlir::StringAttr submitName = name("classical.submit_cycles"),
                   budgetName = name("classical.budget_cycles"),
                   deviceName = name("classical.device"),
                   inFlightName = name("classical.slots"),
                   eagerName = name("classical.eager"),
                   blockingName = name("classical.blocking"),
                   instanceName = name(kInstanceAttr),
                   timelineName = name(kTimelineAttr);
  std::map<int64_t, LinkModel> links;
  auto linkOf = [&](int64_t device) -> const LinkModel & {
    auto [it, inserted] = links.try_emplace(device);
    if (inserted)
      it->second = linkModel(module, device);
    return it->second;
  };
  // Sets an i64 attribute, without making one when it holds the value.
  auto setInt = [&](mlir::NamedAttrList &attrs, mlir::StringAttr key,
                    int64_t value) {
    auto old = mlir::dyn_cast_or_null<mlir::IntegerAttr>(attrs.get(key));
    if (!old || old.getType() != b.getI64Type() || old.getInt() != value)
      attrs.set(key, b.getI64IntegerAttr(value));
  };
  auto calleeOf = [&](mlir::FlatSymbolRefAttr callee) {
    return symbols.lookup<CallableOp>(callee.getAttr());
  };
  // The timelines on each link (and instance), for the submissions that
  // have one.
  using Slots = std::pair<int64_t, int64_t>;
  std::vector<std::tuple<mlir::Operation *, int64_t, Slots>> shared;
  std::map<Slots, std::vector<int64_t>> timelines;
  function.walk([&](mlir::Operation *op) {
    if (auto submit = mlir::dyn_cast<SubmitOp>(op)) {
      auto callable = calleeOf(submit.getCalleeAttr());
      if (!callable)
        return;
      any = true;
      const LinkModel &link = linkOf(callable.getDevice());
      auto instance = instances.find(op);
      Slots slots{callable.getDevice(),
                  instance == instances.end() ? -1 : instance->second};
      mlir::NamedAttrList attrs(op->getDiscardableAttrDictionary());
      setInt(attrs, submitName, submitCycles(callable, link, clock));
      setInt(
          attrs, budgetName,
          budgetCycles(callable, link, clock, waitTimeoutNs(submit),
                       callUncertaintyNs(submit, controllers, trace, takers)));
      setInt(attrs, deviceName, callable.getDevice());
      setInt(attrs, inFlightName, link.slots);
      if (callable.getSubmit() == "eager")
        attrs.set(eagerName, b.getUnitAttr());
      else
        attrs.erase(eagerName);
      if (slots.second >= 0) {
        setInt(attrs, instanceName, slots.second);
        for (WaitOp wait : waitsOf(submit)) {
          mlir::NamedAttrList waitAttrs(wait->getDiscardableAttrDictionary());
          setInt(waitAttrs, instanceName, slots.second);
          wait->setDiscardableAttrs(waitAttrs.getDictionary(context));
        }
      } else {
        attrs.erase(instanceName);
      }
      if (auto timeline = attrs.get(timelineName)) {
        int64_t t = mlir::cast<mlir::IntegerAttr>(timeline).getInt();
        shared.emplace_back(op, t, slots);
        timelines[slots].push_back(t);
      }
      op->setDiscardableAttrs(attrs.getDictionary(context));
    } else if (auto wait = mlir::dyn_cast<WaitOp>(op)) {
      auto callable = calleeOf(wait.getCalleeAttr());
      if (!callable)
        return;
      any = true;
      mlir::NamedAttrList attrs(op->getDiscardableAttrDictionary());
      setInt(attrs, deviceName, callable.getDevice());
      if (callable.getMode() == "blocking")
        attrs.set(blockingName, b.getUnitAttr());
      op->setDiscardableAttrs(attrs.getDictionary(context));
    }
  });
  // A link that T timelines use gives each its share of the slots, the
  // first ones one more when they do not divide evenly. A timeline sends one
  // request at a time, so each is sent at most after one pending send of
  // every other timeline: it holds the link for its own send and T - 1 of
  // the link's longest.
  auto cyclesOf = [&](mlir::Operation *op) {
    return mlir::cast<mlir::IntegerAttr>(op->getDiscardableAttr(submitName))
        .getInt();
  };
  std::map<Slots, int64_t> longest;
  for (auto [op, t, slots] : shared) {
    int64_t &most = longest[slots];
    most = std::max(most, cyclesOf(op));
  }
  for (auto &[slots, list] : timelines) {
    std::sort(list.begin(), list.end());
    list.erase(std::unique(list.begin(), list.end()), list.end());
  }
  for (auto [op, t, key] : shared) {
    const std::vector<int64_t> &list = timelines[key];
    auto count = static_cast<int64_t>(list.size());
    if (count <= 1)
      continue;
    auto rank = static_cast<int64_t>(
        std::lower_bound(list.begin(), list.end(), t) - list.begin());
    int64_t slots = linkOf(key.first).slots;
    int64_t share = slots / count + (rank < slots % count ? 1 : 0);
    int64_t send = cyclesOf(op) + (count - 1) * longest[key];
    mlir::NamedAttrList attrs(op->getDiscardableAttrDictionary());
    setInt(attrs, submitName, send);
    setInt(attrs, inFlightName, std::max<int64_t>(1, share));
    op->setDiscardableAttrs(attrs.getDictionary(context));
  }
  return any;
}

/// Why a wait cannot be in a `pulse.schedule_block`, which is placed whole:
/// the verifier and the scheduler, which may run without it, both say so.
inline constexpr const char *kVariableWaitInBlock =
    "a variable wait cannot be inside a pulse.schedule_block, whose length "
    "is fixed; call the kernel without schedule= or move the wait out of it";
inline constexpr const char *kBlockingWaitInBlock =
    "a blocking wait cannot be inside a pulse.schedule_block, whose length "
    "is fixed; return the future from the kernel and take the answer after "
    "it returns, or call the kernel without schedule=";

/// A controller that plays nothing while a call is out (`blocks_on_calls`
/// in the module's control system, see pulse::blocksOnCalls): each call
/// holds every line from its submission until its answer is due. On several
/// controllers, a call holds the lines of each of those that issues it,
/// reads a measurement it sends or plays an op that uses its answer.
inline constexpr const char *kBlockingCallInBlock =
    "the controller blocks on calls, holding every line while one is out, "
    "which a pulse.schedule_block, placed whole, cannot; call the kernel "
    "without schedule=, or compile with flatten_schedules=True";

inline int64_t intAttr(mlir::Operation *op, llvm::StringRef name,
                       int64_t fallback = 0) {
  if (auto attr = op->getAttrOfType<mlir::IntegerAttr>(name))
    return attr.getInt();
  return fallback;
}

/// intAttr() of a discardable attribute by its interned name.
inline int64_t intAttr(mlir::Operation *op, mlir::StringAttr name,
                       int64_t fallback = 0) {
  if (auto attr = mlir::dyn_cast_or_null<mlir::IntegerAttr>(
          op->getDiscardableAttr(name)))
    return attr.getInt();
  return fallback;
}

} // namespace classical
