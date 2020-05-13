/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/compiler/plugin/poplar/driver/passes/computation_flattener.h"

#include "tensorflow/compiler/plugin/poplar/driver/backend_config.pb.h"
#include "tensorflow/compiler/xla/service/call_graph.h"
#include "tensorflow/compiler/xla/service/call_inliner.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/core/errors.h"

namespace xla {
namespace poplarplugin {

Status ComputationFlattener::FlattenNode(const CallGraphNode& node) {
  if (node.caller_callsites().size() == 1 &&
      !IsPopOpsFusion(node.computation())) {
    CallSite call_site = node.caller_callsites()[0];
    HloInstruction* call_op = call_site.instruction();

    if (call_op->opcode() != HloOpcode::kCall) {
      return Status::OK();
    }
    TF_ASSIGN_OR_RETURN(PoplarBackendConfig config,
                        call_op->backend_config<PoplarBackendConfig>());

    bool inline_computation = false;
    switch (config.call_config().type()) {
      case PoplarBackendConfig::CallConfig::Call: {
        // Always inline default call type.
        inline_computation = true;
        break;
      }
      case PoplarBackendConfig::CallConfig::Function: {
        if (all_function_comps_.count(node.computation()) == 1) {
          // Inline functions if they are called from a single site.
          VLOG(1) << "Inlining function " << node.computation()->name()
                  << " because it is only called from one call site.";
          inline_computation = true;

        } else if (node.computation()->HasSideEffect()) {
          // Inline function calls if they have side effect.
          LOG(INFO)
              << "Inlining function " << node.computation()->name()
              << " because it contains stateful operations which means that "
                 "the Poplar function cannot be reused.";
          inline_computation = true;
        }
        break;
      }
      default:
        break;
    }

    if (inline_computation) {
      TF_ASSIGN_OR_RETURN(CallInliner::InlinedInstructionMap map,
                          CallInliner::Inline(call_op));
    }
  }
  return Status::OK();
}

// Construct a set of Function computations for equality comparison
Status ComputationFlattener::GenerateFunctionSet(const CallGraphNode& node) {
  if (node.caller_callsites().size() == 1 &&
      !IsPopOpsFusion(node.computation())) {
    CallSite call_site = node.caller_callsites()[0];
    const HloInstruction* caller = call_site.instruction();

    if (caller->opcode() != HloOpcode::kCall) {
      return Status::OK();
    }

    TF_ASSIGN_OR_RETURN(PoplarBackendConfig config,
                        caller->backend_config<PoplarBackendConfig>());

    auto type = config.call_config().type();
    if (type == PoplarBackendConfig::CallConfig::Function) {
      all_function_comps_.insert(node.computation());
    }
  }

  return Status::OK();
}

StatusOr<bool> ComputationFlattener::Run(HloModule* module) {
  std::unique_ptr<CallGraph> call_graph = CallGraph::Build(module);

  TF_RETURN_IF_ERROR(call_graph->VisitNodes(
      [this](const CallGraphNode& node) { return GenerateFunctionSet(node); }));
  TF_RETURN_IF_ERROR(call_graph->VisitNodes(
      [this](const CallGraphNode& node) { return FlattenNode(node); }));
  return true;
}

}  // namespace poplarplugin
}  // namespace xla
