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
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/xla/service/call_graph.h"
#include "tensorflow/compiler/xla/service/call_inliner.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/core/errors.h"

namespace xla {
namespace poplarplugin {

namespace {
StatusOr<bool> ShouldInlineComputation(const HloInstruction* inst) {
  if (inst->opcode() != HloOpcode::kCall) {
    return false;
  }
  TF_ASSIGN_OR_RETURN(PoplarBackendConfig config,
                      inst->backend_config<PoplarBackendConfig>());

  switch (config.call_config().type()) {
    case PoplarBackendConfig::CallConfig::Call: {
      // Always inline default call type.
      return true;
    }
    case PoplarBackendConfig::CallConfig::Function: {
      // Inline function calls if they have side effect.
      const bool has_side_effect = inst->to_apply()->HasSideEffect();
      if (has_side_effect) {
        LOG(INFO)
            << "Inlining function " << inst->to_apply()->name()
            << " because it contains stateful operations which means that "
               "the Poplar function cannot be reused.";
      }
      return has_side_effect;
    }
    default: { return false; }
  }
}

// If this computation has only one caller, and the callsite is a kCall
// operation, then merge with the calling computation.
Status FlattenNode(const CallGraphNode& node) {
  if (node.caller_callsites().size() == 1 &&
      !IsPopOpsFusion(node.computation())) {
    CallSite call_site = node.caller_callsites()[0];
    TF_ASSIGN_OR_RETURN(bool inline_computation,
                        ShouldInlineComputation(call_site.instruction()));

    if (inline_computation) {
      TF_ASSIGN_OR_RETURN(CallInliner::InlinedInstructionMap map,
                          CallInliner::Inline(call_site.instruction()));
    }
  }
  return Status::OK();
}

}  // namespace

StatusOr<bool> ComputationFlattener::Run(HloModule* module) {
  std::unique_ptr<CallGraph> call_graph = CallGraph::Build(module);
  TF_RETURN_IF_ERROR(call_graph->VisitNodes(FlattenNode));
  return true;
}

}  // namespace poplarplugin
}  // namespace xla
