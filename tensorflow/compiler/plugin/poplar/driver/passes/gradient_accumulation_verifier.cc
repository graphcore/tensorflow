/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/plugin/poplar/driver/passes/gradient_accumulation_verifier.h"

#include <memory>

#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/stateful_gradient_accumulate.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"

#include "tensorflow/compiler/xla/service/call_graph.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"

namespace xla {
namespace poplarplugin {

StatusOr<bool> GradientAccumulationVerifier::Run(HloModule* module) {
  std::unique_ptr<CallGraph> call_graph = CallGraph::Build(module);

  for (HloComputation* comp : module->MakeComputationPostOrder()) {
    if (IsPopOpsFusion(comp)) {
      continue;
    }
    for (HloInstruction* inst : comp->instructions()) {
      if (inst->opcode() != HloOpcode::kCustomCall) {
        continue;
      }

      if (replication_factor_ > 1 &&
          IsPoplarInstruction(PoplarOp::StatefulGradientAccumulateWithMomentum)(
              inst)) {
        return FailedPrecondition(
            "Detected a gradient accumulation operation with momentum in a "
            "replicated graph which should have been fused with all reduce and "
            "normalisation instructions. Please use the "
            "`CrossReplicaGradientAccumulationOptimizer` optimizer for "
            "replicated graphs and gradient accumulation.\n"
            "This check can be disabled with "
            "`XLA_FLAGS=\"--xla_disable_hlo_passes=gradient-accumulation-"
            "verifier\"`, however the compiler cannot guarantee "
            "correctness of results.");
      }

      // Only do the check for the non-pipelined gradient accumulation ops.
      const bool is_grad_accumulation_op =
          IsPoplarInstruction(PoplarOp::StatefulGradientAccumulate)(inst) ||
          IsPoplarInstruction(PoplarOp::StatefulGradientAccumulateAndAllReduce)(
              inst) ||
          IsPoplarInstruction(PoplarOp::StatefulGradientAccumulateWithMomentum)(
              inst) ||
          IsPoplarInstruction(
              PoplarOp::
                  StatefulGradientAccumulateWithMomentumAndAllReduceWithNorm)(
              inst);

      if (is_grad_accumulation_op) {
        auto* grad_inst = Cast<HloStatefulGradientAccumulate>(inst);
        const int32 num_mini_batches = grad_inst->MiniBatchesToAccumulate();

        // Expect this computation to have a unique call site.
        auto call_graph_node = call_graph->GetNode(comp);
        auto& callsites = call_graph_node.caller_callsites();
        if (callsites.size() == 0) {
          return FailedPrecondition(
              "Detected a gradient accumulation operation which is not inside "
              "a training loop. The model with the gradient accumulation "
              "operation needs to be wrapped in a training loop.\n"
              "This check can be disabled with "
              "`XLA_FLAGS=\"--xla_disable_hlo_passes=gradient-accumulation-"
              "verifier\"`, however the compiler cannot guarantee "
              "correctness of results.");
        } else if (callsites.size() > 1) {
          return FailedPrecondition(
              "The call graph should have been flattened.");
        }

        HloInstruction* caller = callsites[0].instruction();
        // Expect this computation to either have been called form
        // 1. repeat loop, or
        // 2. while loop.
        if (IsRepeatLoop(caller)) {
          // Expect the number of mini batches to accumulate to divide the
          // repeat count.
          TF_ASSIGN_OR_RETURN(PoplarBackendConfig cfg,
                              caller->backend_config<PoplarBackendConfig>());
          const int64 repeat_count =
              cfg.call_config().repeat_config().repeat_count();
          if (repeat_count % num_mini_batches) {
            return FailedPrecondition(
                "Detected a gradient accumulation operation with %d number of "
                "mini batches inside a loop with %d iterations.\n"
                "It is required that the number of mini batches to accumulate "
                "evenly divides the number of loop iterations.\n"
                "This check can be disabled with "
                "`XLA_FLAGS=\"--xla_disable_hlo_passes=gradient-accumulation-"
                "verifier\"`, however the compiler cannot guarantee "
                "correctness of results.",
                num_mini_batches, repeat_count);
          }
        } else if (caller->opcode() == HloOpcode::kWhile) {
          // If called from a while loop then issue a warning.
          LOG(INFO) << "Detected a gradient accumulation operation inside of a "
                       "while loop. This might result in unexpected numerical "
                       "results if the number of mini batches to accumulate "
                       "does not divide the number of iterations.";
        } else {
          return FailedPrecondition(
              "Detected a gradient accumulation operation from an unexpected "
              "callsite. Gradient accumulation operations are only allowed "
              "inside of training loops.\n"
              "This check can be disabled with "
              "`XLA_FLAGS=\"--xla_disable_hlo_passes=gradient-accumulation-"
              "verifier\"`, however the compiler cannot guarantee "
              "correctness of results.");
        }
      }
    }
  }
  return false;
}

}  // namespace poplarplugin
}  // namespace xla
