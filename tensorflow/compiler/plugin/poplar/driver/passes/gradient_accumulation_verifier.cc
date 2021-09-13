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
#include <string>
#include <vector>

#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/stateful_gradient_accumulate.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/inplace_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/offloading_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/xla/service/call_graph.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"

namespace xla {
namespace poplarplugin {

Status GradientAccumulationVerifier::VerifyStatefulGradientAccumulation(
    HloInstruction* const inst, CallGraph* call_graph) {
  if (inst->opcode() != HloOpcode::kCustomCall) {
    return Status::OK();
  }
  HloComputation* comp = inst->parent();

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
          PoplarOp::StatefulGradientAccumulateWithMomentumAndAllReduceWithNorm)(
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
      return FailedPrecondition("The call graph should have been flattened.");
    }

    HloInstruction* caller = callsites[0].instruction();
    // This computation must be called from a repeat loop.
    if (IsRepeatLoop(caller)) {
      // Expect the number of mini batches to accumulate to divide the
      // repeat count.
      const int64 repeat_count = GetRepeatLoopCount(caller);
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
  return Status::OK();
}

namespace {

static StatusOr<HloInstruction*> FindCorrespondingResourceUpdate(
    HloInstruction* sink) {
  // Make sure the sink is only used by a resource update.
  if (sink->user_count() != 1) {
    return InternalErrorStrCat(
        "Expected the gradient accumulation sink to only have a single "
        "user, but has ",
        sink->user_count(), " users.");
  }
  HloInstruction* candidate = sink->users()[0];
  if (IsResourceUpdate(candidate)) {
    return candidate;
  }
  return InternalErrorStrCat(
      "Expected the gradient accumulation sink to be used by a resource "
      "update, but was used by ",
      candidate->name(), " instead.");
}

static Status HasGradientAccumulationInstruction(const HloInstruction* inst) {
  CHECK(IsResourceUpdate(inst));
  const bool check = absl::c_any_of(
      inst->to_apply()->instructions(), [](const HloInstruction* candidate) {
        return IsPoplarInstruction(PoplarOp::GradientAccumulationCount)(
            candidate);
      });
  return check ? Status::OK()
               : InternalErrorStrCat(
                     "No gradient accumulation count instruction found for"
                     " resource update instruction ",
                     inst->name());
}

StatusOr<int64> VerifyGradientAccumulationInsideComputation(
    const HloComputation* computation, int64 parameter_index) {
  // Expect the gradient accumulator to be used serially, with the final use in
  // the root tuple. We expect all the uses to be inplace on the buffer and look
  // inside of repeat loops.
  int64 output_index = parameter_index;
  HloInstruction* inner_user =
      computation->parameter_instruction(parameter_index);
  do {
    if (inner_user->user_count() == 2) {
      HloInstruction *load, *store;
      TF_RETURN_IF_ERROR(GetRemoteLoadStoreUsers(inner_user, &load, &store));
      inner_user = load;
      continue;
    }

    if (inner_user->user_count() != 1) {
      return InternalErrorStrCat(
          "Expected the gradient accumulation buffer to be used "
          "serially, but detected ",
          inner_user->user_count(), " users.");
    }
    HloInstruction* next_user = inner_user->users()[0];

    const auto next_user_indices = next_user->OperandIndices(inner_user);
    if (next_user_indices.size() != 1) {
      return InternalErrorStrCat(
          "Expected the gradient accumulation buffer to only appear as "
          "an operand once, but it is used ",
          next_user_indices.size(), " times.");
    }
    const int64 next_user_index = next_user_indices[0];

    if (IsPoplarInstruction(PoplarOp::RemoteParameterStore)(next_user)) {
      inner_user = next_user;
      output_index = next_user_index - next_user->operand_count() / 2;
      continue;
    }

    auto optional_inplace_modifier = GetInplaceModifier(inner_user);
    if (!optional_inplace_modifier) {
      return InternalError(
          "Expected the gradient accumulation buffer to be used "
          "inplace.");
    }
    HloInstruction* inplace_modifier = *optional_inplace_modifier;
    if (IsRepeatLoop(inplace_modifier)) {
      const auto indices = inplace_modifier->OperandIndices(inner_user);
      if (indices.size() != 1) {
        return InternalErrorStrCat(
            "Expected the gradient accumulation buffer to only appear as an "
            "operand once, but it is used ",
            indices.size(), " times.");
      }
      TF_ASSIGN_OR_RETURN(int64 gte_output_index,
                          VerifyGradientAccumulationInsideComputation(
                              inplace_modifier->to_apply(), indices[0]));
      TF_ASSIGN_OR_RETURN(HloInstruction * gte,
                          GetUniqueGTEUser(inplace_modifier, gte_output_index));
      inner_user = gte;
      output_index = 0;
    } else {
      if (inplace_modifier != next_user) {
        return InternalErrorStrCat(
            "Expected the gradient accumulation inplace user to be ",
            next_user->ToString(), " but it was ", inplace_modifier->ToString(),
            ".");
      }
      inner_user = next_user;
      output_index = next_user_index;
    }
  } while (computation->root_instruction() != inner_user);
  CHECK_EQ(inner_user->opcode(), HloOpcode::kTuple);
  return output_index;
}

std::vector<const HloInstruction*> GetGTEUsers(const HloInstruction* root_gte) {
  std::vector<const HloInstruction*> result;

  for (auto user : root_gte->users()) {
    if (user->opcode() == HloOpcode::kGetTupleElement) {
      auto tmp_result = GetGTEUsers(user);
      result.insert(result.end(), tmp_result.begin(), tmp_result.end());
    } else {
      result.push_back(user);
    }
  }

  return result;
}
}  // namespace

Status GradientAccumulationVerifier::VerifyGenericGradientAccumulation(
    HloInstruction* const inst, CallGraph* call_graph) {
  if (IsResourceUpdate(inst)) {
    return HasGradientAccumulationInstruction(inst);
  }

  if (!IsPoplarInstruction(PoplarOp::GradientAccumulatorCreate)(inst)) {
    return Status::OK();
  }

  if (inst->user_count() == 0) {
    return InternalErrorStrCat("Expected the gradient accumulation buffer ",
                               inst->ToString(), " to have at least one user.");
  }

  // Expect this computation to have a unique call site.
  auto call_graph_node = call_graph->GetNode(inst->parent());
  auto& callsites = call_graph_node.caller_callsites();
  if (callsites.size() == 0) {
    return FailedPrecondition(
        "Detected a gradient accumulation operation which is not inside "
        "a training loop. The model with the gradient accumulation "
        "operation needs to be wrapped in a training loop.\n");
  } else if (callsites.size() > 1) {
    return FailedPrecondition("The call graph should have been flattened.");
  }

  HloInstruction* caller = callsites[0].instruction();
  // Expect this computation to either have been called form
  // 1. repeat loop, or
  // 2. pipeline.
  if (IsRepeatLoop(caller)) {
    if (inst->user_count() != 1) {
      return InternalErrorStrCat(
          "Expected the gradient accumulation buffer to only have a single "
          "user, but has ",
          inst->user_count(), " users.");
    }
    HloInstruction* user = inst->users()[0];
    while (!IsPoplarInstruction(PoplarOp::GradientAccumulatorSink)(user)) {
      if (user->user_count() != 1) {
        return InternalErrorStrCat(
            "Expected the gradient accumulation buffer to be used "
            "serially, but detected ",
            user->user_count(), " users.");
      }
      HloInstruction* next_user = user->users()[0];
      const auto next_user_indices = next_user->OperandIndices(user);

      if (next_user_indices.size() != 1) {
        return InternalErrorStrCat(
            "Expected the gradient accumulation buffer to only appear as "
            "an operand once, but it is used ",
            next_user_indices.size(), " times.");
      }
      auto inplace_modifier = GetInplaceModifier(user);
      if (!inplace_modifier) {
        return InternalError(
            "Expected the gradient accumulation buffer to be used "
            "inplace.");
      }
      if (*inplace_modifier != next_user) {
        return InternalErrorStrCat(
            "Expected the gradient accumulation inplace user to be ",
            next_user->ToString(), " but it was ",
            (*inplace_modifier)->ToString(), ".");
      }
      user = next_user;
    }

    TF_ASSIGN_OR_RETURN(HloInstruction * resource_update,
                        FindCorrespondingResourceUpdate(user));

    TF_RETURN_IF_ERROR(HasGradientAccumulationInstruction(resource_update));

    // Make sure that the number of iterations divides the number of mini
    // batches to accumulate.
    const int64 repeat_count = GetRepeatLoopCount(caller);
    const auto num_batches_to_accumulate =
        GetResourceUpdateBatchesToAccumulate(resource_update);
    if (!num_batches_to_accumulate) {
      // If want to enable this should only have to change the repeat loop
      // visitor program generation
      return FailedPrecondition(
          "Can't have runtime num mini batches for"
          " resource update");
    }
    if (repeat_count % *num_batches_to_accumulate) {
      return FailedPrecondition(
          "Detected a gradient accumulation operation with %d number of "
          "mini batches inside a loop with %d iterations.\n"
          "It is required that the number of mini batches to accumulate "
          "evenly divides the number of loop iterations.",
          *num_batches_to_accumulate, repeat_count);
    }

    // Make sure the resource update is only used by the root (via GTEs).
    const HloInstruction* root_instruction =
        resource_update->parent()->root_instruction();
    if (root_instruction != resource_update) {
      const std::string error_msg =
          "When using the "
          "`gradient_accumulation_optimizer.GradientAccumulationOptimizerV2`,"
          " it must not be wrapped by any other optimizer. Please see the "
          "`Gradient Accumulation` section in the documentation.";
      if (root_instruction->opcode() != HloOpcode::kTuple) {
        return FailedPrecondition("%s", error_msg);
      }

      for (const HloInstruction* user : resource_update->users()) {
        if (user->opcode() != HloOpcode::kGetTupleElement) {
          return FailedPrecondition("%s", error_msg);
        }
        if (absl::c_any_of(user->users(),
                           [root_instruction](const HloInstruction* inst)
                               -> bool { return inst != root_instruction; })) {
          return FailedPrecondition("%s", error_msg);
        }
      }
    }

  } else if (IsPipelineOp(caller)) {
    // We expect all the backward stages to be lowered inplace in order to
    // make sure there is only one gradient accumulation buffer if it is used
    // by multiple stages.
    const bool expect_lowered_inplace = inst->user_count() > 1;

    // We expect the gradient accumulation creators that take in parameters to
    // only be used by backward pipeline stages residing on the same shard.
    // Creators that don't take inputs can be used in forward stages too since
    // they're not modifying any resource.
    for (HloInstruction* user : inst->users()) {
      const auto indices = user->OperandIndices(inst);
      if (indices.size() != 1) {
        return InternalErrorStrCat(
            "Expected the gradient accumulation buffer to only appear as an "
            "operand once, but it is used ",
            indices.size(), " times.");
      }
      if (inst->operand_count() && !IsPipelineStageBackward(user)) {
        return InternalErrorStrCat(
            "Expected the gradient accumulation buffer to only be used by "
            "backward pipeline stages since it takes an input, but detected ",
            user->ToString(), " as a user.");
      }
      if (*user->sharding_unique_device() != *inst->sharding_unique_device()) {
        return InternalError(
            "Expected the gradient accumulation buffer and the backward "
            "pipeline stage to have compatible sharding.");
      }
      if (expect_lowered_inplace && !IsLoweredInplace(user)) {
        return InternalErrorStrCat("Expected the pipeline backward stage ",
                                   user->ToString(),
                                   " to have been lowered inplace.");
      }
      HloComputation* pipeline_stage_comp = user->to_apply();
      TF_ASSIGN_OR_RETURN(int64 output_index,
                          VerifyGradientAccumulationInsideComputation(
                              pipeline_stage_comp, indices[0]));

      // We expect the output at the gradient accumulation buffer location
      // to be only used (via a GTE) by the gradient accumulation sink
      // instruction.
      TF_ASSIGN_OR_RETURN(HloInstruction * gte,
                          GetUniqueGTEUser(user, output_index));

      std::vector<const HloInstruction*> leafs = GetGTEUsers(gte);
      auto itr =
          std::find_if(leafs.begin(), leafs.end(),
                       IsPoplarInstruction(PoplarOp::GradientAccumulatorSink));

      if ((*itr)->user_count() != 1) {
        return InternalErrorStrCat(
            "Expected the gradient accumulation buffer to only have a single "
            "user, but it has ",
            (*itr)->user_count(), " users.");
      }
    }
  } else {
    return FailedPrecondition(
        "Detected a gradient accumulation operation from an unexpected "
        "callsite. Gradient accumulation operations are only allowed "
        "inside of training loops created using `tf.python.ipu.loops.repeat`.");
  }

  return Status::OK();
}

StatusOr<bool> GradientAccumulationVerifier::Run(HloModule* module) {
  std::unique_ptr<CallGraph> call_graph = CallGraph::Build(module);
  for (HloComputation* comp : module->MakeComputationPostOrder()) {
    if (IsPopOpsFusion(comp)) {
      continue;
    }
    for (HloInstruction* inst : comp->instructions()) {
      TF_RETURN_IF_ERROR(
          VerifyStatefulGradientAccumulation(inst, call_graph.get()));
      TF_RETURN_IF_ERROR(
          VerifyGenericGradientAccumulation(inst, call_graph.get()));
    }
  }
  return false;
}

}  // namespace poplarplugin
}  // namespace xla
