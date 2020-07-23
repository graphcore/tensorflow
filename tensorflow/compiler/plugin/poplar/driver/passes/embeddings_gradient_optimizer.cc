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

#include "tensorflow/compiler/plugin/poplar/driver/passes/embeddings_gradient_optimizer.h"

#include <list>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/execution_counter.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/hlo_poplar_instruction.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/multi_slice.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/stateful_gradient_accumulate.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/ops.pb.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/call_graph.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"

namespace xla {
namespace poplarplugin {
namespace {

bool IsMultiUpdateAdd(const HloInstruction* inst) {
  return IsPoplarInstruction(PoplarOp::MultiUpdateAdd)(inst);
}
bool IsGradientAccumulatorAdd(const HloInstruction* inst) {
  return IsPoplarInstruction(PoplarOp::GradientAccumulatorAdd)(inst);
}

bool IsGradientAccumulatorSink(const HloInstruction* inst) {
  return IsPoplarInstruction(PoplarOp::GradientAccumulatorSink)(inst);
}

bool IsGradientAccumulatorCreate(const HloInstruction* inst) {
  return IsPoplarInstruction(PoplarOp::GradientAccumulatorCreate)(inst);
}

std::unique_ptr<HloInstruction> CreateComputationParameter(int64 param_no,
                                                           const Shape& shape) {
  const std::string param_name = absl::StrCat("arg_", param_no);
  VLOG(2) << "Creating parameter " << param_name;
  return HloInstruction::CreateParameter(param_no, shape, param_name);
}

StatusOr<bool> ReplaceGradientAccumulator(HloModule* module,
                                          HloGradientAccumulatorSink* grad_sink,
                                          HloInstruction* repeat_inst) {
  auto repeat_comp = repeat_inst->parent();
  auto grad_comp = repeat_inst->to_apply();
  CHECK_EQ(grad_comp, grad_sink->parent());

  // Rewriting operands to resource update function by replacing
  // old sink argument with grad/indices sinks.
  auto resource_update = grad_sink->users()[0];
  std::size_t resource_update_grad_sink_arg_index;

  // Find index of old sink in resource update function arguments and
  // erase it.
  auto resource_update_arguments = resource_update->operands();
  {
    auto resource_update_grad_sink_op_it =
        absl::c_find(resource_update_arguments, grad_sink);
    CHECK_NE(resource_update_grad_sink_op_it, resource_update_arguments.end());
    resource_update_grad_sink_arg_index = std::distance(
        resource_update_arguments.begin(), resource_update_grad_sink_op_it);
    CHECK_LT(resource_update_grad_sink_arg_index,
             resource_update_arguments.size());
    resource_update_arguments.erase(resource_update_grad_sink_op_it);
    VLOG(2) << "Old gradient sink argument index: "
            << resource_update_grad_sink_arg_index;
  }
  VLOG(2) << "Resource update function: " << resource_update->ToString();

  auto grad_add = grad_sink->mutable_operand(0);
  auto grad_create = grad_add->mutable_operand(0);
  auto multi_update_add =
      Cast<HloMultiUpdateInstruction>(grad_add->mutable_operand(1));

  auto indices = multi_update_add->mutable_operand(1);
  auto grads = multi_update_add->mutable_operand(2);
  const Shape& grads_shape = grads->shape();
  if (grads_shape.dimensions_size() != 2) {
    VLOG(2) << "Invalid gradients shape.";
    return false;
  }
  VLOG(2) << "Gradient add instruction: " << grad_add->ToString();
  VLOG(2) << "Indices argument: " << indices->ToString();
  VLOG(2) << "Gradients argument: " << grads->ToString();
  auto row_num = grad_sink->shape().dimensions(0);
  auto mini_batches_num = grad_sink->MiniBatchesToAccumulate();
  auto mini_batch_size = grads_shape.dimensions(0);
  auto row_size = grads_shape.dimensions(1);
  VLOG(2) << "Mini batches: " << mini_batches_num
          << ", mini batch size: " << mini_batch_size
          << ", row size: " << row_size << ", rows: " << row_num;

  auto accum_grads_shape =
      ShapeUtil::MakeShape(grad_create->shape().element_type(),
                           {mini_batches_num * mini_batch_size, row_size});
  auto accum_indices_shape = ShapeUtil::MakeShape(
      indices->shape().element_type(), {mini_batches_num, mini_batch_size});

  auto current_layout_size = ShapeUtil::ByteSizeOf(grad_sink->shape());
  auto new_layout_size = ShapeUtil::ByteSizeOf(accum_grads_shape) +
                         ShapeUtil::ByteSizeOf(accum_indices_shape);
  VLOG(2) << "Current layout size: " << current_layout_size
          << ", new layout size: " << new_layout_size;

  if (current_layout_size <= new_layout_size) {
    return false;
  }

  // Replace {row_num, row_size} buffer with two buffers of shapes
  // {mini_batches_num * mini_batch_size, row_size} for gradients
  // and { mini_batches_num, mini_batch_size} for indices.
  VLOG(2) << "Replacing with alternative layout...";

  // Patching resource update function.
  HloComputation* new_resource_update_comp;
  {
    auto resource_update_comp = resource_update->to_apply();
    auto old_sink_arg = resource_update_comp->parameter_instruction(
        resource_update_grad_sink_arg_index);

    absl::flat_hash_map<const HloInstruction*, std::unique_ptr<HloInstruction>>
        replacements;

    const Shape& old_sink_shape = old_sink_arg->shape();
    auto params_n = resource_update_comp->num_parameters();
    auto new_grads_arg =
        CreateComputationParameter(params_n - 1, accum_grads_shape);
    auto new_indices_arg =
        CreateComputationParameter(params_n, accum_indices_shape);

    auto new_indices_reshape = HloInstruction::CreateReshape(
        ShapeUtil::MakeShape(new_indices_arg->shape().element_type(),
                             {mini_batches_num * mini_batch_size, 1}),
        new_indices_arg.get());

    auto const_0 = HloInstruction::CreateConstant(
        LiteralUtil::Zero(old_sink_shape.element_type()));
    auto const_1 = HloInstruction::CreateConstant(
        LiteralUtil::One(old_sink_shape.element_type()));
    auto zero_broadcast =
        HloInstruction::CreateBroadcast(old_sink_shape, const_0.get(), {});

    auto accum_update =
        CreateMultiUpdateAdd(old_sink_shape,
                             {zero_broadcast.get(), new_indices_reshape.get(),
                              new_grads_arg.get(), const_1.get()},
                             multi_update_add->GetIndexVectorDimension(),
                             multi_update_add->GetUpdateSliceDimension(),
                             multi_update_add->GetSerializationFactor());

    for (auto old_sink_user : old_sink_arg->users()) {
      VLOG(2) << "Old GradientAccumulatorSink user: "
              << old_sink_user->ToString();

      auto old_sink_user_operands = old_sink_user->operands();
      for (auto& op : old_sink_user_operands) {
        if (op == old_sink_arg) {
          op = accum_update.get();
        }
      }
      auto new_sink_user = old_sink_user->CloneWithNewOperands(
          old_sink_shape, old_sink_user_operands);
      replacements.emplace(old_sink_user, std::move(new_sink_user));
    }

    replacements.emplace(old_sink_arg, nullptr);

    {
      const auto& params = resource_update_comp->parameter_instructions();
      for (std::size_t i = resource_update_grad_sink_arg_index + 1;
           i < params.size(); ++i) {
        auto param_instruction = params[i];
        auto new_param = HloInstruction::CreateParameter(
            i - 1, param_instruction->shape(),
            tensorflow::strings::StrCat("param_", i));
        replacements.emplace(param_instruction, std::move(new_param));
      }
    }

    new_resource_update_comp = module->AddEmbeddedComputation(
        resource_update_comp->CloneWithReplacements(std::move(replacements),
                                                    {}));
    resource_update->set_to_apply(new_resource_update_comp);

    VLOG(2) << "New resource update function: "
            << new_resource_update_comp->ToString();
  }

  {
    // Patching repeat loop body.
    auto accum_grads = grad_comp->AddInstruction(
        CreateGradientAccumulatorCreate(accum_grads_shape));
    auto accum_indices = grad_comp->AddInstruction(
        CreateGradientAccumulatorCreate(accum_indices_shape));

    auto execution_counter =
        grad_comp->AddInstruction(CreateExecutionCounter());
    auto n_const = grad_comp->AddInstruction(HloInstruction::CreateConstant(
        LiteralUtil::CreateR0<int32>(mini_batches_num)));
    auto update_index = grad_comp->AddInstruction(HloInstruction::CreateBinary(
        execution_counter->shape(), HloOpcode::kRemainder, execution_counter,
        n_const));

    auto update_index_broadcast_grads =
        grad_comp->AddInstruction(HloInstruction::CreateBroadcast(
            ShapeUtil::MakeShape(update_index->shape().element_type(),
                                 {mini_batch_size}),
            update_index, {}));
    auto update_index_broadcast_indices =
        grad_comp->AddInstruction(HloInstruction::CreateBroadcast(
            ShapeUtil::MakeShape(update_index->shape().element_type(), {1}),
            update_index, {}));

    auto const_int_1 = grad_comp->AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::One(S32)));

    auto grads_update = grad_comp->AddInstruction(
        CreateMultiUpdateAdd(accum_grads_shape,
                             {accum_grads, update_index_broadcast_grads, grads,
                              multi_update_add->mutable_operand(3)},
                             1, 1, multi_update_add->GetSerializationFactor()));
    auto indices_update = grad_comp->AddInstruction(CreateMultiUpdateAdd(
        accum_indices_shape,
        {accum_indices, update_index_broadcast_indices, indices, const_int_1},
        0, 0, multi_update_add->GetSerializationFactor()));

    auto accum_grads_sink = grad_comp->AddInstruction(
        CreateGradientAccumulatorSink({grads_update}, mini_batches_num));
    auto accum_indices_sink = grad_comp->AddInstruction(
        CreateGradientAccumulatorSink({indices_update}, mini_batches_num));

    // Changing arguments to resource update function (old sink argument is
    // already removed).
    resource_update_arguments.push_back(accum_grads_sink);
    resource_update_arguments.push_back(accum_indices_sink);
    auto resource_update_clone =
        grad_comp->AddInstruction(resource_update->CloneWithNewOperands(
            resource_update->shape(), std::move(resource_update_arguments)));
    resource_update_clone->set_to_apply(new_resource_update_comp);

    resource_update->ReplaceAllUsesWith(resource_update_clone);

    // Removing old instructions in reverse order. Dangling
    // GradientAccumulatorCreate will trigger "Expected the gradient
    // accumulation buffer to have at least one user" exception
    TF_RETURN_IF_ERROR(grad_comp->RemoveInstruction(resource_update));
    TF_RETURN_IF_ERROR(grad_comp->RemoveInstruction(grad_sink));
    TF_RETURN_IF_ERROR(grad_comp->RemoveInstruction(grad_add));
    TF_RETURN_IF_ERROR(grad_comp->RemoveInstruction(multi_update_add));
    TF_RETURN_IF_ERROR(grad_comp->RemoveInstruction(grad_create));
    VLOG(2) << "New repeat computation: " << grad_comp->ToString();
  }
  TF_RETURN_IF_ERROR(module->RemoveUnusedComputations());
  return true;
}

}  // namespace

StatusOr<bool> EmbeddingsGradientOptimizer::Run(HloModule* module) {
  bool changed = false;
  VLOG(2) << "Before the EmbeddingsGradientOptimizer:";
  XLA_VLOG_LINES(2, module->ToString());

  auto callGraph = CallGraph::Build(module);

  struct Candidate {
    HloGradientAccumulatorSink* sink;
    std::vector<HloInstruction*> callsites;
  };
  std::list<Candidate> candidates;

  for (auto comp : module->MakeComputationPostOrder()) {
    if (IsPopOpsFusion(comp)) {
      continue;
    }
    // Go through instructions in post order to make sure we do not change
    // operands.
    for (HloInstruction* inst : comp->MakeInstructionPostOrder()) {
      if (!IsGradientAccumulatorSink(inst) || inst->user_count() > 1) {
        continue;
      }
      VLOG(2) << "Checking potential sink: " << inst->ToString();

      auto accum_add = inst->operand(0);
      if (!IsGradientAccumulatorAdd(accum_add)) {
        VLOG(2) << "No GradientAccumulatorAdd found.";
        continue;
      }
      auto accum_create = accum_add->operand(0);
      auto multi_update_add = accum_add->operand(1);
      if (!IsGradientAccumulatorCreate(accum_create) ||
          !IsMultiUpdateAdd(multi_update_add)) {
        VLOG(2) << "GradientAccumulatorCreate/MultiUpdateAdd pair wasn't "
                   "found.";
        continue;
      }
      VLOG(2) << "Found gradient accumulation candidate " << inst->ToString();

      auto grad_sink = Cast<HloGradientAccumulatorSink>(inst);
      Candidate candidate{grad_sink, {}};
      for (auto& callsite : callGraph->GetNode(comp).caller_callsites()) {
        auto repeat_inst = callsite.instruction();
        VLOG(2) << "Found callsite " << repeat_inst->ToString();
        if (IsRepeatLoop(repeat_inst)) {
          candidate.callsites.push_back(repeat_inst);
        }
      }
      if (!candidate.callsites.empty()) {
        candidates.emplace_back(std::move(candidate));
      }
    }
  }

  for (auto& candidate : candidates) {
    for (auto callsite : candidate.callsites) {
      TF_ASSIGN_OR_RETURN(
          bool computation_replaced,
          ReplaceGradientAccumulator(module, candidate.sink, callsite));
      // GradientAccumulatorSink and Repeat pair was replaced, going to the next
      // match, sink argument was removed from computation parameters list.
      if (computation_replaced) {
        changed = true;
        break;
      }
    }
  }

  if (changed) {
    VLOG(2) << "After the EmbeddingsGradientOptimizer:";
    XLA_VLOG_LINES(2, module->ToString());
  }
  return changed;
}

}  // namespace poplarplugin
}  // namespace xla
