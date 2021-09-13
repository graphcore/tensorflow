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
#include <set>
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
#include "tensorflow/compiler/xla/shape_util.h"

namespace xla {
namespace poplarplugin {
namespace {

using Replacements =
    absl::flat_hash_map<const HloInstruction*, std::unique_ptr<HloInstruction>>;

bool IsTuple(HloInstruction* inst) {
  return inst->opcode() == HloOpcode::kTuple;
}

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

void RemoveComputationParameter(Replacements& replacements,
                                HloComputation* comp, int64 param_no) {
  const auto& params = comp->parameter_instructions();
  CHECK_LT(param_no, params.size())
      << "Invalid parameter number for RemoveComputationParameter";
  replacements.emplace(params[param_no], nullptr);
  for (std::size_t i = param_no + 1; i < params.size(); ++i) {
    auto param_instruction = params[i];
    auto new_param =
        HloInstruction::CreateParameter(i - 1, param_instruction->shape(),
                                        tensorflow::strings::StrCat("arg_", i));
    replacements.emplace(param_instruction, std::move(new_param));
  }
}

std::unique_ptr<HloInstruction> SpliceOperandsImpl(
    HloInstruction* inst, const Shape& shape, const std::set<int64>& remove_ops,
    const std::initializer_list<HloInstruction*>& new_operands) {
  auto operands = inst->operands();

  // This is sorted, going backwards to make indices stable
  for (auto it = remove_ops.rbegin(); it != remove_ops.rend(); ++it) {
    operands.erase(operands.begin() + *it);
  }

  operands.reserve(operands.size() + new_operands.size());
  operands.insert(operands.end(), new_operands.begin(), new_operands.end());
  Shape tuple_shape;
  // Automatically update tuple shape
  bool is_tuple = IsTuple(inst);
  if (is_tuple) {
    std::vector<Shape> operands_shape;
    for (auto operand : operands) {
      operands_shape.push_back(operand->shape());
    }
    tuple_shape = ShapeUtil::MakeTupleShape(operands_shape);
  }
  return inst->CloneWithNewOperands(is_tuple ? tuple_shape : shape,
                                    std::move(operands));
}

std::unique_ptr<HloInstruction> SpliceOperands(
    HloInstruction* inst, const Shape& shape,
    const std::initializer_list<HloInstruction*>& old_operands,
    const std::initializer_list<HloInstruction*>& new_operands) {
  std::set<int64> remove_ops;
  for (auto old_operand : old_operands) {
    remove_ops.insert(inst->operand_index(old_operand));
  }
  return SpliceOperandsImpl(inst, shape, remove_ops, new_operands);
}

std::unique_ptr<HloInstruction> SpliceOperands(
    HloInstruction* inst, const Shape& shape,
    const std::initializer_list<HloInstruction*>& new_operands) {
  return SpliceOperandsImpl(inst, shape, std::set<int64>(), new_operands);
}

std::unique_ptr<HloInstruction> SpliceOperands(
    HloInstruction* inst, const Shape& shape,
    // Using int64 because it is operand_index(...) return type.
    const std::initializer_list<int64>& old_operands,
    const std::initializer_list<HloInstruction*>& new_operands) {
  std::set<int64> remove_ops;
  for (auto op : old_operands) {
    remove_ops.insert(op);
  }
  return SpliceOperandsImpl(inst, shape, remove_ops, new_operands);
}

HloInstruction* AddInstruction(Replacements& replacements,
                               std::unique_ptr<HloInstruction>&& inst) {
  auto result = replacements.emplace(inst.get(), std::move(inst));
  CHECK(result.second) << "Double AddInstruction";
  return result.first->second.get();
}

HloInstruction* ReplaceInstruction(Replacements& replacements,
                                   HloInstruction* from,
                                   std::unique_ptr<HloInstruction>&& to) {
  auto result = replacements.emplace(from, std::move(to));
  CHECK(result.second) << "Double ReplaceInstruction";
  return result.first->second.get();
}

void RemoveInstruction(Replacements& replacements, HloInstruction* inst) {
  replacements.emplace(inst, nullptr);
}

std::unique_ptr<HloInstruction> CreateComputationParameter(int64 param_no,
                                                           const Shape& shape) {
  const std::string param_name = absl::StrCat("arg_", param_no);
  VLOG(2) << "Creating parameter " << param_name;
  return HloInstruction::CreateParameter(param_no, shape, param_name);
}

void AdjustGTEIndices(Replacements& replacements, HloInstruction* call,
                      int64 removed_index) {
  // If we remove element from output tuple, we have to adjust indices for GTEs.
  // For instance, for tuple of four elements returned, we have the following
  // instructions:
  //  gte.0 = get-tuple-element call, index=0
  //  gte.1 = get-tuple-element call, index=1
  //  gte.2 = get-tuple-element call, index=2
  //  gte.3 = get-tuple-element call, index=3
  // GTE with index of 1 has been removed by this pass, so we have
  // adjust all GTEs with indices greater than 1
  //  gte.0 = get-tuple-element call, index=0 [ignored]
  //  gte.2 = get-tuple-element call, index=1 [index adjusted by -1]
  //  gte.3 = get-tuple-element call, index=2 [index adjusted by -1]
  // There should be no users other than GTE.

  // Copy users locally, because Clone will add newly created instruction to the
  // users array.
  auto call_users = call->users();
  for (HloInstruction* gte : call_users) {
    CHECK_EQ(gte->opcode(), HloOpcode::kGetTupleElement);
    if (gte->tuple_index() > removed_index) {
      auto clone = gte->Clone();
      clone->set_tuple_index(gte->tuple_index() - 1);
      replacements.emplace(gte, std::move(clone));
    }
  }
}

struct OptimisationPlan {
  int64 row_num;
  int64 mini_batches_num;
  int64 mini_batch_size;
  int64 row_size;
  Shape accum_grads_shape;
  Shape accum_indices_shape;

  static absl::optional<OptimisationPlan> Build(HloInstruction* grad_create,
                                                HloInstruction* grad_add,
                                                HloInstruction* grad_sink_inst,
                                                const int32 mini_batches_num) {
    auto grad_sink = Cast<HloGradientAccumulatorSink>(grad_sink_inst);
    auto multi_update_add =
        Cast<HloMultiUpdateInstruction>(grad_add->mutable_operand(1));

    auto indices = multi_update_add->mutable_operand(1);
    auto grads = multi_update_add->mutable_operand(2);
    const Shape& grads_shape = grads->shape();
    if (grads_shape.dimensions_size() != 2) {
      VLOG(2) << "Invalid gradients shape.";
      return absl::nullopt;
    }
    VLOG(2) << "Gradient add instruction: " << grad_add->ToString();
    VLOG(2) << "Indices argument: " << indices->ToString();
    VLOG(2) << "Gradients argument: " << grads->ToString();
    auto row_num = grad_sink->shape().dimensions(0);

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

    auto current_shape_size = ShapeUtil::ByteSizeOf(grad_sink->shape());
    auto new_shape_size = ShapeUtil::ByteSizeOf(accum_grads_shape) +
                          ShapeUtil::ByteSizeOf(accum_indices_shape);
    VLOG(2) << "Current shape byte size: " << current_shape_size
            << ", new shape byte size: " << new_shape_size;

    if (new_shape_size >= current_shape_size) {
      VLOG(2) << "New layout is bigger than current, skipping optimisation";
      return absl::nullopt;
    }

    return OptimisationPlan{row_num,  mini_batches_num,  mini_batch_size,
                            row_size, accum_grads_shape, accum_indices_shape};
  }
};

HloComputation* ReplaceResourceUpdateFunction(
    HloModule* module, const OptimisationPlan& plan,
    HloInstruction* resource_update, int64 resource_update_grad_sink_arg_index,
    HloMultiUpdateInstruction* multi_update_add) {
  auto resource_update_comp = resource_update->to_apply();
  auto old_sink_arg = resource_update_comp->parameter_instruction(
      resource_update_grad_sink_arg_index);

  Replacements replacements;

  const Shape& old_sink_shape = old_sink_arg->shape();
  auto params_n = resource_update_comp->num_parameters();
  auto new_grads_arg =
      CreateComputationParameter(params_n - 1, plan.accum_grads_shape);
  auto new_indices_arg =
      CreateComputationParameter(params_n, plan.accum_indices_shape);

  auto new_indices_reshape = HloInstruction::CreateReshape(
      ShapeUtil::MakeShape(new_indices_arg->shape().element_type(),
                           {plan.mini_batches_num * plan.mini_batch_size, 1}),
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
    ReplaceInstruction(replacements, old_sink_user,
                       old_sink_user->CloneWithNewOperands(
                           old_sink_user->shape(), old_sink_user_operands));
  }

  RemoveComputationParameter(replacements, resource_update_comp,
                             resource_update_grad_sink_arg_index);

  auto new_resource_update_comp = module->AddEmbeddedComputation(
      resource_update_comp->CloneWithReplacements(std::move(replacements), {}));
  resource_update->set_to_apply(new_resource_update_comp);

  VLOG(2) << "New resource update function: "
          << new_resource_update_comp->ToString();
  return new_resource_update_comp;
}

StatusOr<HloComputation*> ReplaceAccumulatorCaller(
    HloComputation* grad_comp, const OptimisationPlan& plan,
    HloInstruction* grad_create, HloInstruction* grad_add,
    HloInstruction* grad_sink, HloInstruction* resource_update,
    std::size_t resource_update_grad_sink_arg_index,
    HloInstruction* pipeline_stage, HloInstruction* grad_sink_gte) {
  auto update_comp = pipeline_stage ? pipeline_stage->to_apply() : grad_comp;
  auto module = update_comp->parent();
  auto multi_update_add =
      Cast<HloMultiUpdateInstruction>(grad_add->mutable_operand(1));

  Replacements grad_repl, pipeline_repl;
  Replacements& update_repl = pipeline_stage ? pipeline_repl : grad_repl;

  // Patching repeat loop body.
  auto accum_grads = CreateGradientAccumulatorCreate(plan.accum_grads_shape);
  auto accum_indices =
      CreateGradientAccumulatorCreate(plan.accum_indices_shape);

  auto execution_counter = CreateExecutionCounter();
  auto n_const = HloInstruction::CreateConstant(
      LiteralUtil::CreateR0<int32>(plan.mini_batches_num));
  auto update_index = HloInstruction::CreateBinary(
      execution_counter->shape(), HloOpcode::kRemainder,
      execution_counter.get(), n_const.get());

  std::unique_ptr<HloInstruction> pipeline_stage_accum_grads_param,
      pipeline_stage_accum_indices_param;
  if (pipeline_stage) {
    VLOG(2) << "Adding grads/indices parameter to pipeline stage...";

    auto pipeline_stage_param_n = update_comp->num_parameters() - 1;

    pipeline_stage_accum_grads_param = CreateComputationParameter(
        pipeline_stage_param_n, accum_grads->shape());

    pipeline_stage_accum_indices_param = CreateComputationParameter(
        pipeline_stage_param_n + 1, accum_indices->shape());
  }

  auto update_index_broadcast_grads = HloInstruction::CreateBroadcast(
      ShapeUtil::MakeShape(update_index->shape().element_type(),
                           {plan.mini_batch_size}),
      update_index.get(), {});
  auto update_index_broadcast_indices = HloInstruction::CreateBroadcast(
      ShapeUtil::MakeShape(update_index->shape().element_type(), {1}),
      update_index.get(), {});

  auto const_int_1 = HloInstruction::CreateConstant(LiteralUtil::One(S32));

  auto indices = multi_update_add->mutable_operand(1);
  auto grads = multi_update_add->mutable_operand(2);
  auto scale = multi_update_add->mutable_operand(3);

  auto grads_update = CreateMultiUpdateAdd(
      plan.accum_grads_shape,
      {pipeline_stage_accum_grads_param ? pipeline_stage_accum_grads_param.get()
                                        : accum_grads.get(),
       update_index_broadcast_grads.get(), grads, scale},
      1, 1, multi_update_add->GetSerializationFactor());
  auto indices_update = CreateMultiUpdateAdd(
      plan.accum_indices_shape,
      {pipeline_stage_accum_indices_param
           ? pipeline_stage_accum_indices_param.get()
           : accum_indices.get(),
       update_index_broadcast_indices.get(), indices, const_int_1.get()},
      0, 0, multi_update_add->GetSerializationFactor());

  std::unique_ptr<HloInstruction> grads_update_gte, indices_update_gte;
  if (pipeline_stage) {
    // Pass MultiUpdateAdd updates to the output of the pipeline stage.
    auto output = update_comp->root_instruction();
    CHECK(IsTuple(output))
        << "Root instruction of pipeline stage is not a tuple.";

    auto grad_add_output_index = output->operand_index(grad_add);
    VLOG(2) << "Removing " << grad_add->ToString()
            << " from output tuple at position " << grad_add_output_index;
    auto new_output = ReplaceInstruction(
        update_repl, output,
        SpliceOperands(output, output->shape(), {grad_add},
                       {grads_update.get(), indices_update.get()}));
    AdjustGTEIndices(grad_repl, pipeline_stage, grad_add_output_index);

    auto grad_create_arg_index = pipeline_stage->operand_index(grad_create);
    VLOG(2) << "Fix pipeline stage arguments, removing arg "
            << grad_create_arg_index;

    RemoveComputationParameter(update_repl, update_comp, grad_create_arg_index);

    pipeline_stage = ReplaceInstruction(
        grad_repl, pipeline_stage,
        SpliceOperands(pipeline_stage, new_output->shape(), {grad_create},
                       {accum_grads.get(), accum_indices.get()}));

    auto output_operand_count = new_output->operand_count();
    grads_update_gte = HloInstruction::CreateGetTupleElement(
        grads_update->shape(), pipeline_stage, output_operand_count - 2);
    indices_update_gte = HloInstruction::CreateGetTupleElement(
        indices_update->shape(), pipeline_stage, output_operand_count - 1);
  }

  auto accum_grads_sink = CreateGradientAccumulatorSink(
      {grads_update_gte ? grads_update_gte.get() : grads_update.get()});
  auto accum_indices_sink = CreateGradientAccumulatorSink(
      {indices_update_gte ? indices_update_gte.get() : indices_update.get()});

  ReplaceInstruction(
      grad_repl, resource_update,
      SpliceOperands(resource_update, resource_update->shape(),
                     {resource_update_grad_sink_arg_index},
                     {accum_grads_sink.get(), accum_indices_sink.get()}));

  RemoveInstruction(grad_repl, grad_sink);
  RemoveInstruction(update_repl, grad_add);
  RemoveInstruction(update_repl, multi_update_add);
  RemoveInstruction(grad_repl, grad_create);

  if (pipeline_stage) {
    auto new_pipeline_comp = module->AddEmbeddedComputation(
        update_comp->CloneWithReplacements(std::move(update_repl)));
    VLOG(2) << "New pipeline stage computation: "
            << new_pipeline_comp->ToString();
    pipeline_stage->set_to_apply(new_pipeline_comp);
    if (grad_sink_gte) {
      RemoveInstruction(grad_repl, grad_sink_gte);
    }
  }
  auto new_grad_comp = module->AddEmbeddedComputation(
      grad_comp->CloneWithReplacements(std::move(grad_repl)));
  VLOG(2) << "New repeat computation: " << new_grad_comp->ToString();
  return new_grad_comp;
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
  const int32 num_mini_batches =
      GetResourceUpdateBatchesToAccumulate(resource_update);

  // Find index of old sink in resource update function arguments and
  // erase it.
  auto resource_update_grad_sink_arg_index =
      resource_update->operand_index(grad_sink);

  VLOG(2) << "Old gradient sink argument index: "
          << resource_update_grad_sink_arg_index;
  VLOG(2) << "Resource update function: " << resource_update->ToString();

  auto grad_add = grad_sink->mutable_operand(0);
  auto grad_create = grad_add->mutable_operand(0);
  auto multi_update_add =
      Cast<HloMultiUpdateInstruction>(grad_add->mutable_operand(1));
  auto indices = multi_update_add->mutable_operand(1);
  auto grads = multi_update_add->mutable_operand(2);

  auto plan_opt = OptimisationPlan::Build(grad_create, grad_add, grad_sink,
                                          num_mini_batches);
  if (!plan_opt) {
    return false;
  }
  auto& plan = *plan_opt;

  // Replace {row_num, row_size} buffer with two buffers of shapes
  // {mini_batches_num * mini_batch_size, row_size} for gradients
  // and { mini_batches_num, mini_batch_size} for indices.

  VLOG(2) << "Replacing with alternative layout...";

  // Patching resource update function.
  ReplaceResourceUpdateFunction(module, plan, resource_update,
                                resource_update_grad_sink_arg_index,
                                multi_update_add);

  TF_ASSIGN_OR_RETURN(
      auto new_repeat_body,
      ReplaceAccumulatorCaller(
          grad_comp, plan, grad_create, grad_add, grad_sink, resource_update,
          resource_update_grad_sink_arg_index, nullptr, nullptr));
  repeat_inst->set_to_apply(new_repeat_body);

  TF_RETURN_IF_ERROR(module->RemoveUnusedComputations());
  return true;
}

struct Candidate {
  HloGradientAccumulatorSink* sink;
  std::vector<HloInstruction*> callsites;
};

absl::optional<Candidate> CheckEmbeddingsCandidate(HloInstruction* inst) {
  if (!IsGradientAccumulatorSink(inst) || inst->user_count() != 1) {
    return absl::nullopt;
  }
  VLOG(2) << "Checking potential sink: " << inst->ToString();

  auto accum_add = inst->operand(0);
  if (!IsGradientAccumulatorAdd(accum_add)) {
    VLOG(2) << "No GradientAccumulatorAdd found.";
    return absl::nullopt;
  }
  auto accum_create = accum_add->operand(0);
  auto multi_update_add = accum_add->operand(1);
  if (!IsGradientAccumulatorCreate(accum_create) ||
      !IsMultiUpdateAdd(multi_update_add)) {
    VLOG(2) << "GradientAccumulatorCreate/MultiUpdateAdd pair wasn't "
               "found.";
    return absl::nullopt;
  }
  return Candidate{Cast<HloGradientAccumulatorSink>(inst), {}};
}

struct PipelineCandidate {
  HloInstruction* grad_create;
  HloInstruction* grad_add;
  HloInstruction* multi_update_add;
  HloInstruction* grad_sink;
  HloInstruction* grad_sink_gte;
  HloInstruction* pipeline_stage;
  HloInstruction* resource_update;
};

absl::optional<PipelineCandidate> CheckPipelineEmbeddingsCandidate(
    HloInstruction* grad_create, CallGraph* call_graph) {
  if (!IsGradientAccumulatorCreate(grad_create) ||
      grad_create->user_count() != 1) {
    return absl::nullopt;
  }
  auto pipeline_stage = grad_create->users()[0];
  if (!IsPipelineStageBackward(pipeline_stage)) {
    return absl::nullopt;
  }

  // TODO(T26860): support batch serialization.
  auto callsites =
      call_graph->GetNode(grad_create->parent()).caller_callsites();
  CHECK_EQ(callsites.size(), 1);
  HloInstruction* pipeline = callsites[0].instruction();
  CHECK(IsPipelineOp(pipeline));
  if (GetPipelineBatchSerializationIterations(pipeline) > 1) {
    return absl::nullopt;
  }

  auto grad_create_arg_index = pipeline_stage->operand_index(grad_create);
  VLOG(2) << "GradientAccumulatorCreate argument index: "
          << grad_create_arg_index;
  auto pipeline_stage_comp = pipeline_stage->to_apply();
  auto grad_create_arg =
      pipeline_stage_comp->parameter_instruction(grad_create_arg_index);
  if (grad_create_arg->user_count() != 1) {
    VLOG(2) << "GradientAccumulatorCreate has more than one user";
    return absl::nullopt;
  }
  auto grad_add = grad_create_arg->users()[0];
  if (!IsGradientAccumulatorAdd(grad_add)) {
    VLOG(2) << "GradientAccumulatorCreate user is not GradientAccumulatorAdd "
               "instruction.";
    return absl::nullopt;
  }
  auto multi_update_add = grad_add->mutable_operand(1);
  if (!IsMultiUpdateAdd(multi_update_add)) {
    VLOG(2) << "GradientAccumulatorAdd argument is not MultiUpdateAdd";
    return absl::nullopt;
  }

  auto pipeline_stage_root = pipeline_stage_comp->root_instruction();
  if (!IsTuple(pipeline_stage_root)) {
    VLOG(2) << "Resource update output is not a tuple.";
    return absl::nullopt;
  }

  auto grad_add_output_index = pipeline_stage_root->operand_index(grad_add);
  VLOG(2) << "GradientAccumulatorAdd output tuple index: "
          << grad_add_output_index;

  HloInstruction* grad_sink = nullptr;
  HloInstruction* grad_sink_gte = nullptr;
  for (auto gte : pipeline_stage->users()) {
    VLOG(2) << "Checking result of resource_update: " << gte->ToString();
    if (gte->opcode() == HloOpcode::kGetTupleElement &&
        gte->tuple_index() == grad_add_output_index) {
      VLOG(2) << "Found matching GTE: " << gte->ToString();
      for (auto user : gte->users()) {
        if (IsGradientAccumulatorSink(user)) {
          VLOG(2) << "Found GradientAccumulatorSink: " << user->ToString();
          grad_sink_gte = gte;
          grad_sink = user;
          break;
        }
      }
    }
  }

  if (!grad_sink) {
    VLOG(2) << "Could not find GTE passed to GradientAccumulatorSink";
    return absl::nullopt;
  }

  HloInstruction* resource_update = nullptr;
  for (auto user : grad_sink->users()) {
    if (IsResourceUpdate(user)) {
      resource_update = user;
      break;
    }
  }

  if (!resource_update) {
    VLOG(2) << "Could not find resource update function";
    return absl::nullopt;
  }

  PipelineCandidate candidate{grad_create,    grad_add,      multi_update_add,
                              grad_sink,      grad_sink_gte, pipeline_stage,
                              resource_update};
  return candidate;
}

StatusOr<bool> ReplacePipelineGradientAccumulator(
    HloModule* module, HloInstruction* grad_create, HloInstruction* grad_add,
    HloInstruction* multi_update_add, HloInstruction* grad_sink,
    HloInstruction* grad_sink_gte, HloInstruction* pipeline_stage,
    HloInstruction* resource_update) {
  auto plan_opt = OptimisationPlan::Build(
      grad_create, grad_add, grad_sink,
      GetResourceUpdateBatchesToAccumulate(resource_update));
  if (!plan_opt) {
    return false;
  }
  auto& plan = *plan_opt;
  VLOG(2)
      << "Replacing pipeline gradient accumulator with alternative layout...";

  // Find index of old sink in resource update function arguments and
  // erase it.
  auto resource_update_grad_sink_arg_index =
      resource_update->operand_index(grad_sink);

  VLOG(2) << "Old gradient sink argument index: "
          << resource_update_grad_sink_arg_index;
  VLOG(2) << "Resource update function: " << resource_update->ToString();
  VLOG(2) << "Pipeline stage function: " << pipeline_stage->ToString();

  ReplaceResourceUpdateFunction(
      module, plan, resource_update, resource_update_grad_sink_arg_index,
      Cast<HloMultiUpdateInstruction>(multi_update_add));

  auto pipeline_comp = pipeline_stage->parent();
  TF_ASSIGN_OR_RETURN(auto new_pipeline_comp,
                      ReplaceAccumulatorCaller(
                          pipeline_comp, plan, grad_create, grad_add, grad_sink,
                          resource_update, resource_update_grad_sink_arg_index,
                          pipeline_stage, grad_sink_gte));

  module->ReplaceComputations({{pipeline_comp, new_pipeline_comp}});
  TF_RETURN_IF_ERROR(module->RemoveUnusedComputations());

  return true;
}

}  // namespace

StatusOr<bool> EmbeddingsGradientOptimizer::Run(HloModule* module) {
  bool changed = false;
  VLOG(2) << "Before the EmbeddingsGradientOptimizer:";
  XLA_VLOG_LINES(2, module->ToString());

  auto call_graph = CallGraph::Build(module);

  std::list<Candidate> candidates;
  std::list<PipelineCandidate> pipeline_candidates;

  for (auto comp : module->MakeComputationPostOrder()) {
    if (IsPopOpsFusion(comp)) {
      continue;
    }

    // Go through instructions in post order to make sure we do not change
    // operands.
    for (HloInstruction* inst : comp->MakeInstructionPostOrder()) {
      auto candidate = CheckEmbeddingsCandidate(inst);
      if (candidate) {
        VLOG(2) << "Found gradient accumulation candidate " << inst->ToString();
        for (auto& callsite : call_graph->GetNode(comp).caller_callsites()) {
          auto repeat_inst = callsite.instruction();
          VLOG(2) << "Found callsite " << repeat_inst->ToString();
          if (IsRepeatLoop(repeat_inst)) {
            candidate->callsites.push_back(repeat_inst);
          }
        }
        if (!candidate->callsites.empty()) {
          candidates.emplace_back(std::move(*candidate));
        }
      }

      auto pipeline_candidate =
          CheckPipelineEmbeddingsCandidate(inst, call_graph.get());
      if (pipeline_candidate) {
        VLOG(2) << "Found pipeline gradient accumulation candidate "
                << inst->ToString();
        pipeline_candidates.emplace_back(std::move(*pipeline_candidate));
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

  for (auto& candidate : pipeline_candidates) {
    TF_ASSIGN_OR_RETURN(bool computation_replaced,
                        ReplacePipelineGradientAccumulator(
                            module, candidate.grad_create, candidate.grad_add,
                            candidate.multi_update_add, candidate.grad_sink,
                            candidate.grad_sink_gte, candidate.pipeline_stage,
                            candidate.resource_update));
    // GradientAccumulatorCreate was replaced, going to the next
    // match, create argument was removed from computation parameter list
    if (computation_replaced) {
      changed = true;
      break;
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
