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

#include "tensorflow/compiler/plugin/poplar/driver/passes/seed_hoisting.h"

#include <memory>
#include <vector>

#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/execution_counter.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/pipeline_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/xla/service/call_graph.h"
#include "tensorflow/compiler/xla/service/hlo_creation_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/strings/str_util.h"

namespace xla {
namespace poplarplugin {
namespace {

// Hashing function based on
// tensorflow/core/grappler/graph_analyzer/hash_tools.h
// seed ^ (execution_counter + 0x9E3779B9U + (seed << 6) + (seed >> 2))
StatusOr<HloInstruction*> HashCombine(HloInstruction* seed,
                                      HloInstruction* execution_counter) {
  const Shape& shape = seed->shape();
  const Shape unsigned_shape = ShapeUtil::MakeShape(U32, shape.dimensions());
  HloComputation* comp = seed->parent();

  // All the maths is done in U32.
  seed = comp->AddInstruction(
      HloInstruction::CreateBitcastConvert(unsigned_shape, seed));
  execution_counter = comp->AddInstruction(HloInstruction::CreateBitcastConvert(
      ShapeUtil::MakeShape(U32, {}), execution_counter));

  HloInstruction* large_constant = MakeR0ConstantHlo<uint32>(comp, 0x9E3779B9U);
  TF_ASSIGN_OR_RETURN(HloInstruction * constant_six,
                      MakeR1ConstantHlo<uint32>(comp, U32, {6U, 6U}));
  TF_ASSIGN_OR_RETURN(HloInstruction * constant_two,
                      MakeR1ConstantHlo<uint32>(comp, U32, {2U, 2U}));
  // seed << 6
  TF_ASSIGN_OR_RETURN(HloInstruction * shift_left,
                      MakeBinaryHlo(HloOpcode::kShiftLeft, seed, constant_six));
  // seed >> 2
  TF_ASSIGN_OR_RETURN(
      HloInstruction * shift_right,
      MakeBinaryHlo(HloOpcode::kShiftRightLogical, seed, constant_two));

  // Compute the rhs of xor.
  // execution_counter + 0x9E3779B9U
  TF_ASSIGN_OR_RETURN(
      HloInstruction * rhs,
      MakeBinaryHlo(HloOpcode::kAdd, execution_counter, large_constant));
  rhs = MakeBroadcastHlo(rhs, {}, shape.dimensions());

  // execution_counter + 0x9E3779B9U + (seed << 6)
  TF_ASSIGN_OR_RETURN(rhs, MakeBinaryHlo(HloOpcode::kAdd, rhs, shift_left));
  // execution_counter + 0x9E3779B9U + (seed << 6) + (seed >> 2)
  TF_ASSIGN_OR_RETURN(rhs, MakeBinaryHlo(HloOpcode::kAdd, rhs, shift_right));

  TF_ASSIGN_OR_RETURN(seed, MakeBinaryHlo(HloOpcode::kXor, seed, rhs));
  // Convert back to the original type.
  return comp->AddInstruction(
      HloInstruction::CreateBitcastConvert(shape, seed));
}

StatusOr<HloInstruction*> AddParametersToCall(
    HloInstruction* call, const std::vector<HloInstruction*>& new_parameters,
    HloCloneContext* context, bool add_parameters_as_outputs = false) {
  HloComputation* comp = call->to_apply();
  HloComputation* call_parent = call->parent();
  auto builder = HloComputation::Builder(comp->name());

  // Force add new parameters.
  const int64 num_parameters = comp->num_parameters();
  std::vector<HloInstruction*> added_parameters(new_parameters.size());
  for (int64 i = 0; i != new_parameters.size(); ++i) {
    HloInstruction* inst = new_parameters[i];
    added_parameters[i] =
        builder.AddInstruction(HloInstruction::CreateParameter(
            num_parameters + i, inst->shape(), "parameter." + inst->name()));
  }

  // Clone all the existing instructions.
  for (HloInstruction* old_inst : comp->MakeInstructionPostOrder()) {
    // Get the operands for the instruction we are about to lower.
    std::vector<HloInstruction*> new_operands(old_inst->operand_count());
    absl::c_transform(old_inst->operands(), new_operands.begin(),
                      [context](HloInstruction* old_operand) {
                        return context->GetInstruction(old_operand);
                      });
    Shape new_shape = old_inst->shape();
    if (add_parameters_as_outputs && comp->root_instruction() == old_inst) {
      // Add the parameters as inputs to the root tuple.
      CHECK_EQ(old_inst->opcode(), HloOpcode::kTuple);
      for (int64 i = 0; i != new_parameters.size(); ++i) {
        HloInstruction* inst = added_parameters[i];
        new_operands.push_back(inst);
        ShapeUtil::AppendShapeToTuple(inst->shape(), &new_shape);
      }
    }

    HloInstruction* new_inst = builder.AddInstruction(
        old_inst->CloneWithNewOperands(new_shape, new_operands));
    old_inst->SetupDerivedInstruction(new_inst);
    context->MapInstruction(old_inst, new_inst);
  }

  HloInstruction* new_root = context->GetInstruction(comp->root_instruction());
  HloComputation* new_comp =
      context->module()->AddEmbeddedComputation(builder.Build(new_root));

  // Replace the call with the new call with new operands.
  std::vector<HloInstruction*> call_operands{call->operands().begin(),
                                             call->operands().end()};
  call_operands.insert(call_operands.end(), new_parameters.begin(),
                       new_parameters.end());
  HloInstruction* new_call = call_parent->AddInstruction(
      call->CloneWithNewOperands(new_root->shape(), call_operands));
  new_call->set_to_apply(new_comp);
  call->SetupDerivedInstruction(new_call);
  TF_RETURN_IF_ERROR(call->ReplaceAllUsesWithDifferentShape(new_call));
  TF_RETURN_IF_ERROR(call_parent->RemoveInstruction(call));
  return new_call;
}

Status HoistFromFunction(HloInstruction* seed, HloInstruction* function) {
  HloModule* module = seed->GetModule();
  HloCloneContext context(module);
  HloComputation* old_function_comp = function->to_apply();
  HloComputation* parent_comp = function->parent();

  // When hoisting a seed from a function:
  // 1. Clone the seed instruction to the same scope as the function.
  // 2. Add the cloned seed as an input to the function, and replace all the
  // uses of the old seed with the new parameter.
  HloInstruction* new_seed = parent_comp->AddInstruction(seed->Clone());
  seed->SetupDerivedInstruction(new_seed);

  TF_ASSIGN_OR_RETURN(function,
                      AddParametersToCall(function, {new_seed}, &context));
  HloComputation* function_comp = function->to_apply();

  HloInstruction* lowered_seed = function_comp->parameter_instructions().back();
  seed = context.GetInstruction(seed);
  TF_RETURN_IF_ERROR(seed->ReplaceAllUsesWith(lowered_seed));
  // Force remove as the seed is stateful.
  TF_RETURN_IF_ERROR(function_comp->ForceRemoveInstruction(seed));
  TF_RETURN_IF_ERROR(module->RemoveEmbeddedComputation(old_function_comp));
  return Status::OK();
}

Status HoistFromPipelineStage(HloInstruction* seed,
                              HloInstruction* pipeline_stage,
                              HloInstruction* pipeline) {
  VLOG(3) << "Hoisting " << seed->ToString() << " from a pipeline stage.";
  HloModule* module = seed->GetModule();
  HloCloneContext context(module);
  HloComputation* old_pipeline_comp = pipeline_stage->parent();
  HloComputation* old_stage_comp = pipeline_stage->to_apply();

  // When hoisting a seed from a pipeline stage:
  // 1. Clone the seed instruction to the parent of *the pipeline*.
  // 2. Add the seed as an input to the pipeline, inside of the pipeline add it
  // as an input to the current pipeline stage and add it to the root
  // instruction of the pipeline too (similarly to loops, inputs/outputs have to
  // alias).
  // 3. Add the execution counter for the pipeline as an input to the pipeline
  // stage and hash it with the seed, resulting in seed' - this makes sure that
  // the seed is different between executions of the pipeline.
  // 4. Hash the execution counter for the pipeline stage with seed', resulting
  // in seed'' which is different for each execution of the pipeline stage.

  HloComputation* pipeline_parent = pipeline->parent();
  if (pipeline_parent->root_instruction() == pipeline) {
    // Pipeline can't be the root because new inputs/outputs are added
    TF_RETURN_IF_ERROR(FixRootInstruction(pipeline_parent).status());
  }
  TF_RETURN_IF_ERROR(FixRootInstruction(old_pipeline_comp).status());

  HloInstruction* new_seed = pipeline_parent->AddInstruction(seed->Clone());
  seed->SetupDerivedInstruction(new_seed);

  // Add seed as input to the pipeline - note that pipelines, similarly to
  // loops, require the number of inputs/outputs to match, therefore the seed
  // needs to be added to the root tuple instruction.
  TF_ASSIGN_OR_RETURN(pipeline,
                      AddParametersToCall(pipeline, {new_seed}, &context,
                                          /*add_parameters_as_outputs*/ true));
  HloComputation* pipeline_comp = pipeline->to_apply();
  pipeline_stage = context.GetInstruction(pipeline_stage);

  // Add the seed and pipeline execution counter as inputs to pipeline stage.
  HloInstruction* lowered_seed = pipeline_comp->parameter_instructions().back();
  HloInstruction* pipeline_execution_counter =
      pipeline_comp->AddInstruction(CreateExecutionCounter());

  TF_ASSIGN_OR_RETURN(
      pipeline_stage,
      AddParametersToCall(pipeline_stage,
                          {lowered_seed, pipeline_execution_counter},
                          &context));

  // Hash the seed with the pipeline execution counter inside of the stage.
  HloComputation* stage_comp = pipeline_stage->to_apply();
  const int64 num_parameters = stage_comp->num_parameters();
  lowered_seed = stage_comp->parameter_instruction(num_parameters - 2);
  pipeline_execution_counter =
      stage_comp->parameter_instruction(num_parameters - 1);
  TF_ASSIGN_OR_RETURN(lowered_seed,
                      HashCombine(lowered_seed, pipeline_execution_counter));

  // Hash in the pipeline stage execution counter.
  HloInstruction* stage_execution_counter =
      stage_comp->AddInstruction(CreateExecutionCounter());
  TF_ASSIGN_OR_RETURN(lowered_seed,
                      HashCombine(lowered_seed, stage_execution_counter));

  seed = context.GetInstruction(seed);
  TF_RETURN_IF_ERROR(seed->ReplaceAllUsesWith(lowered_seed));
  // Force remove as the seed is stateful.
  TF_RETURN_IF_ERROR(stage_comp->ForceRemoveInstruction(seed));
  TF_RETURN_IF_ERROR(module->RemoveEmbeddedComputation(old_stage_comp));
  TF_RETURN_IF_ERROR(module->RemoveEmbeddedComputation(old_pipeline_comp));
  return Status::OK();
}

Status HoistFromRepeatLoop(HloInstruction* seed, HloInstruction* loop) {
  HloModule* module = seed->GetModule();
  HloCloneContext context(module);
  HloComputation* old_loop_comp = loop->to_apply();
  HloComputation* parent_comp = loop->parent();
  if (parent_comp->root_instruction() == loop) {
    // Repeat loop can't be the root because new inputs/outputs are added
    TF_RETURN_IF_ERROR(FixRootInstruction(parent_comp).status());
  }
  TF_RETURN_IF_ERROR(FixRootInstruction(old_loop_comp).status());

  // When hoisting a seed from a loop:
  // 1. Clone the seed instruction to the same scope as the loop.
  // 2. Add the cloned seed as an input/output to the loop, and replace all the
  // uses of the old seed with the new parameter hashed with the execution
  // counter.
  HloInstruction* new_seed = parent_comp->AddInstruction(seed->Clone());
  seed->SetupDerivedInstruction(new_seed);

  TF_ASSIGN_OR_RETURN(loop,
                      AddParametersToCall(loop, {new_seed}, &context,
                                          /*add_parameters_as_outputs*/ true));
  HloComputation* loop_comp = loop->to_apply();

  HloInstruction* lowered_seed = loop_comp->parameter_instructions().back();
  // Hash in the execution counter.
  HloInstruction* execution_counter =
      loop_comp->AddInstruction(CreateExecutionCounter());
  TF_ASSIGN_OR_RETURN(lowered_seed,
                      HashCombine(lowered_seed, execution_counter));

  seed = context.GetInstruction(seed);
  TF_RETURN_IF_ERROR(seed->ReplaceAllUsesWith(lowered_seed));
  // Force remove as the seed is stateful.
  TF_RETURN_IF_ERROR(loop_comp->ForceRemoveInstruction(seed));
  TF_RETURN_IF_ERROR(module->RemoveEmbeddedComputation(old_loop_comp));
  return Status::OK();
}
}  // namespace

StatusOr<bool> SeedHoisting::Run(HloModule* module) {
  VLOG(2) << "Before SeedHoisting:";
  XLA_VLOG_LINES(2, module->ToString(HloPrintOptions::ShortParsable()));
  std::unique_ptr<CallGraph> call_graph = CallGraph::Build(module);

  for (HloComputation* comp : module->MakeComputationPostOrder()) {
    if (IsPopOpsFusion(comp)) {
      continue;
    }

    const auto& call_graph_node = call_graph->GetNode(comp);
    const auto& callsites = call_graph_node.caller_callsites();
    if (callsites.empty()) {
      continue;
    }

    // Skip parallel context.
    if (callsites[0].context() == CallContext::kParallel) {
      continue;
    }

    if (callsites.size() > 1) {
      return FailedPrecondition("Expected the call graph to be flat.");
    }
    HloInstruction* callsite = callsites[0].instruction();
    // Currently only outline the seed from scopes.
    if (!(IsRepeatLoop(callsite) || IsFunction(callsite) ||
          IsPipelineStage(callsite))) {
      continue;
    }

    for (HloInstruction* inst : comp->MakeInstructionPostOrder()) {
      if (IsPoplarInstruction(PoplarOp::Seed)(inst)) {
        if (IsFunction(callsite)) {
          TF_RETURN_IF_ERROR(HoistFromFunction(inst, callsite));
        } else if (IsRepeatLoop(callsite)) {
          // Cannot hoist from a repeat loop where the inputs have not been
          // broken up.
          if (callsite->operand_count() !=
              ShapeUtil::TupleElementCount(callsite->shape())) {
            continue;
          }
          TF_RETURN_IF_ERROR(ConvertAllUsersToGTEs(callsite).status());
          TF_RETURN_IF_ERROR(HoistFromRepeatLoop(inst, callsite));
        } else {
          CHECK(IsPipelineStage(callsite));
          // Find the pipeline instruction.
          const auto& callsite_gn = call_graph->GetNode(callsite->parent());
          const auto& stage_callsites = callsite_gn.caller_callsites();
          if (stage_callsites.size() != 1) {
            return FailedPrecondition("Expected the call graph to be flat.");
          }
          HloInstruction* pipeline = stage_callsites[0].instruction();
          CHECK(IsPipelineOp(pipeline));
          TF_RETURN_IF_ERROR(ConvertAllUsersToGTEs(pipeline).status());
          TF_RETURN_IF_ERROR(HoistFromPipelineStage(inst, callsite, pipeline));
        }
        return true;
      }
    }
  }
  return false;
}

}  // namespace poplarplugin
}  // namespace xla
