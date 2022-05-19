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
#include "tensorflow/compiler/plugin/poplar/driver/passes/pipeline_batch_serialization_loop_inserter.h"

#include <list>
#include <vector>

#include "tensorflow/compiler/plugin/poplar/driver/backend_config.pb.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/offloading_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/pipeline_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/xla/service/hlo_creation_utils.h"
#include "tensorflow/compiler/xla/shape_util.h"

namespace xla {
namespace poplarplugin {
namespace {
template <typename InputSequence>
std::vector<HloInstruction*> GetCloneContextOperands(HloCloneContext* context,
                                                     InputSequence& operands) {
  std::vector<HloInstruction*> new_operands(operands.size());
  absl::c_transform(operands, new_operands.begin(),
                    [context](HloInstruction* old_operand) {
                      return context->GetInstruction(old_operand);
                    });
  return new_operands;
}
}  // namespace

Status PipelineBatchSerializationLoopInserter::InsertIntoPipeline(
    HloInstruction* pipeline_op) {
  HloModule* module = pipeline_op->GetModule();
  HloComputation* pipeline_comp = pipeline_op->to_apply();
  const int64_t batch_serialization_iterations =
      GetPipelineBatchSerializationIterations(pipeline_op);

  TF_ASSIGN_OR_RETURN(PipelineStages stages, GetPipelineStages(pipeline_comp));
  // Make sure that the root of each stage is a tuple.
  TF_RETURN_IF_ERROR(FixRootInstructions(stages));
  OrderedPipelineStages ordered_stages(stages,
                                       /*include_resource_update*/ false);

  for (int64_t stage_id = 0; stage_id != ordered_stages.GetNumberOfStages();
       ++stage_id) {
    HloInstruction* stage = ordered_stages.GetStage(stage_id);
    HloComputation* stage_comp = stage->to_apply();
    HloInstruction* root = stage_comp->root_instruction();
    CHECK_EQ(root->opcode(), HloOpcode::kTuple);

    // Get the loop body in post order excluding parameters and the root
    // instruction.
    std::list<HloInstruction*> loop_instructions;
    absl::c_copy_if(stage_comp->MakeInstructionPostOrder(),
                    std::back_inserter(loop_instructions),
                    [root](HloInstruction* inst) -> bool {
                      return inst != root &&
                             inst->opcode() != HloOpcode::kParameter;
                    });

    // For the batch serialization loops inside of the stage, the inputs and
    // outputs need to be connected appropriately.
    std::vector<HloInstruction*> loop_inputs;
    std::vector<HloInstruction*> loop_root_tuple_operands;
    std::vector<HloInstruction*> outputs;
    absl::flat_hash_map<const HloInstruction*, int64_t> outputs_map;

    for (int64_t i = 0; i != stage->operand_count(); ++i) {
      const HloInstruction* operand = stage->operand(i);
      HloInstruction* param = stage_comp->parameter_instruction(i);
      switch (operand->opcode()) {
        case HloOpcode::kGetTupleElement: {
          // Inputs from other stages are expected to be sliced on only and not
          // modified.
          const HloInstruction* source = operand->operand(0);
          CHECK(IsPipelineStageOrBackwardOp(source));
          CHECK_EQ(param->user_count(), 1);
          const HloInstruction* param_user = param->users()[0];
          CHECK(param_user->opcode() == HloOpcode::kDynamicSlice ||
                IsPoplarInstruction(PoplarOp::BufferLoadSlice)(param_user));
          // Not modified hence the loop parameter is unmodified.
          loop_inputs.push_back(param);
          loop_root_tuple_operands.push_back(param);
          outputs.push_back(param);
          break;
        }
        case HloOpcode::kParameter: {
          // Not modified hence the loop parameter is unmodified.
          loop_inputs.push_back(param);
          loop_root_tuple_operands.push_back(param);
          outputs.push_back(param);
          break;
        }
        case HloOpcode::kCustomCall: {
          if (IsPoplarInstruction(PoplarOp::GradientAccumulatorCreate)(
                  operand)) {
            // Gradient accumulation.
            if (param->user_count() == 2) {
              // Offloaded case.
              // Make sure the loads and stores of the gradient accumulation are
              // done outside of the batch serial loop rather than inside, which
              // would cause extra communication every iteration.
              HloInstruction *load, *store;
              TF_RETURN_IF_ERROR(GetRemoteLoadStoreUsers(param, &load, &store));
              // Expect the only user of the load to be the gradient
              // accumulation fusion which is used by the store.
              CHECK_EQ(load->user_count(), 1);
              HloInstruction* user = load->users()[0];
              CHECK(IsSerializedGradientAccumulation(user));
              CHECK_EQ(user->user_count(), 1);
              CHECK_EQ(user->users()[0], store);
              // Map the load to the user as the buffer is updated every
              // iteration.
              loop_inputs.push_back(load);
              loop_root_tuple_operands.push_back(user);
              // The loop output needs to be stored, and the store is the actual
              // loop output.
              outputs.push_back(store);

              // Erase the load and store from being lowered into the batch
              // serialization loop.
              loop_instructions.remove(load);
              loop_instructions.remove(store);
            } else {
              // Non offloaded case.
              CHECK_EQ(param->user_count(), 1);
              HloInstruction* user = param->users()[0];
              if (user == root) {
                // The buffer is just passed through the stage, just map it to
                // itself.
                loop_inputs.push_back(param);
                loop_root_tuple_operands.push_back(param);
                outputs.push_back(param);
              } else {
                // Expect the only user to be the gradient accumulation fusion
                // which is used by the root tuple.
                CHECK(IsSerializedGradientAccumulation(user));
                CHECK_EQ(user->user_count(), 1);
                CHECK_EQ(user->users()[0], root);
                // Map the parameter to the user as it is updated every
                // iteration of the loop.
                loop_inputs.push_back(param);
                loop_root_tuple_operands.push_back(user);
                outputs.push_back(user);
              }
            }
            break;
          }
          if (IsPoplarInstruction(PoplarOp::CreateBuffer)(operand)) {
            // Buffers are used to pass outputs to other stages - these buffers
            // are updated with values every iteration.
            CHECK_EQ(param->user_count(), 1);
            HloInstruction* user = param->users()[0];
            CHECK(user->opcode() == HloOpcode::kDynamicUpdateSlice ||
                  IsPoplarInstruction(PoplarOp::BufferStoreSlice)(user));
            CHECK_EQ(user->user_count(), 1);
            CHECK_EQ(user->users()[0], root);
            // Map the parameter to the user as it is updated every iteration of
            // the loop.
            loop_inputs.push_back(param);
            loop_root_tuple_operands.push_back(user);
            outputs.push_back(user);
            break;
          }
          if (IsPoplarInstruction(PoplarOp::ExecutionCounter)(operand)) {
            // The input value from the pipeline execution counter does not
            // change during the loop execution.
            loop_inputs.push_back(param);
            loop_root_tuple_operands.push_back(param);
            outputs.push_back(param);
            break;
          }
          TF_FALLTHROUGH_INTENDED;
        }
        default: {
          return InternalErrorStrCat("Invalid input ", operand->ToString(),
                                     " to pipeline stage ", stage_id, ".");
        }
      }
      outputs_map[outputs.back()] = i;
    }

    // Go through all the stage outputs and find any which were missing.
    for (int64_t i = 0; i != root->operand_count(); ++i) {
      HloInstruction* output = root->mutable_operand(i);
      if (!outputs_map.contains(output)) {
        // Create zeros as the loop input.
        HloInstruction* zeros =
            BroadcastZeros(stage_comp, output->shape().element_type(),
                           output->shape().dimensions());
        loop_inputs.push_back(zeros);
        loop_root_tuple_operands.push_back(output);
        outputs.push_back(output);

        const int64_t output_index = outputs_map.size();
        outputs_map[output] = output_index;
      }
    }

    // Create the loop body.
    HloCloneContext context(module);
    HloComputation::Builder builder(stage->name() + ".batch_loop");

    // Create inputs.
    for (int64_t i = 0; i != loop_inputs.size(); ++i) {
      HloInstruction* input = loop_inputs[i];
      HloInstruction* parameter =
          builder.AddInstruction(HloInstruction::CreateParameter(
              i, input->shape(), "parameter_" + input->name()));
      input->SetupDerivedInstruction(parameter);
      context.MapInstruction(input, parameter);
    }

    // Copy the body.
    for (HloInstruction* old_inst : loop_instructions) {
      std::vector<HloInstruction*> new_operands =
          GetCloneContextOperands(&context, old_inst->operands());
      HloInstruction* new_inst = builder.AddInstruction(
          old_inst->CloneWithNewOperands(old_inst->shape(), new_operands));
      old_inst->SetupDerivedInstruction(new_inst);
      context.MapInstruction(old_inst, new_inst);
    }
    // Set up the root instruction.
    std::vector<HloInstruction*> new_operands =
        GetCloneContextOperands(&context, loop_root_tuple_operands);
    HloInstruction* loop_root =
        builder.AddInstruction(HloInstruction::CreateTuple(new_operands));

    // Create the actual loop and configure the number of iterations.
    HloComputation* loop_body = module->AddEmbeddedComputation(builder.Build());
    HloInstruction* repeat_loop = stage_comp->AddInstruction(
        HloInstruction::CreateCall(loop_root->shape(), loop_inputs, loop_body));

    auto backend_config =
        repeat_loop->backend_config<PoplarBackendConfig>().ValueOrDie();
    auto* call_config = backend_config.mutable_call_config();
    call_config->set_type(PoplarBackendConfig::CallConfig::RepeatLoop);
    auto* repeat_cfg = call_config->mutable_repeat_config();
    repeat_cfg->set_repeat_count(batch_serialization_iterations);
    repeat_cfg->set_allow_finer_alias_analysis(true);
    TF_RETURN_IF_ERROR(repeat_loop->set_backend_config(backend_config));

    // Set up all the loop outputs.
    for (int64_t i = 0; i != root->operand_count(); ++i) {
      HloInstruction* output = root->mutable_operand(i);
      const int64_t loop_output_index = outputs_map.at(output);
      TF_ASSIGN_OR_RETURN(
          HloInstruction * gte,
          MakeGetTupleElementHlo(repeat_loop, loop_output_index));
      const HloInstruction* loop_root_tuple_operand =
          loop_root_tuple_operands.at(loop_output_index);
      if (loop_root_tuple_operand != output) {
        const auto indices = output->OperandIndices(loop_root_tuple_operand);
        CHECK(indices.size());
        for (auto index : indices) {
          TF_RETURN_IF_ERROR(output->ReplaceOperandWith(index, gte));
        }
      } else {
        TF_RETURN_IF_ERROR(root->ReplaceOperandWith(i, gte));
      }
    }

    // Remove all the old instructions in reverse post order.
    for (auto itr = loop_instructions.rbegin(); itr != loop_instructions.rend();
         ++itr) {
      HloInstruction* inst = *itr;
      CHECK_EQ(inst->user_count(), 0);
      TF_RETURN_IF_ERROR(stage_comp->ForceRemoveInstruction(inst));
    }
  }
  return Status::OK();
}

StatusOr<bool> PipelineBatchSerializationLoopInserter::Run(HloModule* module) {
  TF_ASSIGN_OR_RETURN(auto pipeline_ops, GetPipelines(module));
  if (pipeline_ops.empty()) {
    // No pipeline ops found - nothing to fix.
    return false;
  }
  CHECK_EQ(pipeline_ops.size(), 1);
  if (GetPipelineBatchSerializationIterations(pipeline_ops[0]) < 2) {
    // Nothing to do.
    return false;
  }

  VLOG(2) << "Before PipelineBatchSerializationLoopInserter.";
  XLA_VLOG_LINES(2, module->ToString());

  TF_RETURN_IF_ERROR(InsertIntoPipeline(pipeline_ops[0]));

  VLOG(2) << "After PipelineBatchSerializationLoopInserter.";
  XLA_VLOG_LINES(2, module->ToString());
  return true;
}

}  // namespace poplarplugin
}  // namespace xla
