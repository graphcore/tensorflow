/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/plugin/poplar/driver/passes/pipeline_recomputation_stage_inserter.h"

#include <list>
#include <memory>
#include <utility>

#include "tensorflow/compiler/plugin/poplar/driver/backend_config.pb.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/pipeline_fifo_inserter.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/fifo.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/stateful_noop.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/inplace_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/pipeline_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/xla/service/hlo_creation_utils.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/strings/str_util.h"

namespace xla {
namespace poplarplugin {

namespace {

struct RecomputationInfo {
  ConstHloInstructionSet to_include;
  ConstHloInstructionSet to_exclude;
  HloInstructionSet to_cache;
};

// Is stateful instruction which needs caching for recomputation.
// Note that to prevent DCE each pipeline stage has had a stateful noop
// inserted inside which needs to be ignored here.
bool IsStatefulInstruction(const HloInstruction* inst) {
  return !IsPoplarInstruction(PoplarOp::StatefulNoop)(inst) &&
         inst->HasSideEffect();
}

bool ContainsToken(const Shape& shape) {
  if (shape.IsToken()) {
    return true;
  } else if (shape.IsTuple()) {
    return absl::c_any_of(shape.tuple_shapes(), ContainsToken);
  }
  return false;
}

bool ContainsOpaque(const Shape& shape) {
  if (shape.IsOpaque()) {
    return true;
  } else if (shape.IsTuple()) {
    return absl::c_any_of(shape.tuple_shapes(), ContainsOpaque);
  }
  return false;
}

RecomputationInfo GetRecomputationInfo(const HloComputation* comp) {
  RecomputationInfo info;

  // Root instruction will always be included
  info.to_include.insert(comp->root_instruction());

  // Iterate in reverse post order to guarantee each op processed has already
  // had all of its users processed.
  auto post_order = comp->MakeInstructionPostOrder();
  for (auto iterator = post_order.rbegin(); iterator != post_order.rend();
       ++iterator) {
    HloInstruction* inst = *iterator;
    if (inst == comp->root_instruction()) {
      continue;
    }
    auto is_included = [&](const HloInstruction* inst) {
      return info.to_include.count(inst) > 0;
    };
    // If instruction has no included users it can be excluded.
    if (absl::c_none_of(inst->users(), is_included)) {
      info.to_exclude.insert(inst);
      // If instruction is stateful it will need to be cached.
    } else if (IsStatefulInstruction(inst)) {
      info.to_cache.insert(inst);
    } else {
      info.to_include.insert(inst);
    }
  }
  return info;
}

// Create a new stage which returns the original stage's outputs followed by
// the output of all the stateful ops.
StatusOr<HloComputation*> CloneStageCompWithStates(
    HloInstruction* stage, HloComputation* pipeline_comp) {
  HloComputation* stage_comp = pipeline_comp->parent()->AddEmbeddedComputation(
      stage->to_apply()->Clone("state"));
  auto recomp_info = GetRecomputationInfo(stage_comp);

  // Return the output of all the stateful ops to pipeline.
  auto* root = stage_comp->root_instruction();
  CHECK_EQ(root->opcode(), HloOpcode::kTuple);
  HloInstruction::InstructionVector new_outputs = root->operands();

  // Add the output of the stateful ops to the output of the computation - need
  // to preserve inplaceness here and add copies if the output of the stateful
  // op is used inplace elsewhere.
  for (HloInstruction* inst : recomp_info.to_cache) {
    auto optional_inplace_user = GetInplaceModifier(inst);
    if (optional_inplace_user) {
      // Add a copy for the instruction.
      HloInstruction* copy = stage_comp->AddInstruction(
          HloInstruction::CreateUnary(inst->shape(), HloOpcode::kCopy, inst));
      inst->SetupDerivedInstruction(copy);
      inst = copy;
      if (*optional_inplace_user != root) {
        // Add a control dependency to make sure the copy is executed before the
        // inplace instruction.
        TF_RETURN_IF_ERROR(
            copy->AddControlDependencyTo(*optional_inplace_user));
      }
    }
    new_outputs.push_back(inst);
  }
  // Go through any inputs to the pipeline stage which are not parameters in the
  // pipeline and add copies if they are used inplace to make sure the inputs
  // are preserved when storing them in FIFOs for recomputation.
  for (int64 idx = 0; idx != stage_comp->num_parameters(); ++idx) {
    if (stage->operand(idx)->opcode() == HloOpcode::kParameter) {
      continue;
    }

    HloInstruction* parameter_inst = stage_comp->parameter_instruction(idx);
    if (IsOutputModifiedInplace(parameter_inst)) {
      HloInstruction* copy =
          stage_comp->AddInstruction(HloInstruction::CreateUnary(
              parameter_inst->shape(), HloOpcode::kCopy, parameter_inst));
      parameter_inst->SetupDerivedInstruction(copy);
      TF_RETURN_IF_ERROR(parameter_inst->ReplaceAllUsesWith(copy));
    }
  }

  Shape shape = stage->shape();
  HloInstruction* new_root =
      stage_comp->AddInstruction(HloInstruction::CreateTuple(new_outputs));
  root->SetupDerivedInstruction(new_root);
  auto optional_sharding = stage->sharding().ExtractSingleSharding();
  if (!optional_sharding) {
    return FailedPrecondition("Could not extract single sharding.");
  }
  new_root->CopyAllControlDepsFrom(root);
  new_root->set_sharding(
      HloSharding::SingleTuple(stage->shape(), *optional_sharding));
  stage_comp->set_root_instruction(new_root, true);

  // Make sure that the root tuple preserves the inplace information - any extra
  // outputs had to insert copies if the stateful output was used inplace
  // elsewhere so if the root instruction was inplace before, it can be inplace
  // now. Also note that tuples support inplace even if the same operand appears
  // twice (every occurrence but the first one adds a copy.)
  if (IsLoweredInplace(root)) {
    MakeUsedInplace(new_root);
  }

  root->DropAllControlDeps();
  TF_RETURN_IF_ERROR(stage_comp->RemoveInstruction(root));
  return stage_comp;
}

}  // namespace
PipelineRecomputationStageInserter::PipelineRecomputationStageInserter(
    bool allow_recomputation, bool remote_memory_supported)
    : allow_recomputation_(allow_recomputation),
      remote_memory_supported_(remote_memory_supported) {}

StatusOr<bool> PipelineRecomputationStageInserter::RecomputePipeline(
    HloInstruction* pipeline_op) {
  HloComputation* pipeline_comp = pipeline_op->to_apply();
  TF_ASSIGN_OR_RETURN(PipelineStages stages, GetPipelineStages(pipeline_comp));
  TF_ASSIGN_OR_RETURN(const auto schedule, GetPipelineSchedule(pipeline_op));
  // Do not perform recomputation if there are no backward stages.
  if (stages.backward.empty()) {
    return false;
  }
  // Get whether any Fifos which will be inserted should be offloaded.
  TF_ASSIGN_OR_RETURN(const bool offload_fifos,
                      PipelineFIFOInserter::OffloadFifos(
                          pipeline_op, remote_memory_supported_));

  bool changed = false;
  // Go through all the forward stages (apart from the last one which does not
  // need recomputation).
  for (int64 stage_id = 0;
       stage_id != static_cast<int64>(stages.forward.size()) - 1; ++stage_id) {
    HloInstruction* fwd_stage = stages.forward[stage_id];
    HloInstruction* bwd_stage = stages.backward[stage_id];
    // Do not recompute a stage if it has no outputs which go into the
    // corresponding backward stage (i.e. there is no FIFO).
    const bool bwd_uses_fwd = absl::c_any_of(
        bwd_stage->operands(), IsPoplarInstruction(PoplarOp::Fifo));
    if (!bwd_uses_fwd) {
      continue;
    }
    auto recomp_info = GetRecomputationInfo(fwd_stage->to_apply());

    VLOG(2) << "Recomputation information for pipeline stage " << stage_id
            << " {";
    VLOG(2) << "   Included directly in recomputation {";
    for (auto inst : recomp_info.to_include) {
      VLOG(2) << "      " << inst->ToString();
    }
    VLOG(2) << "   }\n   Outputs cached for use in recomputation {";
    for (auto inst : recomp_info.to_cache) {
      VLOG(2) << "      " << inst->ToString();
    }
    VLOG(2) << "   }\n   Not included in recomputation {";
    for (auto inst : recomp_info.to_exclude) {
      VLOG(2) << "      " << inst->ToString();
    }
    VLOG(2) << "   }\n}";

    auto contains_token = [&](const HloInstruction* inst) {
      return ContainsToken(inst->shape());
    };

    // It does not make sense to recompute a token.
    // Tokens also cannot be cached because we do not support token
    // computation inputs/outputs.
    // There is potential to instead check if the users of tokens are
    // cacheable instead, or even to search for a memory-optimal set of
    // cacheable instructions.
    if (absl::c_any_of(recomp_info.to_include, contains_token) ||
        absl::c_any_of(recomp_info.to_cache, contains_token)) {
      LOG(INFO) << "Cannot recompute pipeline stage " << stage_id
                << " because it contains the following token shaped ops {";
      for (auto inst : recomp_info.to_include) {
        if (contains_token(inst)) {
          LOG(INFO) << "   " << inst->ToString();
        }
      }
      for (auto inst : recomp_info.to_cache) {
        if (contains_token(inst)) {
          LOG(INFO) << "   " << inst->ToString();
        }
      }
      LOG(INFO) << "}";
      continue;
    }

    auto contains_opaque = [](const HloInstruction* inst) {
      return ContainsOpaque(inst->shape());
    };

    if (absl::c_any_of(fwd_stage->operands(), contains_opaque)) {
      LOG(INFO) << "Cannot recompute pipeline stage " << stage_id
                << " because it contains the following opaque shaped inputs {";
      for (auto operand : fwd_stage->operands()) {
        if (contains_opaque(operand)) {
          LOG(INFO) << "   " << operand->ToString();
        }
      }
      LOG(INFO) << "}";
      continue;
    }

    HloComputation* original_fwd_stage_comp = fwd_stage->to_apply();
    auto recomp_operands = fwd_stage->operands();
    const Shape& recomp_shape =
        original_fwd_stage_comp->root_instruction()->shape();

    VLOG(2) << "Original fwd comp for pipeline stage " << stage_id << ":";
    VLOG(2) << original_fwd_stage_comp->ToString();

    // This replacement map will be used to create the recomputation
    // computation by cloning the fwd computation using CloneWithReplacements.
    absl::flat_hash_map<const HloInstruction*, std::unique_ptr<HloInstruction>>
        replacements;

    // Map instructions in to_exclude to nullptr to exclude them from the
    // recomputation.
    for (auto inst : recomp_info.to_exclude) {
      replacements.emplace(inst, nullptr);
    }

    // If there are instructions in to_cache, the original fwd stage
    // is modified to output the results of those instructions,
    // then those outputs are substituted into in the recomputation.
    if (!recomp_info.to_cache.empty()) {
      // Create a copy of the fwd stage with the outputs of the
      // instructions in to_cache appended to the computation output.
      TF_ASSIGN_OR_RETURN(auto comp_states,
                          CloneStageCompWithStates(fwd_stage, pipeline_comp));
      fwd_stage->set_to_apply(comp_states);

      VLOG(2) << "Rewritten fwd comp for pipeline stage " << stage_id << ":";
      VLOG(2) << comp_states->ToString();

      // Update the stage shape.
      *fwd_stage->mutable_shape() = comp_states->root_instruction()->shape();

      CHECK_EQ(comp_states->root_instruction()->sharding(),
               fwd_stage->sharding());

      // Outputs of the stateful ops are passed as extra parameters after
      // the regular stage operands.
      int64 parameter_idx = fwd_stage->operands().size();
      int64 tuple_index =
          original_fwd_stage_comp->root_instruction()->operands().size();

      for (auto inst : recomp_info.to_cache) {
        // Create GTE to pass cached value to the recomputation
        // (Fifos will be created later).
        TF_ASSIGN_OR_RETURN(HloInstruction * gte,
                            MakeGetTupleElementHlo(fwd_stage, tuple_index++));
        gte->set_sharding(fwd_stage->sharding());
        // Mark the new GTE as inplace - it is guaranteed that they are unique.
        MakeUsedInplace(gte);
        // Add GTEs as inputs to the recomputation.
        recomp_operands.push_back(gte);

        // Create parameter for the cached value to be passed into.
        auto param = HloInstruction::CreateParameter(
            parameter_idx++, gte->shape(), inst->name() + "_state");
        param->set_sharding(fwd_stage->sharding());

        // Mark instruction to be replaced by the parameter.
        replacements.emplace(inst, std::move(param));
      }
    }

    // Create recomputation by cloning the original fwd stage with the
    // replacements specified in the replacement map.
    // The resulting computation is similar to the fwd stage but with
    // extraneous instructions removed and stateful instructions replaced by
    // parameters which take cached values output by the modified fwd
    // computation.
    HloComputation* recomp_stage_comp =
        pipeline_comp->parent()->AddEmbeddedComputation(
            original_fwd_stage_comp->CloneWithReplacements(
                std::move(replacements), {}, nullptr, "_recomputation"));
    VLOG(2) << "Recomputation comp for pipeline stage " << stage_id << ":";
    VLOG(2) << recomp_stage_comp->ToString();

    HloInstruction* recomp_stage = pipeline_comp->AddInstruction(
        fwd_stage->CloneWithNewOperands(recomp_shape, recomp_operands));
    recomp_stage->set_sharding(fwd_stage->sharding());
    recomp_stage->set_to_apply(recomp_stage_comp);
    recomp_stage->SetAndSanitizeName(recomp_stage_comp->name());
    TF_RETURN_IF_ERROR(pipeline_comp->parent()->RemoveUnusedComputations());

    // Mark this stage as a Recomputation stage.
    TF_ASSIGN_OR_RETURN(PoplarBackendConfig config,
                        recomp_stage->backend_config<PoplarBackendConfig>());
    config.mutable_call_config()->set_type(
        PoplarBackendConfig::CallConfig::PipelineStageRecomputation);
    TF_RETURN_IF_ERROR(recomp_stage->set_backend_config(config));
    CHECK(IsPipelineStageRecomputation(recomp_stage));

    TF_ASSIGN_OR_RETURN(const int fifo_depth_multiplier,
                        GetFifoDepthMultiplier(pipeline_op));

    if (schedule !=
        PoplarBackendConfig::CallConfig::PipelineConfig::Sequential) {
      // Replace all the non read-only inputs with FIFOs.
      auto recomp_operands = recomp_stage->operands();
      for (size_t op_idx = 0; op_idx != recomp_operands.size(); ++op_idx) {
        HloInstruction* operand = recomp_operands[op_idx];
        if (IsPipelineStageReadOnlyInput(operand)) {
          continue;
        }

        // Create the FIFO.
        HloInstruction* fifo_inst = pipeline_comp->AddInstruction(CreateFifo(
            operand,
            fifo_depth_multiplier * (stages.forward.size() - stage_id - 1),
            offload_fifos));

        fifo_inst->SetAndSanitizeName(operand->name() + ".fifo");
        fifo_inst->set_sharding(operand->sharding());
        // Use the fifo as the input.
        TF_RETURN_IF_ERROR(recomp_stage->ReplaceOperandWith(op_idx, fifo_inst));
        // If there is an inplace user of the operand, then we need to add a
        // control dependency from the new FIFO instruction to that user.
        auto optional_inplace_user = GetInplaceModifier(operand);
        if (optional_inplace_user) {
          TF_RETURN_IF_ERROR(
              fifo_inst->AddControlDependencyTo(*optional_inplace_user));
        }
      }
    }

    // Wire inputs to the bwd stage so that they use the recomp stage outputs.
    auto bwd_operands = bwd_stage->operands();
    for (size_t op_idx = 0; op_idx != bwd_operands.size(); ++op_idx) {
      HloInstruction* operand = bwd_operands[op_idx];
      HloInstruction* gte;
      if (schedule ==
          PoplarBackendConfig::CallConfig::PipelineConfig::Sequential) {
        if (operand->opcode() != HloOpcode::kGetTupleElement) {
          continue;
        }
        gte = operand;
      } else {
        if (!IsPoplarInstruction(PoplarOp::Fifo)(operand)) {
          continue;
        }
        // We expect the FIFO input to be a GTE.
        gte = operand->mutable_operand(0);
        CHECK_EQ(gte->opcode(), HloOpcode::kGetTupleElement);
      }

      CHECK_EQ(gte->opcode(), HloOpcode::kGetTupleElement);
      const HloInstruction* gte_operand = gte->operand(0);
      // If it is not on the fwd stage, then it is a FIFO on some other bwd
      // stage which should be on the same shard.
      if (gte_operand != fwd_stage) {
        CHECK(IsPipelineStageBackward(gte_operand));
        CHECK_EQ(*gte_operand->sharding_unique_device(),
                 *bwd_stage->sharding_unique_device());
        continue;
      }

      // Create a GTE from the recomputation output and wire it to the BWD
      // stage.
      HloInstruction* new_gte = pipeline_comp->AddInstruction(
          gte->CloneWithNewOperands(gte->shape(), {recomp_stage}));
      TF_RETURN_IF_ERROR(bwd_stage->ReplaceOperandWith(op_idx, new_gte));

      // Remove the old operand.
      TF_RETURN_IF_ERROR(operand->DropAllControlDeps());
      TF_RETURN_IF_ERROR(
          pipeline_comp->RemoveInstructionAndUnusedOperands(operand));
    }

    // Make sure that the fwd pass is executed before the recomputation.
    fwd_stage->AddControlDependencyTo(recomp_stage);

    VLOG(1) << "Added recomputation for pipeline stage " << stage_id;
    changed = true;
  }
  return changed;
}

StatusOr<bool> PipelineRecomputationStageInserter::Run(HloModule* module) {
  if (!allow_recomputation_) {
    return false;
  }

  TF_ASSIGN_OR_RETURN(auto pipeline_ops, GetPipelines(module));
  if (pipeline_ops.empty()) {
    // No pipeline ops found - nothing to fix.
    return false;
  }
  CHECK_EQ(pipeline_ops.size(), 1);

  // Check whether this is the requested method of recomputation.
  TF_ASSIGN_OR_RETURN(const auto recomputation_mode,
                      GetPipelineRecomputationMode(pipeline_ops[0]));
  if (recomputation_mode != PoplarBackendConfig::CallConfig::PipelineConfig::
                                Recompute_then_backpropagate) {
    return false;
  }

  VLOG(2) << "Before PipelineRecomputationStageInserter:";
  XLA_VLOG_LINES(2, module->ToString(HloPrintOptions::ShortParsable()));

  TF_ASSIGN_OR_RETURN(bool changed, RecomputePipeline(pipeline_ops[0]));

  if (changed) {
    VLOG(2) << "After PipelineRecomputationStageInserter:";
    XLA_VLOG_LINES(2, module->ToString());
  } else {
    VLOG(2) << "No changes were made to the Pipeline.";
  }
  return changed;
}

}  // namespace poplarplugin
}  // namespace xla
