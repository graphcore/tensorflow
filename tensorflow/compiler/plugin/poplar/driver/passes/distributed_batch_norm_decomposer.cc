/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/plugin/poplar/driver/passes/distributed_batch_norm_decomposer.h"

#include <map>
#include <memory>
#include <queue>
#include <stack>
#include <utility>
#include <vector>

#include "tensorflow/compiler/plugin/poplar/driver/passes/pipeline_recomputation.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/norm.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/recompute.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/pipeline_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_creation_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/strings/str_util.h"

namespace xla {
namespace poplarplugin {

Status DistributedBatchNormDecompose(HloInstruction* const inst) {
  HloComputation* comp = inst->parent();
  HloInstruction* bn_training = Cast<HloBatchNormTrainingInstruction>(inst);
  HloInstruction* operand = inst->mutable_operand(0);
  HloInstruction* scale = inst->mutable_operand(1);
  HloInstruction* offset = inst->mutable_operand(2);
  const float epsilon = bn_training->epsilon();
  const int64 feature_index = bn_training->feature_index();
  const Shape& mean_var_shape = ShapeUtil::GetSubshape(inst->shape(), {1});
  CHECK_EQ(mean_var_shape, ShapeUtil::GetSubshape(inst->shape(), {2}));

  const Shape norm_stats_shape =
      ShapeUtil::MakeTupleShape({mean_var_shape, mean_var_shape});

  // Calculate the stats.
  HloInstruction* norm_stats = comp->AddInstruction(
      CreateBatchNormStats(norm_stats_shape, operand, epsilon, feature_index));
  inst->SetupDerivedInstruction(norm_stats);
  TF_ASSIGN_OR_RETURN(HloInstruction * mean,
                      MakeGetTupleElementHlo(norm_stats, 0));
  TF_ASSIGN_OR_RETURN(HloInstruction * variance,
                      MakeGetTupleElementHlo(norm_stats, 1));

  // Add the checkpoints for mean and variance.
  mean = comp->AddInstruction(CreateRecomputationCheckpoint(mean));
  variance = comp->AddInstruction(CreateRecomputationCheckpoint(variance));

  // Normalize the operand.
  HloInstruction* normalized =
      comp->AddInstruction(HloInstruction::CreateBatchNormInference(
          operand->shape(), operand, scale, offset, mean, variance, epsilon,
          feature_index));
  inst->SetupDerivedInstruction(normalized);

  VLOG(2) << "Replacing " << inst->ToString() << " with a decomposed "
          << norm_stats->ToString() << " and " << normalized->ToString();

  return comp->ReplaceWithNewInstruction(
      inst, HloInstruction::CreateTuple({normalized, mean, variance}));
}

StatusOr<bool> DistributedBatchNormDecomposer::Run(HloModule* module) {
  if (!allow_recomputation_ || replica_group_size_ < 2) {
    return false;
  }

  TF_ASSIGN_OR_RETURN(
      std::vector<HloInstruction*> instructions_to_recompute,
      PipelineRecomputation::GetInstructionsToRecompute(module));

  std::vector<HloInstruction*> batch_norm_training_insts;
  for (HloInstruction* inst : instructions_to_recompute) {
    if (inst->opcode() == HloOpcode::kBatchNormTraining) {
      batch_norm_training_insts.push_back(inst);
    }
  }

  if (batch_norm_training_insts.empty()) {
    return false;
  }

  VLOG(2) << "Before DistributedBatchNormDecomposer:";
  XLA_VLOG_LINES(2, module->ToString());

  for (HloInstruction* inst : batch_norm_training_insts) {
    TF_RETURN_IF_ERROR(DistributedBatchNormDecompose(inst));
  }

  VLOG(2) << "After DistributedBatchNormDecomposer:";
  XLA_VLOG_LINES(2, module->ToString());
  return true;
}

DistributedBatchNormDecomposer::DistributedBatchNormDecomposer(
    bool allow_recomputation, int64 replica_group_size)
    : allow_recomputation_(allow_recomputation),
      replica_group_size_(replica_group_size) {}

}  // namespace poplarplugin
}  // namespace xla
