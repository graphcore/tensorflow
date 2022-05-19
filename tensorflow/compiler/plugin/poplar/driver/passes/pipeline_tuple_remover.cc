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

#include "tensorflow/compiler/plugin/poplar/driver/passes/pipeline_tuple_remover.h"

#include <map>
#include <vector>

#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/pipeline_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/ops.pb.h"
#include "tensorflow/compiler/xla/service/hlo_creation_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/tuple_simplifier.h"
#include "tensorflow/compiler/xla/shape_tree.h"
#include "tensorflow/compiler/xla/shape_util.h"

namespace xla {
namespace poplarplugin {

StatusOr<bool> PipelineTupleRemover::FlattenPipeline(
    HloInstruction* pipeline_op) {
  bool changed = false;
  HloComputation* pipeline_comp = pipeline_op->to_apply();
  TF_ASSIGN_OR_RETURN(PipelineStages stages, GetPipelineStages(pipeline_comp));
  // Make sure that the root of each stage is a tuple.
  TF_RETURN_IF_ERROR(FixRootInstructions(stages));
  OrderedPipelineStages ordered_stages(stages,
                                       /*include_resource_update*/ true);

  for (int64_t i = 0; i != ordered_stages.GetNumberOfStages(); ++i) {
    HloInstruction* stage = ordered_stages.GetStage(i);

    // Note - the stage and its users will change - keep track of original GTEs.
    std::vector<HloInstruction*> gtes = stage->users();
    for (HloInstruction* gte : gtes) {
      CHECK_EQ(gte->opcode(), HloOpcode::kGetTupleElement);
      if (!gte->shape().IsTuple()) {
        // No tuple to flatten.
        continue;
      }

      HloInstructionSet users;
      for (HloInstruction* user : gte->users()) {
        switch (user->opcode()) {
          case HloOpcode::kCall: {
            CHECK(IsAnyPipelineStageOpOrResourceUpdate(user));
            users.insert(user);
            break;
          }
          case HloOpcode::kCustomCall: {
            if (IsPoplarInstruction(PoplarOp::GradientAccumulatorSink)(user)) {
              // We don't need to do anything for gradient accumulator sink.
              break;
            }
            TF_FALLTHROUGH_INTENDED;
          }
          default: {
            return InternalErrorStrCat("Invalid user ", user->ToString(),
                                       " of pipeline stage ", stage->ToString(),
                                       ".");
          }
        }
      }

      // Unused output - will be removed later.
      if (users.empty()) {
        continue;
      }

      CHECK(gte->shape().IsTuple());
      VLOG(3) << "Flattening output " << gte->ToString();
      changed = true;
      // Keep track of all the instructions which need to be lowered into the
      // `user`.
      std::vector<HloInstruction*> new_stage_instructions;
      // Break up any (nested) tuples by inserting GTEs - leaf nodes will become
      // non-tuple tensors.
      ShapeTree<HloInstruction*> gte_tree(gte->shape());
      TF_RETURN_IF_ERROR(gte_tree.ForEachMutableElementWithStatus(
          [&](const ShapeIndex& index, HloInstruction** node) -> Status {
            if (index.empty()) {
              *node = gte;
            } else {
              ShapeIndex parent_index = index;
              const int64_t tuple_index = parent_index.back();
              parent_index.pop_back();
              HloInstruction* parent = gte_tree.element(parent_index);
              TF_ASSIGN_OR_RETURN(*node,
                                  MakeGetTupleElementHlo(parent, tuple_index));
              new_stage_instructions.push_back(*node);
            }
            return Status::OK();
          }));

      for (HloInstruction* user : users) {
        const int64_t user_idx = ordered_stages.GetIndex(user);
        // Keep track of all the instructions which need to be lowered into the
        // `stage`.
        std::vector<HloInstruction*> new_user_instructions;

        ShapeTree<HloInstruction*> tuple_tree(gte->shape());
        // Traverse the shape in reverse order - leaves to root.
        for (auto itr = tuple_tree.rbegin(); itr != tuple_tree.rend(); itr++) {
          const ShapeIndex& index = itr->first;
          HloInstruction* node;
          if (tuple_tree.IsLeaf(index)) {
            // For leaves just get the GTE inputs.
            node = gte_tree.element(index);
          } else {
            const Shape& tuple_shape =
                ShapeUtil::GetSubshape(gte->shape(), index);
            std::vector<HloInstruction*> operands(
                ShapeUtil::TupleElementCount(tuple_shape));
            for (int64_t i = 0; i != operands.size(); ++i) {
              ShapeIndex child_index = index;
              child_index.push_back(i);
              operands[i] = tuple_tree.element(child_index);
            }
            node = pipeline_comp->AddInstruction(
                HloInstruction::CreateTuple(operands));
            new_user_instructions.push_back(node);
          }

          itr->second = node;
        }

        // Replace the GTE with the root tuple.
        std::map<int64_t, HloInstruction*> replacements;
        absl::c_for_each(user->OperandIndices(gte), [&](int64_t operand_idx) {
          replacements[operand_idx] = tuple_tree.element(ShapeIndex{});
        });

        TF_ASSIGN_OR_RETURN(
            user, AddInstructionsToPipelineStage(user, new_user_instructions,
                                                 replacements));

        ordered_stages.UpdateStage(user_idx, user);
      }

      // Finally lower the new instructions into the stage.
      TF_ASSIGN_OR_RETURN(
          stage, AddInstructionsToPipelineStage(stage, new_stage_instructions));
      ordered_stages.UpdateStage(i, stage);
    }
  }

  for (int64_t i = 0; i != ordered_stages.GetNumberOfStages(); ++i) {
    HloInstruction* stage = ordered_stages.GetStage(i);
    TF_ASSIGN_OR_RETURN(bool changed_ts,
                        TupleSimplifier::RunOnComputation(stage->to_apply()));
    changed |= changed_ts;
  }
  return changed;
}

StatusOr<bool> PipelineTupleRemover::Run(HloModule* module) {
  TF_ASSIGN_OR_RETURN(auto pipeline_ops, GetPipelines(module));
  if (pipeline_ops.empty()) {
    // No pipeline ops found - nothing to hoist.
    return false;
  }
  CHECK_EQ(pipeline_ops.size(), 1);
  VLOG(2) << "Before PipelineTupleRemover:";
  XLA_VLOG_LINES(2, module->ToString(HloPrintOptions::ShortParsable()));

  TF_ASSIGN_OR_RETURN(bool changed, FlattenPipeline(pipeline_ops[0]));

  if (changed) {
    VLOG(2) << "After PipelineTupleRemover:";
    XLA_VLOG_LINES(2, module->ToString());
  } else {
    VLOG(2) << "No changes were made to the Pipeline.";
  }
  return changed;
}

}  // namespace poplarplugin
}  // namespace xla
