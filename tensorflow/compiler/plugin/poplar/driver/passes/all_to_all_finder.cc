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

#include "tensorflow/compiler/plugin/poplar/driver/passes/all_to_all_finder.h"

#include <algorithm>
#include <vector>

#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/all_gather.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/hlo_poplar_instruction.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/multi_slice.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/replication_factor.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/hlo_matcher.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/shape_util.h"

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

/*
Find the pattern:

a = MultiUpdateAdd(WideConstant, Indices, Updates, Scale)
a = AllReduce(a)
a = ReplicationNormalise(a)

And transform it into:

gathered_ind = AllGather(Indices)
gathered_up = AllGather(Updates)
gathered_up = Normalize(gathered_up)
a = MultiUpdateAdd(WideConstant, gathered_ind, gathered_up, Scale)

We only perform this when the wide constant tensor is larger than the update
tensor * replication factor.
*/

namespace xla {
namespace poplarplugin {

namespace {
// clang-format off
static const std::vector<HloMatcherPattern> patterns = {
  HloMatcherPattern(
    PatternType("multi_update_add"),
    PatternMetaTarget(2),
    PatternInputs({5, 6, 7}),
    PatternOutputs({0}),
    Pattern({
        {HloOpcode::kCustomCall, NodeOperands({1}), IsPoplarInstruction(PoplarOp::ReplicationNormalise)},
        {HloOpcode::kAllReduce, NodeOperands({2})},
        {HloOpcode::kCustomCall, NodeOperands({3, 5, 6, 7}), IsPoplarInstruction(PoplarOp::MultiUpdateAdd)},
        {HloOpcode::kBroadcast, NodeOperands({4})},
        {HloOpcode::kConstant, NodeOperands({}), IsScalarConstant},
        {HloMatcherOpcode::kAnyOpcode, NodeOperands({})},
        {HloMatcherOpcode::kAnyOpcode, NodeOperands({}), IsF16OrF32},
        {HloMatcherOpcode::kAnyOpcode, NodeOperands({}), IsF16OrF32},
    })
  ),
  HloMatcherPattern(
    PatternType("multi_update"),
    PatternMetaTarget(2),
    PatternInputs({5, 6}),
    PatternOutputs({0}),
    Pattern({
        {HloOpcode::kCustomCall, NodeOperands({1}), IsPoplarInstruction(PoplarOp::ReplicationNormalise)},
        {HloOpcode::kAllReduce, NodeOperands({2})},
        {HloOpcode::kCustomCall, NodeOperands({3, 5, 6}), IsPoplarInstruction(PoplarOp::MultiUpdate)},
        {HloOpcode::kBroadcast, NodeOperands({4})},
        {HloOpcode::kConstant, NodeOperands({}), IsScalarConstant},
        {HloMatcherOpcode::kAnyOpcode, NodeOperands({})},
        {HloMatcherOpcode::kAnyOpcode, NodeOperands({}), IsF16OrF32},
    })
  ),
};
// clang-format on

// Add an all gather and reshape it.
static HloInstruction* AddAllGatherAndReshape(HloInstruction* original,
                                              uint32 replication_factor) {
  HloComputation* comp = original->parent();
  // Extend the old shape to include the replication factor.
  std::vector<int64> new_update_dims = original->shape().dimensions();
  new_update_dims.insert(new_update_dims.begin(), replication_factor);

  // Create the new update output shape.
  Shape new_update_shape =
      ShapeUtil::MakeShape(original->shape().element_type(), new_update_dims);

  // Gather the updates from all replicas.
  HloInstruction* gathered =
      comp->AddInstruction(CreatePoplarAllGather({original}, new_update_shape));

  Shape flattened_shape = original->shape();
  flattened_shape.set_dimensions(
      0, flattened_shape.dimensions(0) * replication_factor);

  HloInstruction* reshaped_updates = comp->AddInstruction(
      HloInstruction::CreateReshape(flattened_shape, gathered));

  return reshaped_updates;
}

// We say the operation is cost effective if the size of the updates and indices
// multiplided by the replication_factor together are smaller than the size of
// the buffer sent by the all reduce.
static bool IsSwapCostEffective(HloInstruction* multi_update,
                                HloInstruction* all_reduce,
                                uint32 replication_factor) {
  // Get the shape and size of the updates which are sent by the multi_update.
  const Shape& updates_shape = multi_update->operand(2)->shape();
  const int64 updates_size = ShapeUtil::ByteSizeOf(updates_shape);

  // Get the shape of the indices.
  const Shape& indices_shape = multi_update->operand(1)->shape();
  const int64 indices_size = ShapeUtil::ByteSizeOf(indices_shape);

  // This is how much data each replica would send if we do the optimization.
  const int64 size_sent_by_opt =
      replication_factor * (updates_size + indices_size);

  // Get the size of the data which would be send if we don't do the
  // optimization.
  const Shape& all_reduce_shape = all_reduce->shape();
  const int64 all_reduce_size = ShapeUtil::ByteSizeOf(all_reduce_shape);

  VLOG(1) << "Seeing if cost of performing all reduce of "
          << all_reduce->ToString()
          << " is less than the cost of performing an all gather of the "
             "updates and indices of "
          << multi_update->ToString();
  VLOG(1) << "All reduce cost estimate: " << all_reduce_size
          << " optimization cost estimate: " << size_sent_by_opt;

  // If the allreduce is bigger than the other two, perform the optimization.
  return all_reduce_size > size_sent_by_opt;
}

// Actually apply the transformation.
static Status ApplyTransformation(HloMatcherMatched& match,
                                  uint32 replication_factor) {
  HloComputation* comp = match.computation;

  HloMultiUpdateInstruction* multi_update =
      Cast<HloMultiUpdateInstruction>(match.instruction_mapping[2]);

  HloInstruction* broadcast = match.instruction_mapping[3];
  HloInstruction* indices = match.instruction_mapping[5];
  CHECK_EQ(indices->shape().rank(), 2);
  HloInstruction* updates = match.instruction_mapping[6];
  CHECK_EQ(updates->shape().rank(), 2);

  // Take the indices parameter to the multi update add and all gather it
  // across all replicas.
  HloInstruction* reduced_indices =
      AddAllGatherAndReshape(indices, replication_factor);

  // Repeat the above with the updates as well.
  HloInstruction* reduced_updates =
      AddAllGatherAndReshape(updates, replication_factor);

  // Replace the old normalization of the zero tensor with a normalization of
  // the updates.
  HloInstruction* normalized_updates =
      comp->AddInstruction(CreateReplicationNormalise(reduced_updates));

  // TODO(T12935) - Serialize multi updates for large replication factors.
  uint32 serialization_factor = multi_update->GetSerializationFactor();

  if (replication_factor > 4 && replication_factor % 4 == 0) {
    serialization_factor =
        std::max(serialization_factor, replication_factor / 4);
  }

  HloInstruction* output;
  if (match.pattern_idx == 0) {
    // Create MultiUpdateAdd.
    HloInstruction* scale = match.instruction_mapping[7];
    output = comp->AddInstruction(CreateMultiUpdateAdd(
        broadcast->shape(),
        {broadcast, reduced_indices, normalized_updates, scale},
        serialization_factor));
  } else {
    // Create MultiUpdate.
    CHECK_EQ(match.pattern_idx, 1);
    output = comp->AddInstruction(CreateMultiUpdate(
        broadcast->shape(), {broadcast, reduced_indices, normalized_updates},
        serialization_factor));
  }

  // Replace with the new output.
  TF_RETURN_IF_ERROR(
      comp->ReplaceInstruction(match.instruction_mapping[0], output));
  return Status::OK();
}

};  // namespace

AllToAllFinder::AllToAllFinder(CompilerAnnotations& annotations,
                               uint32 replication_factor)
    : replication_factor(replication_factor),
      HloMatcher(patterns, annotations, false, false) {}

StatusOr<bool> AllToAllFinder::HandleMatch(HloMatcherMatched& match,
                                           const absl::optional<int64>) {
  HloInstruction* all_reduce = match.instruction_mapping[1];
  HloInstruction* multi_update = match.instruction_mapping[2];
  if (IsSwapCostEffective(multi_update, all_reduce, replication_factor)) {
    TF_RETURN_IF_ERROR(ApplyTransformation(match, replication_factor));
    return true;
  }

  return false;
}

}  // namespace poplarplugin
}  // namespace xla
