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

add = MultiUpdateAdd(ZeroTensor, Indices, Updates)
add = AllReduce(add)
add = ReplicationNormalise(add)
mul = add * learning_rate
embedding = embedding - mul

And transform it into:

gathered_ind = AllGather(Indices)
gathered_up = AllGather(Updates)
gathered_up = AllReduce(gathered_up)
gathered_up = gathered_up *lr
add = MultiUpdateAdd(embedding, gathered_ind, gathered_up, -1.0)

We only perfom this when the zero tensor is larger than the update tensor. That
is to say, when it is more expensive to send a large zero tensor to be all
reduces on all replicas than to send a small update to every replica. This:
1. Eliminates the zero tensor
2. Sends less data between replicas.
3. Shrinks the size of the operations inbetween the all reduce and the update.
*/

namespace xla {
namespace poplarplugin {

namespace {
// clang-format off
static const std::vector<HloMatcherPattern> patterns = {
    HloMatcherPattern(
        PatternType("all_reduce_to_multi_update_add"), PatternMetaTarget(1),
        PatternInputs({10, 9, 8, 7, 6, 5}), PatternOutputs({0}),
        Pattern({
            // 0. Subtract that from the actual update.
            {HloOpcode::kSubtract, NodeOperands({10, 1})},
            // 1. Multiply by the learning rate.
            {HloOpcode::kMultiply, NodeOperands({5, 2})},
            // 2. Normalizing the output on each replica.
            {HloOpcode::kCustomCall, NodeOperands({3}),
                IsPoplarInstruction(PoplarOp::ReplicationNormalise)},
            // 3. All reduce the zero tensor.
            {HloOpcode::kAllReduce, NodeOperands({4})},
            // 4. Updating the zero tensor.
            {HloOpcode::kCustomCall, NodeOperands({9, 7, 8, 6}),
                IsPoplarInstruction(PoplarOp::MultiUpdateAdd)},
            // 5. Learning Rate
            {HloOpcode::kBroadcast, NodeOperands({}), IsF16OrF32},
            // 6. Scale
            {HloMatcherOpcode::kAnyOpcode, NodeOperands({}), IsF16OrF32},
            // 7. Indices
            {HloMatcherOpcode::kAnyOpcode, NodeOperands({})},
            // 8. Updates
            {HloMatcherOpcode::kAnyOpcode, NodeOperands({}), IsF16OrF32},
            // 9. Zero Tensor.
            {HloMatcherOpcode::kAnyOpcode, NodeOperands({}), IsF16OrF32},
            // 10. Actual embedding.
            {HloMatcherOpcode::kAnyOpcode, NodeOperands({}), IsF16OrF32},
        })),
};
// clang-format on

// Add an all gather and reshape it.
static HloInstruction* AddAllGatherAndReshape(HloComputation* comp,
                                              HloInstruction* original,
                                              uint32 replication_factor) {
  // Extend the old shape to include the replication factor.
  auto original_dims = original->shape().dimensions();
  std::vector<int64> new_update_dims(original_dims.begin(),
                                     original_dims.end());
  new_update_dims.insert(new_update_dims.begin(), replication_factor);

  // Create the new update output shape.
  Shape new_update_shape =
      ShapeUtil::MakeShape(original->shape().element_type(), new_update_dims);

  // Gather the updates from all replicas.
  HloInstruction* gathered =
      comp->AddInstruction(CreateAllGather(original, new_update_shape));

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
static bool IsSwapCostEffective(HloCustomCallInstruction* multi_update_add,
                                HloAllReduceInstruction* all_reduce) {
  // Get the shape and size of the updates which are sent by the multi_update.
  const Shape& updates_shape = multi_update_add->operand(1)->shape();

  const std::size_t updates_size = ShapeUtil::ElementsIn(updates_shape);

  // Get the shape of the indices.
  const Shape& indices_shape = multi_update_add->operand(2)->shape();

  // If the updates are only float 16 then the indices will be twice as big.
  const std::size_t indices_modifier =
      updates_shape.element_type() == PrimitiveType::F16 ? 2 : 1;
  const std::size_t indices_size =
      ShapeUtil::ElementsIn(indices_shape) * indices_modifier;

  // This is how much data each replica would send if we do the optimization.
  std::size_t size_sent_by_opt = updates_size + indices_size;

  // Get the size of the data which would be send if we don't do the
  // optimization.
  const Shape& all_reduce_shape = all_reduce->operand(0)->shape();
  std::size_t all_reduce_size = ShapeUtil::ElementsIn(all_reduce_shape);

  VLOG(1) << "Seeing if cost of performing all reduce of "
          << all_reduce->ToString()
          << " is less than the cost of performing an all gather of the "
             "updates and indices of "
          << multi_update_add->ToString();
  VLOG(1) << "All reduce cost estimate: " << all_reduce_size
          << " optimization cost estimate: " << size_sent_by_opt;

  // If the allreduce is bigger than the other two, perform the optimization.
  return all_reduce_size > size_sent_by_opt;
}

// Actually apply the transformation.
static Status ApplyTransformation(HloMatcherMatched& match,
                                  uint32 replication_factor) {
  // Get the instructions we need to reference from the original pattern.
  HloInstruction* original_subtract = match.instruction_mapping[0];

  HloCustomCallInstruction* multi_update_add =
      Cast<HloCustomCallInstruction>(match.instruction_mapping[4]);
  HloInstruction* original_broadcast = match.instruction_mapping[5];

  HloInstruction* original_scale = match.instruction_mapping[6];

  // Get the parent computation.
  HloComputation* comp = multi_update_add->parent();

  // Take the indices parameter to the multi update add and all gather it
  // across all replicas.
  HloInstruction* original_indices = multi_update_add->operands()[1];
  HloInstruction* reshaped_indices =
      AddAllGatherAndReshape(comp, original_indices, replication_factor);

  // Repeat the above with the updates as well.
  HloInstruction* original_updates = multi_update_add->operands()[2];
  HloInstruction* reshaped_updates =
      AddAllGatherAndReshape(comp, original_updates, replication_factor);

  // Replace the old normalization of the zero tensor with a normalization of
  // the updates.
  HloInstruction* new_normalize =
      comp->AddInstruction(CreateReplicationNormalise(reshaped_updates));

  // Broadcast the learning rate to the new dimensions.
  HloInstruction* broadcast_lr_to_new_shape =
      comp->AddInstruction(HloInstruction::CreateBroadcast(
          reshaped_updates->shape(), original_broadcast->operands()[0], {}));

  HloInstruction* multiply_by_learning_rate = comp->AddInstruction(
      HloInstruction::CreateBinary(new_normalize->shape(), HloOpcode::kMultiply,
                                   new_normalize, broadcast_lr_to_new_shape));

  // Keep track of the series of multi update adds which are to replace
  // the subtract operation.
  std::vector<HloInstruction*> multi_update_adds;

  // We need to invert the multiupdate add to be a subtract so we have to
  // jump through the XLA hoops to create a literal in the right type.
  xla::Literal literal_minus_one = xla::LiteralUtil::CreateR0<float>(-1.0f);

  if (original_scale->shape().element_type() == PrimitiveType::F16) {
    TF_ASSIGN_OR_RETURN(literal_minus_one,
                        literal_minus_one.Convert(PrimitiveType::F16));
  }

  HloInstruction* constant_literal = comp->AddInstruction(
      HloInstruction::CreateConstant(literal_minus_one.Clone()));

  // If the replication factor is potentially too big then we will
  // seralise the multi update. This should be eventually handled by the poplar
  // planner once T12935 has been fixed.
  if (replication_factor > 4 && replication_factor % 4 == 0) {
    std::size_t division_amount = replication_factor / 4;

    Shape indices_shape = reshaped_indices->shape();
    indices_shape.set_dimensions(0,
                                 indices_shape.dimensions(0) / division_amount);

    Shape updates_shape = multiply_by_learning_rate->shape();
    updates_shape.set_dimensions(0,
                                 updates_shape.dimensions(0) / division_amount);

    // We are operating on the same input each time but since it is
    // inplace and we only return the last one we need to pass it along on
    // XLA "sees" the ones inbetween and puts them in the right place.
    HloInstruction* active_embedding = original_subtract->operands()[0];

    // Slice the updates and the index an peform the MultiUpdateAdd operation on
    // those then repeat to cover all slices.
    for (int slice = 0; slice < division_amount; ++slice) {
      // Get a slice of the index.
      const std::size_t index_slice_size = indices_shape.dimensions(0);
      HloInstruction* indices_slice =
          comp->AddInstruction(HloInstruction::CreateSlice(
              indices_shape, reshaped_indices, {slice * index_slice_size},
              {(slice * index_slice_size) + index_slice_size}, {}));

      // Get a slice of the update.
      const std::size_t updates_slice_size = updates_shape.dimensions(0);

      std::vector<int64> start_slice = {slice * updates_slice_size};
      std::vector<int64> end_slice = {(slice * updates_slice_size) +
                                      updates_slice_size};

      // We are creating a slice into the first dimension so additional
      // dimensions need to be fully sliced.
      for (int i = 1; i < updates_shape.dimensions_size(); ++i) {
        start_slice.push_back(0);
        end_slice.push_back(updates_shape.dimensions(i));
      }

      HloInstruction* updates_slice = comp->AddInstruction(
          HloInstruction::CreateSlice(updates_shape, multiply_by_learning_rate,
                                      start_slice, end_slice, {}));

      // Subtract them from the embedding.
      active_embedding = comp->AddInstruction(CreateMultiUpdateAdd(
          multi_update_add->shape(),
          {active_embedding, indices_slice, updates_slice, constant_literal}, 1,
          1));

      multi_update_adds.push_back(active_embedding);
    }

  } else {
    // Add the multi update add.
    multi_update_adds.push_back(comp->AddInstruction(CreateMultiUpdateAdd(
        multi_update_add->shape(),
        {original_subtract->operands()[0], reshaped_indices,
         multiply_by_learning_rate, constant_literal},
        1, 1)));
  }

  // Replace the all reduce with the new multi update add.
  TF_RETURN_IF_ERROR(
      comp->ReplaceInstruction(original_subtract, multi_update_adds.back()));

  return Status::OK();
}

};  // namespace

AllToAllFinder::AllToAllFinder(CompilerAnnotations& annotations, uint32 rep_fac)
    : replication_factor(rep_fac),
      HloMatcher(patterns, annotations, false, false) {}

bool AllToAllFinder::HandleMatch(HloMatcherMatched& match,
                                 const absl::optional<int64>) {
  HloCustomCallInstruction* multi_update_add =
      Cast<HloCustomCallInstruction>(match.instruction_mapping[4]);
  HloAllReduceInstruction* all_reduce =
      Cast<HloAllReduceInstruction>(match.instruction_mapping[3]);

  if (IsSwapCostEffective(multi_update_add, all_reduce)) {
    Status err = ApplyTransformation(match, replication_factor);

    if (err != Status::OK()) {
      LOG(FATAL) << err.ToString();
    } else {
      return true;
    }
  }

  return false;
}

}  // namespace poplarplugin
}  // namespace xla
