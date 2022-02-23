/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#include <algorithm>
#include <memory>
#include <vector>

#include "tensorflow/compiler/plugin/poplar/driver/tools/slice_util.h"

#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/multi_slice.h"

#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/shape_util.h"

namespace xla {
namespace poplarplugin {
namespace {

std::unique_ptr<HloInstruction> CreateShuffledInput(HloInstruction* input,
                                                    int64 dim,
                                                    bool reverse = false) {
  const auto Shuffle = [&](absl::Span<int64> values) {
    auto front_index = reverse ? 1 : dim;
    std::rotate(values.begin(), values.begin() + front_index,
                values.begin() + dim + 1);
  };

  auto shuffled_shape = input->shape();
  Shuffle(shuffled_shape.mutable_dimensions());

  std::vector<int64> shuffled_dim_order(shuffled_shape.dimensions_size(), 0);
  std::iota(shuffled_dim_order.begin(), shuffled_dim_order.end(), 0);
  Shuffle(absl::MakeSpan(shuffled_dim_order));

  return HloInstruction::CreateTranspose(shuffled_shape, input,
                                         shuffled_dim_order);
}

std::unique_ptr<HloInstruction> CreateFlattened2DInput(HloInstruction* input) {
  auto shape = input->shape();
  const auto dim0 = shape.dimensions(0);
  const auto flattened_elements = ShapeUtil::ElementsIn(shape) / dim0;
  const auto flattened_2d_shape =
      ShapeUtil::MakeShape(shape.element_type(), {dim0, flattened_elements});
  return HloInstruction::CreateReshape(flattened_2d_shape, input);
}

StatusOr<HloInstruction*> Replace1DDynamicWithMultiSlice(
    HloDynamicIndexInstruction* dynamic_slice, int64 slice_dim) {
  auto comp = dynamic_slice->parent();
  // Shuffle the dimension being sliced to the front, since
  // HloMultiSliceInstruction is hard coded to slice dim 0.
  auto input = dynamic_slice->mutable_operand(0);
  auto shuffled_input =
      comp->AddInstruction(CreateShuffledInput(input, slice_dim));
  // Flatten since the HloMultiSliceInstruction planner
  // (EmbeddingPlansPreplanning) expects a 2D input tensor.
  auto flattened_2d_input =
      comp->AddInstruction(CreateFlattened2DInput(shuffled_input));
  // Reshape the offset since dynamic-slice offsets are scalar.
  auto dim_offset = dynamic_slice->mutable_operand(
      dynamic_slice->first_index_operand_number() + slice_dim);
  auto offset = comp->AddInstruction(HloInstruction::CreateReshape(
      ShapeUtil::MakeShape(S32, {1, 1}), dim_offset));

  HloInstruction* multislice = nullptr;
  if (dynamic_slice->opcode() == HloOpcode::kDynamicSlice) {
    multislice = comp->AddInstruction(
        CreateMultiSlice(dynamic_slice->shape(), flattened_2d_input, offset));
    TF_RETURN_IF_ERROR(dynamic_slice->ReplaceAllUsesWith(multislice));
  } else {
    CHECK_EQ(dynamic_slice->opcode(), HloOpcode::kDynamicUpdateSlice);
    auto slice = dynamic_slice->mutable_operand(1);
    // For the slice to fit into the input it must be shuffled
    // in the same way.
    auto shuffled_slice =
        comp->AddInstruction(CreateShuffledInput(slice, slice_dim));
    // HloMultiUpdateInstruction expects the slice operand to be 2D, in our case
    // the outer dimension will always be 1 though, since we're replacing a
    // size 1 update.
    auto flattened_slice =
        comp->AddInstruction(CreateFlattened2DInput(shuffled_slice));
    CHECK_EQ(flattened_slice->shape().dimensions(0), 1);

    multislice = comp->AddInstruction(
        CreateMultiUpdate(flattened_2d_input->shape(),
                          {flattened_2d_input, offset, flattened_slice}));

    // Restore the input to its original shape..
    auto unflatten = comp->AddInstruction(
        HloInstruction::CreateReshape(shuffled_input->shape(), multislice));
    auto unshuffle = comp->AddInstruction(
        CreateShuffledInput(unflatten, slice_dim, /*reverse*/ true));
    TF_RETURN_IF_ERROR(dynamic_slice->ReplaceAllUsesWith(unshuffle));
  }

  dynamic_slice->SetupDerivedInstruction(multislice);
  TF_RETURN_IF_ERROR(comp->RemoveInstruction(dynamic_slice));
  return multislice;
}

}  // namespace

StatusOr<HloInstruction*> TryReplaceDynamicWithMultiSlice(
    HloDynamicIndexInstruction* dynamic_slice) {
  const auto dynamic_slice_helper = DynamicSliceHelper(dynamic_slice);
  if (dynamic_slice_helper.has_dynamic_slice) {
    const auto& slice_info = dynamic_slice_helper.dynamic_slice_info;
    const auto& slice_dims = slice_info.sliced_dims;
    const auto& slice_sizes = slice_info.slice_sizes;

    // Limiting to size 1 slices, since HloMultiSliceInstruction is hard coded
    // to size 1 slices. If needed we can work around this by doing an N
    // multi-slice for a size N slice.
    const auto can_replace = slice_dims.size() == 1 && slice_sizes.front() == 1;
    if (can_replace) {
      const auto slice_dim = slice_dims.front();
      return Replace1DDynamicWithMultiSlice(dynamic_slice, slice_dim);
    }
  }

  return nullptr;
}

DynamicSliceHelper::DynamicSliceHelper(const HloDynamicIndexInstruction* inst) {
  auto index_operands = inst->index_operands();
  const Shape input_shape = inst->operand(0)->shape();

  // Get the slice sizes in the dynamic and constant dimensions.
  std::vector<int64> slice_sizes;
  if (inst->opcode() == HloOpcode::kDynamicSlice) {
    auto* dyn_slice = Cast<HloDynamicSliceInstruction>(inst);
    slice_sizes = dyn_slice->dynamic_slice_sizes();
  } else {
    auto* dyn_update_slice = Cast<HloDynamicUpdateSliceInstruction>(inst);
    auto dims = dyn_update_slice->operand(1)->shape().dimensions();
    slice_sizes = std::vector<int64>(dims.begin(), dims.end());
  }

  Shape dynamic_slice_shape = input_shape;
  Shape constant_slice_shape = input_shape;

  // For each operand find whether it is a slice dimension.
  for (uint64 dim = 0; dim != index_operands.size(); ++dim) {
    size_t slice_size = slice_sizes[dim];
    if (input_shape.dimensions(dim) != static_cast<int64>(slice_size)) {
      if (index_operands[dim]->opcode() == HloOpcode::kConstant) {
        constant_slice_shape.set_dimensions(dim, slice_size);
        has_constant_slice = true;
      } else {
        dynamic_slice_shape.set_dimensions(dim, slice_size);
        has_dynamic_slice = true;
      }
    }
  }
  dynamic_slice_info = GetSliceInfo(input_shape, dynamic_slice_shape);
  constant_slice_info = GetSliceInfo(input_shape, constant_slice_shape);
}

bool Is1DSliceInFirstDimension(const HloInstruction* slice) {
  auto* broadcast = slice->operand(0);
  return ShapeUtil::DeleteDimension(0, broadcast->shape()) ==
             ShapeUtil::DeleteDimension(0, slice->shape()) &&
         ShapeUtil::GetDimension(slice->shape(), 0) == 1;
}

}  // namespace poplarplugin
}  // namespace xla
