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

#include "absl/types/span.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/multi_slice.h"

#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_creation_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/shape_util.h"

namespace xla {
namespace poplarplugin {
namespace {

void ShuffleSpan(absl::Span<int64> values, int64 dim, bool reverse = false) {
  auto front_index = reverse ? 1 : dim;
  std::rotate(values.begin(), values.begin() + front_index,
              values.begin() + dim + 1);
}

std::unique_ptr<HloInstruction> CreateShuffledInput(HloInstruction* input,
                                                    int64 dim,
                                                    bool reverse = false) {
  auto shuffled_shape = input->shape();
  ShuffleSpan(shuffled_shape.mutable_dimensions(), dim, reverse);

  std::vector<int64> shuffled_dim_order(shuffled_shape.dimensions_size(), 0);
  std::iota(shuffled_dim_order.begin(), shuffled_dim_order.end(), 0);
  ShuffleSpan(absl::MakeSpan(shuffled_dim_order), dim, reverse);

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

HloInstruction* Transform1DSliceInput(HloComputation* comp,
                                      HloInstruction* input, int64 slice_dim) {
  // Shuffle⋅the⋅dimension⋅being⋅sliced⋅to⋅the⋅front,⋅since
  // the multi slice ops⋅are⋅hard⋅coded⋅to⋅slice⋅dim⋅0.
  auto* shuffled_input =
      comp->AddInstruction(CreateShuffledInput(input, slice_dim));
  // Flatten since the HloMultiSliceInstruction planner
  // (EmbeddingPlansPreplanning) expects a 2D input tensor.
  // Additionally popops::multiUpdateAdd requires input tensors
  // to be rank2.
  auto* flattened_2d_input =
      comp->AddInstruction(CreateFlattened2DInput(shuffled_input));
  return flattened_2d_input;
}

HloInstruction* Transform1DSliceOffset(HloComputation* comp,
                                       HloDynamicIndexInstruction* inst,
                                       int64 slice_dim) {
  // Reshape the offset since dynamic-slice offsets are scalar.
  auto* dim_offset =
      inst->mutable_operand(inst->first_index_operand_number() + slice_dim);
  // Hardcoded offset shape since we're always doing a single slice
  // of a 2D input.
  auto* offset = comp->AddInstruction(HloInstruction::CreateReshape(
      ShapeUtil::MakeShape(S32, {1, 1}), dim_offset));
  return offset;
}

HloInstruction* Restore1DSliceOutput(HloComputation* comp,
                                     HloInstruction* replacement,
                                     const HloDynamicIndexInstruction* instr,
                                     int64 slice_dim) {
  // Undo the input transformation to get back a tensor of the original
  // shape.
  auto shuffled_output_shape = instr->shape();
  ShuffleSpan(shuffled_output_shape.mutable_dimensions(), slice_dim);

  auto* unflatten = comp->AddInstruction(
      HloInstruction::CreateReshape(shuffled_output_shape, replacement));
  auto* unshuffle = comp->AddInstruction(
      CreateShuffledInput(unflatten, slice_dim, /*reverse=*/true));
  return unshuffle;
}

HloInstruction* Create1DMultiSlice(HloComputation* comp,
                                   HloInstruction* input_tensor,
                                   HloInstruction* offset) {
  // MultiSlice is always 2d and it slices the outer dimension.
  // Patch slice dim with provided value.
  auto multislice_shape = input_tensor->shape();
  multislice_shape.set_dimensions(0, 1);

  auto multislice = comp->AddInstruction(
      CreateMultiSlice(multislice_shape, input_tensor, offset));
  return multislice;
}

HloInstruction* CreateMultiUpdate(HloComputation* comp,
                                  HloInstruction* input_tensor,
                                  HloInstruction* update_slice,
                                  HloInstruction* offset) {
  auto* multiupdate = comp->AddInstruction(xla::poplarplugin::CreateMultiUpdate(
      input_tensor->shape(), {input_tensor, offset, update_slice}));

  return multiupdate;
}

Status ReplaceAndCleanup(HloComputation* comp, HloInstruction* replacement,
                         HloInstruction* original,
                         std::vector<HloInstruction*> to_remove) {
  TF_RETURN_IF_ERROR(original->ReplaceAllUsesWith(replacement));
  original->SetupDerivedInstruction(replacement);

  for (auto* inst : to_remove) {
    TF_RETURN_IF_ERROR(comp->RemoveInstruction(inst));
  }

  return Status::OK();
}
StatusOr<HloInstruction*> Replace1DDynamicWithMultiSlice(
    HloDynamicIndexInstruction* dynamic_slice, int64 slice_dim) {
  auto comp = dynamic_slice->parent();

  auto* transformed_input =
      Transform1DSliceInput(comp, dynamic_slice->mutable_operand(0), slice_dim);
  auto* offset = Transform1DSliceOffset(comp, dynamic_slice, slice_dim);

  HloInstruction* multislice = nullptr;
  if (dynamic_slice->opcode() == HloOpcode::kDynamicSlice) {
    const int64 slice_dim_size = dynamic_slice->shape().dimensions(slice_dim);
    CHECK_EQ(slice_dim_size, 1);
    multislice = Create1DMultiSlice(comp, transformed_input, offset);
  } else {
    CHECK_EQ(dynamic_slice->opcode(), HloOpcode::kDynamicUpdateSlice);
    auto* update_slice = Transform1DSliceInput(
        comp, dynamic_slice->mutable_operand(1), slice_dim);
    multislice =
        CreateMultiUpdate(comp, transformed_input, update_slice, offset);
  }

  auto* output =
      Restore1DSliceOutput(comp, multislice, dynamic_slice, slice_dim);
  TF_RETURN_IF_ERROR(
      ReplaceAndCleanup(comp, output, dynamic_slice, {dynamic_slice}));
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

StatusOr<HloInstruction*> ReduceIndices(HloInstruction* indices, int64 dim,
                                        absl::Span<const int64> sizes) {
  CHECK(dim < indices->shape().rank());
  int64 num_index_dims = indices->shape().dimensions(dim);
  CHECK(sizes.size() == num_index_dims);

  int64 size = 1;
  std::vector<int64> rhs_values(num_index_dims);
  // Generate coefficients to multiply the indices with before reducing them.
  // If the sizes of the dims being indexed are a, b, and c then the
  // coefficients will be [b*c, c, 1].
  for (int64 i = num_index_dims - 1; i != -1; --i) {
    rhs_values[i] = size;
    size *= sizes[i];
  }
  auto type = indices->shape().element_type();
  TF_ASSIGN_OR_RETURN(
      HloInstruction * coefficients,
      MakeR1ConstantHlo<int64>(indices->parent(), type, rhs_values));

  coefficients =
      MakeBroadcastHlo(coefficients, {dim}, indices->shape().dimensions());

  // Multiply the coefficients into the indices.
  TF_ASSIGN_OR_RETURN(
      indices, MakeBinaryHlo(HloOpcode::kMultiply, indices, coefficients));

  // Reduce the index dimension.
  HloInstruction* zero = MakeR0ConstantHlo<int32>(indices->parent(), 0);
  return MakeReduceHlo(indices, zero, {dim}, HloOpcode::kAdd);
}

}  // namespace poplarplugin
}  // namespace xla
