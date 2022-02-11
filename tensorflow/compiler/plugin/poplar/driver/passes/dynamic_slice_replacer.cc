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

#include "tensorflow/compiler/plugin/poplar/driver/passes/dynamic_slice_replacer.h"

#include "tensorflow/compiler/plugin/poplar/driver/tools/slice_util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/ops.pb.h"

#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/shape_util.h"

namespace xla {
namespace poplarplugin {
namespace {

const Shape& GetSliceShape(const HloDynamicIndexInstruction* dynamic_slice) {
  const auto& slice_shape = dynamic_slice->opcode() == HloOpcode::kDynamicSlice
                                ? dynamic_slice->shape()
                                : dynamic_slice->operand(1)->shape();
  return slice_shape;
}

int64 UnslicedElementCount(const HloDynamicIndexInstruction* dynamic_slice) {
  const auto& input_shape = dynamic_slice->operand(0)->shape();
  const auto& slice_shape = GetSliceShape(dynamic_slice);

  int unsliced_elements = 1;
  for (auto i = 0; i < input_shape.dimensions_size(); ++i) {
    const auto input_dim_size = input_shape.dimensions(i);
    const auto sliced_dim_size = slice_shape.dimensions(i);

    const auto is_unsliced_dim = input_dim_size == sliced_dim_size;
    if (is_unsliced_dim) {
      unsliced_elements *= input_dim_size;
    }
  }

  return unsliced_elements;
}

bool DynamicSliceMightGoOOM(const HloDynamicIndexInstruction* dynamic_slice,
                            uint32_t bytes_per_tile) {
  // popops::dynamicSlice doesn't support planning, so the input tensor being
  // sliced will get allocated over the number of tiles used (via
  // popops::createSliceableTensor). The number of unsliced elements is a factor
  // of this, so if the unsliced dims are small then few tiles will be used.
  // This can lead to OOM when the slice dims are large, as too many elements
  // are placed on a tile. The following is a rough way of trying to check and
  // prevent this situation from occuring, as we can replace an unplanned
  // dynamicSlice with a planned multiSlice.
  const auto input_size =
      ShapeUtil::ByteSizeOfElements(dynamic_slice->operand(0)->shape());
  const auto tiles_used = UnslicedElementCount(dynamic_slice);
  if (tiles_used == 0) {
    return false;
  }

  const auto usage_per_tile_estimate = input_size / tiles_used;
  return usage_per_tile_estimate >= (0.25 * bytes_per_tile);
}
}  // namespace

StatusOr<bool> DynamicSliceReplacer::Run(HloModule* module) {
  bool changed = false;

  for (auto* comp : module->MakeComputationPostOrder()) {
    for (auto* inst : comp->MakeInstructionPostOrder()) {
      if (auto dynamic_slice = DynCast<HloDynamicIndexInstruction>(inst)) {
        // We don't want to try and replace all dynamiceSlices, since multiSlice
        // is slower. Ideallly we only try and replace those which will cause
        // us to go OOM.
        if (DynamicSliceMightGoOOM(dynamic_slice, bytes_per_tile_)) {
          TF_ASSIGN_OR_RETURN(auto replaced,
                              TryReplaceDynamicWithMultiSlice(dynamic_slice));
          if (replaced) {
            changed = true;
          }
        }
      }
    }
  }

  return changed;
}

}  // namespace poplarplugin
}  // namespace xla
