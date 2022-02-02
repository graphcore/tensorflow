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
#include <vector>

#include "tensorflow/compiler/plugin/poplar/driver/tools/slice_util.h"

#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/shape_util.h"

namespace xla {
namespace poplarplugin {

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
