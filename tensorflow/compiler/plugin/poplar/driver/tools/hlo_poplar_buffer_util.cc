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

#include "tensorflow/compiler/plugin/poplar/driver/tools/hlo_poplar_buffer_util.h"

#include <set>

#include "tensorflow/compiler/plugin/poplar/driver/tools/alias_info.pb.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/hlo_poplar_buffer.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/shape_util.h"

namespace xla {
namespace poplarplugin {
HloPoplarUseDescriptions UseDescriptionsNoInputOutputAlias() { return {}; }

HloPoplarUseDescriptions UseDescriptionsSimpleNoTupleAliasing(
    const HloInstruction* inst, int64 num_operands, BufferUseKind kind) {
  auto is_tuple = [](const HloInstruction* inst) {
    return inst->shape().IsTuple();
  };
  CHECK(!is_tuple(inst));
  CHECK(!absl::c_any_of(inst->operands(), is_tuple));
  HloPoplarUseDescriptions outputs;
  for (int64 i = 0; i != num_operands; ++i) {
    outputs.push_back(HloPoplarUseDescription{i, /*operand_index=*/ShapeIndex{},
                                              /*output_index=*/ShapeIndex{},
                                              kind});
  }
  return outputs;
}

HloPoplarUseDescriptions UseDescriptionsSimpleNoTuple0thOperandAliasing(
    const HloInstruction* inst, BufferUseKind kind) {
  return UseDescriptionsSimpleNoTupleAliasing(inst, 1, kind);
}

HloPoplarUseDescriptions UseDescriptionsForwardsBuffers(
    const HloInstruction* inst, int64 num_operands, BufferUseKind kind) {
  if (!inst->shape().IsTuple()) {
    CHECK_EQ(num_operands, 1);
    return UseDescriptionsSimpleNoTuple0thOperandAliasing(inst, kind);
  }

  HloPoplarUseDescriptions outputs;
  CHECK_EQ(num_operands, ShapeUtil::TupleElementCount(inst->shape()));
  for (int64 i = 0; i != num_operands; ++i) {
    const HloInstruction* operand = inst->operand(i);
    CHECK_EQ(ShapeUtil::GetSubshape(inst->shape(), ShapeIndex{i}),
             operand->shape());

    for (auto& indexed_shape : ShapeUtil::GetLeafShapes(operand->shape())) {
      const ShapeIndex operand_index = indexed_shape.index;
      // Prefix the output with the corresponding tuple index.
      ShapeIndex output_index = operand_index;
      output_index.push_front(i);

      outputs.push_back(
          HloPoplarUseDescription{i, operand_index, output_index, kind});
    }
  }
  return outputs;
}

HloPoplarBufferDescriptions BufferDescriptionsNoAllocations() { return {}; }

HloPoplarBufferDescriptions BufferDescriptionsAllocatesAllOutputs(
    const HloInstruction* inst, BufferLocality locality) {
  HloPoplarBufferDescriptions outputs;
  for (auto& indexed_shape : ShapeUtil::GetLeafShapes(inst->shape())) {
    outputs.push_back(
        HloPoplarBufferDescription{indexed_shape.index, locality});
  }
  return outputs;
}

HloPoplarBufferDescriptions BufferDescriptionsAllocatesAllUnaliasedBuffers(
    const HloInstruction* inst, const HloPoplarUseDescriptions& descriptions,
    BufferLocality locality) {
  std::set<ShapeIndex> unaliased_outputs;
  absl::c_transform(ShapeUtil::GetLeafShapes(inst->shape()),
                    std::inserter(unaliased_outputs, unaliased_outputs.begin()),
                    [](const ShapeUtil::IndexedShape& indexed_shape) {
                      return indexed_shape.index;
                    });

  // Erase all the aliased outputs.
  for (const auto& description : descriptions) {
    CHECK(unaliased_outputs.erase(description.output_index()));
  }

  // Create the buffer descriptions.
  HloPoplarBufferDescriptions outputs;
  for (const ShapeIndex& output_index : unaliased_outputs) {
    outputs.push_back(HloPoplarBufferDescription{output_index, locality});
  }
  return outputs;
}
}  // namespace poplarplugin
}  // namespace xla
