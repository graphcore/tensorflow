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

#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/f8_convert.h"

#include <memory>
#include <string>
#include <vector>

namespace xla {
namespace poplarplugin {

HloConvertFromF8Instruction::HloConvertFromF8Instruction(
    const Shape& shape, HloInstruction* operand)
    : HloF8ConvertInstruction(shape, operand) {
  CHECK_EQ(shape, GetShape(operand));
}

Shape HloConvertFromF8Instruction::GetShape(const HloInstruction* operand) {
  // Result shape is f16[<input-dimensions>].
  const Shape& op_shape = operand->shape();
  CHECK(op_shape.IsTuple());

  // Expect data to be in U8.
  const Shape& input_shape = op_shape.tuple_shapes(0);
  CHECK(input_shape.element_type() == U8);

  const Shape& metadata_shape = op_shape.tuple_shapes(1);
  CHECK(metadata_shape.element_type() == U8);
  CHECK(ShapeUtil::IsScalar(metadata_shape));

  // The only supported type now is F16.
  return ShapeUtil::MakeShape(F16, input_shape.dimensions());
}

std::unique_ptr<HloInstruction>
HloConvertFromF8Instruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext*) const {
  CHECK_EQ(new_operands.size(), 1);
  return std::unique_ptr<HloInstruction>(
      new HloConvertFromF8Instruction(shape, new_operands[0]));
}

HloConvertToF8Instruction::HloConvertToF8Instruction(const Shape& shape,
                                                     HloInstruction* operand)
    : HloF8ConvertInstruction(shape, operand) {
  CHECK(ShapeUtil::Compatible(shape, GetShape(operand)));
}

Shape HloConvertToF8Instruction::GetShape(const HloInstruction* operand) {
  // Result shape is (u8[<input-dimensions>], u8 metadata).
  const Shape& op_shape = operand->shape();
  CHECK(op_shape.IsTuple());

  // The only supported type now is F16.
  const Shape& input_shape = op_shape.tuple_shapes(0);
  CHECK(input_shape.element_type() == F16);

  const Shape& metadata_shape = op_shape.tuple_shapes(1);
  CHECK(metadata_shape.element_type() == U8);
  CHECK(ShapeUtil::IsScalar(metadata_shape));

  return ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeShape(U8, input_shape.dimensions()), metadata_shape});
}

std::unique_ptr<HloInstruction>
HloConvertToF8Instruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext*) const {
  CHECK_EQ(new_operands.size(), 1);
  return std::unique_ptr<HloInstruction>(
      new HloConvertToF8Instruction(shape, new_operands[0]));
}

namespace {
StatusOr<std::unique_ptr<HloInstruction>>
HloConvertFromF8InstructionFactoryFunc(HloCustomCallInstruction* call) {
  return std::unique_ptr<HloInstruction>(
      new HloConvertFromF8Instruction(call->shape(), call->mutable_operand(0)));
}

static HloPoplarInstructionFactory fp8_convert_from_factory(
    PoplarOp::ConvertFromF8, HloConvertFromF8InstructionFactoryFunc);

StatusOr<std::unique_ptr<HloInstruction>> HloConvertToF8InstructionFactoryFunc(
    HloCustomCallInstruction* call) {
  return std::unique_ptr<HloInstruction>(
      new HloConvertToF8Instruction(call->shape(), call->mutable_operand(0)));
}

static HloPoplarInstructionFactory fp8_convert_to_factory(
    PoplarOp::ConvertToF8, HloConvertToF8InstructionFactoryFunc);
}  // namespace

}  // namespace poplarplugin
}  // namespace xla
