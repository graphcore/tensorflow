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

#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/fp8_ops.h"

#include <memory>
#include <string>

namespace xla {
namespace poplarplugin {

HloF8MatMulInstruction::HloF8MatMulInstruction(const Shape& shape,
                                               HloInstruction* lhs,
                                               HloInstruction* lhs_metadata,
                                               HloInstruction* rhs,
                                               HloInstruction* rhs_metadata)
    : HloPoplarInstruction(shape, {lhs, lhs_metadata, rhs, rhs_metadata},
                           PoplarOp::F8MatMul) {}

std::unique_ptr<HloInstruction>
HloF8MatMulInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    HloCloneContext* context) const {
  return absl::make_unique<HloF8MatMulInstruction>(
      shape, operands[0], operands[1], operands[2], operands[3]);
}

namespace {

static HloPoplarInstructionFactory f8_mat_mul_factory(
    PoplarOp::F8MatMul,
    [](HloCustomCallInstruction* call)
        -> StatusOr<std::unique_ptr<HloInstruction>> {
      return StatusOr<std::unique_ptr<HloInstruction>>(
          absl::make_unique<HloF8MatMulInstruction>(
              call->shape(), call->mutable_operand(0), call->mutable_operand(1),
              call->mutable_operand(2), call->mutable_operand(3)));
    });
}  // namespace

}  // namespace poplarplugin
}  // namespace xla
