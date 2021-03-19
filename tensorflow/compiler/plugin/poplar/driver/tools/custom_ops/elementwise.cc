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

#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/elementwise.h"

#include <memory>
#include <string>
#include <vector>

namespace xla {
namespace poplarplugin {

// Square
std::unique_ptr<HloInstruction> HloSquareInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext*) const {
  return absl::make_unique<HloSquareInstruction>(new_operands[0]);
}

std::unique_ptr<HloInstruction> CreateSquare(HloInstruction* const operand) {
  return absl::make_unique<HloSquareInstruction>(operand);
}

// Inverse
std::unique_ptr<HloInstruction> HloInverseInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext*) const {
  return absl::make_unique<HloInverseInstruction>(new_operands[0]);
}

std::unique_ptr<HloInstruction> CreateInverse(HloInstruction* const operand) {
  return absl::make_unique<HloInverseInstruction>(operand);
}
}  // namespace poplarplugin
}  // namespace xla
