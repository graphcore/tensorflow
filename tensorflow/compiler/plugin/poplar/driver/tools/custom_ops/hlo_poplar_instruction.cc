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

// All HloInstruction subclasses are put in this file.

#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/hlo_poplar_instruction.h"
#include <utility>
#include "tensorflow/compiler/plugin/poplar/kernels/ops.pb.h"

#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/window_util.h"
#include "tensorflow/core/lib/core/errors.h"

#include "absl/strings/escaping.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"

namespace xla {
namespace poplarplugin {

std::vector<string> HloPoplarInstruction::ExtraAttributesToStringImpl(
    const HloPrintOptions& options) const {
  std::vector<string> extras;
  extras.push_back(absl::StrCat("custom_call_target=\"",
                                absl::CEscape(custom_call_target()), "\""));

  if (custom_call_has_side_effect()) {
    extras.push_back("custom_call_has_side_effect=true");
  }

  std::vector<string> attributes = ExtraPoplarAttributesToStringImpl(options);
  extras.insert(extras.end(), attributes.begin(), attributes.end());

  return extras;
}

HloPoplarInplaceDescription HloPoplarInstruction::GetInplaceDescription()
    const {
  auto use_descriptions = GetUseDescriptions();
  if (use_descriptions.size()) {
    HloPoplarInplaceDescription::OperandIndices inplace_operands;
    for (const HloPoplarUseDescription& description : use_descriptions) {
      inplace_operands.push_back(description.operand_number());
    }
    absl::c_sort(inplace_operands);
    return HloPoplarInplaceDescription(HloInstructionType::kInplaceReadWrite,
                                       std::move(inplace_operands));
  } else {
    return HloPoplarInplaceDescription(HloInstructionType::kNotInplace, {});
  }
}

HloPoplarInstructionFactory::HloPoplarInstructionFactory(
    PoplarOp op, HloPoplarInstructionFactory::FactoryType factory) {
  GetFactoryMap()[PoplarOp_Name(op)] = factory;
}

bool HloPoplarInstructionFactory::IsCreatable(HloCustomCallInstruction* inst) {
  return GetFactoryMap().count(inst->custom_call_target()) == 1;
}

std::unordered_map<std::string, HloPoplarInstructionFactory::FactoryType>&
HloPoplarInstructionFactory::GetFactoryMap() {
  static std::unordered_map<std::string, FactoryType>
      poplar_instruction_factory;
  return poplar_instruction_factory;
}

StatusOr<std::unique_ptr<HloInstruction>> HloPoplarInstructionFactory::Create(
    HloCustomCallInstruction* inst) {
  std::string target_name = inst->custom_call_target();
  if (HloPoplarInstructionFactory::IsCreatable(inst)) {
    return GetFactoryMap().at(target_name)(inst);
  } else {
    return tensorflow::errors::FailedPrecondition(tensorflow::strings::StrCat(
        target_name, " does not have a HloPoplarInstructionFactory instance"));
  }
}

const bool IsPoplibsHloCustomOp(const HloInstruction* inst) {
  return DynCast<HloPoplarInstruction>(inst) != nullptr;
}

Shape GetHloPoplarInstructionShape(absl::Span<HloInstruction* const> operands) {
  if (operands.size() > 1) {
    std::vector<Shape> shapes(operands.size());
    absl::c_transform(operands, shapes.begin(),
                      [](HloInstruction* inst) { return inst->shape(); });
    return ShapeUtil::MakeTupleShape(shapes);
  } else {
    CHECK_EQ(operands.size(), 1);
    return operands[0]->shape();
  }
}

}  // namespace poplarplugin
}  // namespace xla
