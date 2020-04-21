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

#include "tensorflow/compiler/plugin/poplar/driver/passes/slice_optimizer.h"

#include <vector>

#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/slice_apply.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/hlo_matcher.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/inplace_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

namespace xla {
namespace poplarplugin {

namespace {
std::function<bool(const HloInstruction*)> IsInplaceHloElementwiseBinary() {
  return [](const HloInstruction* inst) -> bool {
    auto inplace_description = HloInstructionDescription(inst);
    return inst->IsElementwiseBinary() &&
           inplace_description.GetType() ==
               HloInstructionType::kInplaceReadWrite;
  };
}

// clang-format off
static const std::vector<HloMatcherPattern> patterns = {
  // Serialize fn(A * x, Concat(...) * y), where fn is add/subtract.
  HloMatcherPattern(
    PatternType("slice_apply_axby"),
    PatternMetaTarget(0),
    PatternInputs({5, 6, 7, 8}),
    PatternOutputs({0}),
    Pattern({
      {HloMatcherOpcode::kAnyOpcode, NodeOperands({1, 2}), IsAddOrSubtract},
      {HloOpcode::kMultiply, NodeOperands({5, 3})},
      {HloOpcode::kMultiply, NodeOperands({6, 4})},
      {HloOpcode::kBroadcast, NodeOperands({7})},
      {HloOpcode::kBroadcast, NodeOperands({8})},
      {HloMatcherOpcode::kAnyOpcode, NodeOperands({})},
      {HloOpcode::kConcatenate, NodeOperands({})},
      {HloMatcherOpcode::kAnyOpcode, NodeOperands({}), IsScalar},
      {HloMatcherOpcode::kAnyOpcode, NodeOperands({}), IsScalar}
    })
  ),
  // Serialize fn(A, Concat(...) * y), where fn is add/subtract.
  HloMatcherPattern(
    PatternType("slice_apply_aby"),
    PatternMetaTarget(0),
    PatternInputs({3, 4, 5}),
    PatternOutputs({0}),
    Pattern({
      {HloMatcherOpcode::kAnyOpcode, NodeOperands({3, 1}), IsAddOrSubtract},
      {HloOpcode::kMultiply, NodeOperands({4, 2})},
      {HloOpcode::kBroadcast, NodeOperands({5})},
      {HloMatcherOpcode::kAnyOpcode, NodeOperands({})},
      {HloOpcode::kConcatenate, NodeOperands({})},
      {HloMatcherOpcode::kAnyOpcode, NodeOperands({}), IsScalar}
    })
  ),
  // Serialize fn(A * x, Concat(...)), where fn is add/subtract.
  HloMatcherPattern(
    PatternType("slice_apply_axb"),
    PatternMetaTarget(0),
    PatternInputs({3, 4, 5}),
    PatternOutputs({0}),
    Pattern({
      {HloMatcherOpcode::kAnyOpcode, NodeOperands({1, 4}), IsAddOrSubtract},
      {HloOpcode::kMultiply, NodeOperands({3, 2})},
      {HloOpcode::kBroadcast, NodeOperands({5})},
      {HloMatcherOpcode::kAnyOpcode, NodeOperands({})},
      {HloOpcode::kConcatenate, NodeOperands({})},
      {HloMatcherOpcode::kAnyOpcode, NodeOperands({}), IsScalar}
    })
  ),
  // Serialize fn(A, Concat(...)), where fn is a binary op which can be inplace.
  HloMatcherPattern(
    PatternType("slice_apply"),
    PatternMetaTarget(0),
    PatternInputs({1, 2}),
    PatternOutputs({0}),
    Pattern({
      {HloMatcherOpcode::kAnyOpcode, NodeOperands({1, 2}), IsInplaceHloElementwiseBinary()},
      {HloMatcherOpcode::kAnyOpcode, NodeOperands({})},
      {HloOpcode::kConcatenate, NodeOperands({})},
    })
  )
};
// clang-format on

Status ReplaceMatch(HloMatcherMatched& match) {
  HloComputation* comp = match.computation;
  auto pattern = patterns[match.pattern_idx];
  HloInstruction* root = match.instruction_mapping.at(0);
  HloInstruction* input = match.instruction_mapping.at(pattern.GetInputs()[0]);
  HloInstruction* updates =
      match.instruction_mapping.at(pattern.GetInputs()[1]);

  HloInstruction* output = input;
  int64 start_index = 0;
  const int64 apply_dimension = updates->concatenate_dimension();
  const HloOpcode operation = root->opcode();

  for (HloInstruction* update : updates->operands()) {
    switch (match.pattern_idx) {
      case 0: {
        // Handle SliceApplyaXbY.
        HloInstruction* scale_input =
            match.instruction_mapping.at(pattern.GetInputs()[2]);
        HloInstruction* scale_update =
            match.instruction_mapping.at(pattern.GetInputs()[3]);

        output = comp->AddInstruction(
            CreateSliceApplyaXbY(output, update, scale_input, scale_update,
                                 apply_dimension, start_index, operation));
        break;
      }
      case 1: {
        // Handle SliceApplyabY.
        HloInstruction* scale_update =
            match.instruction_mapping.at(pattern.GetInputs()[2]);

        output = comp->AddInstruction(
            CreateSliceApplyabY(output, update, scale_update, apply_dimension,
                                start_index, operation));
        break;
      }
      case 2: {
        // Handle SliceApplyaXb.
        HloInstruction* scale_input =
            match.instruction_mapping.at(pattern.GetInputs()[2]);

        output = comp->AddInstruction(
            CreateSliceApplyaXb(output, update, scale_input, apply_dimension,
                                start_index, operation));
        break;
      }
      case 3: {
        // Handle SliceApply.
        output = comp->AddInstruction(CreateSliceApply(
            output, update, apply_dimension, start_index, operation));
        break;
      }
      default: { return InternalError("Invalid pattern index."); }
    }

    start_index += update->shape().dimensions(apply_dimension);
  }

  TF_RETURN_IF_ERROR(comp->ReplaceInstruction(root, output));
  return Status::OK();
}
};  // namespace

SliceOptimizer::SliceOptimizer(CompilerAnnotations& annotations)
    : HloMatcher(patterns, annotations, false, false) {}

bool SliceOptimizer::HandleMatch(HloMatcherMatched& match,
                                 const absl::optional<int64>) {
  auto status = ReplaceMatch(match);
  if (!status.ok()) {
    LOG(FATAL) << status;
  }
  return true;
}

}  // namespace poplarplugin
}  // namespace xla
