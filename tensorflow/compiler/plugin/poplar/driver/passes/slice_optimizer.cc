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
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

namespace xla {
namespace poplarplugin {

namespace {
std::function<bool(const HloInstruction*)> IsInplaceHloElementwiseBinary() {
  return [](const HloInstruction* inst) -> bool {
    return inst->IsElementwiseBinary() &&
           GetInplaceDescription(inst).GetType() ==
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

// Function which is callled when creating a slice.
using SliceApplyFn = std::function<StatusOr<HloInstruction*>(
    HloOpcode, HloInstruction* const, HloInstruction* const, int64_t, int64_t)>;

StatusOr<HloInstruction*> ConvertToSliceApplyBase(HloOpcode opcode,
                                                  HloInstruction* const input,
                                                  HloInstruction* const update,
                                                  SliceApplyFn fn) {
  if (update->opcode() != HloOpcode::kConcatenate) {
    return FailedPrecondition(
        "Expected the update to be a concatenate instruction");
  }
  HloInstruction* output = input;
  int64_t start_index = 0;
  const int64_t apply_dimension = update->concatenate_dimension();
  for (HloInstruction* update_slice : update->operands()) {
    TF_ASSIGN_OR_RETURN(
        output, fn(opcode, output, update_slice, apply_dimension, start_index));
    CopyShardingIfPresent(input, output);
    start_index += update_slice->shape().dimensions(apply_dimension);
  }
  return output;
}

Status ReplaceMatch(HloMatcherMatched& match) {
  HloComputation* comp = match.computation;
  const auto& pattern = match.pattern;
  HloInstruction* root = match.instruction_mapping.at(0);
  HloInstruction* input = match.instruction_mapping.at(pattern.GetInputs()[0]);
  HloInstruction* update = match.instruction_mapping.at(pattern.GetInputs()[1]);
  const HloOpcode opcode = root->opcode();

  HloInstruction* output;
  switch (match.pattern_idx) {
    case 0: {
      // Handle SliceApplyaXbY.
      HloInstruction* scale_input =
          match.instruction_mapping.at(pattern.GetInputs()[2]);
      HloInstruction* scale_update =
          match.instruction_mapping.at(pattern.GetInputs()[3]);
      TF_ASSIGN_OR_RETURN(
          output, SliceOptimizer::ConvertToSliceApplyaXbY(
                      opcode, input, update, scale_input, scale_update));
      break;
    }
    case 1: {
      // Handle SliceApplyabY.
      HloInstruction* scale_update =
          match.instruction_mapping.at(pattern.GetInputs()[2]);
      TF_ASSIGN_OR_RETURN(output, SliceOptimizer::ConvertToSliceApplyabY(
                                      opcode, input, update, scale_update));
      break;
    }
    case 2: {
      // Handle SliceApplyaXb.
      HloInstruction* scale_input =
          match.instruction_mapping.at(pattern.GetInputs()[2]);
      TF_ASSIGN_OR_RETURN(output, SliceOptimizer::ConvertToSliceApplyaXb(
                                      opcode, input, update, scale_input));
      break;
    }
    case 3: {
      // Handle SliceApply.
      TF_ASSIGN_OR_RETURN(
          output, SliceOptimizer::ConvertToSliceApply(opcode, input, update));
      break;
    }
    default: { return InternalError("Invalid pattern index."); }
  }

  TF_RETURN_IF_ERROR(comp->ReplaceInstruction(root, output));
  return Status::OK();
}
};  // namespace

StatusOr<HloInstruction*> SliceOptimizer::ConvertToSliceApply(
    HloOpcode opcode, HloInstruction* const input,
    HloInstruction* const update) {
  return ConvertToSliceApplyBase(
      opcode, input, update,
      [](HloOpcode opcode, HloInstruction* const input,
         HloInstruction* const update_slice, int64_t apply_dimension,
         int64_t start_index) -> StatusOr<HloInstruction*> {
        HloInstruction* output = input;
        // We can skip this update if the slice is just zeros.
        if (!IsWideConstantZero(update_slice)) {
          output = input->parent()->AddInstruction(CreateSliceApply(
              input, update_slice, apply_dimension, start_index, opcode));
        }
        return output;
      });
}

StatusOr<HloInstruction*> SliceOptimizer::ConvertToSliceApplyabY(
    HloOpcode opcode, HloInstruction* const input, HloInstruction* const update,
    HloInstruction* const scale_update) {
  return ConvertToSliceApplyBase(
      opcode, input, update,
      [scale_update](HloOpcode opcode, HloInstruction* const input,
                     HloInstruction* const update_slice,
                     int64_t apply_dimension,
                     int64_t start_index) -> StatusOr<HloInstruction*> {
        HloInstruction* output = input;
        // We can skip this update if the slice is just zeros.
        if (!IsWideConstantZero(update_slice)) {
          output = input->parent()->AddInstruction(
              CreateSliceApplyabY(input, update_slice, scale_update,
                                  apply_dimension, start_index, opcode));
        }
        return output;
      });
}

StatusOr<HloInstruction*> SliceOptimizer::ConvertToSliceApplyaXb(
    HloOpcode opcode, HloInstruction* const input, HloInstruction* const update,
    HloInstruction* const scale_input) {
  return ConvertToSliceApplyBase(
      opcode, input, update,
      [scale_input](HloOpcode opcode, HloInstruction* const input,
                    HloInstruction* const update_slice, int64_t apply_dimension,
                    int64_t start_index) -> StatusOr<HloInstruction*> {
        // TODO(T19611) when the update slice is all zeros, just create a scalar
        // multiply on the part of the input.
        return input->parent()->AddInstruction(
            CreateSliceApplyaXb(input, update_slice, scale_input,
                                apply_dimension, start_index, opcode));
      });
}

StatusOr<HloInstruction*> SliceOptimizer::ConvertToSliceApplyaXbY(
    HloOpcode opcode, HloInstruction* const input, HloInstruction* const update,
    HloInstruction* const scale_input, HloInstruction* const scale_update) {
  return ConvertToSliceApplyBase(
      opcode, input, update,
      [scale_input, scale_update](
          HloOpcode opcode, HloInstruction* const input,
          HloInstruction* const update_slice, int64_t apply_dimension,
          int64_t start_index) -> StatusOr<HloInstruction*> {
        // TODO(T19611) when the update slice is all zeros, just create a scalar
        // multiply on the part of the input.
        return input->parent()->AddInstruction(
            CreateSliceApplyaXbY(input, update_slice, scale_input, scale_update,
                                 apply_dimension, start_index, opcode));
      });
}

SliceOptimizer::SliceOptimizer(CompilerAnnotations& annotations)
    : HloMatcher(patterns, annotations, false, false) {}

StatusOr<bool> SliceOptimizer::HandleMatch(HloMatcherMatched& match,
                                           const absl::optional<int64_t>) {
  TF_RETURN_IF_ERROR(ReplaceMatch(match));
  return true;
}

}  // namespace poplarplugin
}  // namespace xla
