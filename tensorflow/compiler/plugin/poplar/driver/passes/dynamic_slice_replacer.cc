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

#include <vector>

#include "tensorflow/compiler/plugin/poplar/driver/tools/slice_util.h"

#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"

namespace xla {
namespace poplarplugin {
namespace {

bool IsDynamicUpdateAdd(const HloInstruction* inst) {
  if (inst->opcode() == HloOpcode::kDynamicUpdateSlice) {
    return DynamicUpdateAdd::IsDynamicUpdateAdd(
        Cast<HloDynamicUpdateSliceInstruction>(inst));
  }
  return false;
}

// These patterns are simple but it's easier to use a HloMatcher than
// manually iterating over the instructions - due to complications from
// prescedence and overlap between the dynamic_update_add and other patterns.
static const std::vector<HloMatcherPattern> patterns = {
    // dynamic_update_add should be first since it matches to the other patterns
    // too
    HloMatcherPattern(PatternType("dynamic_update_add"), PatternMetaTarget(1),
                      PatternInputs({}), PatternOutputs({0}),
                      Pattern({
                          {HloOpcode::kDynamicUpdateSlice, NodeOperands({}),
                           IsDynamicUpdateAdd},
                      })),
    HloMatcherPattern(PatternType("dynamic_slice"), PatternMetaTarget(1),
                      PatternInputs({}), PatternOutputs({0}),
                      Pattern({
                          {HloOpcode::kDynamicSlice, NodeOperands({})},
                      })),
    HloMatcherPattern(PatternType("dynamic_update"), PatternMetaTarget(1),
                      PatternInputs({}), PatternOutputs({0}),
                      Pattern({
                          {HloOpcode::kDynamicUpdateSlice, NodeOperands({})},
                      })),
};
}  // namespace

DynamicSliceReplacer::DynamicSliceReplacer(CompilerAnnotations& annotations)
    : HloMatcher(patterns, annotations, false, true) {}

StatusOr<bool> DynamicSliceReplacer::HandleMatch(
    HloMatcherMatched& match, const absl::optional<int64_t> shard) {
  auto* root = match.instruction_mapping.at(0);

  auto matched = false;
  switch (match.pattern_idx) {
    case 0: {
      TF_ASSIGN_OR_RETURN(matched, HandleDynamicUpdateAdd(root));
      break;
    }
    case 1: {
      TF_ASSIGN_OR_RETURN(matched, HandleDynamicSlice(root));
      break;
    }
    case 2: {
      TF_ASSIGN_OR_RETURN(matched, HandleDynamicUpdate(root));
      break;
    }
    default: {
      return InternalError("Invalid pattern index for %s",
                           match.pattern.GetType());
    }
  }

  return matched;
}

StatusOr<bool> DynamicSliceReplacer::HandleDynamicUpdateAdd(
    HloInstruction* match_root) const {
  auto* inst = Cast<HloDynamicUpdateSliceInstruction>(match_root);
  DynamicUpdateAdd dynamic_update_add(inst);
  TF_ASSIGN_OR_RETURN(
      auto replacement,
      TryReplaceDynamicUpdateAddWithMultiUpdateAdd(dynamic_update_add));
  return replacement != nullptr;
}

StatusOr<bool> DynamicSliceReplacer::HandleDynamicSlice(
    HloInstruction* match_root) const {
  auto* inst = Cast<HloDynamicSliceInstruction>(match_root);
  TF_ASSIGN_OR_RETURN(auto replacement,
                      TryReplaceDynamicSliceWithMultiSlice(inst));

  return replacement != nullptr;
}

StatusOr<bool> DynamicSliceReplacer::HandleDynamicUpdate(
    HloInstruction* match_root) const {
  auto* inst = Cast<HloDynamicUpdateSliceInstruction>(match_root);
  TF_ASSIGN_OR_RETURN(auto replacement,
                      TryReplaceDynamicUpdateWithMultiUpdate(inst));

  return replacement != nullptr;
}

}  // namespace poplarplugin
}  // namespace xla
