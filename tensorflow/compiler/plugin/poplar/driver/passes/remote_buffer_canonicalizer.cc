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

#include "tensorflow/compiler/plugin/poplar/driver/passes/remote_buffer_canonicalizer.h"

#include <vector>

#include "tensorflow/compiler/plugin/poplar/driver/tools/hlo_matcher.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"

#include "tensorflow/compiler/xla/service/hlo_instruction.h"

namespace xla {
namespace poplarplugin {
namespace {
bool IsRemoteBufferLoad(const HloInstruction* inst) {
  return IsPoplarInstruction(RemoteParameterLoad, inst);
}

bool IsRemoteBufferStore(const HloInstruction* inst) {
  return IsPoplarInstruction(RemoteParameterStore, inst);
}

// clang-format off
static const std::vector<HloMatcherPattern> patterns = {
  HloMatcherPattern(
    PatternType("remote_buffer_load_store"),
    PatternMetaTarget(0),
    PatternInputs({4, 5}),
    PatternOutputs({0, 1}),
    Pattern({
      {HloOpcode::kReshape, NodeOperands({3})},
      {HloOpcode::kCustomCall, NodeOperands({5, 2}), IsRemoteBufferStore},
      {HloOpcode::kReshape, NodeOperands({4})},
      {HloOpcode::kCustomCall, NodeOperands({5}), IsRemoteBufferLoad},
      {HloMatcherOpcode::kAnyOpcode, NodeOperands({})},
      {HloMatcherOpcode::kAnyOpcode, NodeOperands({})},
    })),

  HloMatcherPattern(
    PatternType("remote_buffer_load_only"),
    PatternMetaTarget(0),
    PatternInputs({2}),
    PatternOutputs({0}),
    Pattern({
      {HloOpcode::kReshape, NodeOperands({1})},
      {HloOpcode::kCustomCall, NodeOperands({2}), IsRemoteBufferLoad},
      {HloMatcherOpcode::kAnyOpcode, NodeOperands({})},
    })),
};
// clang-format on

// TODO(T31039): Move these functions into the pattern.
StatusOr<bool> HandleRemoteBufferLoadStore(HloMatcherMatched& match) {
  HloComputation* computation = match.computation;
  HloInstruction* load_reshape = match.instruction_mapping.at(0);
  HloInstruction* store_reshape = match.instruction_mapping.at(2);
  const Shape shape = load_reshape->shape();

  if (shape != store_reshape->operand(0)->shape()) {
    return false;
  }

  // The use of the load and the input to store have a matching shape - move the
  // reshape onto the remote buffer.
  HloInstruction* remote_buffer = match.instruction_mapping.at(5);
  HloInstruction* load = match.instruction_mapping.at(3);
  HloInstruction* store = match.instruction_mapping.at(1);

  // Make sure there are no other users.
  if (remote_buffer->user_count() != 2 || load->user_count() != 1 ||
      store->user_count() != 1) {
    return false;
  }

  HloInstruction* remote_buffer_reshape = computation->AddInstruction(
      HloInstruction::CreateReshape(shape, remote_buffer));
  HloInstruction* new_load = computation->AddInstruction(
      load->CloneWithNewOperands(shape, {remote_buffer_reshape}));
  TF_RETURN_IF_ERROR(load_reshape->ReplaceAllUsesWith(new_load));

  HloInstruction* store_value = store_reshape->mutable_operand(0);
  HloInstruction* new_store = computation->AddInstruction(
      store->CloneWithNewOperands(shape, {remote_buffer_reshape, store_value}));

  // Reshape the remote buffer into the original shape.
  HloInstruction* remote_buffer_updated_reshape = computation->AddInstruction(
      HloInstruction::CreateReshape(store->shape(), new_store));
  TF_RETURN_IF_ERROR(store->ReplaceAllUsesWith(remote_buffer_updated_reshape));

  // Remove the old instructions.
  TF_RETURN_IF_ERROR(computation->RemoveInstruction(store));
  TF_RETURN_IF_ERROR(computation->RemoveInstruction(store_reshape));
  TF_RETURN_IF_ERROR(computation->RemoveInstruction(load_reshape));
  TF_RETURN_IF_ERROR(computation->RemoveInstruction(load));
  return true;
}

StatusOr<bool> HandleRemoteBufferLoadOnly(HloMatcherMatched& match) {
  HloComputation* computation = match.computation;
  HloInstruction* load_reshape = match.instruction_mapping.at(0);
  HloInstruction* load = match.instruction_mapping.at(1);
  HloInstruction* remote_buffer = match.instruction_mapping.at(2);
  const Shape shape = load_reshape->shape();

  if (remote_buffer->user_count() != 1 || load->user_count() != 1) {
    return false;
  }

  HloInstruction* remote_buffer_reshape = computation->AddInstruction(
      HloInstruction::CreateReshape(shape, remote_buffer));
  HloInstruction* new_load = computation->AddInstruction(
      load->CloneWithNewOperands(shape, {remote_buffer_reshape}));
  TF_RETURN_IF_ERROR(load_reshape->ReplaceAllUsesWith(new_load));

  TF_RETURN_IF_ERROR(computation->RemoveInstruction(load_reshape));
  TF_RETURN_IF_ERROR(computation->RemoveInstruction(load));
  return true;
}
}  // namespace

RemoteBufferCanonicalizer::RemoteBufferCanonicalizer(
    struct CompilerAnnotations& annotations)
    : HloMatcher(patterns, annotations, /*root_only=*/false,
                 /*requires_unique_sharding=*/true) {}

StatusOr<bool> RemoteBufferCanonicalizer::HandleMatch(
    HloMatcherMatched& match, const absl::optional<int64_t>) {
  bool handled;
  switch (match.pattern_idx) {
    case 0: {
      TF_ASSIGN_OR_RETURN(handled, HandleRemoteBufferLoadStore(match));
      break;
    }
    case 1: {
      TF_ASSIGN_OR_RETURN(handled, HandleRemoteBufferLoadOnly(match));
      break;
    }
    default: { return InternalError("Invalid pattern index."); }
  }

  return handled;
}
}  // namespace poplarplugin
}  // namespace xla
