/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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
#include <memory>
#include <tuple>
#include <utility>

#include "tensorflow/compiler/plugin/poplar/driver/backend_config.pb.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/inplace_finder.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/hlo_poplar_instruction.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/hlo_poplar_dataflow_analysis.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/inplace_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"

#include "tensorflow/compiler/xla/service/hlo_module.h"

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/strings/str_util.h"

namespace xla {
namespace poplarplugin {
namespace {
enum class InplacePriority {
  kHigh = 0,
  kMedium = 1,
  kLow = 2,
};

using InplaceCandidates =
    std::map<InplacePriority, std::vector<HloInstruction*>>;

static bool IsInplaceCandidate(HloInstruction* instruction,
                               const HloPoplarDataflowAnalysis& dataflow) {
  // if any of the outputs could be inplaced consider instruction as a candidate
  if (instruction->opcode() == HloOpcode::kParameter) {
    return false;  // Don't handle these later down the line
  }
  if (instruction->opcode() == HloOpcode::kCopy) {
    return false;  // Can't inplace a copy
  }
  if (IsFunction(instruction)) {
    if (GetFunctionNumberModifiedRemoteBufferInputs(instruction) +
        GetFunctionNumberUnmodifiedRemoteBufferInputs(instruction)) {
      // In this case we must inline so don't even check buffers
      return true;
    }
  }
  const auto& instruction_set = dataflow.GetInstructionBufferSet(instruction);
  const auto& buffer_sets = instruction_set.GetBufferSets();
  bool result = false;
  buffer_sets.ForEachElement(
      [&](const ShapeIndex& index, const HloPoplarBufferSet& data) {
        result |= (data.GetUseKind() > BufferUseKind::USE_NO_ALIAS);
      });
  return result;
}

static std::vector<HloInstruction*> FindAllCandidates(
    HloComputation* computation, const HloPoplarDataflowAnalysis& dataflow) {
  std::vector<HloInstruction*> result;
  result.reserve(computation->instruction_count());
  for (HloInstruction* instruction : computation->MakeInstructionPostOrder()) {
    if (IsInplaceCandidate(instruction, dataflow)) {
      result.emplace_back(instruction);
    }
  }
  return result;
}

static HloInstructionType GetInplaceType(
    const HloInstruction* inst,
    const HloPoplarInplaceDescription& inst_description) {
  auto result = inst_description.GetType();
  CHECK(result == HloInstructionType::kInplaceGetTupleElement ||
        result == HloInstructionType::kInplaceReadWrite ||
        result == HloInstructionType::kInplaceReadOnly)
      << "Not allowed to inplace " << static_cast<int>(result)
      << inst->ToString();
  return result;
}

static InplacePriority GetPriority(const HloInstruction* inst) {
  if (inst->parent()->root_instruction() == inst) {
    return InplacePriority::kHigh;
  }
  switch (inst->opcode()) {
    case HloOpcode::kAdd:
    case HloOpcode::kFusion:
    case HloOpcode::kDynamicUpdateSlice:
    case HloOpcode::kMultiply:
    case HloOpcode::kSubtract:
    case HloOpcode::kAddDependency:
    case HloOpcode::kGetTupleElement:
      return InplacePriority::kHigh;
    case HloOpcode::kCustomCall:
      return InplacePriority::kMedium;
    default:
      return InplacePriority::kLow;
  }
}

static int64 GetBufferOperandsSize(
    const HloInstruction* inst,
    const HloPoplarInplaceDescription& inst_description) {
  int64 result = 0;
  for (const int64 index : inst_description.GetInplaceOperandIndices()) {
    const auto* inplace_operand = (inst->operand(index));
    result += GetByteSizeOfTotalShapeSafe(inplace_operand->shape());
  }
  return result;
}

using SortKeyType = std::tuple<HloInstructionType, InplacePriority, int64>;
static SortKeyType CreateSortKey(const HloInstruction* inst) {
  auto inst_description = GetInplaceDescription(inst);
  // Prioritise instructions by how they'll be inplaced, how important they
  // are to inplace, and then by how large they are
  return std::make_tuple(GetInplaceType(inst, inst_description),
                         GetPriority(inst),
                         -1 * GetBufferOperandsSize(inst, inst_description));
}

static absl::flat_hash_map<const HloInstruction*, SortKeyType> CreateKeyMap(
    const std::vector<HloInstruction*>& candidates) {
  absl::flat_hash_map<const HloInstruction*, SortKeyType> result;
  result.reserve(candidates.size());
  for (const auto* inst : candidates) {
    result.emplace(inst, CreateSortKey(inst));
  }
  return result;
}

static std::vector<HloInstruction*> OrderInplaceCandidates(
    std::vector<HloInstruction*> candidates, HloComputation* comp) {
  const auto key_map = CreateKeyMap(candidates);
  absl::c_stable_sort(candidates,
                      [&](const HloInstruction* a, const HloInstruction* b) {
                        return key_map.at(a) < key_map.at(b);
                      });
  return candidates;
}

static std::vector<HloInstruction*> FindInplaceCandidates(
    HloComputation* comp, const HloPoplarDataflowAnalysis& dataflow) {
  std::vector<HloInstruction*> candidates = FindAllCandidates(comp, dataflow);
  return OrderInplaceCandidates(std::move(candidates), comp);
}

}  // namespace

StatusOr<bool> InplaceFinder::Run(HloModule* module) {
  TF_ASSIGN_OR_RETURN(auto dataflow,
                      HloPoplarDataflowAnalysis::Run(module, annotations));
  bool changed = false;
  for (auto* comp : module->MakeComputationPostOrder()) {
    if (IsPopOpsFusion(comp)) {
      continue;
    }

    InplaceWorkList worklist;
    // The reachability map is used for adding and finding control dependencies
    // in order to allow for inplace ops to be executed after other instructions
    // which are using the inplace input.
    std::unique_ptr<HloReachabilityMap> reachability_map =
        HloReachabilityMap::Build(comp);
    // Because we are using a map, we first inplace GTEs, then Read/Write and
    // then Read-Only.
    const auto inplace_candidates = FindInplaceCandidates(comp, *dataflow);
    for (auto* inst : inplace_candidates) {
      HloPoplarInplaceDescription::ConvertToInplace(
          inst, reachability_map.get(), worklist);
    }
    changed |= (!inplace_candidates.empty());
  }

  return changed;
}

}  // namespace poplarplugin
}  // namespace xla
