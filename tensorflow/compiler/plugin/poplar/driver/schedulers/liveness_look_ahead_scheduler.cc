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
#include "tensorflow/compiler/plugin/poplar/driver/schedulers/liveness_look_ahead_scheduler.h"

#include <map>
#include <queue>
#include <set>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "tensorflow/compiler/plugin/poplar/driver/compiler_information.h"
#include "tensorflow/compiler/plugin/poplar/driver/schedulers/schedule_tree.h"
#include "tensorflow/compiler/plugin/poplar/driver/schedulers/schedule_utils.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/flags.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/platform/logging.h"

namespace xla {
namespace poplarplugin {
namespace {

// TF's calculation for bytes defined
int64 BytesIfScheduled(HloInstruction* instruction,
                       const absl::flat_hash_map<const HloComputation*, int64>&
                           memory_by_computation_,
                       const HloPoplarDataflowAnalysis& dataflow_analysis) {
  auto opcode = instruction->opcode();

  // We only count the memory usage of the largest subcomputation, instead of
  // adding them all, because subcomputations won't execute in parallel.
  int64 max_subcomputation_bytes = 0;
  for (const auto* c : instruction->called_computations()) {
    auto it = memory_by_computation_.find(c);
    if (it != memory_by_computation_.end()) {
      int64 subcomputation_bytes = it->second;
      if (subcomputation_bytes > max_subcomputation_bytes) {
        max_subcomputation_bytes = subcomputation_bytes;
      }
    }
  }

  int64 bytes_defined = 0;
  dataflow_analysis.GetInstructionBufferSet(instruction)
      .ForEachElement([&](const ShapeIndex& /*index*/,
                          const HloPoplarBufferSet& buffer_set) {
        for (auto* buffer : buffer_set.buffers()) {
          if (buffer->DefinedBy(instruction)) {
            bytes_defined += buffer->SizeInBytes();
          }
        }
      });

  if (max_subcomputation_bytes > 0 &&
      (opcode == HloOpcode::kWhile || opcode == HloOpcode::kCall ||
       opcode == HloOpcode::kConditional)) {
    // The output buffer of while/call/conditional is always aliased with the
    // output buffer of the root instruction in the body. Don't double count.
    bytes_defined = max_subcomputation_bytes;
  } else {
    bytes_defined += max_subcomputation_bytes;
  }

  return bytes_defined;
}

struct GrossCost {
  const absl::flat_hash_map<const HloComputation*, int64>&
      memory_by_computation_;
  const HloPoplarDataflowAnalysis& dataflow_analysis;

  int64 operator()(HloInstruction* inst) const {
    return BytesIfScheduled(inst, memory_by_computation_, dataflow_analysis);
  }
};

struct TempCost {
  GrossCost cost_f_;

  template <typename Set>
  int64 operator()(const Set& set, HloInstruction* inst) const {
    int64 result = 0;

    // Consider operands that will be killed, as a temporary cost.
    for (auto operand : inst->unique_operands()) {
      if (set.count(operand) == 0) {
        result += cost_f_(operand);
      }
    }

    return result;
  }
};

// HloInstruction* comparison should not depend on the address, which is
// psuedo-random.
struct HloInstructionPtrComparison {
  size_t operator()(xla::HloInstruction* const& a,
                    xla::HloInstruction* const& b) const {
    if (!a) {
      return true;
    }

    if (!b) {
      return false;
    }

    return a->unique_id() < b->unique_id();
  }
};

using HloScheduleTree =
    ScheduleTree<HloInstruction*, HloInstructionForEachPredecessor,
                 HloInstructionForEachSucessor, GrossCost, TempCost,
                 HloInstructionPtrComparison>;

StatusOr<HloInstructionSequence> ScheduleInstructions(
    HloComputation* comp, const HloPoplarDataflowAnalysis& dataflow_analysis,
    const absl::flat_hash_map<const HloComputation*, int64>&
        memory_by_computation,
    int64 max_search_depth, int64 max_search_size) {
  auto instructions = comp->MakeInstructionPostOrder();
  auto schedule_tree = std::make_shared<HloScheduleTree const>(
      instructions, HloInstructionForEachPredecessor{},
      HloInstructionForEachSucessor{},
      GrossCost{memory_by_computation, dataflow_analysis},
      TempCost{GrossCost{memory_by_computation, dataflow_analysis}});

  schedule_tree =
      schedule_tree->TakeAllReady()->Grow(max_search_depth, max_search_size);
  while (!schedule_tree->IsLeaf()) {
    schedule_tree =
        schedule_tree->BestChild(max_search_size)->Grow(1, max_search_size);
  }

  auto schedule = schedule_tree->GetSchedule();
  return HloInstructionSequence(schedule);
}

StatusOr<HloInstructionSequence> LivenessLookAheadMemoryScheduler(
    HloComputation* computation,
    const HloPoplarDataflowAnalysis& dataflow_analysis,
    const absl::flat_hash_map<const HloComputation*, int64>&
        memory_by_computation,
    int64 max_search_depth, int64 max_search_size) {
  TF_ASSIGN_OR_RETURN(
      auto sched, ScheduleInstructions(computation, dataflow_analysis,
                                       memory_by_computation, max_search_depth,
                                       max_search_size));

  return sched;
}

}  // namespace

// Create a functor which performs the look-ahead scheduling.
IpuSchedulerAlgorithm CreateLivenessLookAheadMemoryScheduler(
    const CompilerInformation& information) {
  return [=](HloComputation* computation,
             const HloPoplarDataflowAnalysis& dataflow_analysis,
             const absl::flat_hash_map<const HloComputation*, int64>&
                 memory_by_computation) {
    return LivenessLookAheadMemoryScheduler(
        computation, dataflow_analysis, memory_by_computation,
        information.max_scheduler_lookahead_depth,
        information.max_scheduler_search_space_size);
  };
}

}  // namespace poplarplugin
}  // namespace xla
