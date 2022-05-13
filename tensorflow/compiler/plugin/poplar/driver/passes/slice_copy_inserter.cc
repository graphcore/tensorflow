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

#include "tensorflow/compiler/plugin/poplar/driver/passes/slice_copy_inserter.h"

#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/hlo_poplar_dataflow_analysis.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/hlo_poplar_liveness.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/xla/service/call_graph.h"
#include "tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/core/platform/logging.h"

namespace xla {
namespace poplarplugin {

/**
 * Generate a map from instruction to a set of buffers that exist at the
 * instruction output.
 */
static absl::flat_hash_map<HloInstruction*,
                           absl::flat_hash_set<HloPoplarBuffer*>>
FindOutputBuffers(HloModule* module,
                  const HloPoplarDataflowAnalysis& dataflow) {
  absl::flat_hash_map<HloInstruction*, absl::flat_hash_set<HloPoplarBuffer*>>
      instr_to_buffers;

  for (auto* comp : module->computations()) {
    for (auto* instr : comp->instructions()) {
      auto& buffers = instr_to_buffers[instr];
      for (auto& buffer_set :
           dataflow.GetInstructionBufferSet(instr).GetBufferSets()) {
        for (auto* buffer : buffer_set.second.buffers()) {
          buffers.insert(buffer);
        }
      }
    }
  }

  return instr_to_buffers;
}

/**
 * The `xla::FlattenCallGraph` creates clones of computations that are called
 * multiple times. This makes these computations not-unique. This function finds
 * all the rest of the unique computations.
 */
static absl::flat_hash_set<HloComputation*> FindUniqueComputations(
    HloModule* module) {
  absl::flat_hash_set<HloComputation*> result;
  absl::flat_hash_map<HloComputation*, int64_t, HloComputationHash,
                      HloComputationEquals>
      comp_counter;

  for (auto* comp : module->computations()) {
    comp_counter[comp] += 1;
  }

  for (auto& entry : comp_counter) {
    auto comp = entry.first;
    auto count = entry.second;
    if (count == 1) {
      result.insert(comp);
    }
  }

  return result;
}

/**
 * For each buffer in the graph, this class monitors the number of different
 * paths in which the buffer is currently live. You need to call
 * `BufferLivenessCounter::EnterInstruction` and
 * `BufferLivenessCounter::ExitInstruction` for each instruction as you traverse
 * the graph.
 *
 * As an example, consider the graph below in which `c_0` creates a buffer. The
 * right column shows the number of paths in which that buffer is live after
 * each instruction.
 *
 *   | Instruction                            | Live path count |
 *   | -------------------------------------- | --------------- |
 *   | c_0 = s32[3] constant({1, 2, 3})       | 3               |
 *   | s_0 = s32[1] slice(c_0), slice={[0:1]} | 3               |
 *   | s_1 = s32[1] slice(c_0), slice={[1:2]} | 3               |
 *   | s_2 = s32[1] slice(c_0), slice={[2:3]} | 3               |
 *   | a_0 = s32[1] add(s_0, s_1)             | 2               |
 *   | ROOT a_1 = s32[1] add(a_0, s_2)        | 1               |
 */
class BufferLivenessCounter {
 public:
  explicit BufferLivenessCounter(
      absl::flat_hash_map<HloInstruction*,
                          absl::flat_hash_set<HloPoplarBuffer*>>&
          output_buffers)
      : output_buffers_(output_buffers) {}

  /**
   * Update the number of live paths for all buffers that flow into the `instr`
   * instruction.
   *
   * For each producer that flows into `instr`, the corresponding live path
   * count for each buffer is decremented by one.
   */
  void EnterInstruction(HloInstruction* instr) {
    for (std::size_t i = 0; i < instr->operand_count(); i++) {
      for (auto& buffer : output_buffers_[instr->mutable_operand(i)]) {
        live_path_counts_[buffer] -= 1;
      }
    }
  }

  /**
   * Update the number of live paths for all buffers that flow out of the
   * `instr` instruction.
   *
   * For each consumer that flows out of `instr`, the corresponding live path
   * count for each buffer is incremented by one.
   */
  void ExitInstruction(HloInstruction* instr) {
    int64_t use_count = 0;
    for (auto* user : instr->users()) {
      for (std::size_t i = 0; i < user->operand_count(); i++) {
        if (user->operand(i) != instr) {
          continue;
        }
        use_count += 1;
      }
    }
    for (auto& buffer : output_buffers_[instr]) {
      // If the buffer is dead, it would have a negative count at this point.
      // The condition below prevents from accidentally making it appear "live"
      // again.
      if (live_path_counts_[buffer] >= 0) {
        live_path_counts_[buffer] += use_count;
      }
    }
  }

  bool IsBufferDead(HloPoplarBuffer* buffer) {
    return live_path_counts_[buffer] == 0;
  }

 private:
  absl::flat_hash_map<HloPoplarBuffer*, int64_t> live_path_counts_;
  absl::flat_hash_map<HloInstruction*, absl::flat_hash_set<HloPoplarBuffer*>>&
      output_buffers_;
};

class PathLivenessVisitor : public DfsHloVisitorWithDefault {
 public:
  explicit PathLivenessVisitor(HloModule* module,
                               const HloPoplarDataflowAnalysis& dataflow)
      : module_(module),
        schedule_(module->schedule()),
        liveness_counter_(output_buffers_) {
    output_buffers_ = FindOutputBuffers(module, dataflow);
    unique_comps_ = FindUniqueComputations(module);
  }

  Status Run() {
    auto* entry = module_->entry_computation();
    auto& instructions = schedule_.sequence(entry).instructions();
    TF_RETURN_IF_ERROR(entry->AcceptOrdered(this, instructions));
    return Status::OK();
  }

  Status HandleSlice(HloInstruction* instr) override {
    liveness_counter_.EnterInstruction(instr);
    auto is_comp_unique = unique_comps_.count(instr->parent());
    auto is_entry_root =
        instr == module_->entry_computation()->root_instruction();
    auto is_already_copied =
        absl::c_any_of(instr->users(), [](HloInstruction* user) {
          return user->opcode() == HloOpcode::kCopy;
        });
    auto memory_change = CalculatePotentialMemoryChange(instr);
    auto copying_reduces_memory = (is_comp_unique && !is_entry_root &&
                                   !is_already_copied && memory_change < 0);
    if (copying_reduces_memory) {
      instructions_to_copy_.push_back(instr);
      // No need to call `liveness_counter_.ExitInstruction` if the instruction
      // is going to be copied, because the copy will stop the flow of buffers.
    } else {
      liveness_counter_.ExitInstruction(instr);
    }
    return Status::OK();
  }

  Status HandleCall(HloInstruction* instr) override {
    liveness_counter_.EnterInstruction(instr);
    auto* comp = instr->to_apply();
    auto& instructions = schedule_.sequence(comp).instructions();
    TF_RETURN_IF_ERROR(comp->AcceptOrdered(this, instructions));
    liveness_counter_.ExitInstruction(instr);
    return Status::OK();
  }

  Status HandleWhile(HloInstruction* instr) override {
    liveness_counter_.EnterInstruction(instr);
    {
      auto* comp = instr->while_condition();
      auto& instructions = schedule_.sequence(comp).instructions();
      TF_RETURN_IF_ERROR(comp->AcceptOrdered(this, instructions));
    }
    {
      auto* comp = instr->while_body();
      auto& instructions = schedule_.sequence(comp).instructions();
      TF_RETURN_IF_ERROR(comp->AcceptOrdered(this, instructions));
    }
    liveness_counter_.ExitInstruction(instr);
    return Status::OK();
  }

  Status HandleConditional(HloInstruction* instr) override {
    liveness_counter_.EnterInstruction(instr);
    for (auto* comp : instr->branch_computations()) {
      auto& instructions = schedule_.sequence(comp).instructions();
      TF_RETURN_IF_ERROR(comp->AcceptOrdered(this, instructions));
    }
    liveness_counter_.ExitInstruction(instr);
    return Status::OK();
  }

  std::vector<HloInstruction*>& instructions_to_copy() {
    return instructions_to_copy_;
  }

 private:
  HloModule* module_;
  HloSchedule& schedule_;
  BufferLivenessCounter liveness_counter_;
  absl::flat_hash_map<HloInstruction*, absl::flat_hash_set<HloPoplarBuffer*>>
      output_buffers_;
  absl::flat_hash_set<HloComputation*> unique_comps_;
  std::vector<HloInstruction*> instructions_to_copy_;

  /**
   * Copying the given instruction will increase live memory based on the
   * shape of the instruction. All the buffers that become dead after this
   * instruction will reduce memory based on their shape. Considering both of
   * these, this function calculates by how much memory will change in bytes. A
   * negative value indicates a memory reduction.
   */
  int64_t CalculatePotentialMemoryChange(HloInstruction* instr) {
    int64_t memory_change = ShapeUtil::ByteSizeOf(instr->shape(), 1);
    for (auto& buffer : output_buffers_[instr]) {
      if (liveness_counter_.IsBufferDead(buffer)) {
        memory_change -= ShapeUtil::ByteSizeOf(buffer->shape(), 1);
      }
    }
    return memory_change;
  }

  Status DefaultAction(HloInstruction* instr) override {
    liveness_counter_.EnterInstruction(instr);
    liveness_counter_.ExitInstruction(instr);
    return Status::OK();
  }
};

static StatusOr<bool> CopyInstructions(
    const std::vector<HloInstruction*>& instrs_to_copy) {
  for (auto* instr : instrs_to_copy) {
    HloComputation* comp = instr->parent();
    HloInstruction* copy = comp->AddInstruction(
        HloInstruction::CreateUnary(instr->shape(), HloOpcode::kCopy, instr));
    instr->SetupDerivedInstruction(copy);
    TF_RETURN_IF_ERROR(instr->ReplaceAllUsesWith(copy));
  }
  return instrs_to_copy.size();
}

SliceCopyInserter::SliceCopyInserter(CompilerAnnotations& annotations)
    : annotations_(annotations) {}

/**
 * This pass starts by traversing the scheduled graph. It counts the number of
 * branches through which each buffer flows in and out of an instruction,
 * essentially keeping a count of the number of different paths in which each
 * buffer is live. An instruction is copied if all conditions below are met:
 * - The instruction is `HloOpcode::kSlice`.
 * - Copying the instruction would result in reduction of live memory.
 * - The parent computation for that instruction is unique (i.e. not a clone
 *   that `xla::ModuleFlatten` created).
 */
StatusOr<bool> SliceCopyInserter::Run(HloModule* module) {
  auto call_graph = CallGraph::Build(module);
  TF_RET_CHECK(call_graph->IsFlattened()) << absl::StrFormat(
      "'%s' can't be run on module '%s' which is not flattened.", name(),
      module->name());

  TF_ASSIGN_OR_RETURN(auto dataflow, HloPoplarDataflowAnalysis::Run(
                                         module, annotations_, *call_graph));

  return Run(module, *call_graph, *dataflow);
}

StatusOr<bool> SliceCopyInserter::Run(
    HloModule* module, const CallGraph& call_graph,
    const HloPoplarDataflowAnalysis& dataflow) {
  TF_RET_CHECK(module->has_schedule()) << absl::StrFormat(
      "'%s' can't be run on module '%s' which has no schedule.", name(),
      module->name());

  PathLivenessVisitor visitor{module, dataflow};
  TF_RETURN_IF_ERROR(visitor.Run());
  TF_ASSIGN_OR_RETURN(auto changed,
                      CopyInstructions(visitor.instructions_to_copy()));
  TF_RETURN_IF_ERROR(module->schedule().Update());
  return changed;
}

}  // namespace poplarplugin
}  // namespace xla
