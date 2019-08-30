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
#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_PIPELINE_UTIL_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_TOOLS_PIPELINE_UTIL_H_

#include "tensorflow/compiler/xla/service/hlo_value.h"

#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"

#include "absl/container/flat_hash_map.h"

#include <map>

namespace xla {

class HloInstruction;
class HloComputation;

namespace poplarplugin {
class PipelineDataflowAnalysis;

// Returns whether the instruction is a PipelineStage or a
// PipelineStageBackward.
bool IsPiplineStageOrBackwardOp(const HloInstruction* inst);

// Helper function for the PipelineDataflowAnalysis. Is used to identify whether
// an instruction is allowed to produce outputs in a PipelineOp.
bool IsProducerOp(const HloInstruction* inst);

// Helper struct for holding onto both forward and backward (if any)
// PipelineStages.
struct PipelineStages {
  std::vector<HloInstruction*> forward;
  std::vector<HloInstruction*> backward;
};

// Get the forward and backward pipeline stages from the pipeline_computation.
StatusOr<PipelineStages> GetPipelineStages(
    HloComputation* pipeline_computation);

// Verifies that Pipeline stages are suitable for lowering.
// This means that we expect the Pipeline to not have been modified and so
// the root instruction for all the stages is a tuple and all the users of
// stages are GTEs.
Status VerifyPipelineStagesBeforeLowering(
    const PipelineStages& pipeline_stages);

// Verifies that the Pipeline is legal and is suitable for lowering to Poplar.
// We make sure that the data flow is legal and that all instructions have been
// lowered.
Status VerifyPipelineStagesAfterLowering(HloInstruction* pipeline_op);

// A function which makes sure there are unique output edges for each output of
// a pipeline stage.
// This makes the analysis easier as we ever only need to consider a single use
// for each edge.
// Returns true if edges were added.
// Note that CSE will tidy this up later.
StatusOr<bool> DuplicateGTEEdges(PipelineStages& pipeline_stages);

// Make sure each PiplineStage has a unique HloComputation.
// Returns true is a new HloComputation has been added.
StatusOr<bool> UniquifyPipelineStageCallsites(PipelineStages& pipeline_stages);

// Add the instruction in ordered_lowering to the PipelineStage stage  Note that
// the instructions in ordered_lowering are sorted in post order. Optionally
// takes a map from a parameter index to an instruction which is being lowered
// which means that the lowered instruction will be used rather than the
// parameter (it does not remove the parameter).
// We can also force parameters to be added as inputs, which are threaded
// through the pipeline stage, and then any users of those inputs are replaced
// with GTEs on the output.
// This function also keeps track  of any uses of the instructions which are
// being lowered and replaces those uses with GTEs to the new output of this
// stage.
StatusOr<HloInstruction*> AddInstructionsToPipelineStage(
    HloInstruction* stage,
    const std::vector<HloInstruction*>& ordered_lowering = {},
    std::map<int64, HloInstruction*>
        replace_parameter_with_lowered_instruction = {},
    absl::flat_hash_set<HloInstruction*> forced_parameters = {});

// Get output tuple indices for unused stage outputs.
StatusOr<std::set<int64>> GetUnusedPipelineStageOutputIndices(
    const HloInstruction* stage);

// Get parameter numbers for parameter instructions in the stage which have no
// users.
StatusOr<std::set<int64>> GetUnusedParametersInPipelineStage(
    const HloInstruction* stage);

// Get tuple indices for stage outputs which are used in multiple places.
// Returns a map from the tuple index of first occurrence to a set of all other
// occurrences.
StatusOr<std::map<int64, std::set<int64>>> GetDuplicatePipelineStageOutputs(
    const HloInstruction* stage);

// Get tuple indices for stage operands which are used in multiple places.
// Returns a map from the tuple index of first occurrence to a set of all other
// occurrences.
StatusOr<std::map<int64, std::set<int64>>> GetDuplicatePipelineStageInputs(
    const HloInstruction* stage);

// Removes parameters from the stage, and any operands which now have no users.
StatusOr<HloInstruction*> RemoveParametersFromStage(
    HloInstruction* stage, const std::set<int64>& parameters_to_remove);

// Removes outputs from the stage, and GTEs which are not used by anything.
Status RemoveOutputsFromStage(HloInstruction* stage,
                              const std::set<int64>& outputs_to_remove);

// Helper struct for identifying pipeline stages.
struct StageID {
  StageID(bool is_forward, int64 id) : is_forward(is_forward), id(id) {}

  bool operator==(const StageID& other) const {
    return is_forward == other.is_forward && id == other.id;
  }

  std::string ToString() const;

  bool is_forward;
  int64 id;
};

std::ostream& operator<<(std::ostream& stream, const StageID& stage_id);

// Simple dataflow analysis for instructions outside of pipeline stages.
// This analysis is simple and could be improved in the future to handle
// tuples/sub buffers - however current pipelining API should not produce such
// HLO graphs.
class PipelineDataflowAnalysis {
 public:
  static StatusOr<std::unique_ptr<PipelineDataflowAnalysis>> GetAnalysis(
      const PipelineStages& pipeline_stages,
      bool allow_duplicate_gte_edges = false);

  explicit PipelineDataflowAnalysis(const PipelineStages& pipeline_stages,
                                    bool allow_duplicate_gte_edges);

  // Returns whether the instruction needs to be lowered given the current
  // analysis.
  StatusOr<bool> HasToBeLowered(const HloInstruction* inst) const;

  // Given the PipelineStage(Backward) instruction, get whether is is a FWD or
  // BWD stage and it's ID.
  StatusOr<StageID> GetStageID(const HloInstruction* inst) const;

  // Given the PipelineStage(Backward) instruction, get the
  // PipelineStage(Backward) which will be executed next.
  StatusOr<StageID> GetPreviousStageID(const HloInstruction* inst) const;

  // Verifies that the dataflow between Pipeline Stages is legal.
  Status VerifyPipelineUsage(const HloInstruction* pipeline_stage,
                             const HloInstruction* pipeline_stage_user) const;

  // Verifies that the parameter is only used the one stage (fwd and/or bwd).
  Status VerifyParameterUsage(const HloInstruction* parameter,
                              const HloInstruction* pipeline_stage_user);

  // Verifies that the inputs to the Pipeline Stage are allowed.
  Status VerifyPipelineStageOperands(
      const HloInstruction* pipeline_stage,
      const HloValueSet& new_inputs = HloValueSet());

  // Wrapper function for getting the value set.
  HloValueSet* GetMutableValueSet(HloInstruction* inst);

  // Wrapper function for getting the value set.
  const HloValueSet& GetValueSet(const HloInstruction* inst) const;

  // Wrapper function for getting the value set.
  HloValueSet* CreateValueSet(HloInstruction* inst);

  // Get the value set which is the union of all operands of inst.
  HloValueSet GetOperandsValueSet(const HloInstruction* inst);

 private:
  // Updates the analysis
  Status UpdateThroughInstruction(HloInstruction* inst);

  // Create value and maintain it internally.
  HloValue* CreateValue(HloInstruction* inst);

  // Internal storage of values.
  std::map<HloValue::Id, HloValue> values_;

  // Value sets for each instruction.
  absl::flat_hash_map<HloInstruction*, InstructionValueSet> inst_to_value_set_;
  // A map used to indicate in what stages it is used.
  absl::flat_hash_map<const HloValue*, absl::flat_hash_set<HloInstruction*>>
      used_by_stages_;

  // Next buffer id.
  HloValue::Id next_value_id_ = 0;

  const PipelineStages pipeline_stages_;
  // Hash maps to speed up lookup.
  absl::flat_hash_map<HloInstruction*, int64> fwd_stages_lookup_;
  absl::flat_hash_map<HloInstruction*, int64> bwd_stages_lookup_;

  bool allow_duplicate_gte_edges_;
};
}  // namespace poplarplugin
}  // namespace xla

#endif
