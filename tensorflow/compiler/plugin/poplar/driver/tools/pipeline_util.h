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

#include <map>
#include <set>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/inlined_vector.h"
#include "tensorflow/compiler/plugin/poplar/driver/backend_config.pb.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"
#include "tensorflow/compiler/xla/service/hlo_value.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"

namespace xla {

class HloComputation;
class HloModule;
class CallGraph;

namespace poplarplugin {
class PipelineDataflowAnalysis;

// Returns whether the instruction is a PipelineStage or a
// PipelineStageBackward.
bool IsPipelineStageOrBackwardOp(const HloInstruction* inst);

// Returns whether the instruction is a PipelineStage op of any kind.
bool IsAnyPipelineStageOp(const HloInstruction* inst);

// Returns whether the instruction is a PipelineStage op of any kind or the
// ResourceUpdate.
bool IsAnyPipelineStageOpOrResourceUpdate(const HloInstruction* inst);

// Helper function for the PipelineDataflowAnalysis. Is used to identify whether
// an instruction is allowed to produce outputs in a PipelineOp.
bool IsProducerOp(const HloInstruction* inst);

// Returns true if there is guarantee that the given input instruction to a
// pipeline stage will not be modified inplace.
bool IsPipelineStageReadOnlyInput(const HloInstruction* inst);

// Helper struct for holding onto forward, backward and recomputation (if any)
// PipelineStages.
// Note that we might not have a recomputation for every pipeline stage
// therefore need to use a map.
struct PipelineStages {
  std::vector<HloInstruction*> forward;
  std::vector<HloInstruction*> backward;
  absl::flat_hash_map<int64, HloInstruction*> recomputation;
  absl::optional<HloInstruction*> resource_update;
};

// Helper struct for storing stages in order from forward to backward.
class OrderedPipelineStages {
 public:
  OrderedPipelineStages(const PipelineStages& stages,
                        bool include_resource_update);
  int64 GetNumberOfStages() const;
  HloInstruction* GetStage(int64 index) const;
  int64 GetIndex(HloInstruction* stage) const;
  void UpdateStage(int64 index, HloInstruction* stage);

 private:
  absl::flat_hash_map<int64, HloInstruction*> id_to_stage;
  absl::flat_hash_map<HloInstruction*, int64> stage_to_id;
};

// Get all the pipelines in the module.
StatusOr<absl::InlinedVector<HloInstruction*, 1>> GetPipelines(
    const HloModule* module);

// Get the forward and backward pipeline stages from the pipeline_computation.
StatusOr<PipelineStages> GetPipelineStages(HloComputation* pipeline_computation,
                                           bool validate_stages = true);

// Get all the computations called by the pipeline stage or which are reachable
// from it. Ignores computations which are called in the Parallel context.
StatusOr<absl::flat_hash_set<HloComputation*>> GetAllComputationsCalledBy(
    HloInstruction* pipeline_stage, const CallGraph* call_graph);

// Convert an instruction which has a tuple shape such that all the users of
// that instruction are GetTupleElement instructions.
StatusOr<HloInstruction*> ConvertAllUsersToGTEs(HloInstruction* const inst);

// Make sure that the root instruction of the computation is a Tuple
// instruction.
StatusOr<bool> FixRootInstruction(HloComputation* comp);

// Makes sure that the root instruction of each stage is a Tuple instruction
// (not just tuple shaped).
Status FixRootInstructions(const PipelineStages& pipeline_stages);

// Verifies that Pipeline stages are suitable for fixing.
// This means that we expect the Pipeline to not have been modified and so
// the root instruction for all the stages is a tuple and all the users of
// stages are GTEs.
Status VerifyPipelineStagesBeforeFixing(const PipelineStages& pipeline_stages);

// Verifies that the data flow in the Pipeline is legal and that all
// instructions have been lowered. Ignores sharding.
Status VerifyPipelineAfterFixing(HloInstruction* pipeline_op);

// A function which makes sure that every user of a pipeline stage is a GTE by
// inserting GTEs and tuples into the graph.
StatusOr<bool> InsertGTEEdges(PipelineStages& pipeline_stages);

// A function which makes sure there are unique output edges for each output of
// a pipeline stage.
// This makes the analysis easier as we ever only need to consider a single use
// for each edge.
// Returns true if edges were added.
// Note that CSE will tidy this up later.
StatusOr<bool> DuplicateGTEEdges(PipelineStages& pipeline_stages);

// Make sure each PipelineStage has a unique HloComputation.
// Returns true is a new HloComputation has been added.
StatusOr<bool> UniquifyPipelineStageCallsites(PipelineStages& pipeline_stages);

// Create an empty pipeline stage inside of the pipeline computation
StatusOr<HloInstruction*> CreatePipelineStage(
    HloComputation* pipeline, const std::vector<HloInstruction*> operands,
    HloComputation* stage_comp, PoplarBackendConfig_CallConfig_Type stage_type,
    int64 stage_id, const std::string& name);

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
    HloInstructionSet forced_parameters = {},
    bool replace_resource_update_uses = true);

// Inlines the provided computation and replaces the output at caller site with
// the inlined root instruction.
StatusOr<absl::flat_hash_map<HloInstruction*, HloInstruction*>>
InlineComputation(HloInstruction* caller, HloComputation* comp_to_inline,
                  bool copy_sharding = false);

// Get a schedule from a pipeline.
StatusOr<PoplarBackendConfig::CallConfig::PipelineConfig::Schedule>
GetPipelineSchedule(const HloInstruction* pipeline_op);

StatusOr<PoplarBackendConfig::CallConfig::PipelineConfig::RecomputationMode>
GetPipelineRecomputationMode(const HloInstruction* pipeline_op);

// Compute the fifo depth multiplier for the given schedule of a pipeline
// operation.
StatusOr<int> GetFifoDepthMultiplier(const HloInstruction* pipeline_op);

// Helper struct for identifying pipeline stages.
enum class StageType {
  kForward,
  kBackward,
  kRecomputation,
};

struct StageID {
  StageID(StageType stage_type, int64 id) : stage_type(stage_type), id(id) {}

  bool operator==(const StageID& other) const {
    return stage_type == other.stage_type && id == other.id;
  }

  bool operator!=(const StageID& other) const { return !operator==(other); }

  std::string ToString() const;

  StageType stage_type;
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
      bool allow_duplicate_gte_edges = false,
      bool allow_communication_ops = false, bool allow_feeds = false,
      bool allow_recomputation = false,
      bool allow_communication_optimizations = false, bool use_io_tiles = true);

  explicit PipelineDataflowAnalysis(const PipelineStages& pipeline_stages,
                                    bool allow_duplicate_gte_edges,
                                    bool allow_communication_ops,
                                    bool allow_feeds, bool allow_recomputation,
                                    bool allow_communication_optimizations,
                                    bool use_io_tiles);

  // Returns whether the instruction needs to be lowered into a stage given the
  // current analysis.
  StatusOr<bool> HasToBeLoweredIntoStage(const HloInstruction* stage,
                                         const HloInstruction* inst) const;

  // Returns whether the instruction needs to be lowered given the current
  // analysis.
  StatusOr<bool> HasToBeLowered(const HloInstruction* inst) const;

  // Given the PipelineStage(Backward) instruction, get whether is is a FWD or
  // BWD stage and it's ID.
  StatusOr<StageID> GetStageID(const HloInstruction* inst) const;

  // Given the PipelineStage(Backward) instruction, get the
  // PipelineStage(Backward) which will be executed next.
  StatusOr<StageID> GetPreviousStageID(const HloInstruction* inst) const;

  // Get the sharding device a pipeline stage resides on.
  StatusOr<int64> GetShardForStage(const StageID& stage_id) const;

  // Verifies that the dataflow between Pipeline Stages is legal.
  Status VerifyPipelineUsage(const HloInstruction* pipeline_stage,
                             const HloInstruction* pipeline_stage_user) const;

  // Verifies that the parameter is only used by stages (fwd, recomp and/or
  // bwd) on the same shard.
  Status VerifyParameterUsage(const HloInstruction* parameter,
                              const HloInstruction* pipeline_stage_user);

  // Verifies that the gradient accumulator creator is only used by backward
  // pipeline stages on the same shard.
  Status VerifyGradientAccumulatorCreateUsage(
      const HloInstruction* gradient_accumulator_creator,
      const HloInstruction* pipeline_stage_user);

  // Verifies that the execution counter is only used by stages on the same
  // shard.
  Status VerifyExecutionCounterUsage(const HloInstruction* execution_counter,
                                     const HloInstruction* pipeline_stage_user);

  // Verifies that the infeed is only used by one stage (fwd and possibly
  // recomp).
  Status VerifyInfeedUsage(const HloInstruction* infeed,
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
  HloValueSet GetOperandsValueSet(const HloInstruction* inst) const;

 private:
  // Updates the analysis
  Status UpdateThroughInstruction(HloInstruction* inst);

  // Create value and maintain it internally.
  HloValue* CreateValue(HloInstruction* inst);

  // Internal storage of values.
  // Using std::map as absl maps lack pointer stability.
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
  absl::flat_hash_map<HloInstruction*, int64> recomputation_stages_lookup_;

  bool allow_duplicate_gte_edges_;
  bool allow_communication_ops_;
  bool allow_feeds_;
  bool allow_recomputation_;
  bool allow_communication_optimizations_;
  bool use_io_tiles_;
};

// A helper class used to represent a tensor being passed through pipeline
// stages.
class PipelinePath {
 public:
  // Type used to describe the path.
  enum class Type {
    // A path is between two backward stages on the same shard.
    kBackward,
    // A path is between two forward stages on the same shard.
    kForward,
    // A path is between a forward and a backward stage - same pipeline stage
    // id.
    kForwardToBackward,
    // A path between any two stages.
    kAny,
  };

  PipelinePath(
      HloInstruction* new_consumer, uint64 stage_idx, uint64 input_idx,
      uint64 output_idx,
      PoplarBackendConfig::CallConfig::PipelineConfig::Schedule schedule);
  bool FinishPath(PipelineStages& stages);
  std::vector<uint64>& GetVisitedStages();
  std::vector<uint64>& GetInputsPath();
  std::vector<uint64>& GetOutputsPath();
  StatusOr<int64> GetFifoDepth();
  // The pipeline stage which should now be consuming the value.
  HloInstruction* GetNewConsumerStage() const;
  // The old pipeline stage which is currently consuming the value.
  HloInstruction* GetOldConsumerStage() const;
  Type GetType() const;

 private:
  // The fields below are populated by the FinishPath function.
  bool finished_ = false;
  int64 fifo_depth_ = -1;
  bool fifo_between_fwd_and_bwd_ = false;
  HloInstruction* old_consumer_ = nullptr;
  Type type_;

  std::vector<uint64> visited_stages_;
  std::vector<uint64> inputs_path_;
  std::vector<uint64> outputs_path_;
  HloInstruction* new_consumer_;
  const PoplarBackendConfig::CallConfig::PipelineConfig::Schedule schedule_;
};

// A function used to find pipeline paths for operations which are passed
// through multiple stages.
StatusOr<std::vector<PipelinePath>> FindPassthroughPipelinePaths(
    PipelineStages& stages,
    PoplarBackendConfig::CallConfig::PipelineConfig::Schedule schedule);

}  // namespace poplarplugin
}  // namespace xla

#endif
