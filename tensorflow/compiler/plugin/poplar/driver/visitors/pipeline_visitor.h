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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_VISITORS_PIPELINE_VISITOR_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_VISITORS_PIPELINE_VISITOR_H_

#include <memory>
#include <string>

#include "tensorflow/compiler/plugin/poplar/driver/ops/ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/visitors/deferred_visitor.h"
#include "tensorflow/compiler/plugin/poplar/driver/visitors/pipeline_stage_visitor.h"

#define HLO_PIPELINE_VISITOR_NOT_IMPLEMENTED(X) \
  Status X(HloInstruction* hlo) override { return HandleNotImplemented(hlo); }

namespace xla {
namespace poplarplugin {

namespace pipelinevisitorutils {
struct PipelineSchedulerUtil;
}

struct CompilerResources;

class PipelineVisitor : public InplaceDeferredVisitor {
 public:
  PipelineVisitor(
      PoplarBackendConfig::CallConfig::PipelineConfig::Schedule schedule,
      int64 stage_count, const std::vector<int>& stage_ipu_mapping,
      const absl::flat_hash_map<const HloInstruction*, int>& inst_stage_mapping,
      const absl::flat_hash_set<int> stages_with_recomputation,
      int64 num_backward_stages, CompilerResources& res,
      const DeferredArgRBVectors& inputs,
      const HloInstructionDescription& description,
      const poplar::DebugNameAndId& debug_name_and_id);

  PipelineVisitor(const HloInstruction* pipeline, CompilerResources& res,
                  const DeferredArgRBVectors& inputs,
                  const HloInstructionDescription& description,
                  const poplar::DebugNameAndId& debug_name_and_id);

  ~PipelineVisitor();

  HLO_PIPELINE_VISITOR_NOT_IMPLEMENTED(HandleClamp);
  HLO_PIPELINE_VISITOR_NOT_IMPLEMENTED(HandleSelect);
  HLO_PIPELINE_VISITOR_NOT_IMPLEMENTED(HandleTupleSelect);
  HLO_PIPELINE_VISITOR_NOT_IMPLEMENTED(HandleConcatenate);
  HLO_PIPELINE_VISITOR_NOT_IMPLEMENTED(HandleDot);
  HLO_PIPELINE_VISITOR_NOT_IMPLEMENTED(HandleConvolution);
  HLO_PIPELINE_VISITOR_NOT_IMPLEMENTED(HandleFft);
  HLO_PIPELINE_VISITOR_NOT_IMPLEMENTED(HandleTriangularSolve);
  HLO_PIPELINE_VISITOR_NOT_IMPLEMENTED(HandleCholesky);
  HLO_PIPELINE_VISITOR_NOT_IMPLEMENTED(HandleAllReduce);
  HLO_PIPELINE_VISITOR_NOT_IMPLEMENTED(HandleAllToAll);
  HLO_PIPELINE_VISITOR_NOT_IMPLEMENTED(HandleCollectivePermute);
  HLO_PIPELINE_VISITOR_NOT_IMPLEMENTED(HandleReplicaId);
  HLO_PIPELINE_VISITOR_NOT_IMPLEMENTED(HandlePartitionId);
  HLO_PIPELINE_VISITOR_NOT_IMPLEMENTED(HandleGetDimensionSize);
  HLO_PIPELINE_VISITOR_NOT_IMPLEMENTED(HandleRng);
  HLO_PIPELINE_VISITOR_NOT_IMPLEMENTED(HandleRngGetAndUpdateState);
  HLO_PIPELINE_VISITOR_NOT_IMPLEMENTED(HandleReverse);
  HLO_PIPELINE_VISITOR_NOT_IMPLEMENTED(HandleSort);
  HLO_PIPELINE_VISITOR_NOT_IMPLEMENTED(HandleIota);
  HLO_PIPELINE_VISITOR_NOT_IMPLEMENTED(HandleReduce);
  HLO_PIPELINE_VISITOR_NOT_IMPLEMENTED(HandleBitcast);
  HLO_PIPELINE_VISITOR_NOT_IMPLEMENTED(HandleBroadcast);
  HLO_PIPELINE_VISITOR_NOT_IMPLEMENTED(HandleReshape);
  HLO_PIPELINE_VISITOR_NOT_IMPLEMENTED(HandleTranspose);
  HLO_PIPELINE_VISITOR_NOT_IMPLEMENTED(HandleFusion);
  HLO_PIPELINE_VISITOR_NOT_IMPLEMENTED(HandleSlice);
  HLO_PIPELINE_VISITOR_NOT_IMPLEMENTED(HandleDynamicSlice);
  HLO_PIPELINE_VISITOR_NOT_IMPLEMENTED(HandleDynamicUpdateSlice);
  HLO_PIPELINE_VISITOR_NOT_IMPLEMENTED(HandleMap);
  HLO_PIPELINE_VISITOR_NOT_IMPLEMENTED(HandleReduceWindow);
  HLO_PIPELINE_VISITOR_NOT_IMPLEMENTED(HandleSelectAndScatter);
  HLO_PIPELINE_VISITOR_NOT_IMPLEMENTED(HandleGather);
  HLO_PIPELINE_VISITOR_NOT_IMPLEMENTED(HandleScatter);
  HLO_PIPELINE_VISITOR_NOT_IMPLEMENTED(HandlePad);
  HLO_PIPELINE_VISITOR_NOT_IMPLEMENTED(HandleCopyStart);
  HLO_PIPELINE_VISITOR_NOT_IMPLEMENTED(HandleCopyDone);
  HLO_PIPELINE_VISITOR_NOT_IMPLEMENTED(HandleSend);
  HLO_PIPELINE_VISITOR_NOT_IMPLEMENTED(HandleSendDone);
  HLO_PIPELINE_VISITOR_NOT_IMPLEMENTED(HandleRecv);
  HLO_PIPELINE_VISITOR_NOT_IMPLEMENTED(HandleRecvDone);
  HLO_PIPELINE_VISITOR_NOT_IMPLEMENTED(HandleBatchNormTraining);
  HLO_PIPELINE_VISITOR_NOT_IMPLEMENTED(HandleBatchNormInference);
  HLO_PIPELINE_VISITOR_NOT_IMPLEMENTED(HandleBatchNormGrad);
  HLO_PIPELINE_VISITOR_NOT_IMPLEMENTED(HandleAddDependency);
  HLO_PIPELINE_VISITOR_NOT_IMPLEMENTED(HandleConstant);

  Status HandleCopy(HloInstruction* hlo) override;
  Status HandleOutfeed(HloInstruction* hlo) override;

  virtual Status HandleExecutionCounter(HloInstruction* hlo);
  virtual Status HandleFifo(HloInstruction* hlo);
  virtual Status HandleInterIpuCopy(HloInstruction* hlo);
  virtual Status HandleGradientAccumulatorSink(HloInstruction* hlo);
  virtual Status HandleInterTilesetCopy(HloInstruction* hlo);

  virtual StatusOr<poplar::program::Sequence> VerifyPipelineArguments(
      const HloInstruction* accumulation_count,
      poplar::Tensor accumulation_count_tensor, poplar::Graph& graph) const;

  StatusOr<poplar::program::Sequence> GetPipelineSequence(
      int64 iterations) const;

 protected:
  Status AddSequenceForInstruction(
      const HloInstruction* inst,
      const poplar::program::Sequence& seq) override;

  Status AppendSequenceGroupedByInstruction(
      const HloInstruction* inst,
      const poplar::program::Sequence& seq) override;

  Status PrependSequenceGroupedByInstruction(
      const HloInstruction* inst,
      const poplar::program::Sequence& seq) override;

  void AddSequenceForAliasingCopy(
      const HloInstruction* inst,
      const poplar::program::Sequence& seq) override;

  Status HandleDeferredAllocationCall(HloInstruction* inst) override;
  Status HandleNonDeferredCustomCall(HloInstruction* hlo) override;
  Status HandleDeferredAllocationTuple(HloInstruction* inst) override;
  Status HandleDeferredAllocationWhile(HloInstruction* inst) override;
  Status FinishDeferedAllocationVisit(HloInstruction* inst) override;

  std::unique_ptr<pipelinevisitorutils::PipelineSchedulerUtil>
      pipeline_scheduler_util_;
  std::vector<poplar::program::Sequence> copy_sequences_;
  std::vector<poplar::program::Sequence> inter_ipu_copy_sequences_;
  std::vector<poplar::program::Sequence> fifo_sequences_;
  std::vector<poplar::program::Sequence> infeed_sequences_;
  std::vector<poplar::program::Sequence> outfeed_sequences_;
  std::vector<poplar::program::Sequence> program_sequences_;
  std::vector<poplar::program::Sequence> recomputation_sequences_;
  std::vector<poplar::program::Sequence> inter_tileset_copy_in_sequences_;
  std::vector<poplar::program::Sequence> inter_tileset_copy_out_sequences_;
  poplar::program::Sequence resource_update_;

  // Sequence which sets the initial values for all the execution counters.
  poplar::program::Sequence pipeline_execution_counters_initialize_sequence_;

  // Sequence which zeros pipeline specific tensors before the pipeline is
  // executed.
  poplar::program::Sequence pipeline_tensors_zeroing_sequence_;

  // Sequence which write undefs pipeline specific tensors which are not fully
  // written to before the pipeline is executed.
  poplar::program::Sequence pipeline_write_undef_sequence_;

  std::vector<int> stage_ipu_mapping_;
  absl::flat_hash_map<const HloInstruction*, int> inst_stage_mapping_;
  absl::flat_hash_set<int> stages_with_recomputation_;
  const int64 num_backward_stages_;
  absl::flat_hash_map<int, std::unique_ptr<PipelineStageVisitor>>
      fwd_stage_visitors_;

  struct RepeatBlock {
    poplar::program::Program program;
    int64 iterations;
  };

  virtual RepeatBlock GetPipelineRampUpSequence(
      const poplar::DebugNameAndId& debug_name_and_id) const = 0;
  virtual RepeatBlock GetPipelineRampDownSequence(
      const poplar::DebugNameAndId& debug_name_and_id,
      int additional_iterations = 0) const = 0;
  virtual RepeatBlock GetPipelineRepeatBlockSequence(
      const poplar::DebugNameAndId& debug_name_and_id,
      int64 iterations) const = 0;

  // Function which indicates whether stage outputs should be copied.
  virtual bool StageOutputsRequireCopies() const = 0;

  Status HandleNotImplemented(HloInstruction* hlo);

  // Creator for PipelineStage(Backward).
  StatusOr<poplar::program::Sequence> CreatePipelineStageOp(
      const HloInstruction* inst,
      const poplar::DebugNameAndId& debug_name_and_id);

  // Creator for PipelineStageRecomputation.
  StatusOr<poplar::program::Sequence> CreatePipelineStageRecomputationOp(
      const HloInstruction* inst,
      const poplar::DebugNameAndId& debug_name_and_id);
};

#undef HLO_PIPELINE_VISITOR_NOT_IMPLEMENTED

class ParallelPipelineVisitor : public PipelineVisitor {
 public:
  using PipelineVisitor::PipelineVisitor;

  static std::unique_ptr<PipelineVisitor> Create(
      const HloInstruction* pipeline, CompilerResources& res,
      const DeferredArgRBVectors& inputs,
      const HloInstructionDescription& description,
      const poplar::DebugNameAndId& debug_name_and_id);

 protected:
  RepeatBlock GetPipelineRampUpSequence(
      const poplar::DebugNameAndId& debug_name_and_id) const override;
  RepeatBlock GetPipelineRampDownSequence(
      const poplar::DebugNameAndId& debug_name_and_id,
      int additional_iterations = 0) const override;
  RepeatBlock GetPipelineRepeatBlockSequence(
      const poplar::DebugNameAndId& debug_name_and_id,
      int64 iterations) const override;

  bool StageOutputsRequireCopies() const override { return true; }
};

class SequentialPipelineVisitor : public PipelineVisitor {
 public:
  using PipelineVisitor::PipelineVisitor;

  Status HandleFifo(HloInstruction* hlo) override;

  static std::unique_ptr<PipelineVisitor> Create(
      const HloInstruction* pipeline, CompilerResources& res,
      const DeferredArgRBVectors& inputs,
      const HloInstructionDescription& description,
      const poplar::DebugNameAndId& debug_name_and_id);

 protected:
  RepeatBlock GetPipelineRampUpSequence(
      const poplar::DebugNameAndId& debug_name_and_id) const override;
  RepeatBlock GetPipelineRampDownSequence(
      const poplar::DebugNameAndId& debug_name_and_id,
      int additional_iterations = 0) const override;
  RepeatBlock GetPipelineRepeatBlockSequence(
      const poplar::DebugNameAndId& debug_name_and_id,
      int64 iterations) const override;

  bool StageOutputsRequireCopies() const override { return false; }
};

}  // namespace poplarplugin
}  // namespace xla

#endif
