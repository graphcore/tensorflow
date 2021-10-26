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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_VISITORS_VISITOR_BASE_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_VISITORS_VISITOR_BASE_H_

#include <list>
#include <string>
#include <vector>

#include <poplar/Program.hpp>

#include "tensorflow/compiler/plugin/poplar/driver/ops/ops_helper.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/debug_info.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/execution_counter_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/tensor_map.h"
#include "tensorflow/compiler/xla/service/dfs_hlo_visitor.h"

namespace xla {
namespace poplarplugin {

struct CompilerResources;

/*
 * The base visitor handles all operations that are element-wise.
 * This includes all explicitly element-wise ops, and also operations
 * Select, Convert, Clamp, Rng, Constant.  All of these have no element
 * to element dependencies.
 */
class BaseVisitor : public DfsHloVisitor {
 public:
  BaseVisitor(CompilerResources& resources,
              const poplar::DebugNameAndId& debug_name_and_id);

  virtual const Shape& GetOutputShape(HloInstruction*) const;

  Status HandleTupleSelect(HloInstruction* inst) override;

  Status HandleConvert(HloInstruction* inst) override;

  Status HandleBitcastConvert(HloInstruction* inst) override;

  Status HandleAllReduce(HloInstruction* crs) override;

  Status HandleConstant(HloInstruction* inst) override;

  Status HandleGetTupleElement(HloInstruction* inst) override;

  Status HandleFusion(HloInstruction* inst) override;

  Status HandleCall(HloInstruction* inst) override;

  Status HandleCustomCall(HloInstruction* inst) override;

  Status HandleTuple(HloInstruction* inst) override;

  Status HandleMap(HloInstruction* inst) override;

  Status HandleConditional(HloInstruction* inst) override;

  Status HandleInfeed(HloInstruction* inst) override;

  Status HandleAfterAll(HloInstruction* inst) override;

  Status HandleReal(HloInstruction* inst) override;

  Status HandleAllToAll(HloInstruction* hlo) override;

  Status HandleAddDependency(HloInstruction* hlo) override;

  Status Postprocess(HloInstruction* hlo) override;

  Status HandleHloOp(HloInstruction* hlo);

  Status FinishVisit(HloInstruction* root) final;

  // Called by the FinishVisit.
  virtual Status FinishScopedVisit(HloInstruction* root) {
    return Status::OK();
  }

#define HANDLE_AS_HLO_OP(Name) \
  Status Name(HloInstruction* inst) override { return HandleHloOp(inst); }

  HANDLE_AS_HLO_OP(HandleElementwiseUnary)
  HANDLE_AS_HLO_OP(HandleElementwiseBinary)
  HANDLE_AS_HLO_OP(HandleClamp)
  HANDLE_AS_HLO_OP(HandleSelect)
  HANDLE_AS_HLO_OP(HandleCompare)
  HANDLE_AS_HLO_OP(HandleRng)
  HANDLE_AS_HLO_OP(HandleReplicaId)

  /*
   * Operations not processed by this visitor.
   */
#define UNIMPLEMENTED(Name) \
  Status Name(HloInstruction* inst) override { return Unimplemented(inst); };

  UNIMPLEMENTED(HandleSlice)
  UNIMPLEMENTED(HandleDynamicSlice)
  UNIMPLEMENTED(HandleDynamicUpdateSlice)
  UNIMPLEMENTED(HandleSelectAndScatter)
  UNIMPLEMENTED(HandleWhile)
  UNIMPLEMENTED(HandlePad)
  UNIMPLEMENTED(HandleReverse)
  UNIMPLEMENTED(HandleSort)
  UNIMPLEMENTED(HandleReduce)
  UNIMPLEMENTED(HandleBitcast)
  UNIMPLEMENTED(HandleBroadcast)
  UNIMPLEMENTED(HandleReshape)
  UNIMPLEMENTED(HandleTranspose)
  UNIMPLEMENTED(HandleReducePrecision)
  UNIMPLEMENTED(HandleOutfeed)
  UNIMPLEMENTED(HandleSend)
  UNIMPLEMENTED(HandleSendDone)
  UNIMPLEMENTED(HandleRecv)
  UNIMPLEMENTED(HandleRecvDone)
  UNIMPLEMENTED(HandleBatchNormInference)
  UNIMPLEMENTED(HandleBatchNormTraining)
  UNIMPLEMENTED(HandleBatchNormGrad)
  UNIMPLEMENTED(HandleFft)
  UNIMPLEMENTED(HandleGather)
  UNIMPLEMENTED(HandleCopy)
  UNIMPLEMENTED(HandleIota)
  UNIMPLEMENTED(HandleScatter)
  UNIMPLEMENTED(HandleCollectivePermute)
  UNIMPLEMENTED(HandleConcatenate)
  UNIMPLEMENTED(HandleGetDimensionSize)
  UNIMPLEMENTED(HandleTriangularSolve)
  UNIMPLEMENTED(HandleCholesky)
  UNIMPLEMENTED(HandlePartitionId)
  UNIMPLEMENTED(HandleRngGetAndUpdateState)
  UNIMPLEMENTED(HandleCopyStart)
  UNIMPLEMENTED(HandleCopyDone)
  UNIMPLEMENTED(HandleDot)
  UNIMPLEMENTED(HandleConvolution)
  UNIMPLEMENTED(HandleReduceWindow)

  Status Preprocess(HloInstruction* hlo) override;

  // Add the sequence produced for the given instruction. Note that a
  // poplar::program::Program can be passed directly since it is implicitly
  // convertible to a poplar::program::Sequence.
  virtual Status AddSequenceForInstruction(
      const HloInstruction* inst, const poplar::program::Sequence& seq);

  // Add the sequence produced for the given instruction. This differs from
  // the function above by grouping all the sequences coming from the same
  // instruction together. In other words, it allows for reordering of the
  // sequences by appending to the grouped sequence for the given instruction,
  // rather than always appending to the end. Note that only the sequences added
  // using this function are grouped; if a sequence was already added for the
  // same instruction with the above function, that sequence will be left
  // ungrouped.
  virtual Status AppendSequenceGroupedByInstruction(
      const HloInstruction* inst, const poplar::program::Sequence& seq);

  // Same as above, however the sequence is prepended rather than appended to
  // the given instruction sequence.
  virtual Status PrependSequenceGroupedByInstruction(
      const HloInstruction* inst, const poplar::program::Sequence& seq);

  // Get the sequence generated by this visitor. If `copy_execution_counters` is
  // set to true, then prepend the sequence with copies for populating the
  // execution counters with the values from the outer scope.
  virtual poplar::program::Sequence GetSequence(
      bool copy_execution_counters = true);

  // Return the execution counters for the sequence built by this visitor.
  ExecutionCounters& GetExecutionCounters();

  // Get a copy of the sequence without any execution counter logic.
  poplar::program::Sequence GetRawSequence() const;

 protected:
  Status Unimplemented(HloInstruction* inst);

  poplar::DebugNameAndId GetDebugNameAndId(const HloInstruction* inst) const;

  CompilerResources& resources_;

  TensorMap tensor_map;

  bool has_infeed_ = false;

  const poplar::DebugNameAndId dnai_;

  // Scope execution counters.
  ExecutionCounters execution_counters_;

  // Control whether changing seeds is allowed during instruction lowering,
  // used to improve prng stability.
  bool allow_seed_changes_ = false;

 private:
  Status CreateSequenceGroupedByInstruction(
      const HloInstruction* inst, const poplar::program::Sequence& seq);

  std::vector<std::list<poplar::program::Sequence>> sequences_;

  // The index of a grouped sequence for an instruction in the above vector.
  ConstHloInstructionMap<std::size_t> grouped_sequence_indices_;
};

}  // namespace poplarplugin
}  // namespace xla

#endif
