/* Copyright 2017 Graphcore Ltd
 */

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

#include <poplar/Program.hpp>

#include "tensorflow/compiler/plugin/poplar/driver/ops/ops_helper.h"
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
  BaseVisitor(CompilerResources&);

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

  Status HandleHloOp(HloInstruction* hlo);

#define HANDLE_AS_HLO_OP(Name) \
  Status Name(HloInstruction* inst) override { return HandleHloOp(inst); }

  HANDLE_AS_HLO_OP(HandleElementwiseUnary)
  HANDLE_AS_HLO_OP(HandleElementwiseBinary)
  HANDLE_AS_HLO_OP(HandleClamp)
  HANDLE_AS_HLO_OP(HandleSelect)
  HANDLE_AS_HLO_OP(HandleCompare)
  HANDLE_AS_HLO_OP(HandleRng)

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
  UNIMPLEMENTED(HandleReplicaId)
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

  virtual poplar::program::Sequence GetSequence() const { return sequence; }

  // This should only be used for unit tests
  virtual poplar::program::Sequence& GetMutableSequence() { return sequence; }

 protected:
  Status Unimplemented(HloInstruction* inst);

  CompilerResources& resources_;

  TensorMap tensor_map;

  poplar::program::Sequence sequence;

  bool has_infeed_ = false;
  bool stochastic_rounding_enabled_;
};

}  // namespace poplarplugin
}  // namespace xla

#endif
