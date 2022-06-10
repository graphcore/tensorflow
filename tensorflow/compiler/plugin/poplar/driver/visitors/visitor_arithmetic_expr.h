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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_VISITORS_VISITOR_ARITHMETIC_EXPR_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_VISITORS_VISITOR_ARITHMETIC_EXPR_H_

#include <map>
#include <memory>
#include <string>
#include <vector>

#include <popops/Expr.hpp>

#include "tensorflow/compiler/plugin/poplar/driver/visitors/visitor_base.h"

namespace poplar {
class Graph;
class Tensor;
}  // namespace poplar

namespace xla {
namespace poplarplugin {

class ArithmeticExprVisitor : public BaseVisitor {
 public:
  ArithmeticExprVisitor(CompilerResources& res, const TensorVectors& inputs,
                        const HloInstruction* caller,
                        const poplar::DebugNameAndId& debug_name_and_id);

  Status HandleElementwiseUnary(HloInstruction* inst) override;
  Status HandleElementwiseBinary(HloInstruction* inst) override;
  Status HandleCustomCall(HloInstruction* inst) override;
  Status HandleCompare(HloInstruction* inst) override;
  Status HandleConvert(HloInstruction* inst) override;
  Status HandleSelect(HloInstruction* inst) override;
  Status HandleClamp(HloInstruction* inst) override;
  Status HandleParameter(HloInstruction* inst) override;
  Status FinishScopedVisit(HloInstruction* inst) override;

#define ARITHMETIC_EXPR_VISITOR_UNIMPLEMENTED(Name) \
  Status Name(HloInstruction* inst) override { return Unimplemented(inst); };

  ARITHMETIC_EXPR_VISITOR_UNIMPLEMENTED(HandleCopy);
  ARITHMETIC_EXPR_VISITOR_UNIMPLEMENTED(HandleConcatenate);
  ARITHMETIC_EXPR_VISITOR_UNIMPLEMENTED(HandleBitcastConvert);
  ARITHMETIC_EXPR_VISITOR_UNIMPLEMENTED(HandleDot);
  ARITHMETIC_EXPR_VISITOR_UNIMPLEMENTED(HandleConvolution);
  ARITHMETIC_EXPR_VISITOR_UNIMPLEMENTED(HandleAllReduce);
  ARITHMETIC_EXPR_VISITOR_UNIMPLEMENTED(HandleReverse);
  ARITHMETIC_EXPR_VISITOR_UNIMPLEMENTED(HandleSort);
  ARITHMETIC_EXPR_VISITOR_UNIMPLEMENTED(HandleConstant);
  ARITHMETIC_EXPR_VISITOR_UNIMPLEMENTED(HandleGetTupleElement);
  ARITHMETIC_EXPR_VISITOR_UNIMPLEMENTED(HandleReduce);
  ARITHMETIC_EXPR_VISITOR_UNIMPLEMENTED(HandleBitcast);
  ARITHMETIC_EXPR_VISITOR_UNIMPLEMENTED(HandleBroadcast);
  ARITHMETIC_EXPR_VISITOR_UNIMPLEMENTED(HandleReshape);
  ARITHMETIC_EXPR_VISITOR_UNIMPLEMENTED(HandleTranspose);
  ARITHMETIC_EXPR_VISITOR_UNIMPLEMENTED(HandleFusion);
  ARITHMETIC_EXPR_VISITOR_UNIMPLEMENTED(HandleCall);
  ARITHMETIC_EXPR_VISITOR_UNIMPLEMENTED(HandleSlice);
  ARITHMETIC_EXPR_VISITOR_UNIMPLEMENTED(HandleDynamicSlice);
  ARITHMETIC_EXPR_VISITOR_UNIMPLEMENTED(HandleDynamicUpdateSlice);
  ARITHMETIC_EXPR_VISITOR_UNIMPLEMENTED(HandleTuple);
  ARITHMETIC_EXPR_VISITOR_UNIMPLEMENTED(HandleReduceWindow);
  ARITHMETIC_EXPR_VISITOR_UNIMPLEMENTED(HandleMap);
  ARITHMETIC_EXPR_VISITOR_UNIMPLEMENTED(HandleSelectAndScatter);
  ARITHMETIC_EXPR_VISITOR_UNIMPLEMENTED(HandleConditional);
  ARITHMETIC_EXPR_VISITOR_UNIMPLEMENTED(HandlePad);
  ARITHMETIC_EXPR_VISITOR_UNIMPLEMENTED(HandleReducePrecision);
  ARITHMETIC_EXPR_VISITOR_UNIMPLEMENTED(HandleInfeed);
  ARITHMETIC_EXPR_VISITOR_UNIMPLEMENTED(HandleOutfeed);
  ARITHMETIC_EXPR_VISITOR_UNIMPLEMENTED(HandleSend);
  ARITHMETIC_EXPR_VISITOR_UNIMPLEMENTED(HandleSendDone);
  ARITHMETIC_EXPR_VISITOR_UNIMPLEMENTED(HandleRecv);
  ARITHMETIC_EXPR_VISITOR_UNIMPLEMENTED(HandleRecvDone);
  ARITHMETIC_EXPR_VISITOR_UNIMPLEMENTED(HandleBatchNormInference);
  ARITHMETIC_EXPR_VISITOR_UNIMPLEMENTED(HandleBatchNormTraining);
  ARITHMETIC_EXPR_VISITOR_UNIMPLEMENTED(HandleBatchNormGrad);
  ARITHMETIC_EXPR_VISITOR_UNIMPLEMENTED(HandleFft);
  ARITHMETIC_EXPR_VISITOR_UNIMPLEMENTED(HandleAllGather);
  ARITHMETIC_EXPR_VISITOR_UNIMPLEMENTED(HandleCollectivePermuteStart);
  ARITHMETIC_EXPR_VISITOR_UNIMPLEMENTED(HandleCollectivePermuteDone);
  ARITHMETIC_EXPR_VISITOR_UNIMPLEMENTED(HandleRngBitGenerator);
  ARITHMETIC_EXPR_VISITOR_UNIMPLEMENTED(HandleDynamicReshape);
  ARITHMETIC_EXPR_VISITOR_UNIMPLEMENTED(HandleAllReduceScatter);
  ARITHMETIC_EXPR_VISITOR_UNIMPLEMENTED(HandleAllReduceStart);
  ARITHMETIC_EXPR_VISITOR_UNIMPLEMENTED(HandleAllReduceDone);
#undef ARITHMETIC_EXPR_VISITOR_UNIMPLEMENTED

  const TensorVector& outputs() { return outputs_; }

 private:
  StatusOr<std::unique_ptr<popops::expr::Expr>> FindExpressionInput(
      const HloInstruction* inst);
  TensorVectors inputs_;
  TensorVector outputs_;
  std::map<const HloInstruction*, std::unique_ptr<popops::expr::Expr>>
      expressions_map_;
  std::map<const DriverTensor*, std::unique_ptr<popops::expr::Expr>>
      tensors_map_;
  std::vector<DriverTensor> ts_;
  const HloInstruction* caller_;
};

}  // namespace poplarplugin
}  // namespace xla

#endif
