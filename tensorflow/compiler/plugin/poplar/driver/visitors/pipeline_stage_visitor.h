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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_VISITORS_PIPELINE_STAGE_VISITOR_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_VISITORS_PIPELINE_STAGE_VISITOR_H_

#include <string>

#include "tensorflow/compiler/plugin/poplar/driver/ops/ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/visitors/deferred_visitor.h"

namespace xla {
namespace poplarplugin {

struct CompilerResources;

class PipelineStageVisitor : public InplaceDeferredVisitor {
 public:
  PipelineStageVisitor(CompilerResources& res, const DeferredArgVectors& inputs,
                       const std::string& name);

  bool TupleOutputsNeedToPreserveAliasing(const HloInstruction* inst) override;

  poplar::program::Sequence GetCachedSequence();

  // Returns whether the output needs a copy.
  virtual ShapeTree<bool> GetOutputCopies(const HloInstruction* inst) const;

 private:
  // Caching fields for the GetSequence call
  bool has_function_ = false;
  poplar::Function function_;
};

// Similar to PipelineStageVisitor, however it adds copies for any non-inplace
// which allows its sequence to be reused with different inputs.
class ReusablePipelineStageVisitor : public PipelineStageVisitor {
 public:
  ReusablePipelineStageVisitor(CompilerResources& res,
                               const DeferredArgVectors& inputs,
                               const std::string& name);

  // A function which propagates any tensors which were not allocated at call
  // site but now have a tensor.
  Status PropagateDeferredAllocations(const HloInstruction* callsite);

  // Get the sequence for the forward stage, adding any copies for inplace
  // inputs.
  poplar::program::Sequence GetForwardStageSequence(
      const HloInstruction* callsite, const DeferredArgVectors& inputs,
      TensorMap& callsite_tensor_map);

  // Get the sequence for the recomputation stage.
  poplar::program::Sequence GetRecomputationStageSequence(
      const HloInstruction* callsite, const TensorVectors& inputs);

  // Returns whether the output needs a copy.
  ShapeTree<bool> GetOutputCopies(const HloInstruction* inst) const override;

 private:
  poplar::program::Sequence GetCachedSequence(const HloInstruction* callsite,
                                              const TensorVectors& inputs);

  const HloInstruction* callsite_;
};

}  // namespace poplarplugin
}  // namespace xla

#endif
