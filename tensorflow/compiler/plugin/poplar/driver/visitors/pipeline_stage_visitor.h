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
  PipelineStageVisitor(CompilerResources& res,
                       const DeferredArgRBVectors& inputs,
                       const HloPoplarInplaceDescription& description,
                       const poplar::DebugNameAndId& debug_name_and_id);

  DriverProgramSequence GetCachedSequence(DriverGraph& graph);

  // Returns whether the output needs a copy.
  virtual ShapeTree<bool> GetOutputCopies(const HloInstruction* inst) const;

 private:
  // Caching fields for the GetSequence call
  bool has_function_ = false;
  DriverFunction function_;
};

// Similar to PipelineStageVisitor, however it adds copies for any non-inplace
// which allows its sequence to be reused with different inputs.
class ReusablePipelineStageVisitor : public PipelineStageVisitor {
 public:
  ReusablePipelineStageVisitor(CompilerResources& res,
                               const DeferredArgRBVectors& inputs,
                               const HloPoplarInplaceDescription& description,
                               const poplar::DebugNameAndId& debug_name_and_id);

  // A function which propagates any tensors which were not allocated at call
  // site but now have a tensor.
  Status PropagateDeferredAllocations(
      const HloInstruction* callsite,
      const DeferredArgRBVectors& callsite_inputs,
      const poplar::DebugNameAndId& debug_name_and_id) override;

  // Get the sequence for the forward stage, adding any copies for inplace
  // inputs.
  DriverProgramSequence GetForwardStageSequence(
      const HloInstruction* callsite, const DeferredArgRBVectors& inputs,
      TensorMap& callsite_tensor_map);

  // Get the sequence for the recomputation stage.
  DriverProgramSequence GetRecomputationStageSequence(
      const HloInstruction* callsite,
      const TensorOrRemoteBufferVectors& inputs);

  // Returns whether the output needs a copy.
  ShapeTree<bool> GetOutputCopies(const HloInstruction* inst) const override;

 private:
  DriverProgramSequence GetCachedSequence(
      const HloInstruction* callsite,
      const TensorOrRemoteBufferVectors& inputs);

  const HloInstruction* callsite_;
};

}  // namespace poplarplugin
}  // namespace xla

#endif
