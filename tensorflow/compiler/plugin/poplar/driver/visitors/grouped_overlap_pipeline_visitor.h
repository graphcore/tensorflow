/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_VISITORS_GROUPED_OVERLAP_PIPELINE_VISITOR_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_VISITORS_GROUPED_OVERLAP_PIPELINE_VISITOR_H_

#include <memory>
#include <string>

#include "tensorflow/compiler/plugin/poplar/driver/ops/ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/visitors/deferred_visitor.h"
#include "tensorflow/compiler/plugin/poplar/driver/visitors/pipeline_stage_visitor.h"
#include "tensorflow/compiler/plugin/poplar/driver/visitors/pipeline_visitor.h"

namespace xla {
namespace poplarplugin {

struct CompilerResources;

class GroupedOverlapPipelineVisitor : public ParallelPipelineVisitor {
 public:
  using ParallelPipelineVisitor::ParallelPipelineVisitor;

  StatusOr<DriverProgramSequence> VerifyPipelineArguments(
      const HloInstruction* accumulation_count,
      DriverTensor accumulation_count_tensor,
      DriverGraph& graph) const override;

  PipelineVisitor::IterationsType RampDownAdditionalIterations(
      IterationsType iterations, const size_t overlap_length,
      DriverProgramSequence& program) const override;

  static std::unique_ptr<PipelineVisitor> Create(
      DriverGraph& graph, const HloInstruction* pipeline,
      CompilerResources& res, const DeferredArgRBVectors& inputs,
      const HloPoplarInplaceDescription& description,
      const poplar::DebugNameAndId& debug_name_and_id);

 protected:
  RepeatBlock GetPipelineRampUpSequence(
      const poplar::DebugNameAndId& debug_name_and_id) const override;
  DriverProgramSequence GetPipelineRampDownSequence(
      const poplar::DebugNameAndId& debug_name_and_id,
      const IterationsType& additional_iterations = 0) const override;
  DriverProgramSequence GetPipelineRepeatBlockSequence(
      const poplar::DebugNameAndId& debug_name_and_id,
      const IterationsType& iterations) const override;
};

}  // namespace poplarplugin
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_VISITORS_GROUPED_OVERLAP_PIPELINE_VISITOR_H_
