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

#include "tensorflow/compiler/plugin/poplar/driver/visitors/pipeline_visitor_creator.h"

#include <stddef.h>
#include <string.h>

#include <limits>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <poplar/Engine.hpp>
#include <poplar/GraphElements.hpp>
#include <poplar/Tensor.hpp>
#include <poplar/exceptions.hpp>
#include <poputil/Util.hpp>

#include "tensorflow/compiler/plugin/poplar/driver/compiler_resources.h"
#include "tensorflow/compiler/plugin/poplar/driver/ops/ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/poplar_executor.h"
#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/hlo_poplar_instruction.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/debug_info.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/inplace_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/pipeline_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/poplar_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/plugin/poplar/driver/visitors/deferred_visitor.h"
#include "tensorflow/compiler/plugin/poplar/driver/visitors/grouped_overlap_pipeline_visitor.h"
#include "tensorflow/compiler/plugin/poplar/driver/visitors/pipeline_stage_visitor.h"
#include "tensorflow/compiler/plugin/poplar/driver/visitors/pipeline_visitor.h"
#include "tensorflow/compiler/plugin/poplar/kernels/custom_kernels_util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/ops.pb.h"
#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/service/buffer_assignment.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/stream_executor/lib/initialize.h"

namespace xla {
namespace poplarplugin {

StatusOr<std::unique_ptr<PipelineVisitor>> GetPipelineVisitor(
    const HloInstruction* pipeline, CompilerResources& res,
    const DeferredArgRBVectors& inputs,
    const HloPoplarInplaceDescription& description,
    const poplar::DebugNameAndId& debug_name_and_id) {
  auto& graph = GetGraph(res, pipeline);
  TF_ASSIGN_OR_RETURN(auto schedule, GetPipelineSchedule(pipeline));
  switch (schedule) {
    case PoplarBackendConfig::CallConfig::PipelineConfig::Grouped:
      if (HasIOTiles(res)) {
        return GroupedOverlapPipelineVisitor::Create(
            graph, pipeline, res, inputs, description, debug_name_and_id);
      } else {
        return ParallelPipelineVisitor::Create(graph, pipeline, res, inputs,
                                               description, debug_name_and_id);
      }
    case PoplarBackendConfig::CallConfig::PipelineConfig::Interleaved:
      return ParallelPipelineVisitor::Create(graph, pipeline, res, inputs,
                                             description, debug_name_and_id);
    case PoplarBackendConfig::CallConfig::PipelineConfig::Sequential:
      return SequentialPipelineVisitor::Create(graph, pipeline, res, inputs,
                                               description, debug_name_and_id);
    default:
      return FailedPrecondition("Unknown pipeline schedule.");
  }
}

}  // namespace poplarplugin
}  // namespace xla
