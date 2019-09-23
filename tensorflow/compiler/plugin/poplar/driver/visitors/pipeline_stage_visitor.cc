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

#include "tensorflow/compiler/plugin/poplar/driver/visitors/pipeline_stage_visitor.h"
#include "tensorflow/compiler/plugin/poplar/driver/compiler_resources.h"
#include "tensorflow/compiler/plugin/poplar/driver/ops/ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/pipeline_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"

#include <poplar/Tensor.hpp>

namespace xla {
namespace poplarplugin {

PipelineStageVisitor::PipelineStageVisitor(CompilerResources& res,
                                           const ArgVectors& inputs)
    : InplaceSubComputationVisitor(res, inputs) {}

Status PipelineStageVisitor::HandleTuple(HloInstruction* inst) {
  VLOG(1) << "Processing " << inst->name();
  // We preserve constants/aliasing when this is the output tuple.
  const bool is_root = inst->parent()->root_instruction() == inst;
  const bool expand_constants = !is_root;
  const bool preserve_aliasing = is_root;
  TF_ASSIGN_OR_RETURN(poplar::program::Program prog,
                      CreateTuple(resources_, inst, tensor_map,
                                  expand_constants, preserve_aliasing));
  sequence.add(prog);
  return Status::OK();
}

}  // namespace poplarplugin
}  // namespace xla
