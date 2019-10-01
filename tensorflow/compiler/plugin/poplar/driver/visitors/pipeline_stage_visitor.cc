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

StatusOr<std::vector<bool>> PipelineStageVisitor::GetOutputCopies(
    const HloInstruction* inst, bool used_for_recomputation) {
  std::vector<bool> result;
  const Shape& output_shape = inst->shape();
  if (!output_shape.IsTuple()) {
    return FailedPrecondition(
        "Expected the output of the PipelineStage to be a tuple shape.");
  }

  // Get which outputs of the stage are used.
  absl::flat_hash_set<int64> used_gtes_indicies;
  for (const HloInstruction* user : inst->users()) {
    CHECK_EQ(user->opcode(), HloOpcode::kGetTupleElement);
    used_gtes_indicies.insert(user->tuple_index());
  }

  // We need to add copies if the subshape has a user (i.e. there is a tuple
  // index).
  for (int64 i = 0; i < ShapeUtil::TupleElementCount(output_shape); i++) {
    const Shape& subshape = ShapeUtil::GetTupleElementShape(output_shape, i);
    result.resize(result.size() + FlattenedXlaShape(subshape).size(),
                  used_for_recomputation && used_gtes_indicies.contains(i));
  }
  return result;
}

}  // namespace poplarplugin
}  // namespace xla
