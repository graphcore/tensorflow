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

#include "tensorflow/compiler/plugin/poplar/driver/ops/ops_helper.h"

#include <algorithm>
#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "tensorflow/compiler/plugin/poplar/driver/compiler_resources.h"
#include "tensorflow/compiler/plugin/poplar/driver/ops/custom_ops/poplar_ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/ops/ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/allocation_finder.h"
#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/debug_info.h"
#include "tensorflow/compiler/plugin/poplar/driver/vertex_templates.h"
#include "tensorflow/compiler/plugin/poplar/kernels/custom_kernels_util.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/core/errors.h"

namespace xla {
namespace poplarplugin {

StatusOr<DriverTensor> AllocatePoplarOpTensor(
    DriverGraph& graph, CompilerResources& res,
    const poplar::DebugNameAndId& parent_debug_name_and_id,
    const TensorTarget& tensor_target, const xla::Shape& shape,
    const TensorMap& tensor_map) {
  const HloInstruction* inst = tensor_target.tgt;

  // This is the other way round than AllocateHloOpTensor which is
  // named after the target hlo instruction.
  poplar::DebugContext debug_context(parent_debug_name_and_id);
  PoplarOpDefDebugInfo debug_info(debug_context, "AllocatePoplarOpTensor");

  TF_ASSIGN_OR_RETURN(auto op_def, PoplarOpManager::GetOp(inst));
  TF_ASSIGN_OR_RETURN(poplar::Tensor out,
                      op_def->Allocator(graph, res, "", tensor_target,
                                        tensor_map, {debug_info}));
  return DriverTensor(out, graph);
}

StatusOr<DriverProgramSequence> CreatePoplarOp(
    DriverGraph& graph, CompilerResources& res, const HloInstruction* inst,
    const xla::Shape& output_shape, TensorMap& tensor_map,
    const poplar::DebugNameAndId& debug_name_and_id) {
  poplar::DebugContext debug_context(debug_name_and_id);
  PoplarOpDefDebugInfo debug_info(debug_context, "CreatePoplarOp");

  TF_ASSIGN_OR_RETURN(auto op_def, PoplarOpManager::GetOp(inst));
  TF_ASSIGN_OR_RETURN(DriverProgramSequence prog,
                      op_def->Creator(graph, res, inst, output_shape,
                                      tensor_map, {debug_info}));
  return prog;
}

StatusOr<DriverTensor> AllocateHloOpTensor(
    DriverGraph& graph, CompilerResources& res,
    const poplar::DebugNameAndId& parent_debug_name_and_id,
    const TensorTarget& tensor_target, const xla::Shape& shape,
    const TensorMap& tensor_map) {
  const HloInstruction* inst = tensor_target.tgt;

  // This function has a choice of debug information to use
  // The target instruction (i.e. convolution.7)
  // The parent instruction (i.e. arg2.3)
  // Currently debug context's can not have multiple parents.
  // We will keep the existing implementation and use the target instruction
  // and include the parent instruction name

  // The name in the "Allocator" api call is now redundent.
  // TODO(T32947) : Remove name from PoplarOpDef::Allocator method

  poplar::DebugNameAndId target_debug_name_and_id =
      GetDebugNameAndId(res, inst);
  poplar::DebugContext debug_context(target_debug_name_and_id);
  PoplarOpDefDebugInfo debug_info(debug_context, "AllocateHloOpTensor");
  debug_info.setValue("parent_op", parent_debug_name_and_id.getPathName());

  TF_ASSIGN_OR_RETURN(auto op_def, HloOpManager::GetOp(inst));
  TF_ASSIGN_OR_RETURN(poplar::Tensor out,
                      op_def->Allocator(graph, res, "", tensor_target,
                                        tensor_map, {debug_info}));
  return DriverTensor(out, graph);
}

StatusOr<DriverProgramSequence> CreateHloOp(DriverGraph& graph,
                                            CompilerResources& res,
                                            const HloInstruction* inst,
                                            const xla::Shape& output_shape,
                                            TensorMap& tensor_map) {
  poplar::DebugNameAndId debug_name_and_id = GetDebugNameAndId(res, inst);
  poplar::DebugContext debug_context(debug_name_and_id);
  PoplarOpDefDebugInfo debug_info(debug_context, "CreateHloOp");

  TF_ASSIGN_OR_RETURN(auto op_def, HloOpManager::GetOp(inst));
  TF_ASSIGN_OR_RETURN(DriverProgramSequence prog,
                      op_def->Creator(graph, res, inst, output_shape,
                                      tensor_map, {debug_info}));
  return prog;
}

}  // namespace poplarplugin
}  // namespace xla
