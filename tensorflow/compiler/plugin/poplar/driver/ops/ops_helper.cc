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

StatusOr<poplar::Tensor> AllocatePoplarOpTensor(
    poplar::Graph& graph, CompilerResources& res, const std::string& name,
    const TensorTarget& tensor_target, const xla::Shape& shape,
    const TensorMap& tensor_map) {
  const HloInstruction* inst = tensor_target.tgt;
  TF_ASSIGN_OR_RETURN(auto op_def, PoplarOpManager::GetOp(inst));
  TF_ASSIGN_OR_RETURN(
      poplar::Tensor out,
      op_def->Allocator(graph, res, name, tensor_target, tensor_map));
  return out;
}

StatusOr<poplar::program::Program> CreatePoplarOp(
    poplar::Graph& graph, CompilerResources& res, const HloInstruction* inst,
    const xla::Shape& output_shape, TensorMap& tensor_map) {
  TF_ASSIGN_OR_RETURN(auto op_def, PoplarOpManager::GetOp(inst));
  TF_ASSIGN_OR_RETURN(
      poplar::program::Program prog,
      op_def->Creator(graph, res, inst, output_shape, tensor_map));
  return prog;
}

StatusOr<poplar::Tensor> AllocateHloOpTensor(poplar::Graph& graph,
                                             CompilerResources& res,
                                             const std::string& name,
                                             const TensorTarget& tensor_target,
                                             const xla::Shape& shape,
                                             const TensorMap& tensor_map) {
  const HloInstruction* inst = tensor_target.tgt;
  TF_ASSIGN_OR_RETURN(auto op_def, HloOpManager::GetOp(inst));
  TF_ASSIGN_OR_RETURN(
      poplar::Tensor out,
      op_def->Allocator(graph, res, name, tensor_target, tensor_map));
  return out;
}

StatusOr<poplar::program::Program> CreateHloOp(poplar::Graph& graph,
                                               CompilerResources& res,
                                               const HloInstruction* inst,
                                               const xla::Shape& output_shape,
                                               TensorMap& tensor_map) {
  TF_ASSIGN_OR_RETURN(auto op_def, HloOpManager::GetOp(inst));
  TF_ASSIGN_OR_RETURN(
      poplar::program::Program prog,
      op_def->Creator(graph, res, inst, output_shape, tensor_map));
  return prog;
}

}  // namespace poplarplugin
}  // namespace xla
