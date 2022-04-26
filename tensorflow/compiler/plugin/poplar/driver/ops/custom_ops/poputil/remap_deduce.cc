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

#include <poplar/DebugContext.hpp>
#include <poputil/TileMapping.hpp>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "tensorflow/compiler/plugin/poplar/driver/ops/custom_ops/poplar_ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/pooling.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/debug_info.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/custom_kernels_util.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/core/errors.h"

namespace xla {
namespace poplarplugin {
namespace {
class RemapDeduceOp : public PoplarOpDef {
  StatusOr<DriverProgramSequence> Creator(
      DriverGraph& graph, CompilerResources& res, const HloInstruction* inst,
      const xla::Shape& output_shape, TensorMap& tensor_map,
      const poplar::DebugContext& debug_context) override {
    PoplarOpDefDebugInfo debug_info(debug_context, "RemapDeduceOp");
    DriverProgramSequence seq(graph, debug_info);

    // Get the original tensor which is to be remapped.
    TF_ASSIGN_OR_RETURN(auto input,
                        FindInstructionInput(tensor_map, res, inst, 0, seq,
                                             {debug_info}, false));

    // Create a new tensor using "AddTensor" to get a good layout.
    TF_ASSIGN_OR_RETURN(auto output,
                        AddTensor(graph, TensorLocation{inst, 0}, output_shape,
                                  res, tensor_map, {debug_info, "output"}));

    // Copy the original into the new layout.
    seq.add(poplar::program::Copy(input, output, false, {debug_info}));
    TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, output));
    return seq;
  }
};

REGISTER_POPLAR_OP(RemapDeduce, RemapDeduceOp);

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
