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

#include "absl/container/flat_hash_map.h"
#include "tensorflow/compiler/plugin/poplar/driver/ops/custom_ops/poplar_ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/debug_info.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/custom_kernels_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/core/errors.h"

namespace xla {
namespace poplarplugin {
namespace {

class BarrierOp : public PoplarOpDef {
  StatusOr<DriverProgramSequence> Creator(
      DriverGraph& graph, CompilerResources& res, const HloInstruction* inst,
      const xla::Shape& output_shape, TensorMap& tensor_map,
      const poplar::DebugContext& debug_context) override {
    PoplarOpDefDebugInfo debug_info(debug_context, "BarrierOp");
    DriverProgramSequence seq(graph, debug_info);

    // Just pass the tensors through.
    TF_ASSIGN_OR_RETURN(TensorVectors inputs,
                        FindInplaceOutputTensors(tensor_map, res, inst, seq,
                                                 debug_info, false));
    int64_t flat_tuple_index = 0;
    for (auto& operand_tensors : inputs) {
      for (auto& tensor : operand_tensors) {
        TF_CHECK_OK(
            AddOutputTensor(tensor_map, inst, flat_tuple_index++, tensor));
      }
    }

    return seq;
  }
};
REGISTER_POPLAR_OP(Barrier, BarrierOp);

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
