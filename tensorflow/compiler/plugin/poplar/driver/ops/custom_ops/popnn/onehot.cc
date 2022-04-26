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

#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/onehot.h"

#include <poplar/DebugContext.hpp>
#include <popops/Encoding.hpp>

#include "absl/container/flat_hash_map.h"
#include "tensorflow/compiler/plugin/poplar/driver/ops/custom_ops/poplar_ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"
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

class OneHotOp : public PoplarOpDef {
  StatusOr<DriverProgramSequence> Creator(
      DriverGraph& graph, CompilerResources& res, const HloInstruction* inst,
      const xla::Shape& output_shape, TensorMap& tensor_map,
      const poplar::DebugContext& debug_context) override {
    PoplarOpDefDebugInfo debug_info(debug_context, "OneHotOp");
    // Create the control program.
    DriverProgramSequence seq(graph, debug_info);
    const HloOneHotInstruction* one_hot_op = Cast<HloOneHotInstruction>(inst);

    // Get the inputs.
    TF_ASSIGN_OR_RETURN(poplar::Tensor indices,
                        FindInstructionInput(tensor_map, res, inst, 0, seq,
                                             {debug_info}, false));

    TF_ASSIGN_OR_RETURN(poplar::Tensor on,
                        FindInstructionInput(tensor_map, res, inst, 1, seq,
                                             {debug_info}, false));

    TF_ASSIGN_OR_RETURN(poplar::Tensor off,
                        FindInstructionInput(tensor_map, res, inst, 2, seq,
                                             {debug_info}, false));

    const int64 axis = one_hot_op->Axis();

    // Create the output tensor to store the result in (as popops takes this by
    // reference to make sure the output is in this layout).
    TF_ASSIGN_OR_RETURN(auto output,
                        AddTensor(graph, TensorLocation{inst, 0}, inst->shape(),
                                  res, tensor_map, {debug_info, "output"}));

    // popops::encodeOneHot expects the tensor to be 2D with the `axis` at the
    // back - create a permuted 2D view of that tensor.
    const std::size_t rank = output.rank();
    poplar::Tensor output_2d =
        output.dimRoll(axis, rank - 1).flatten(0, rank - 1);

    // Encode one hot returns void but stores output in "output".
    popops::encodeOneHot(graph, indices.flatten(), output_2d, seq, on, off,
                         {debug_info, "OneHot"});

    TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, output));

    return seq;
  }
};

REGISTER_POPLAR_OP(OneHot, OneHotOp);

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
