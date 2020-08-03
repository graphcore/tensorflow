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

#include <poputil/Util.hpp>
#include <random>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "tensorflow/compiler/plugin/poplar/driver/ops/custom_ops/poplar_ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/dropout_hlo.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/pooling.h"
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
namespace pe = popops::expr;

class DropoutOp : public PoplarOpDef {
  StatusOr<poplar::program::Program> Creator(poplar::Graph& graph,
                                             CompilerResources& res,
                                             const HloInstruction* inst,
                                             const Shape& output_shape,
                                             TensorMap& tensor_map) override {
    const std::string debug_name = GetDebugName(inst);
    const HloDropout* dropout_instruction = Cast<HloDropout>(inst);
    const float rate = dropout_instruction->Rate();
    const float scale = dropout_instruction->Scale();

    poplar::program::Sequence seq;
    TF_ASSIGN_OR_RETURN(
        poplar::Tensor input,
        FindInstructionInput(tensor_map, res, inst, 0, seq, false));

    TF_ASSIGN_OR_RETURN(
        poplar::Tensor seed,
        FindInstructionInput(tensor_map, res, inst, 1, seq, false));

    // Clone the seed as it is an output.
    seed = poputil::duplicate(graph, seed, seq, debug_name + "/Seed");

    // Dropout expects an unsigned int but tensorflow takes in int32 when
    // targeting IPU.
    poplar::Tensor seed_unsigned = seed.reinterpret(poplar::UNSIGNED_INT);

    poplar::Tensor output;
    if (dropout_instruction->HasNoiseShape()) {
      // Pull out noise_shape if it is set. At this point it is a std::vector,
      // but poprand::dropout expects a poplar::Tensor*
      // noise_shape becomes a constant tensor residing on tile 0.
      const Shape ns_ref_shape =
          ShapeUtil::MakeShape(inst->operand(0)->shape().element_type(),
                               dropout_instruction->NoiseShape());
      TF_ASSIGN_OR_RETURN(poplar::Tensor reference,
                          AddPlainTensor(graph, debug_name + "/ShapedReference",
                                         ns_ref_shape, res, false));

      output = poprand::shapedDropout(graph, &seed_unsigned, 1U, input,
                                      reference, rate, scale, seq, debug_name);
    } else {
      // Create an empty tensor for the dropout. This is internal to the poprand
      // implementation but is exposed anyway so we need to provide it.
      TF_ASSIGN_OR_RETURN(
          poplar::Tensor reference,
          AddPlainTensor(graph, debug_name + "/Reference",
                         inst->operand(0)->shape(), res, false));

      // Perform the actual dropout by calling into the poprand function.
      output = poprand::dropout(graph, &seed_unsigned, 1U, input, reference,
                                rate, scale, seq, debug_name);
    }

    // Mark that tensor as our output.
    TF_RETURN_IF_ERROR(AddOutputTensor(tensor_map, inst, 0, output));
    TF_RETURN_IF_ERROR(AddOutputTensor(tensor_map, inst, 1, seed));

    return seq;
  }
};
REGISTER_POPLAR_OP(Dropout, DropoutOp);
}  // namespace
}  // namespace poplarplugin
}  // namespace xla
