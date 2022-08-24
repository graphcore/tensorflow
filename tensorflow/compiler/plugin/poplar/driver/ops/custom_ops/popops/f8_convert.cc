/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#include <poplar/Graph.hpp>
#include <popops/Cast.hpp>
#include <popops/ElementWise.hpp>
#include <poputil/TileMapping.hpp>
#include <poputil/Util.hpp>

#include "tensorflow/compiler/plugin/poplar/driver/ops/custom_ops/poplar_ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"

namespace xla {
namespace poplarplugin {
namespace {

class ConvertToF8Op : public PoplarOpDef {
 public:
  StatusOr<DriverProgramSequence> Creator(
      DriverGraph& graph, CompilerResources& res, const HloInstruction* inst,
      const xla::Shape& output_shape, TensorMap& tensor_map,
      const poplar::DebugContext& debug_context) override {
    PoplarOpDefDebugInfo debug_info(debug_context, "Fp8Convert");
    DriverProgramSequence seq(debug_info);
    auto inputs = FindInstructionInputs(tensor_map, res, inst, 0, seq,
                                        debug_info, /*expand_aliasing=*/true);
    CHECK_EQ(inputs.size(), 2);
    DriverTensor input = inputs[0].AsTensor();
    DriverTensor input_metadata = inputs[1].AsTensor();
    // We can't reinterpret to neither QUARTER_METADATA nor QUARTER type.
    // Instead, clone them and copy raw unsigned char data over.
    // This copy will be elided by poplar.
    DriverTensor metadata =
        graph.clone(poplar::QUARTER_METADATA, input_metadata, debug_context,
                    poplar::TensorCloneMethod::PRESERVE_ORDER_AND_ALIASES);
    seq.add(DriverProgramCopy(input_metadata,
                              metadata.reinterpret(poplar::UNSIGNED_CHAR),
                              /*dont_outline=*/false, debug_info));
    poplar::Tensor out =
        popops::cast(graph, input, poplar::QUARTER, metadata, seq, debug_info);
    TF_CHECK_OK(
        AddOutputTensor(tensor_map, inst, 1,
                        out.getMetadata().reinterpret(poplar::UNSIGNED_CHAR)));
    out = out.reinterpret(poplar::UNSIGNED_CHAR);
    TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, out));
    return seq;
  }
};

class ConvertFromF8Op : public PoplarOpDef {
 public:
  StatusOr<DriverProgramSequence> Creator(
      DriverGraph& graph, CompilerResources& res, const HloInstruction* inst,
      const xla::Shape& output_shape, TensorMap& tensor_map,
      const poplar::DebugContext& debug_context) override {
    PoplarOpDefDebugInfo debug_info(debug_context, "Fp8Convert");
    DriverProgramSequence seq(debug_info);
    TF_ASSIGN_OR_RETURN(poplar::Type poplar_type, PoplarDataType(output_shape));
    TF_ASSIGN_OR_RETURN(
        auto input,
        FindF8InstructionInput(tensor_map, res, inst, 0, seq, debug_info));
    poplar::Tensor out =
        popops::cast(graph, input, poplar_type, seq, debug_info);
    TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, out));
    return seq;
  }
};

REGISTER_POPLAR_OP(ConvertToF8, ConvertToF8Op);
REGISTER_POPLAR_OP(ConvertFromF8, ConvertFromF8Op);

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
