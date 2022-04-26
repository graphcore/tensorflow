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

#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/codelet_expression_op.h"
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

#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"

#include <poplar/DebugContext.hpp>
#include <poputil/Broadcast.hpp>

#include <fstream>

namespace xla {
namespace poplarplugin {
namespace {

std::stringstream GenerateVertexSource(
    const std::string& name, std::vector<poplar::Tensor> input_tensors,
    const std::string& body) {
  std::stringstream result;
  result << "#include <poplar/HalfFloat.hpp>" << std::endl
         << "#include <poplar/Vertex.hpp>" << std::endl
         << "#include <cmath>" << std::endl
         << "namespace tf {" << std::endl
         << "class " << name << " : public poplar::Vertex {" << std::endl
         << "public:" << std::endl;

  for (auto i = 0ul; i < input_tensors.size(); ++i) {
    result << "poplar::Input<poplar::Vector<" << input_tensors[i].elementType()
           << ">> input" << i << ";" << std::endl;
  }
  result << "poplar::Output<poplar::Vector<" << input_tensors[0].elementType()
         << ">> output;" << std::endl
         << "bool compute() {" << std::endl
         << "for (auto i = 0ul; i < input0.size(); ++i) {" << std::endl;

  for (auto i = 0ul; i < input_tensors.size(); ++i) {
    result << "auto in" << i << " = input" << i << "[i];" << std::endl;
  }

  result << "output[i] = " << body << ';' << std::endl
         << "}" << std::endl
         << "return true;" << std::endl
         << "}" << std::endl
         << "};" << std::endl
         << "}" << std::endl;

  return result;
}

class CodeletExpressionOpOp : public PoplarOpDef {
  StatusOr<DriverProgramSequence> Creator(
      DriverGraph& graph, CompilerResources& res, const HloInstruction* inst,
      const xla::Shape& output_shape, TensorMap& tensor_map,
      const poplar::DebugContext& debug_context) override {
    PoplarOpDefDebugInfo debug_info(debug_context, "CodeletExpressionOpOp");
    auto op_inst = Cast<HloCodeletExpressionOpInstruction>(inst);
    DriverProgramSequence seq(graph, debug_info);

    std::vector<poplar::Tensor> input_tensors(inst->operand_count());
    for (int i = 0; i < inst->operand_count(); ++i) {
      TF_ASSIGN_OR_RETURN(
          input_tensors[i],
          FindInstructionInput(tensor_map, res, inst, i, seq, {debug_info}));

      if (i > 0) {
        poputil::broadcastToMatch(input_tensors[0], input_tensors[i]);
      }
    }

    for (int i = 0; i < inst->operand_count(); ++i) {
      poputil::broadcastToMatch(input_tensors[i], input_tensors[0]);
    }

    static int static_count = 0;
    const std::string vertex_name = "UserVertex" + std::to_string(static_count);
    std::stringstream vertex_source =
        GenerateVertexSource(vertex_name, input_tensors, op_inst->source());
    static_count++;

    graph.addCodelets(vertex_source);

    auto output = graph.clone(input_tensors[0], {debug_info});

    auto cs = graph.addComputeSet({debug_info, "cs"});
    const auto tile_mapping = graph.getTileMapping(input_tensors[0]);
    auto tile = 0u;
    for (const auto& intervals : tile_mapping) {
      for (const auto& interval : intervals) {
        auto v = graph.addVertex(cs, "tf::" + vertex_name);
        graph.setTileMapping(v, tile);
        graph.setPerfEstimate(v, 1);

        for (auto i = 0ul; i < input_tensors.size(); ++i) {
          graph.connect(v["input" + std::to_string(i)],
                        input_tensors[i].flatten().slice(interval));
        }
        graph.connect(v["output"], output.flatten().slice(interval));
      }
      tile++;
    }

    seq.add(poplar::program::Execute(cs));
    TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, output));

    return seq;
  }
};
REGISTER_POPLAR_OP(CodeletExpressionOp, CodeletExpressionOpOp);

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
