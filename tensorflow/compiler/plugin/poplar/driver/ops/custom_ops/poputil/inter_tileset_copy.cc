/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/inter_tileset_copy.h"

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

class InterTilesetCopyOp : public PoplarOpDef {
  StatusOr<poplar::program::Sequence> Creator(
      DriverGraph& graph, CompilerResources& res, const HloInstruction* inst,
      const xla::Shape& output_shape, TensorMap& tensor_map,
      const poplar::DebugContext& debug_context) override {
    const auto* copy_inst = Cast<HloInterTilesetCopy>(inst);

    PoplarOpDefDebugInfo debug_info(debug_context, "InterTilesetCopyOp");
    poplar::program::Sequence seq({}, debug_info);

    CHECK(!IsLoweredInplace(inst)) << inst->ToString();

    TensorOrRemoteBufferVector inputs = FindInstructionInputs(
        tensor_map, res, inst, 0, seq, {debug_info}, true);
    CHECK_EQ(inputs.size(), 1);
    CHECK(inputs[0].IsTensor())
        << "Expected to copy a poplar::Tensor: " << inst->ToString();

    poplar::Tensor input = inputs[0];
    poplar::Tensor output;
    if (copy_inst->IsCopyToIoTiles()) {
      // Sanity check that we are allocating on IO tiles.
      CHECK_EQ(graph.getTarget().getNumTiles(), res.num_io_tiles);

      TF_ASSIGN_OR_RETURN(output,
                          AddHostCopyTensor(graph, {debug_info}, output_shape));
    } else {
      TF_ASSIGN_OR_RETURN(
          output, AddTensor(graph, TensorLocation{inst, 0}, output_shape, res,
                            tensor_map, {debug_info, "output"}));
    }

    seq.add(poplar::program::Copy(input, output, false, {debug_info}));

    TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, output));

    return seq;
  }

  StatusOr<poplar::Tensor> Allocator(
      DriverGraph& graph, CompilerResources& res, const std::string& name,
      const TensorTarget& tensor_target, const TensorMap& tensor_map,
      const poplar::DebugContext& debug_context) override {
    PoplarOpDefDebugInfo debug_info(debug_context, "InterTilesetCopyOp");
    const auto* copy_inst = Cast<HloInterTilesetCopy>(tensor_target.tgt);
    if (copy_inst->IsCopyToIoTiles()) {
      // Sanity check that we are allocating on IO tiles.
      CHECK_EQ(graph.getTarget().getNumTiles(), res.num_io_tiles);
    }

    const int64 input_index = tensor_target.input_index;
    const Shape& input_shape = tensor_target.tgt->operand(input_index)->shape();
    return AddHostCopyTensor(graph, {debug_info}, input_shape);
  }
};

REGISTER_POPLAR_OP(InterTilesetCopy, InterTilesetCopyOp);

}  // namespace

}  // namespace poplarplugin
}  // namespace xla
