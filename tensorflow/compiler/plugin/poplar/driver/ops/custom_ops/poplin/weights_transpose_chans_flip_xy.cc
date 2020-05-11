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

#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/weights_transpose_chans_flip_xy.h"
#include "tensorflow/compiler/plugin/poplar/driver/ops/custom_ops/poplar_ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/conv_poplar_util.h"

#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"

namespace xla {
namespace poplarplugin {
namespace {
class WeightsTransposeChansFlipXYOp : public PoplarOpDef {
  StatusOr<poplar::program::Program> Creator(poplar::Graph& graph,
                                             CompilerResources& res,
                                             const HloInstruction* inst,
                                             const xla::Shape& output_shape,
                                             TensorMap& tensor_map) override {
    poplar::program::Sequence seq;

    TF_ASSIGN_OR_RETURN(poplar::Tensor in_weights,
                        FindInstructionInput(tensor_map, res, inst, 0, seq));

    const HloWeightsTransposeChansFlipXYInstruction* weights_transpose_inst =
        Cast<HloWeightsTransposeChansFlipXYInstruction>(inst);

    const ConvolutionDimensionNumbers& conv_dimension_numbers =
        weights_transpose_inst->convolution_dimension_numbers();

    in_weights = ShuffleConvolutionWeightsToPoplar(conv_dimension_numbers,
                                                   in_weights, true);

    const std::vector<size_t>& conv_input_shape =
        weights_transpose_inst->ConvInputShape();

    const std::vector<size_t>& conv_output_shape =
        weights_transpose_inst->ConvOutputShape();

    TF_ASSIGN_OR_RETURN(
        poplin::ConvParams conv_params,
        GetConvolutionParametersForWeightsTranspose(
            weights_transpose_inst, conv_input_shape, conv_output_shape));

    in_weights = AddGroupsDimensionToWeights(conv_params, in_weights, true);

    const std::string debug_prefix = GetDebugName(inst);
    auto func = [&graph, &res, conv_dimension_numbers, conv_params,
                 debug_prefix](std::vector<poplar::Tensor>& args,
                               poplar::program::Sequence& prog) {
      poplar::Tensor in_weights_f = args[0];

      poplar::Tensor out_weights_f = poplin::createWeights(
          graph, conv_params, absl::StrCat(debug_prefix, "/CreateWeights"));

      poplin::weightsTransposeChansFlipXY(
          graph, in_weights_f, out_weights_f, prog,
          absl::StrCat(debug_prefix, "/WeightsTransposeChansFlipXY"));

      out_weights_f =
          RemoveGroupsDimensionFromWeights(conv_params, out_weights_f);

      out_weights_f = ShuffleConvolutionWeightsToTensorflow(
          conv_dimension_numbers, out_weights_f);

      args[1] = out_weights_f;
    };

    poplar::Tensor out_weights;
    std::vector<poplar::Tensor> args = {in_weights, out_weights};

    poputil::graphfn::Signature signature = {
        poputil::graphfn::input(in_weights, "in_weights"),
        poputil::graphfn::created("out_weights")};

    TF_RETURN_IF_ERROR(res.graph_cache.ExecuteCached(
        inst, graph, res, seq, func, signature, args,
        weights_transpose_inst->AllocatingIndices(),
        weights_transpose_inst->LayoutDependencies()));

    out_weights = args[1];

    TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, out_weights));

    return seq;
  }
};

REGISTER_POPLAR_OP(WeightsTransposeChansFlipXY, WeightsTransposeChansFlipXYOp);

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
