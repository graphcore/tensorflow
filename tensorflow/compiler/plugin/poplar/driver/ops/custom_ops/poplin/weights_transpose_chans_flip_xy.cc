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

#include "tensorflow/compiler/plugin/poplar/driver/ops/custom_ops/poplar_ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/conv_poplar_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/weights_transpose_chans_flip_xy.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/debug_info.h"

#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"

#include "absl/strings/str_cat.h"

namespace xla {
namespace poplarplugin {
namespace {
StatusOr<DriverTensor> AddConvWeightsTransposeChansFlipXY(
    DriverGraph& graph, const poplar::DebugNameAndId& debug_name_and_id,
    const HloInstruction* inst, CompilerResources& resources) {
  const HloWeightsTransposeChansFlipXYInstruction* weights_transpose_inst =
      Cast<HloWeightsTransposeChansFlipXYInstruction>(inst);

  const ConvolutionDimensionNumbers& conv_dimension_numbers =
      weights_transpose_inst->convolution_dimension_numbers();

  TF_ASSIGN_OR_RETURN(
      poplin::ConvParams conv_params,
      GetConvolutionParametersForWeightsTranspose(weights_transpose_inst));

  TF_ASSIGN_OR_RETURN(poplar::OptionFlags opts,
                      GetConvolutionOptionsForInst(inst, resources));

  DriverTensor out_weights = DriverTensor(
      poplin::createWeights(
          graph, conv_params,
          {debug_name_and_id, "createWeights_TransposeChansFlipXY"}, opts,
          &resources.planning_cache),
      graph);

  out_weights = RemoveGroupsDimensionFromWeights(conv_params, out_weights);

  out_weights = ShuffleConvolutionWeightsToTensorflow(conv_dimension_numbers,
                                                      out_weights);

  return out_weights;
}

class WeightsTransposeChansFlipXYOp : public PoplarOpDef {
  StatusOr<DriverProgramSequence> Creator(
      DriverGraph& graph, CompilerResources& res, const HloInstruction* inst,
      const xla::Shape& output_shape, TensorMap& tensor_map,
      const poplar::DebugContext& debug_context) override {
    PoplarOpDefDebugInfo debug_info(debug_context,
                                    "WeightsTransposeChansFlipXYOp");
    DriverProgramSequence seq(graph, debug_info);

    TF_ASSIGN_OR_RETURN(
        poplar::Tensor in_weights,
        FindInstructionInput(tensor_map, res, inst, 0, seq, debug_info, false));

    const HloWeightsTransposeChansFlipXYInstruction* weights_transpose_inst =
        Cast<HloWeightsTransposeChansFlipXYInstruction>(inst);

    const ConvolutionDimensionNumbers& conv_dimension_numbers =
        weights_transpose_inst->convolution_dimension_numbers();

    in_weights = ShuffleConvolutionWeightsToPoplar(
        conv_dimension_numbers, in_weights, /* swap_features= */ true);

    TF_ASSIGN_OR_RETURN(
        poplin::ConvParams conv_params,
        GetConvolutionParametersForWeightsTranspose(weights_transpose_inst));

    TF_ASSIGN_OR_RETURN(poplar::OptionFlags opts,
                        GetConvolutionOptionsForInst(inst, res));

    in_weights = AddGroupsDimensionToWeights(conv_params, in_weights,
                                             /* swap_features= */ true);

    DriverTensor out_weights = DriverTensor(
        poplin::createWeights(graph, conv_params, {debug_info, "CreateWeights"},
                              opts, &res.planning_cache),
        graph);

    poplin::weightsTransposeChansFlipXY(
        graph, in_weights, out_weights, seq,
        {debug_info, "WeightsTransposeChansFlipXY"});

    out_weights = RemoveGroupsDimensionFromWeights(conv_params, out_weights);

    out_weights = ShuffleConvolutionWeightsToTensorflow(
        conv_dimension_numbers, out_weights, /* swap_features = */ true);

    TF_CHECK_OK(
        AddOutputTensor(tensor_map, inst, 0, DriverTensor(out_weights, graph)));

    return seq;
  }

  StatusOr<DriverTensor> Allocator(
      DriverGraph& graph, CompilerResources& res, const std::string& name,
      const TensorTarget& tensor_target, const TensorMap& tensor_map,
      const poplar::DebugContext& debug_context) override {
    PoplarOpDefDebugInfo debug_info(debug_context,
                                    "WeightsTransposeChansFlipXYOp");
    const int64_t input_index = tensor_target.input_index;
    const HloInstruction* inst = tensor_target.tgt;

    DriverTensor out;
    switch (input_index) {
      case 0: {
        TF_ASSIGN_OR_RETURN(out, AddConvWeightsTransposeChansFlipXY(
                                     graph, {debug_info}, inst, res));
        break;
      }
      default:
        return xla::FailedPrecondition(
            "Input index %d of weights transpose chans flipxy op shouldn't be "
            "allocating",
            input_index);
    }

    return out;
  }
};

REGISTER_POPLAR_OP(WeightsTransposeChansFlipXY, WeightsTransposeChansFlipXYOp);

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
