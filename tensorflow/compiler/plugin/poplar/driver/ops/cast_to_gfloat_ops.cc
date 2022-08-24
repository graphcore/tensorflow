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

#include <algorithm>
#include <popfloat/experimental/CastToGfloat.hpp>

#include "tensorflow/compiler/plugin/poplar/driver/compiler_resources.h"
#include "tensorflow/compiler/plugin/poplar/driver/ops/ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/cast_to_gfloat_hlo.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"

namespace xla {
namespace poplarplugin {
StatusOr<DriverProgramSequence> CreatePoplibsGfloatParams(
    CompilerResources& res, const HloInstruction* inst,
    const xla::Shape& output_shape, TensorMap& tensor_map,
    poplar::Type gf_calc_type, const unsigned gf_packed_cfg,
    const poplar::DebugNameAndId& debug_name_and_id) {
  VLOG(1) << "Processing GfloatParams.";

  auto& graph = GetGraph(res, inst);

  DriverProgramSequence seq(debug_name_and_id);

  poplar::Tensor gf_param =
      popfloat::experimental::GfloatCast::createCastOpParamsTensor(
          graph, seq, gf_calc_type, gf_packed_cfg, {debug_name_and_id});

  TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, DriverTensor(gf_param)));
  return seq;
}

StatusOr<DriverProgramSequence> CreatePoplibsCastNativeToGfloat(
    CompilerResources& res, const HloInstruction* inst,
    const xla::Shape& output_shape, TensorMap& tensor_map,
    popfloat::experimental::GfloatCast::CastConfig& gf_cast_cfg,
    const poplar::DebugNameAndId& debug_name_and_id) {
  const HloCastNativeToGfloatInstruction* cast_inst =
      Cast<HloCastNativeToGfloatInstruction>(inst);

  VLOG(1) << "Processing CastNativeToGfloat.";

  auto& graph = GetGraph(res, inst);

  DriverProgramSequence seq(debug_name_and_id);

  auto tf_in_type = cast_inst->InputType();

  xla::PrimitiveType in_type_;
  TF_RETURN_IF_ERROR(DataTypeToPrimitiveType(tf_in_type, &in_type_));

  poplar::Type in_type;
  TF_ASSIGN_OR_RETURN(in_type, PoplarDataType(in_type_));

  TF_ASSIGN_OR_RETURN(
      poplar::Tensor gf_params,
      FindInstructionInput(tensor_map, res, inst, 1, seq, debug_name_and_id));

  if (cast_inst->GetUseDescriptions().size() &&
      gf_cast_cfg.inPlaceOp(in_type)) {
    TF_ASSIGN_OR_RETURN(TensorVectors inputs,
                        FindInplaceOutputTensors(tensor_map, res, inst, seq,
                                                 debug_name_and_id));
    CHECK_EQ(inputs.size(), 1);
    CHECK_EQ(inputs[0].size(), 1);
    auto operand = inputs[0][0];

    popfloat::experimental::GfloatCast::castNativeToGfloatInPlace(
        graph, operand, gf_params, seq, gf_cast_cfg, {debug_name_and_id});

    TF_ASSIGN_OR_RETURN(operand, BroadcastTensor(operand, output_shape));

    TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, operand));
  } else {
    TF_ASSIGN_OR_RETURN(
        auto operand,
        FindInstructionInput(tensor_map, res, inst, 0, seq, debug_name_and_id));

    auto out = popfloat::experimental::GfloatCast::castNativeToGfloat(
        graph, operand, gf_params, seq, gf_cast_cfg, {debug_name_and_id});
    TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, DriverTensor(out)));
  }

  return seq;
}

StatusOr<DriverProgramSequence> CreatePoplibsCastGfloatToNative(
    CompilerResources& res, const HloInstruction* inst,
    const xla::Shape& output_shape, TensorMap& tensor_map,
    popfloat::experimental::GfloatCast::CastConfig& gf_cast_cfg,
    const poplar::DebugNameAndId& debug_name_and_id) {
  VLOG(1) << "Processing Unpack Gfloat.";

  auto& graph = GetGraph(res, inst);

  DriverProgramSequence seq(debug_name_and_id);

  TF_ASSIGN_OR_RETURN(
      poplar::Tensor operand,
      FindInstructionInput(tensor_map, res, inst, 0, seq, debug_name_and_id));
  TF_ASSIGN_OR_RETURN(
      poplar::Tensor params,
      FindInstructionInput(tensor_map, res, inst, 1, seq, debug_name_and_id));

  auto out = popfloat::experimental::GfloatCast::castGfloatToNative(
      graph, operand, params, seq, gf_cast_cfg, {debug_name_and_id});

  TF_CHECK_OK(AddOutputTensor(tensor_map, inst, 0, DriverTensor(out)));

  return seq;
}

}  // namespace poplarplugin
}  // namespace xla
