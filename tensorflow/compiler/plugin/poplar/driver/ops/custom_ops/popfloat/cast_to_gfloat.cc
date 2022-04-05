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

#include "tensorflow/compiler/plugin/poplar/driver/ops/custom_ops/popfloat/gfloat_ops_utils.h"
#include "tensorflow/compiler/plugin/poplar/driver/ops/custom_ops/poplar_ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/debug_info.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/custom_kernels_util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/popfloat/gfloat_config_utils.pb.h"

#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/cast_to_gfloat_hlo.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"

#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/core/errors.h"

#include "tensorflow/compiler/tf2xla/type_util.h"

#include "tensorflow/core/platform/human_readable_json.h"

#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"

#include <popfloat/experimental/CastToGfloat.hpp>
#include <poplar/DebugContext.hpp>

#include <string>

namespace xla {
namespace poplarplugin {
namespace {

class CalcGfloatParamsOp : public PoplarOpDef {
  StatusOr<poplar::program::Sequence> Creator(
      DriverGraph& graph, CompilerResources& res, const HloInstruction* inst,
      const xla::Shape& output_shape, TensorMap& tensor_map,
      const poplar::DebugContext& debug_context) override {
    PoplarOpDefDebugInfo debug_info(debug_context, "CalcGfloatParamsOp");

    const HloGfloatParamsInstruction* param_inst =
        Cast<HloGfloatParamsInstruction>(inst);

    auto tf_calc_type = param_inst->CalculationType();

    xla::PrimitiveType calc_type_;
    TF_RETURN_IF_ERROR(DataTypeToPrimitiveType(tf_calc_type, &calc_type_));

    poplar::Type calc_type;
    TF_ASSIGN_OR_RETURN(calc_type, PoplarDataType(calc_type_));

    auto gf_format_cfg = popfloat::experimental::GfloatCast::FormatConfig(
        param_inst->NumberMantissaBits(), param_inst->NumberExponentBits(),
        param_inst->ExponentBias(), param_inst->IsDenormEnabled(),
        param_inst->InfAndNansEnabled(), calc_type);

    auto gf_packed_params = gf_format_cfg.getPackedFloatParameters();
    return CreatePoplibsGfloatParams(res, inst, output_shape, tensor_map,
                                     calc_type, gf_packed_params, {debug_info});
  }
};
REGISTER_POPLAR_OP(CalcGfloatParams, CalcGfloatParamsOp);

class CastNativeToGfloatOp : public PoplarOpDef {
  StatusOr<poplar::program::Sequence> Creator(
      DriverGraph& graph, CompilerResources& res, const HloInstruction* inst,
      const xla::Shape& output_shape, TensorMap& tensor_map,
      const poplar::DebugContext& debug_context) override {
    PoplarOpDefDebugInfo debug_info(debug_context, "CastNativeToGfloatOp");

    const HloCastNativeToGfloatInstruction* cast_inst =
        Cast<HloCastNativeToGfloatInstruction>(inst);

    auto tf_in_type = cast_inst->InputType();

    xla::PrimitiveType in_type_;
    TF_RETURN_IF_ERROR(DataTypeToPrimitiveType(tf_in_type, &in_type_));

    poplar::Type in_type;
    TF_ASSIGN_OR_RETURN(in_type, PoplarDataType(in_type_));

    auto tf_out_type = cast_inst->OutputType();

    xla::PrimitiveType out_type_;
    TF_RETURN_IF_ERROR(DataTypeToPrimitiveType(tf_out_type, &out_type_));

    poplar::Type out_type;
    TF_ASSIGN_OR_RETURN(out_type, PoplarDataType(out_type_));

    auto tf_calc_type = cast_inst->CalculationType();

    xla::PrimitiveType calc_type_;
    TF_RETURN_IF_ERROR(DataTypeToPrimitiveType(tf_calc_type, &calc_type_));

    poplar::Type calc_type;
    TF_ASSIGN_OR_RETURN(calc_type, PoplarDataType(calc_type_));

    PopfloatCastConfig gf_cast_config;
    tensorflow::HumanReadableJsonToProto(cast_inst->CastOpConfig(),
                                         &gf_cast_config);

    auto cast_op_cfg = gfloatutils::CreateCastNativeToGfloatConfig(
        gf_cast_config, calc_type, out_type);
    return CreatePoplibsCastNativeToGfloat(res, inst, output_shape, tensor_map,
                                           cast_op_cfg, {debug_info});
  }
};
REGISTER_POPLAR_OP(CastNativeToGfloat, CastNativeToGfloatOp);

class CastGfloatToNativeOp : public PoplarOpDef {
  StatusOr<poplar::program::Sequence> Creator(
      DriverGraph& graph, CompilerResources& res, const HloInstruction* inst,
      const xla::Shape& output_shape, TensorMap& tensor_map,
      const poplar::DebugContext& debug_context) override {
    PoplarOpDefDebugInfo debug_info(debug_context, "CastGfloatToNativeOp");

    const HloCastGfloatToNativeInstruction* cast_inst =
        Cast<HloCastGfloatToNativeInstruction>(inst);

    auto gfloat_format =
        gfloatutils::GetPopfloatFormatType(cast_inst->GfloatFormat());

    auto tf_calc_type = cast_inst->CalculationType();

    xla::PrimitiveType calc_type_;
    TF_RETURN_IF_ERROR(DataTypeToPrimitiveType(tf_calc_type, &calc_type_));

    poplar::Type calc_type;
    TF_ASSIGN_OR_RETURN(calc_type, PoplarDataType(calc_type_));

    auto tf_out_type = cast_inst->OutputType();

    xla::PrimitiveType out_type_;
    TF_RETURN_IF_ERROR(DataTypeToPrimitiveType(tf_out_type, &out_type_));

    poplar::Type out_type;
    TF_ASSIGN_OR_RETURN(out_type, PoplarDataType(out_type_));

    auto cast_op_cfg =
        popfloat::experimental::GfloatCast::CastConfig::createCastGFToNative(
            gfloat_format, calc_type, out_type);

    return CreatePoplibsCastGfloatToNative(res, inst, output_shape, tensor_map,
                                           cast_op_cfg, {debug_info});
  }
};
REGISTER_POPLAR_OP(CastGfloatToNative, CastGfloatToNativeOp);

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
