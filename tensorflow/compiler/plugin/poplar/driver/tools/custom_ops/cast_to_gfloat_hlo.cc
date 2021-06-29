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
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/cast_to_gfloat_hlo.h"

#include "tensorflow/compiler/plugin/poplar/driver/tools/hlo_poplar_buffer_util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/custom_kernels_util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/ops.pb.h"

#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/human_readable_json.h"

namespace xla {
namespace poplarplugin {

HloGfloatParamsInstruction::HloGfloatParamsInstruction(
    const Shape& shape, int32 mantissa, int32 exponent, int32 bias,
    bool en_denorm, bool en_inf, tensorflow::DataType calc_type)
    : HloPoplarInstruction(shape, {}, PoplarOp::CalcGfloatParams, mantissa,
                           exponent, bias, en_denorm, en_inf, calc_type),
      mantissa_(mantissa),
      exponent_(exponent),
      bias_(bias),
      en_denorm_(en_denorm),
      en_inf_(en_inf),
      calc_type_(calc_type) {}

int32 HloGfloatParamsInstruction::NumberMantissaBits() const {
  return mantissa_;
}

int32 HloGfloatParamsInstruction::NumberExponentBits() const {
  return exponent_;
}

int32 HloGfloatParamsInstruction::ExponentBias() const { return bias_; }

bool HloGfloatParamsInstruction::IsDenormEnabled() const { return en_denorm_; }

bool HloGfloatParamsInstruction::InfAndNansEnabled() const { return en_inf_; }

tensorflow::DataType HloGfloatParamsInstruction::CalculationType() const {
  return calc_type_;
}

std::vector<std::string>
HloGfloatParamsInstruction::ExtraPoplarAttributesToStringImpl(
    const HloPrintOptions& options) const {
  std::vector<std::string> attributes;
  attributes.push_back("mantissa=" + std::to_string(mantissa_));
  attributes.push_back("exponent=" + std::to_string(exponent_));
  attributes.push_back("bias=" + std::to_string(bias_));
  attributes.push_back("en_denorm=" + std::to_string(en_denorm_));
  attributes.push_back("en_inf=" + std::to_string(en_inf_));
  attributes.push_back("calc_type=" + tensorflow::DataTypeString(calc_type_));

  return attributes;
}

absl::flat_hash_set<int64> HloGfloatParamsInstruction::AllocatingIndices()
    const {
  return {};
}

bool HloGfloatParamsInstruction::AllocatingOutput() const { return false; }

absl::flat_hash_map<int64, int64>
HloGfloatParamsInstruction::LayoutDependencies() const {
  return {};
}

HloPoplarUseDescriptions HloGfloatParamsInstruction::GetUseDescriptions()
    const {
  return UseDescriptionsNoInputOutputAlias();
}

HloPoplarBufferDescriptions HloGfloatParamsInstruction::GetBufferDescriptions()
    const {
  return BufferDescriptionsAllocatesAllOutputs(this);
}

const FindConsumersExtensionResults HloGfloatParamsInstruction::FindConsumers(
    FindConsumersExtensionParams params) const {
  return FindConsumersExtensionResults::DoNotFindConsumers();
}

bool HloGfloatParamsInstruction::IsPopOpsElementwise() const { return false; }

std::unique_ptr<HloInstruction>
HloGfloatParamsInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext*) const {
  return absl::make_unique<HloGfloatParamsInstruction>(
      shape, NumberMantissaBits(), NumberExponentBits(), ExponentBias(),
      IsDenormEnabled(), InfAndNansEnabled(), CalculationType());
}

std::unique_ptr<HloInstruction> CreateGfloatParams(
    const Shape& shape, int32 mantissa, int32 exponent, int32 bias,
    bool en_denorm, bool en_inf, tensorflow::DataType calc_type) {
  return absl::make_unique<HloGfloatParamsInstruction>(
      shape, mantissa, exponent, bias, en_denorm, en_inf, calc_type);
}

HloCastNativeToGfloatInstruction::HloCastNativeToGfloatInstruction(
    const Shape& shape, HloInstruction* const operand,
    HloInstruction* const params, tensorflow::DataType in_type,
    tensorflow::DataType out_type, tensorflow::DataType calc_type,
    std::string cast_op_config)
    : HloPoplarInstruction(shape, {operand, params},
                           PoplarOp::CastNativeToGfloat, in_type, out_type,
                           calc_type, cast_op_config),
      cast_op_config_(cast_op_config),
      in_type_(in_type),
      out_type_(out_type),
      calc_type_(calc_type) {}

std::string HloCastNativeToGfloatInstruction::CastOpConfig() const {
  return cast_op_config_;
}

tensorflow::DataType HloCastNativeToGfloatInstruction::InputType() const {
  return in_type_;
}

tensorflow::DataType HloCastNativeToGfloatInstruction::OutputType() const {
  return out_type_;
}

tensorflow::DataType HloCastNativeToGfloatInstruction::CalculationType() const {
  return calc_type_;
}

std::vector<std::string>
HloCastNativeToGfloatInstruction::ExtraPoplarAttributesToStringImpl(
    const HloPrintOptions& options) const {
  std::vector<std::string> attributes;

  PopfloatCastConfig gf_cast_config;
  tensorflow::HumanReadableJsonToProto(CastOpConfig(), &gf_cast_config);

  attributes.push_back("in_type=" + tensorflow::DataTypeString(in_type_));

  attributes.push_back("out_type=" + tensorflow::DataTypeString(out_type_));

  attributes.push_back("calc_type=" + tensorflow::DataTypeString(calc_type_));

  bool nanoo_enabled = gf_cast_config.fp_config().enable_nanoo();
  attributes.push_back("en_nanoo=" + std::to_string(nanoo_enabled));

  FPConfig::RoundMode round_mode = gf_cast_config.fp_config().round_mode();
  std::string round_mode_str = FPConfig_RoundMode_Name(round_mode).c_str();
  attributes.push_back("round_mode=" + round_mode_str);

  SRConfig::Density sr_density = gf_cast_config.sr_config().sr_density();
  std::string sr_density_str = SRConfig_Density_Name(sr_density).c_str();
  attributes.push_back("sr_density=" + sr_density_str);

  int32 sr_bits = gf_cast_config.sr_config().sr_bits();
  attributes.push_back("sr_bits=" + sr_bits);

  float sr_norm_offset = gf_cast_config.sr_config().sr_norm_offset();
  attributes.push_back("sr_norm_offset=" + std::to_string(sr_norm_offset));

  float sr_norm_scale = gf_cast_config.sr_config().sr_norm_scale();
  attributes.push_back("sr_norm_scale=" + std::to_string(sr_norm_scale));

  float sr_norm_min = gf_cast_config.sr_config().sr_norm_min();
  attributes.push_back("sr_norm_min=" + std::to_string(sr_norm_min));

  float sr_norm_max = gf_cast_config.sr_config().sr_norm_max();
  attributes.push_back("sr_norm_max=" + std::to_string(sr_norm_max));

  float sr_bernoulli_probability =
      gf_cast_config.sr_config().sr_bernoulli_prob();
  attributes.push_back("sr_prob=" + std::to_string(sr_bernoulli_probability));

  GFConfig::GfloatFormat gfloat_format =
      gf_cast_config.gf_config().gfloat_format();
  std::string gfloat_format_str =
      GFConfig_GfloatFormat_Name(gfloat_format).c_str();
  attributes.push_back("gfloat_format=" + gfloat_format_str);

  return attributes;
}

absl::flat_hash_set<int64> HloCastNativeToGfloatInstruction::AllocatingIndices()
    const {
  return {};
}

bool HloCastNativeToGfloatInstruction::AllocatingOutput() const {
  return false;
}

absl::flat_hash_map<int64, int64>
HloCastNativeToGfloatInstruction::LayoutDependencies() const {
  return {};
}

HloPoplarUseDescriptions HloCastNativeToGfloatInstruction::GetUseDescriptions()
    const {
  if (in_type_ == out_type_) {
    return UseDescriptionsSimpleNoTuple0thOperandAliasing(this);
  } else {
    return UseDescriptionsNoInputOutputAlias();
  }
}

HloPoplarBufferDescriptions
HloCastNativeToGfloatInstruction::GetBufferDescriptions() const {
  return BufferDescriptionsAllocatesAllUnaliasedBuffers(this,
                                                        GetUseDescriptions());
}

const FindConsumersExtensionResults
HloCastNativeToGfloatInstruction::FindConsumers(
    FindConsumersExtensionParams params) const {
  return FindConsumersExtensionResults::DoNotFindConsumers();
}

bool HloCastNativeToGfloatInstruction::IsPopOpsElementwise() const {
  return false;
}

std::unique_ptr<HloInstruction>
HloCastNativeToGfloatInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext*) const {
  return absl::make_unique<HloCastNativeToGfloatInstruction>(
      shape, new_operands[0], new_operands[1], InputType(), OutputType(),
      CalculationType(), CastOpConfig());
}

std::unique_ptr<HloInstruction> CreateCastNativeToGfloat(
    const Shape& shape, HloInstruction* const operand,
    HloInstruction* const params, tensorflow::DataType in_type_,
    tensorflow::DataType out_type_, tensorflow::DataType calc_type_,
    std::string cast_op_config_) {
  return absl::make_unique<HloCastNativeToGfloatInstruction>(
      shape, operand, params, in_type_, out_type_, calc_type_, cast_op_config_);
}

HloCastGfloatToNativeInstruction::HloCastGfloatToNativeInstruction(
    const Shape& shape, HloInstruction* const operand,
    HloInstruction* const params, const tensorflow::DataType in_type,
    const tensorflow::DataType out_type, const tensorflow::DataType calc_type,
    GFConfig::GfloatFormat gfloat_format)
    : HloPoplarInstruction(shape, {operand, params},
                           PoplarOp::CastGfloatToNative, in_type, out_type,
                           calc_type, gfloat_format),
      in_type_(in_type),
      out_type_(out_type),
      calc_type_(calc_type),
      gfloat_format_(gfloat_format) {}

const tensorflow::DataType HloCastGfloatToNativeInstruction::InputType() const {
  return in_type_;
}

const tensorflow::DataType HloCastGfloatToNativeInstruction::OutputType()
    const {
  return out_type_;
}
const tensorflow::DataType HloCastGfloatToNativeInstruction::CalculationType()
    const {
  return calc_type_;
};

GFConfig::GfloatFormat HloCastGfloatToNativeInstruction::GfloatFormat() const {
  return gfloat_format_;
}

std::vector<std::string>
HloCastGfloatToNativeInstruction::ExtraPoplarAttributesToStringImpl(
    const HloPrintOptions& options) const {
  std::vector<std::string> attributes;
  attributes.push_back("in_type=" + tensorflow::DataTypeString(in_type_));
  attributes.push_back("out_type=" + tensorflow::DataTypeString(out_type_));
  attributes.push_back("calc_type=" + tensorflow::DataTypeString(calc_type_));
  std::string gfloat_format_str;
  gfloat_format_str = GFConfig_GfloatFormat_Name(GfloatFormat()).c_str();
  attributes.push_back("gfloat_format=" + gfloat_format_str);

  return attributes;
}

absl::flat_hash_set<int64> HloCastGfloatToNativeInstruction::AllocatingIndices()
    const {
  return {};
}

bool HloCastGfloatToNativeInstruction::AllocatingOutput() const {
  return false;
}

absl::flat_hash_map<int64, int64>
HloCastGfloatToNativeInstruction::LayoutDependencies() const {
  return {};
}

HloPoplarUseDescriptions HloCastGfloatToNativeInstruction::GetUseDescriptions()
    const {
  return UseDescriptionsNoInputOutputAlias();
}

HloPoplarBufferDescriptions
HloCastGfloatToNativeInstruction::GetBufferDescriptions() const {
  return BufferDescriptionsAllocatesAllOutputs(this);
}

const FindConsumersExtensionResults
HloCastGfloatToNativeInstruction::FindConsumers(
    FindConsumersExtensionParams params) const {
  return FindConsumersExtensionResults::DoNotFindConsumers();
}

bool HloCastGfloatToNativeInstruction::IsPopOpsElementwise() const {
  return true;
}
std::unique_ptr<HloInstruction>
HloCastGfloatToNativeInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext*) const {
  return absl::make_unique<HloCastGfloatToNativeInstruction>(
      shape, new_operands[0], new_operands[1], InputType(), OutputType(),
      CalculationType(), GfloatFormat());
}

std::unique_ptr<HloInstruction> CreateCastGfloatToNative(
    const Shape& shape, HloInstruction* const operand,
    HloInstruction* const params, const tensorflow::DataType in_type,
    const tensorflow::DataType out_type, const tensorflow::DataType calc_type,
    GFConfig::GfloatFormat gfloat_format) {
  return absl::make_unique<HloCastGfloatToNativeInstruction>(
      shape, operand, params, in_type, out_type, calc_type, gfloat_format);
}

namespace {
StatusOr<std::unique_ptr<HloInstruction>> HloGfloatParamsInstructionFactoryFunc(
    HloCustomCallInstruction* call) {
  auto attribute_map = IPUCustomKernelsUtil::AttributeMap(call);
  // Get the attribute values
  TF_ASSIGN_OR_RETURN(int32 mantissa,
                      attribute_map.GetAttributeAsInt("mantissa"));
  TF_ASSIGN_OR_RETURN(int32 exponent,
                      attribute_map.GetAttributeAsInt("exponent"));
  TF_ASSIGN_OR_RETURN(int32 bias, attribute_map.GetAttributeAsInt("bias"));
  TF_ASSIGN_OR_RETURN(bool en_denorm,
                      attribute_map.GetAttributeAsBool("en_denorm"));
  TF_ASSIGN_OR_RETURN(bool en_inf, attribute_map.GetAttributeAsBool("en_inf"));
  TF_ASSIGN_OR_RETURN(tensorflow::DataType calc_type,
                      attribute_map.GetAttributeAsTFDataType("calc_type"));
  return CreateGfloatParams(call->shape(), mantissa, exponent, bias, en_denorm,
                            (en_inf && (exponent > 0)), calc_type);
}

static HloPoplarInstructionFactory gfloat_params_factory(
    PoplarOp::CalcGfloatParams, HloGfloatParamsInstructionFactoryFunc);

StatusOr<std::unique_ptr<HloInstruction>>
HloCastNativeToGfloatInstructionFactoryFunc(HloCustomCallInstruction* call) {
  auto attribute_map = IPUCustomKernelsUtil::AttributeMap(call);
  // Get the attribute values
  TF_ASSIGN_OR_RETURN(tensorflow::DataType in_type,
                      attribute_map.GetAttributeAsTFDataType("in_type"));
  TF_ASSIGN_OR_RETURN(tensorflow::DataType out_type,
                      attribute_map.GetAttributeAsTFDataType("out_type"));
  TF_ASSIGN_OR_RETURN(tensorflow::DataType calc_type,
                      attribute_map.GetAttributeAsTFDataType("calc_type"));
  TF_ASSIGN_OR_RETURN(std::string cast_op_config,
                      attribute_map.GetAttributeAsString("cast_op_config"));

  return CreateCastNativeToGfloat(call->shape(), call->mutable_operand(0),
                                  call->mutable_operand(1), in_type, out_type,
                                  calc_type, cast_op_config);
}

static HloPoplarInstructionFactory cast_native_to_gfloat_factory(
    PoplarOp::CastNativeToGfloat, HloCastNativeToGfloatInstructionFactoryFunc);

StatusOr<std::unique_ptr<HloInstruction>>
HloCastGfloatToNativeInstructionFactoryFunc(HloCustomCallInstruction* call) {
  auto attribute_map = IPUCustomKernelsUtil::AttributeMap(call);
  // Get the attribute values
  TF_ASSIGN_OR_RETURN(tensorflow::DataType in_type,
                      attribute_map.GetAttributeAsTFDataType("in_type"));
  TF_ASSIGN_OR_RETURN(tensorflow::DataType out_type,
                      attribute_map.GetAttributeAsTFDataType("out_type"));
  TF_ASSIGN_OR_RETURN(tensorflow::DataType calc_type,
                      attribute_map.GetAttributeAsTFDataType("calc_type"));
  TF_ASSIGN_OR_RETURN(std::string gfloat_format_str,
                      attribute_map.GetAttributeAsString("gfloat_format"));

  GFConfig::GfloatFormat gfloat_format{};
  GFConfig_GfloatFormat_Parse(gfloat_format_str, &gfloat_format);

  return CreateCastGfloatToNative(call->shape(), call->mutable_operand(0),
                                  call->mutable_operand(1), in_type, out_type,
                                  calc_type, gfloat_format);
}

static HloPoplarInstructionFactory cast_gfloat_to_native_factory(
    PoplarOp::CastGfloatToNative, HloCastGfloatToNativeInstructionFactoryFunc);

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
