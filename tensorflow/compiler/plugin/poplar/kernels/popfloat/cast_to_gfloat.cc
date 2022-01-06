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

#include "tensorflow/compiler/plugin/poplar/driver/poplar_platform.h"
#include "tensorflow/compiler/plugin/poplar/driver/trace.pb.h"
#include "tensorflow/compiler/plugin/poplar/driver/xla_ipu_common.h"
#include "tensorflow/compiler/plugin/poplar/kernels/custom_kernels_util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/ipu_kernels_common.h"
#include "tensorflow/compiler/plugin/poplar/kernels/popfloat/gfloat_config_utils.pb.h"

#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/human_readable_json.h"
#include "tensorflow/core/util/stream_executor_util.h"
#include "tensorflow/core/util/tensor_format.h"

#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/literal_util.h"

#include "absl/container/flat_hash_set.h"

using namespace xla::poplarplugin;

namespace tensorflow {

class PopfloatCastNativeToGfloatOp : public XlaOpKernel, IpuOpKernel {
 public:
  explicit PopfloatCastNativeToGfloatOp(OpKernelConstruction* ctx)
      : XlaOpKernel(ctx), IpuOpKernel() {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("in_type", &in_type_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("out_type", &out_type_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("calc_type", &calc_type_));
    std::string gfloat_format_str_;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("gfloat_format", &gfloat_format_str_));
    bool en_nanoo_;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("en_nanoo", &en_nanoo_));
    std::string sr_density_str_;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("sr_density", &sr_density_str_));
    int32 sr_bits_;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("sr_bits", &sr_bits_));
    float sr_norm_offset_;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("sr_norm_offset", &sr_norm_offset_));
    float sr_norm_scale_;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("sr_norm_scale", &sr_norm_scale_));
    float sr_norm_min_;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("sr_norm_min", &sr_norm_min_));
    float sr_norm_max_;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("sr_norm_max", &sr_norm_max_));
    float sr_prob_;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("sr_prob", &sr_prob_));
    std::string round_mode_str_;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("round_mode", &round_mode_str_));

    FPConfig::RoundMode round_mode_;
    FPConfig_RoundMode_Parse(round_mode_str_, &round_mode_);

    auto cast_op_fp_config = cast_op_config_.mutable_fp_config();
    cast_op_fp_config->set_enable_nanoo(en_nanoo_);
    cast_op_fp_config->set_round_mode(round_mode_);

    SRConfig::Density sr_density_{};
    SRConfig_Density_Parse(sr_density_str_, &sr_density_);

    auto cast_op_sr_config = cast_op_config_.mutable_sr_config();
    cast_op_sr_config->set_sr_density(sr_density_);
    cast_op_sr_config->set_sr_bits(sr_bits_);
    cast_op_sr_config->set_sr_norm_offset(sr_norm_offset_);
    cast_op_sr_config->set_sr_norm_scale(sr_norm_scale_);
    cast_op_sr_config->set_sr_norm_min(sr_norm_min_);
    cast_op_sr_config->set_sr_norm_max(sr_norm_max_);
    cast_op_sr_config->set_sr_bernoulli_prob(sr_prob_);

    GFConfig::GfloatFormat gfloat_format_;
    GFConfig_GfloatFormat_Parse(gfloat_format_str_, &gfloat_format_);

    auto cast_op_gf_config = cast_op_config_.mutable_gf_config();
    cast_op_gf_config->set_gfloat_format(gfloat_format_);

    std::string cast_op_config_str;
    ProtoToHumanReadableJson(cast_op_config_, &cast_op_config_str, true);
    attribute_map_.AddAttribute("cast_op_config", cast_op_config_str);
    attribute_map_.AddAttribute("in_type", in_type_);
    attribute_map_.AddAttribute("out_type", out_type_);
    attribute_map_.AddAttribute("calc_type", calc_type_);
  }

 public:
  ~PopfloatCastNativeToGfloatOp() override{};

  void Compile(XlaOpKernelContext* ctx) override {
    xla::XlaOp input = ctx->Input(0);
    xla::XlaOp param = ctx->Input(1);

    // Get the input shape
    xla::PrimitiveType output_type;
    OP_REQUIRES_OK(ctx, DataTypeToPrimitiveType(out_type_, &output_type));
    xla::Shape output_shape =
        TensorShapeToXLAShape(output_type, ctx->InputShape(0));

    xla::XlaBuilder& b = *ctx->builder();

    std::vector<xla::XlaOp> args = {input, param};
    xla::XlaOp output =
        xla::CustomCall(&b, PoplarOp_Name(PoplarOp::CastNativeToGfloat), args,
                        output_shape, attribute_map_.Serialise());

    ctx->SetOutput(0, output);
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(PopfloatCastNativeToGfloatOp);

  PopfloatCastConfig cast_op_config_;

  tensorflow::DataType in_type_;
  tensorflow::DataType out_type_;
  tensorflow::DataType calc_type_;
};
REGISTER_IPU_OP("CastNativeToGfloat", PopfloatCastNativeToGfloatOp);

class PopfloatCastGfloatToNativeOp : public XlaOpKernel, IpuOpKernel {
 public:
  explicit PopfloatCastGfloatToNativeOp(OpKernelConstruction* ctx)
      : XlaOpKernel(ctx), IpuOpKernel() {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("in_type", &in_type_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("out_type", &out_type_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("calc_type", &calc_type_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("gfloat_format", &gfloat_format_));

    attribute_map_.AddAttribute("in_type", in_type_);
    attribute_map_.AddAttribute("out_type", out_type_);
    attribute_map_.AddAttribute("calc_type", calc_type_);
    attribute_map_.AddAttribute("gfloat_format", gfloat_format_);
  }

 public:
  ~PopfloatCastGfloatToNativeOp() override{};

  void Compile(XlaOpKernelContext* ctx) override {
    xla::XlaOp input = ctx->Input(0);
    xla::XlaOp param = ctx->Input(1);

    // Get the input shape
    xla::PrimitiveType output_type;
    OP_REQUIRES_OK(ctx, DataTypeToPrimitiveType(out_type_, &output_type));
    xla::Shape output_shape =
        TensorShapeToXLAShape(output_type, ctx->InputShape(0));

    xla::XlaBuilder& b = *ctx->builder();

    std::vector<xla::XlaOp> args = {input, param};
    xla::XlaOp output =
        xla::CustomCall(&b, PoplarOp_Name(PoplarOp::CastGfloatToNative), args,
                        output_shape, attribute_map_.Serialise());

    ctx->SetOutput(0, output);
  }

 private:
  std::string gfloat_format_;
  tensorflow::DataType in_type_;
  tensorflow::DataType out_type_;
  tensorflow::DataType calc_type_;

  TF_DISALLOW_COPY_AND_ASSIGN(PopfloatCastGfloatToNativeOp);
};
REGISTER_IPU_OP("CastGfloatToNative", PopfloatCastGfloatToNativeOp);

class PopfloatGfloatParams : public XlaOpKernel, IpuOpKernel {
 public:
  explicit PopfloatGfloatParams(OpKernelConstruction* ctx)
      : XlaOpKernel(ctx), IpuOpKernel() {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("mantissa", &mantissa_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("exponent", &exponent_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("bias", &bias_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("en_denorm", &en_denorm_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("en_inf", &en_inf_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("calc_type", &calc_type_));

    attribute_map_.AddAttribute("mantissa", mantissa_);
    attribute_map_.AddAttribute("exponent", exponent_);
    attribute_map_.AddAttribute("bias", bias_);
    attribute_map_.AddAttribute("en_denorm", en_denorm_);
    attribute_map_.AddAttribute("en_inf", en_inf_);
    attribute_map_.AddAttribute("calc_type", calc_type_);
  }

 public:
  ~PopfloatGfloatParams() override{};

  void Compile(XlaOpKernelContext* ctx) override {
    xla::PrimitiveType param_type;
    OP_REQUIRES_OK(ctx, DataTypeToPrimitiveType(tensorflow::DataType::DT_INT32,
                                                &param_type));

    TensorShape paramShape;
    OP_REQUIRES_OK(ctx, ctx->ConstantInputAsShape(0, &paramShape));

    xla::Shape param_shape = TensorShapeToXLAShape(param_type, paramShape);
    xla::XlaBuilder& b = *ctx->builder();

    std::vector<xla::XlaOp> args = {};
    xla::XlaOp output =
        xla::CustomCall(&b, PoplarOp_Name(PoplarOp::CalcGfloatParams), args,
                        param_shape, attribute_map_.Serialise());

    ctx->SetOutput(0, output);
  }

 private:
  int32 mantissa_;
  int32 exponent_;
  int32 bias_;
  bool en_denorm_;
  bool en_inf_;
  tensorflow::DataType calc_type_;

  TF_DISALLOW_COPY_AND_ASSIGN(PopfloatGfloatParams);
};
REGISTER_IPU_OP("CalcGfloatParams", PopfloatGfloatParams);

}  // namespace tensorflow
