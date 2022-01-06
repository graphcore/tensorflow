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

// XLA implementations of Categorical op for IPU

#include "tensorflow/compiler/plugin/poplar/driver/xla_ipu_common.h"
#include "tensorflow/compiler/plugin/poplar/kernels/custom_kernels_util.h"
#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/lib/arithmetic.h"
#include "tensorflow/compiler/xla/client/lib/constants.h"
#include "tensorflow/compiler/xla/client/lib/prng.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.pb.h"

namespace tensorflow {
namespace {

class IpuCategoricalOp : public XlaOpKernel {
 public:
  explicit IpuCategoricalOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {}

  void Compile(XlaOpKernelContext* ctx) override {
    // Get the logits
    const xla::XlaOp& logits = ctx->Input(0);
    TensorShape logits_shape = ctx->InputShape(0);
    int64 num_samples;
    OP_REQUIRES_OK(ctx, ctx->ConstantInputAsIntScalar(1, &num_samples));
    OP_REQUIRES(ctx, TensorShapeUtils::IsMatrix(logits_shape),
                errors::InvalidArgument("logits should be a matrix, got shape ",
                                        logits_shape.DebugString()));
    OP_REQUIRES(ctx, num_samples >= 0,
                errors::InvalidArgument(
                    "num_samples should be nonnegative, got ", num_samples));

    for (int i = 0; i < 2; i++) {
      const int64 dim = logits_shape.dim_size(i);
      OP_REQUIRES(
          ctx, static_cast<int>(dim) == dim,
          errors::InvalidArgument("logits.shape = ", logits_shape.DebugString(),
                                  " too large for int"));
    }

    const int64 batch_size = logits_shape.dim_size(0);
    const int64 num_classes = logits_shape.dim_size(1);

    xla::Shape uniform_shape;
    int class_dimension;
    if (num_samples != 1) {
      std::array<int64, 3> uniform_shape_array = {
          {batch_size, num_samples, num_classes}};
      xla::PrimitiveType uniform_xla_type;
      OP_REQUIRES_OK(ctx,
                     DataTypeToPrimitiveType(input_type(0), &uniform_xla_type));
      uniform_shape =
          xla::ShapeUtil::MakeShape(uniform_xla_type, uniform_shape_array);
      class_dimension = 2;
    } else {
      // Have a special case for when we only need one sample, because
      // dimensions may be padded on architectures with tiled memory layouts, so
      // if the num_classes or batch size is large then this can lead to
      // expensive wasted memory.
      std::array<int64, 2> uniform_shape_array = {{batch_size, num_classes}};
      xla::PrimitiveType uniform_xla_type;
      OP_REQUIRES_OK(ctx,
                     DataTypeToPrimitiveType(input_type(0), &uniform_xla_type));
      uniform_shape =
          xla::ShapeUtil::MakeShape(uniform_xla_type, uniform_shape_array);
      class_dimension = 1;
    }
    xla::PrimitiveType type;
    OP_REQUIRES_OK(ctx, DataTypeToPrimitiveType(input_type(0), &type));

    xla::StatusOr<xla::XlaOp> log_uniforms_or_error =
        GetLogUniforms(uniform_shape, type, ctx);

    OP_REQUIRES_OK(ctx, log_uniforms_or_error.status());

    xla::XlaOp log_uniforms = std::move(log_uniforms_or_error.ValueOrDie());

    auto softmax_entries =
        xla::Sub(logits, log_uniforms,
                 /*broadcast_dimensions=*/{0, class_dimension});

    xla::PrimitiveType xla_output_type;
    OP_REQUIRES_OK(ctx,
                   DataTypeToPrimitiveType(output_type(0), &xla_output_type));
    xla::XlaOp argmax;
    argmax = xla::ArgMaxTwoPass(softmax_entries, xla_output_type,
                                /*axis=*/class_dimension);

    if (num_samples == 1) {
      argmax = xla::Reshape(argmax, {batch_size, 1});
    }

    ctx->SetOutput(0, argmax);
  }

  virtual xla::StatusOr<xla::XlaOp> GetLogUniforms(xla::Shape uniform_shape,
                                                   xla::PrimitiveType type,
                                                   XlaOpKernelContext* ctx) {
    xla::XlaBuilder* builder = ctx->builder();

    auto uniforms = xla::RngUniform(
        xla::MinPositiveNormalValue(builder, type),
        xla::One(builder, uniform_shape.element_type()), uniform_shape);
    return xla::Log(-xla::Log(uniforms));
  }

 private:
  bool is_gpu_;
  TF_DISALLOW_COPY_AND_ASSIGN(IpuCategoricalOp);
};

REGISTER_XLA_OP(Name("Multinomial")
                    .Device(DEVICE_IPU_XLA_JIT)
                    .CompileTimeConstantInput("num_samples"),
                IpuCategoricalOp);

template <typename T>
struct FillAttributeMap {};

template <>
struct FillAttributeMap<float> {
  void operator()(
      xla::poplarplugin::IPUCustomKernelsUtil::AttributeMap& attribute_map) {
    attribute_map.AddAttribute("min_val", std::numeric_limits<float>::min());
    attribute_map.AddAttribute("max_val", std::nexttoward(1.0f, 0.0f));
  }
};

template <>
struct FillAttributeMap<Eigen::half> {
  void operator()(
      xla::poplarplugin::IPUCustomKernelsUtil::AttributeMap& attribute_map) {
    const Eigen::half slightly_more_than_zero{0.0000699162483215332};
    const Eigen::half smallest_value_less_than_one{0.99951171875};

    // Generate random numbers within the range of MinVal to MaxVal instead of 0
    // to 1.
    attribute_map.AddAttribute("min_val",
                               static_cast<float>(slightly_more_than_zero));
    attribute_map.AddAttribute(
        "max_val", static_cast<float>(smallest_value_less_than_one));
  }
};

class IpuStatelessCategoricalOp : public IpuCategoricalOp {
 public:
  explicit IpuStatelessCategoricalOp(OpKernelConstruction* ctx)
      : IpuCategoricalOp(ctx),
        device_type_string_(ctx->device_type().type_string()) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("T", &dtype_));
  }

  xla::StatusOr<xla::XlaOp> GetLogUniforms(xla::Shape uniform_shape,
                                           xla::PrimitiveType type,
                                           XlaOpKernelContext* ctx) override {
    xla::XlaOp seed = ctx->Input(2);

    xla::poplarplugin::IPUCustomKernelsUtil::AttributeMap attribute_map;

    switch (type) {
      case xla::PrimitiveType::F32: {
        FillAttributeMap<float>{}(attribute_map);
        break;
      }
      case xla::PrimitiveType::BF16: {
        FillAttributeMap<Eigen::half>{}(attribute_map);
        break;
      }
      default: {
        const Status status{error::INTERNAL,
                            "UNREACHABLE: Type should be checked by "
                            "the TypeConstraint on the operation"};
        return status;
      }
    }

    xla::XlaOp output = xla::CustomCall(
        ctx->builder(), PoplarOp_Name(PoplarOp::StatelessRandomUniform), {seed},
        uniform_shape, attribute_map.Serialise());

    // We want a number in (0, 1) rather than [0, 1) or (0, 1]:
    // * log(-log(0)) is ∞.
    // * log(-log(1)) is -∞.
    return xla::Log(-xla::Log(output));
  }

  void Compile(XlaOpKernelContext* ctx) override {
    TensorShape seed_shape = ctx->InputShape(2);
    OP_REQUIRES(ctx, seed_shape.dims() == 1 && seed_shape.dim_size(0) == 2,
                errors::InvalidArgument("seed must have shape [2], not ",
                                        seed_shape.DebugString()));
    IpuCategoricalOp::Compile(ctx);
  }

 private:
  DataType dtype_;
  string device_type_string_;

  TF_DISALLOW_COPY_AND_ASSIGN(IpuStatelessCategoricalOp);
};

REGISTER_XLA_OP(Name("StatelessMultinomial")
                    .Device(DEVICE_IPU_XLA_JIT)
                    .CompileTimeConstantInput("num_samples")
                    .TypeConstraint("T", {DT_FLOAT, DT_BFLOAT16})
                    .TypeConstraint("Tseed", DT_INT32),
                IpuStatelessCategoricalOp);

}  // anonymous namespace
}  // namespace tensorflow
