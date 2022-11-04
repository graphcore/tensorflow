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

#include "tensorflow/compiler/plugin/poplar/driver/xla_ipu_common.h"
#include "tensorflow/compiler/plugin/poplar/kernels/ipu_kernels_common.h"

#include "tensorflow/compiler/tf2xla/kernels/conv_op_helpers.h"
#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"

#include "tensorflow/compiler/xla/util.h"

namespace tensorflow {

class PoplinF8MatMulOp : public XlaOpKernel, IpuOpKernel {
 public:
  using XlaOpKernel::XlaOpKernel;

  explicit PoplinF8MatMulOp(OpKernelConstruction* ctx)
      : XlaOpKernel(ctx), IpuOpKernel() {}

  void Compile(XlaOpKernelContext* ctx) override {
    std::vector<xla::XlaOp> args = {ctx->Input(0), ctx->Input(1), ctx->Input(2),
                                    ctx->Input(3)};

    const TensorShape lhs_shape = ctx->InputShape(0);
    const TensorShape rhs_shape = ctx->InputShape(2);
    TensorShape matmul_shape_(
        {lhs_shape.dim_size(0), lhs_shape.dim_size(1), rhs_shape.dim_size(2)});

    xla::Shape matmul_shape;
    OP_REQUIRES_OK(ctx, TensorShapeToXLAShape(ctx->expected_output_dtype(0),
                                              matmul_shape_, &matmul_shape));
    xla::Shape metadata_shape =
        xla::ShapeUtil::MakeShape(xla::PrimitiveType::U8, {});

    xla::Shape output_shape =
        xla::ShapeUtil::MakeTupleShape({matmul_shape, metadata_shape});

    xla::XlaOp call_output =
        xla::CustomCall(ctx->builder(), PoplarOp_Name(PoplarOp::F8MatMul), args,
                        output_shape, "");
    xla::XlaOp matmul_out = xla::GetTupleElement(call_output, 0);
    xla::XlaOp metadata_out = xla::GetTupleElement(call_output, 1);
    ctx->SetOutput(0, matmul_out);
    ctx->SetOutput(1, metadata_out);
  }
};

REGISTER_IPU_OP("IpuF8Matmul", PoplinF8MatMulOp);

template <uint16_t D>
class PoplinF8ConvOp : public XlaOpKernel, IpuOpKernel {
 private:
  ConvOpAttrs _conv_attrs;

  TF_DISALLOW_COPY_AND_ASSIGN(PoplinF8ConvOp);

 public:
  using XlaOpKernel::XlaOpKernel;

  explicit PoplinF8ConvOp(OpKernelConstruction* ctx)
      : XlaOpKernel(ctx), IpuOpKernel() {
    std::vector<int32> strides;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("strides", &strides));
    attribute_map_.AddAttribute("strides", strides);

    std::string padding;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("padding", &padding));
    attribute_map_.AddAttribute("padding", padding);

    std::vector<int32> explicit_paddings;
    if (D == 2) {
      OP_REQUIRES_OK(ctx,
                     ctx->GetAttr("explicit_paddings", &explicit_paddings));
    }
    attribute_map_.AddAttribute("explicit_paddings", explicit_paddings);

    std::string data_format;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("data_format", &data_format));
    attribute_map_.AddAttribute("data_format", data_format);

    std::vector<int32> dilations;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("dilations", &dilations));
    attribute_map_.AddAttribute("dilations", dilations);

    auto conv_attrs = ConvOpAttrs::Create(D, false, ctx);
    OP_REQUIRES_OK(ctx, conv_attrs.status());
    _conv_attrs = conv_attrs.ValueOrDie();
  }

  void Compile(XlaOpKernelContext* ctx) override {
    // First generate a dummy convolution
    auto dummy_conv_op_sor =
        MakeXlaForwardConvOp(ctx->op_kernel().type_string(), ctx->Input(0),
                             ctx->Input(1), _conv_attrs);
    OP_REQUIRES_OK(ctx, dummy_conv_op_sor.status());
    auto dummy_conv_op = dummy_conv_op_sor.ValueOrDie();

    auto* dummy_builder = dummy_conv_op.builder();
    auto out_shape_sor = dummy_builder->GetShape(dummy_conv_op);
    OP_REQUIRES_OK(ctx, out_shape_sor.status());
    auto conv_out_shape = out_shape_sor.ValueOrDie();
    conv_out_shape.set_element_type(xla::PrimitiveType::F16);

    const auto metadata_shape =
        xla::ShapeUtil::MakeShape(xla::PrimitiveType::U8, {});

    auto out_shape =
        xla::ShapeUtil::MakeTupleShape({conv_out_shape, metadata_shape});

    std::vector<xla::XlaOp> args = {ctx->Input(0), ctx->Input(1), ctx->Input(2),
                                    ctx->Input(3)};

    PoplarOp op_type;
    switch (D) {
      case 2: {
        op_type = PoplarOp::F8Conv2D;
        break;
      }
      case 3: {
        op_type = PoplarOp::F8Conv3D;
        break;
      }
      default: {
        op_type = PoplarOp::Unknown;
        break;
      }
    }

    OP_REQUIRES(
        ctx, op_type != PoplarOp::Unknown,
        xla::InvalidArgument("Unsupported F8 Convolution Dimension ", D));

    auto call_output =
        xla::CustomCall(ctx->builder(), PoplarOp_Name(op_type), args, out_shape,
                        attribute_map_.Serialise());

    auto conv_out = xla::GetTupleElement(call_output, 0);
    auto metadata_out = xla::GetTupleElement(call_output, 1);
    ctx->SetOutput(0, conv_out);
    ctx->SetOutput(1, metadata_out);
  }
};

REGISTER_IPU_OP("IpuF8Conv2D", PoplinF8ConvOp<2>);
REGISTER_IPU_OP("IpuF8Conv3D", PoplinF8ConvOp<3>);

}  // namespace tensorflow
