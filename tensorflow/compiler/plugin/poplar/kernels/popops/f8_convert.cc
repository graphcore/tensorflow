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

#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"

namespace tensorflow {

class IpuConvertToF8Op : public XlaOpKernel, public IpuOpKernel {
 public:
  using XlaOpKernel::XlaOpKernel;

  explicit IpuConvertToF8Op(OpKernelConstruction* ctx)
      : XlaOpKernel(ctx), IpuOpKernel() {}

  void Compile(XlaOpKernelContext* ctx) override {
    xla::XlaBuilder* b = ctx->builder();
    auto input_data = ctx->Input(0);
    auto input_metadata = ctx->Input(1);
    if (input_type(0) != DT_HALF) {
      // If input tensor type is not half, convert it to half first.
      // Hardware supports conversion to the half only.
      input_data = xla::ConvertElementType(input_data, xla::F16);
    }

    xla::Shape output_data_shape;
    OP_REQUIRES_OK(ctx, TensorShapeToXLAShape(DT_UINT8, ctx->InputShape(0),
                                              &output_data_shape));
    xla::Shape output_metadata_shape;
    OP_REQUIRES_OK(ctx, TensorShapeToXLAShape(DT_UINT8, ctx->InputShape(1),
                                              &output_metadata_shape));
    xla::Shape output_shape = xla::ShapeUtil::MakeTupleShape(
        {output_data_shape, output_metadata_shape});

    auto packed_input = xla::Tuple(b, {input_data, input_metadata});
    auto output_tuple =
        xla::CustomCall(b, PoplarOp_Name(PoplarOp::ConvertToF8), {packed_input},
                        {output_shape}, attribute_map_.Serialise());

    ctx->SetOutput(0, xla::GetTupleElement(output_tuple, 0));
    ctx->SetOutput(1, xla::GetTupleElement(output_tuple, 1));
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(IpuConvertToF8Op);
};
REGISTER_IPU_OP("IpuConvertToF8", IpuConvertToF8Op);

class IpuConvertFromF8Op : public XlaOpKernel, public IpuOpKernel {
 public:
  using XlaOpKernel::XlaOpKernel;

  explicit IpuConvertFromF8Op(OpKernelConstruction* ctx)
      : XlaOpKernel(ctx), IpuOpKernel() {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("dtype", &output_type_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
    xla::XlaBuilder* b = ctx->builder();
    auto input_data = ctx->Input(0);
    auto input_metadata = ctx->Input(1);
    OP_REQUIRES(ctx, input_type(0) == DT_UINT8,
                errors::InvalidArgument("Input must be DT_UINT8 tensor."));
    OP_REQUIRES(ctx, input_type(1) == DT_UINT8,
                errors::InvalidArgument("Metadata must be DT_UINT8 tensor."));

    // Always convert to half type, it's the only conversion supported by
    // hardware now.
    xla::Shape output_shape;
    OP_REQUIRES_OK(
        ctx, TensorShapeToXLAShape(DT_HALF, ctx->InputShape(0), &output_shape));
    xla::XlaOp input = xla::Tuple(b, {input_data, input_metadata});
    auto output =
        xla::CustomCall(b, PoplarOp_Name(PoplarOp::ConvertFromF8), {input},
                        {output_shape}, attribute_map_.Serialise());

    if (output_type_ != DT_HALF) {
      // If output tensor type is not half, convert from half to desired type.
      OP_REQUIRES_OK(ctx, TensorShapeToXLAShape(
                              output_type_, ctx->InputShape(0), &output_shape));
      output = xla::ConvertElementType(output, output_shape.element_type());
    }

    ctx->SetOutput(0, output);
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(IpuConvertFromF8Op);
  DataType output_type_;
};
REGISTER_IPU_OP("IpuConvertFromF8", IpuConvertFromF8Op);

}  // namespace tensorflow
