/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/plugin/poplar/kernels/ipu_kernels_common.h"
#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"

namespace tensorflow {

/*
 * Histogram.
 */
class PopopsHistogramOp : public XlaOpKernel, public IpuOpKernel {
 public:
  using XlaOpKernel::XlaOpKernel;

  explicit PopopsHistogramOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
    bool absolute_of_input;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("absolute_of_input", &absolute_of_input));

    attribute_map_.AddAttribute("absolute_of_input", absolute_of_input);
  }

  void Compile(XlaOpKernelContext* ctx) override {
    auto b = ctx->builder();

    auto input = ctx->Input(0);
    auto levels = ctx->Input(1);

    const auto num_levels = ctx->InputShape(1).dim_size(0);

    TensorShape out_shape;
    out_shape.AddDim(num_levels + 1);

    xla::Shape out_shape_xla;
    OP_REQUIRES_OK(
        ctx, TensorShapeToXLAShape(output_type(0), out_shape, &out_shape_xla));

    auto output =
        xla::CustomCall(b, PoplarOp_Name(PoplarOp::Histogram), {input, levels},
                        {out_shape_xla}, attribute_map_.Serialise());

    ctx->SetOutput(0, output);
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(PopopsHistogramOp);
};
REGISTER_IPU_OP("IpuHistogram", PopopsHistogramOp);

/*
 * Histogram update.
 */
class PopopsHistogramUpdateOp : public XlaOpKernel, public IpuOpKernel {
 public:
  using XlaOpKernel::XlaOpKernel;

  explicit PopopsHistogramUpdateOp(OpKernelConstruction* ctx)
      : XlaOpKernel(ctx) {
    bool absolute_of_input;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("absolute_of_input", &absolute_of_input));

    attribute_map_.AddAttribute("absolute_of_input", absolute_of_input);
  }

  void Compile(XlaOpKernelContext* ctx) override {
    auto b = ctx->builder();

    auto hist = ctx->Input(0);
    auto input = ctx->Input(1);
    auto levels = ctx->Input(2);

    const auto out_shape = ctx->InputShape(0);
    const auto out_type = input_type(0);

    xla::Shape out_shape_xla;
    OP_REQUIRES_OK(ctx,
                   TensorShapeToXLAShape(out_type, out_shape, &out_shape_xla));

    auto output = xla::CustomCall(b, PoplarOp_Name(PoplarOp::HistogramUpdate),
                                  {hist, input, levels}, {out_shape_xla},
                                  attribute_map_.Serialise());

    ctx->SetOutput(0, output);
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(PopopsHistogramUpdateOp);
};
REGISTER_IPU_OP("IpuHistogramUpdate", PopopsHistogramUpdateOp);

}  // namespace tensorflow
