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

class NormaliseImageOp : public XlaOpKernel, IpuOpKernel {
 public:
  explicit NormaliseImageOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
    float scale;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("scale", &scale));
    attribute_map_.AddAttribute("scale", scale);
  }

  void Compile(XlaOpKernelContext* ctx) override {
    xla::XlaBuilder* b = ctx->builder();

    auto image = ctx->Input(0);
    auto image_shape = ctx->InputShape(0);
    auto channel_offsets = ctx->Input(1);
    auto channel_scales = ctx->Input(2);

    // Image must have 3 channels on the innermost dimension and offsets and
    // scales must be the same size.
    const int num_dims = image_shape.dims();
    const int num_channels = image_shape.dim_size(num_dims - 1);
    OP_REQUIRES(ctx, num_channels == 3,
                errors::InvalidArgument(absl::StrFormat(
                    "The image has %u channels, expected 3.", num_channels)));
    const int num_channel_offsets = ctx->InputShape(1).dim_size(0);
    OP_REQUIRES(ctx, num_channel_offsets == num_channels,
                errors::InvalidArgument(absl::StrFormat(
                    "The channel_offsets must be the same size as the number of"
                    " image channels %u, but was %u",
                    num_channels, num_channel_offsets)));
    const int num_channel_scales = ctx->InputShape(2).dim_size(0);
    OP_REQUIRES(ctx, num_channel_scales == num_channels,
                errors::InvalidArgument(absl::StrFormat(
                    "The channel_scales must be the same size as the number of"
                    " image channels %u, but was %u",
                    num_channels, num_channel_scales)));

    // Channel and scale offsets must be the same type as the output.
    DataType out_type = input_type(0);
    // UINT8 is converted to FLOAT16 by the function.
    if (out_type == DataType::DT_UINT8) {
      out_type = DataType::DT_HALF;
    }
    const DataType channel_offsets_dtype = input_type(1);
    OP_REQUIRES(ctx, channel_offsets_dtype == out_type,
                errors::InvalidArgument("The channel_offsets must be the same"
                                        " type as the output image"));
    const DataType channel_scales_dtype = input_type(2);
    OP_REQUIRES(ctx, channel_scales_dtype == out_type,
                errors::InvalidArgument("The channel_scales must be the same"
                                        " type as the output image"));

    // Output image has 4 channels.
    xla::Shape out_shape;
    OP_REQUIRES_OK(ctx,
                   TensorShapeToXLAShape(out_type, image_shape, &out_shape));
    out_shape.set_dimensions(num_dims - 1, 4);

    xla::XlaOp output =
        xla::CustomCall(b, PoplarOp_Name(PoplarOp::NormaliseImage),
                        {image, channel_offsets, channel_scales}, out_shape,
                        attribute_map_.Serialise());
    ctx->SetOutput(0, output);
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(NormaliseImageOp);
};

REGISTER_XLA_OP(Name("NormaliseImage")
                    .Device(DEVICE_IPU_XLA_JIT)
                    .TypeConstraint("Tin", {DT_FLOAT, DT_HALF, DT_UINT8})
                    .CompileTimeConstantInput("channel_offsets")
                    .CompileTimeConstantInput("channel_scales"),
                NormaliseImageOp);

}  // namespace tensorflow
