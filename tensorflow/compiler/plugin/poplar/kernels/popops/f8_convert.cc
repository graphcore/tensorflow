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

#include <poplar/Quarter.hpp>
#include <poplar/TypeConversion.hpp>

namespace tensorflow {

namespace {

template <typename T>
gccs::ArrayRef<T> AsArray(const Tensor& t) {
  return gccs::ArrayRef<T>{static_cast<T*>(t.data()),
                           static_cast<size_t>(t.NumElements())};
}

class CpuConvertToF8Op : public OpKernel {
 public:
  using OpKernel::OpKernel;

  explicit CpuConvertToF8Op(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    OP_REQUIRES(ctx, input_type(0) == DT_HALF,
                errors::InvalidArgument("Input must be DT_HALF tensor."));
    Tensor input = ctx->input(0);
    Tensor input_metadata = ctx->input(1);
    OP_REQUIRES(
        ctx, input_metadata.NumElements() == 1,
        errors::InvalidArgument("Metadata must have single DT_UINT8 element."));
    Tensor* output;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, input.shape(), &output));
    OP_REQUIRES(ctx, input.NumElements() == output->NumElements(),
                errors::InvalidArgument(
                    "Input and output number of elements must match."));
    OP_REQUIRES(ctx, output->dtype() == DT_UINT8,
                errors::InvalidArgument("Output must be D_UINT8 tensor."));
    ctx->set_output(1, input_metadata);
    poplar::QuarterMetadata metadata(input_metadata.scalar<uint8_t>()());
    poplar::convertToDeviceType(poplar::QUARTER, metadata,
                                AsArray<const poplar::Half>(input),
                                output->data());
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(CpuConvertToF8Op);
};
REGISTER_KERNEL_BUILDER(Name("IpuConvertToF8").Device(DEVICE_CPU),
                        CpuConvertToF8Op);

class CpuConvertFromF8Op : public OpKernel {
 public:
  using OpKernel::OpKernel;

  explicit CpuConvertFromF8Op(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    auto input = ctx->input(0);
    auto input_metadata = ctx->input(1);
    OP_REQUIRES(ctx, input_type(0) == DT_UINT8,
                errors::InvalidArgument("Input must be DT_UINT8 tensor."));
    OP_REQUIRES(ctx, input_type(1) == DT_UINT8,
                errors::InvalidArgument("Metadata must be DT_UINT8 tensor."));

    Tensor* output;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, input.shape(), &output));
    OP_REQUIRES(ctx, input.NumElements() == output->NumElements(),
                errors::InvalidArgument(
                    "Input and output number of elements must match."));
    OP_REQUIRES(ctx, output->dtype() == DT_HALF,
                errors::InvalidArgument("Output must be DT_HALF tensor."));
    poplar::QuarterMetadata metadata(input_metadata.scalar<uint8_t>()());
    poplar::convertFromDeviceType(poplar::QUARTER, metadata, input.data(),
                                  AsArray<poplar::Half>(*output));
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(CpuConvertFromF8Op);
};
REGISTER_KERNEL_BUILDER(Name("IpuConvertFromF8").Device(DEVICE_CPU),
                        CpuConvertFromF8Op);

class IpuConvertToF8Op : public XlaOpKernel, public IpuOpKernel {
 public:
  using XlaOpKernel::XlaOpKernel;

  explicit IpuConvertToF8Op(OpKernelConstruction* ctx)
      : XlaOpKernel(ctx), IpuOpKernel() {}

  void Compile(XlaOpKernelContext* ctx) override {
    xla::XlaBuilder* b = ctx->builder();
    auto input_data = ctx->Input(0);
    auto input_metadata = ctx->Input(1);

    xla::Shape output_data_shape;
    OP_REQUIRES_OK(ctx, TensorShapeToXLAShape(DT_UINT8, ctx->InputShape(0),
                                              &output_data_shape));
    xla::Shape output_metadata_shape;
    OP_REQUIRES_OK(ctx, TensorShapeToXLAShape(DT_UINT8, ctx->InputShape(1),
                                              &output_metadata_shape));
    xla::Shape output_shape = xla::ShapeUtil::MakeTupleShape(
        {output_data_shape, output_metadata_shape});

    auto output_tuple = xla::CustomCall(
        b, PoplarOp_Name(PoplarOp::ConvertToF8), {input_data, input_metadata},
        output_shape, attribute_map_.Serialise());

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
      : XlaOpKernel(ctx), IpuOpKernel() {}

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
    auto output = xla::CustomCall(b, PoplarOp_Name(PoplarOp::ConvertFromF8),
                                  {input_data, input_metadata}, output_shape,
                                  attribute_map_.Serialise());

    ctx->SetOutput(0, output);
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(IpuConvertFromF8Op);
};
REGISTER_IPU_OP("IpuConvertFromF8", IpuConvertFromF8Op);

}  // namespace
}  // namespace tensorflow
