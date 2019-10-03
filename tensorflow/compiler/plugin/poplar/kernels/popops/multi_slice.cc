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

#include "tensorflow/compiler/plugin/poplar/driver/xla_ipu_common.h"
#include "tensorflow/compiler/plugin/poplar/kernels/ipu_kernels_common.h"

#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"

using namespace xla::poplarplugin;

namespace tensorflow {

class PopopsMultiSliceOp : public XlaOpKernel, IpuOpKernel {
 public:
  using XlaOpKernel::XlaOpKernel;

  void Compile(XlaOpKernelContext* ctx) override {
    const TensorShape input_shape = ctx->InputShape(0);
    TensorShape output_shape = ctx->InputShape(1);
    output_shape.AddDim(input_shape.dim_size(1));

    xla::PrimitiveType input_type;
    OP_REQUIRES_OK(ctx,
                   DataTypeToPrimitiveType(ctx->input_type(0), &input_type));

    xla::XlaBuilder& b = *ctx->builder();
    std::vector<xla::XlaOp> args = {ctx->Input(0), ctx->Input(1)};
    xla::Shape xla_output_shape =
        TensorShapeToXLAShape(input_type, output_shape);

    xla::XlaOp call_output =
        xla::CustomCall(&b,
                        GetPoplibsCustomOpTargetString(PoplibsOp::Popops,
                                                       PoplibsOp::MultiSlice),
                        args, xla_output_shape, attribute_map_.Serialise());
    ctx->SetOutput(0, call_output);
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(PopopsMultiSliceOp);
};

REGISTER_IPU_OP("IpuMultiSlice", PopopsMultiSliceOp);

class PopopsMultiUpdateOp : public XlaOpKernel, IpuOpKernel {
 public:
  using XlaOpKernel::XlaOpKernel;

  void Compile(XlaOpKernelContext* ctx) override {
    const TensorShape input_shape = ctx->InputShape(0);
    xla::PrimitiveType input_type;
    OP_REQUIRES_OK(ctx,
                   DataTypeToPrimitiveType(ctx->input_type(0), &input_type));

    xla::XlaBuilder& b = *ctx->builder();
    xla::Shape xla_output_shape =
        TensorShapeToXLAShape(input_type, input_shape);
    std::vector<xla::XlaOp> args = {ctx->Input(0), ctx->Input(1),
                                    ctx->Input(2)};

    xla::XlaOp call_output =
        xla::CustomCall(&b,
                        GetPoplibsCustomOpTargetString(PoplibsOp::Popops,
                                                       PoplibsOp::MultiUpdate),
                        args, xla_output_shape, attribute_map_.Serialise());
    ctx->SetOutput(0, call_output);
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(PopopsMultiUpdateOp);
};

REGISTER_IPU_OP("IpuMultiUpdate", PopopsMultiUpdateOp);
}  // namespace tensorflow
