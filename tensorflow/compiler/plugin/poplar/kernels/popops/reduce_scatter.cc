/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/compiler/plugin/poplar/driver/xla_ipu_common.h"
#include "tensorflow/compiler/plugin/poplar/kernels/ipu_kernels_common.h"
#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"

namespace tensorflow {

class PopopsReduceScatter : public XlaOpKernel, IpuOpKernel {
 public:
  using XlaOpKernel::XlaOpKernel;

  explicit PopopsReduceScatter(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx,
                   ctx->GetAttr("replication_factor", &replica_group_size_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("op", &op_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
    const TensorShape input_shape = ctx->InputShape(0);
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(input_shape),
                errors::InvalidArgument("Input must be vector"));

    attribute_map_.AddAttribute("op", op_);
    attribute_map_.AddAttribute("replica_group_size", replica_group_size_);

    const int64 input_length = input_shape.dim_size(0);
    const int64 output_length =
        MathUtil::CeilOfRatio<int64>(input_length, replica_group_size_);

    TensorShape output_shape;
    output_shape.AddDim(output_length);

    xla::PrimitiveType input_type;
    OP_REQUIRES_OK(ctx,
                   DataTypeToPrimitiveType(ctx->input_type(0), &input_type));

    const xla::Shape xla_output_shape =
        TensorShapeToXLAShape(input_type, output_shape);

    const xla::XlaOp call_output = xla::CustomCall(
        ctx->builder(), PoplarOp_Name(PoplarOp::ReduceScatter), {ctx->Input(0)},
        xla_output_shape, attribute_map_.Serialise());

    ctx->SetOutput(0, call_output);
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(PopopsReduceScatter);

  int64 replica_group_size_;
  std::string op_;
};

REGISTER_XLA_OP(Name("IpuReduceScatter").Device(DEVICE_IPU_XLA_JIT),
                PopopsReduceScatter);

}  // namespace tensorflow
