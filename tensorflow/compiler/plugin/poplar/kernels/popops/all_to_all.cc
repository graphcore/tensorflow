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
#include "tensorflow/compiler/plugin/poplar/driver/xla_ipu_common.h"
#include "tensorflow/compiler/plugin/poplar/kernels/ipu_kernels_common.h"

#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"

namespace tensorflow {

class PopopsAllGather : public XlaOpKernel, IpuOpKernel {
 public:
  using XlaOpKernel::XlaOpKernel;

  explicit PopopsAllGather(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx,
                   ctx->GetAttr("replication_factor", &replica_group_size_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
    attribute_map_.AddAttribute("replica_group_size", replica_group_size_);

    // The output should be [replicationFactor][inputShape] shaped.
    const TensorShape input_shape = ctx->InputShape(0);
    TensorShape output_shape = ctx->InputShape(0);
    output_shape.InsertDim(0, replica_group_size_);

    // Get the element type of the input
    xla::PrimitiveType input_type;
    OP_REQUIRES_OK(ctx,
                   DataTypeToPrimitiveType(ctx->input_type(0), &input_type));

    xla::XlaBuilder& b = *ctx->builder();

    // Convert the output shape into the XLA format.
    xla::Shape xla_output_shape =
        TensorShapeToXLAShape(input_type, output_shape);

    xla::XlaOp call_output =
        xla::CustomCall(&b, PoplarOp_Name(PoplarOp::AllGather), {ctx->Input(0)},
                        xla_output_shape, attribute_map_.Serialise());
    ctx->SetOutput(0, call_output);
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(PopopsAllGather);

  tensorflow::int64 replica_group_size_;
};

REGISTER_XLA_OP(Name("IpuAllGather").Device(DEVICE_IPU_XLA_JIT),
                PopopsAllGather);

class PopopsAllToAll : public XlaOpKernel {
 public:
  using XlaOpKernel::XlaOpKernel;

  explicit PopopsAllToAll(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("split_dimension", &split_dimension));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("concat_dimension", &concat_dimension));
    OP_REQUIRES_OK(ctx,
                   ctx->GetAttr("number_of_replicas", &number_of_replicas));
  }

  void Compile(XlaOpKernelContext* ctx) override {
    // Just create the XLA instruction directly, we will catch this in the
    // vistors and lower it to poplar from there. The last parameter, replica
    // groups is ignored because for IPU we assume that we are always targeting
    // all replicas likewise we ignore the split count.
    ctx->SetOutput(0, xla::AllToAll(ctx->Input(0), split_dimension,
                                    concat_dimension, number_of_replicas, {}));
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(PopopsAllToAll);

  tensorflow::int64 split_dimension, concat_dimension, number_of_replicas;
};

REGISTER_XLA_OP(Name("IpuAllToAll").Device(DEVICE_IPU_XLA_JIT), PopopsAllToAll);

}  // namespace tensorflow
