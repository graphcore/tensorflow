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

#include "tensorflow/compiler/plugin/poplar/driver/tools/poplar_replica_groups.h"
#include "tensorflow/compiler/plugin/poplar/driver/xla_ipu_common.h"
#include "tensorflow/compiler/plugin/poplar/kernels/ipu_kernels_common.h"
#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"

namespace tensorflow {

class PopopsCrossReplicaSumOp : public XlaOpKernel {
 public:
  using XlaOpKernel::XlaOpKernel;

  explicit PopopsCrossReplicaSumOp(OpKernelConstruction* ctx)
      : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx,
                   ctx->GetAttr("replica_group_size", &replica_group_size_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
    const auto replica_groups =
        xla::poplarplugin::PoplarReplicaGroups::Consecutive(
            replica_group_size_);

    ctx->SetOutput(0, xla::CrossReplicaSum(
                          ctx->Input(0), replica_groups.ToXlaReplicaGroups()));
  }

 private:
  int64_t replica_group_size_;

  TF_DISALLOW_COPY_AND_ASSIGN(PopopsCrossReplicaSumOp);
};

REGISTER_XLA_OP(Name("IpuCrossReplicaSum").Device(DEVICE_IPU_XLA_JIT),
                PopopsCrossReplicaSumOp);

class PopopsCrossReplicaMeanOp : public XlaOpKernel {
 public:
  using XlaOpKernel::XlaOpKernel;

  explicit PopopsCrossReplicaMeanOp(OpKernelConstruction* ctx)
      : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx,
                   ctx->GetAttr("replica_group_size", &replica_group_size_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
    const auto replica_groups =
        xla::poplarplugin::PoplarReplicaGroups::Consecutive(
            replica_group_size_);

    xla::XlaBuilder* builder = ctx->builder();

    const auto input = ctx->Input(0);

    const auto& xla_shape_status = builder->GetShape(input);
    OP_REQUIRES_OK(ctx, xla_shape_status.status());
    const auto& xla_shape = xla_shape_status.ValueOrDie();

    const auto b = builder->CreateSubBuilder("mean_reduce_op");
    const auto& scalar_shape =
        xla::ShapeUtil::MakeShape(xla_shape.element_type(), {});
    const auto acc = xla::Parameter(b.get(), 0, scalar_shape, "acc", {});
    const auto x = xla::Parameter(b.get(), 1, scalar_shape, "x", {});
    const auto norm_x =
        xla::CustomCall(b.get(), PoplarOp_Name(PoplarOp::ReplicationNormalise),
                        {x}, scalar_shape, {});
    Add(acc, norm_x);
    const auto& computation_status = b->Build();
    OP_REQUIRES_OK(ctx, computation_status.status());
    const auto& computation = computation_status.ValueOrDie();

    auto output =
        xla::AllReduce(input, computation, replica_groups.ToXlaReplicaGroups());

    ctx->SetOutput(0, output);
  }

 private:
  int64_t replica_group_size_;

  TF_DISALLOW_COPY_AND_ASSIGN(PopopsCrossReplicaMeanOp);
};

REGISTER_XLA_OP(Name("IpuCrossReplicaMean").Device(DEVICE_IPU_XLA_JIT),
                PopopsCrossReplicaMeanOp);

}  // namespace tensorflow
