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
  int64 replica_group_size_;

  TF_DISALLOW_COPY_AND_ASSIGN(PopopsCrossReplicaSumOp);
};

REGISTER_XLA_OP(Name("IpuCrossReplicaSum").Device(DEVICE_IPU_XLA_JIT),
                PopopsCrossReplicaSumOp);

}  // namespace tensorflow
