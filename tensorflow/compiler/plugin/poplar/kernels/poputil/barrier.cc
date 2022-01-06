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

#include "tensorflow/compiler/plugin/poplar/driver/xla_ipu_common.h"
#include "tensorflow/compiler/plugin/poplar/kernels/ipu_kernels_common.h"
#include "tensorflow/compiler/plugin/poplar/kernels/ops.pb.h"

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/xla/shape_util.h"

namespace tensorflow {

class BarrierOp : public XlaOpKernel, IpuOpKernel {
 public:
  explicit BarrierOp(OpKernelConstruction* ctx)
      : XlaOpKernel(ctx), IpuOpKernel() {}

  void Compile(XlaOpKernelContext* ctx) override {
    xla::XlaBuilder* builder = ctx->builder();

    std::vector<xla::Shape> input_shapes;
    std::vector<xla::XlaOp> inputs;
    for (int64 i = 0; i != ctx->num_inputs(); ++i) {
      auto input = ctx->Input(i);
      const auto& xla_shape_status = builder->GetShape(input);
      OP_REQUIRES_OK(ctx, xla_shape_status.status());
      input_shapes.push_back(xla_shape_status.ValueOrDie());
      inputs.push_back(input);
    }

    xla::XlaOp outputs =
        xla::CustomCall(builder, PoplarOp_Name(PoplarOp::Barrier), inputs,
                        xla::ShapeUtil::MakeTupleShape(input_shapes),
                        attribute_map_.Serialise());

    for (int64 i = 0; i != ctx->num_inputs(); ++i) {
      ctx->SetOutput(i, xla::GetTupleElement(outputs, i));
    }
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(BarrierOp);
};

REGISTER_IPU_OP("IpuBarrier", BarrierOp);

}  // namespace tensorflow
