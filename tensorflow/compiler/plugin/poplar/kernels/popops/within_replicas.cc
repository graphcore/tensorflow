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

#include "tensorflow/compiler/plugin/poplar/driver/poplar_platform.h"
#include "tensorflow/compiler/plugin/poplar/driver/xla_ipu_common.h"
#include "tensorflow/compiler/plugin/poplar/kernels/ipu_kernels_common.h"

#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"

namespace tensorflow {

class PopopsAllGatherWithinReplica : public XlaOpKernel, IpuOpKernel {
 public:
  using XlaOpKernel::XlaOpKernel;

  explicit PopopsAllGatherWithinReplica(OpKernelConstruction* ctx)
      : XlaOpKernel(ctx) {}

  void Compile(XlaOpKernelContext* ctx) override {
    xla::XlaBuilder* builder = ctx->builder();

    int64 element_count = 0;
    std::vector<xla::XlaOp> inputs;
    for (int64 i = 0; i != ctx->num_inputs(); ++i) {
      auto input = ctx->Input(i);
      inputs.push_back(input);

      const auto& xla_shape_status = builder->GetShape(input);
      OP_REQUIRES_OK(ctx, xla_shape_status.status());
      const auto input_shape = xla_shape_status.ValueOrDie();
      element_count += input_shape.dimensions(0);
    }

    TensorShape output_shape;
    output_shape.AddDim(element_count);

    xla::Shape output_shape_xla;
    OP_REQUIRES_OK(ctx, TensorShapeToXLAShape(output_type(0), output_shape,
                                              &output_shape_xla));
    const std::vector<xla::Shape> output_shapes(ctx->num_inputs(),
                                                output_shape_xla);

    xla::XlaOp outputs = xla::CustomCall(
        builder, PoplarOp_Name(PoplarOp::AllGatherWithinReplica), inputs,
        xla::ShapeUtil::MakeTupleShape(output_shapes),
        attribute_map_.Serialise());

    for (auto i = 0; i != ctx->num_inputs(); ++i) {
      ctx->SetOutput(i, xla::GetTupleElement(outputs, i));
    }
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(PopopsAllGatherWithinReplica);
};

REGISTER_XLA_OP(Name("IpuAllGatherWithinReplica").Device(DEVICE_IPU_XLA_JIT),
                PopopsAllGatherWithinReplica);

}  // namespace tensorflow
