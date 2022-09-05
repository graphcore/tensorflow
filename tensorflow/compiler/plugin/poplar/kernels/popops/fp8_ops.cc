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

namespace tensorflow {

class PoplinF8MatMulOp : public XlaOpKernel, IpuOpKernel {
 public:
  using XlaOpKernel::XlaOpKernel;

  explicit PoplinF8MatMulOp(OpKernelConstruction* ctx)
      : XlaOpKernel(ctx), IpuOpKernel() {}

  void Compile(XlaOpKernelContext* ctx) override {
    std::vector<xla::XlaOp> args = {ctx->Input(0), ctx->Input(1), ctx->Input(2),
                                    ctx->Input(3)};

    const TensorShape lhs_shape = ctx->InputShape(0);
    const TensorShape rhs_shape = ctx->InputShape(2);
    TensorShape matmul_shape_(
        {lhs_shape.dim_size(0), lhs_shape.dim_size(1), rhs_shape.dim_size(2)});

    xla::Shape matmul_shape;
    OP_REQUIRES_OK(ctx, TensorShapeToXLAShape(ctx->expected_output_dtype(0),
                                              matmul_shape_, &matmul_shape));
    xla::Shape metadata_shape =
        xla::ShapeUtil::MakeShape(xla::PrimitiveType::U8, {});

    xla::Shape output_shape =
        xla::ShapeUtil::MakeTupleShape({matmul_shape, metadata_shape});

    xla::XlaOp call_output =
        xla::CustomCall(ctx->builder(), PoplarOp_Name(PoplarOp::F8MatMul), args,
                        output_shape, "");
    xla::XlaOp matmul_out = xla::GetTupleElement(call_output, 0);
    xla::XlaOp metadata_out = xla::GetTupleElement(call_output, 1);
    ctx->SetOutput(0, matmul_out);
    ctx->SetOutput(1, metadata_out);
  }
};

REGISTER_IPU_OP("IpuF8Matmul", PoplinF8MatMulOp);

}  // namespace tensorflow
