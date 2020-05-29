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

#include "absl/container/flat_hash_set.h"
#include "tensorflow/compiler/plugin/poplar/driver/poplar_platform.h"
#include "tensorflow/compiler/plugin/poplar/driver/trace.pb.h"
#include "tensorflow/compiler/plugin/poplar/driver/xla_ipu_common.h"
#include "tensorflow/compiler/plugin/poplar/kernels/custom_kernels_util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/ipu_kernels_common.h"
#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/util/stream_executor_util.h"

using namespace xla::poplarplugin;

namespace tensorflow {

template <PoplarOp Op>
class NonLinearity : public XlaOpKernel, IpuOpKernel {
 public:
  explicit NonLinearity(OpKernelConstruction* ctx)
      : XlaOpKernel(ctx), IpuOpKernel() {}

  void Compile(XlaOpKernelContext* ctx) final {
    const DataType dtype = output_type(0);
    xla::XlaBuilder* b = ctx->builder();

    auto input = ctx->Input(0);
    auto shape = ctx->InputShape(0);

    xla::Shape xla_shape;
    OP_REQUIRES_OK(ctx, TensorShapeToXLAShape(dtype, shape, &xla_shape));

    xla::XlaOp output = xla::CustomCall(b, PoplarOp_Name(Op), {input},
                                        xla_shape, attribute_map_.Serialise());

    ctx->SetOutput(0, output);
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(NonLinearity);
};

REGISTER_IPU_OP("Relu", NonLinearity<PoplarOp::Relu>);
REGISTER_IPU_OP("Sigmoid", NonLinearity<PoplarOp::Sigmoid>);
REGISTER_IPU_OP("IpuGelu", NonLinearity<PoplarOp::Gelu>);

template <PoplarOp Op, bool SwapInputs>
class NonLinearityGrad : public XlaOpKernel, IpuOpKernel {
 public:
  explicit NonLinearityGrad(OpKernelConstruction* ctx)
      : XlaOpKernel(ctx), IpuOpKernel() {}

  void Compile(XlaOpKernelContext* ctx) final {
    const DataType dtype = output_type(0);
    xla::XlaBuilder* b = ctx->builder();

    auto lhs = ctx->Input(0);
    auto rhs = ctx->Input(1);
    auto shape = ctx->InputShape(0);

    xla::Shape xla_shape;
    OP_REQUIRES_OK(ctx, TensorShapeToXLAShape(dtype, shape, &xla_shape));

    // Some non-linearties have the operands as (acts, grad) and some have it as
    // (grad, acts). Standardize it into (acts, grads).
    const std::vector<xla::XlaOp> operands =
        SwapInputs ? std::vector<xla::XlaOp>({rhs, lhs})
                   : std::vector<xla::XlaOp>({lhs, rhs});

    xla::XlaOp output = xla::CustomCall(b, PoplarOp_Name(Op), operands,
                                        xla_shape, attribute_map_.Serialise());

    ctx->SetOutput(0, output);
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(NonLinearityGrad);
};

REGISTER_IPU_OP("ReluGrad", (NonLinearityGrad<PoplarOp::ReluGrad, true>));
REGISTER_IPU_OP("SigmoidGrad",
                (NonLinearityGrad<PoplarOp::SigmoidGrad, false>));
REGISTER_IPU_OP("TanhGrad", (NonLinearityGrad<PoplarOp::TanhGrad, false>));
REGISTER_IPU_OP("IpuGeluGrad", (NonLinearityGrad<PoplarOp::GeluGrad, true>));

}  // namespace tensorflow
