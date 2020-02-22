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

#include "tensorflow/compiler/plugin/poplar/driver/xla_ipu_common.h"
#include "tensorflow/compiler/tf2xla/kernels/cwise_ops.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/lib/constants.h"
#include "tensorflow/compiler/xla/client/lib/math.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/core/framework/kernel_def_builder.h"
#include "tensorflow/core/framework/types.h"

// Re-implementation of ops in
// `tensorflow/compiler/tf2xla/kernels/training_ops.cc` which are better suited
// for the IPU.
namespace tensorflow {
namespace {

class ResourceApplyAdam : public XlaOpKernel {
 public:
  explicit ResourceApplyAdam(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("T", &dtype_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_nesterov", &use_nesterov_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
    TensorShape var_shape, m_shape, v_shape;
    xla::XlaOp var, m, v;
    OP_REQUIRES_OK(ctx, ctx->ReadVariableInput(0, dtype_, &var_shape, &var));
    OP_REQUIRES_OK(ctx, ctx->ReadVariableInput(1, dtype_, &m_shape, &m));
    OP_REQUIRES_OK(ctx, ctx->ReadVariableInput(2, dtype_, &v_shape, &v));

    TensorShape beta1_power_shape = ctx->InputShape(3);
    TensorShape beta2_power_shape = ctx->InputShape(4);
    TensorShape lr_shape = ctx->InputShape(5);
    TensorShape beta1_shape = ctx->InputShape(6);
    TensorShape beta2_shape = ctx->InputShape(7);
    TensorShape epsilon_shape = ctx->InputShape(8);
    TensorShape grad_shape = ctx->InputShape(9);

    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(beta1_power_shape),
                errors::InvalidArgument("beta1_power is not a scalar: ",
                                        beta1_power_shape.DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(beta2_power_shape),
                errors::InvalidArgument("beta2_power is not a scalar: ",
                                        beta2_power_shape.DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(lr_shape),
                errors::InvalidArgument("lr is not a scalar : ",
                                        lr_shape.DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(beta1_shape),
                errors::InvalidArgument("beta1 is not a scalar: ",
                                        beta1_shape.DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(beta2_shape),
                errors::InvalidArgument("beta2 is not a scalar: ",
                                        beta2_shape.DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(epsilon_shape),
                errors::InvalidArgument("epsilon is not a scalar: ",
                                        epsilon_shape.DebugString()));

    OP_REQUIRES(ctx, var_shape.IsSameSize(m_shape),
                errors::InvalidArgument("var and m do not have the same shape",
                                        var_shape.DebugString(), " ",
                                        m_shape.DebugString()));
    OP_REQUIRES(ctx, var_shape.IsSameSize(v_shape),
                errors::InvalidArgument("var and v do not have the same shape",
                                        var_shape.DebugString(), " ",
                                        v_shape.DebugString()));
    OP_REQUIRES(ctx, var_shape.IsSameSize(grad_shape),
                errors::InvalidArgument(
                    "var and grad do not have the same shape",
                    var_shape.DebugString(), " ", grad_shape.DebugString()));

    xla::XlaOp beta1_power = ctx->Input(3);
    xla::XlaOp beta2_power = ctx->Input(4);
    xla::XlaOp lr = ctx->Input(5);
    xla::XlaOp beta1 = ctx->Input(6);
    xla::XlaOp beta2 = ctx->Input(7);
    xla::XlaOp epsilon = ctx->Input(8);
    xla::XlaOp grad = ctx->Input(9);

    // alpha <- learning_rate * sqrt(1 - beta2^t) / (1 - beta1^t)
    // m_t <- beta1 * m_{t-1} + (1 - beta1) * g_t
    // v_t <- beta2 * v_{t-1} + (1 - beta2) * g_t * g_t
    // if use_nesterov:
    //   variable <- variable - alpha * (m_t * beta1 + (1 - beta1) * g_t) /
    //   (sqrt(v_t) + epsilon)
    // else:
    //   variable <- variable - alpha * m_t / (sqrt(v_t) + epsilon)

    xla::XlaBuilder* b = ctx->builder();
    xla::XlaOp one = XlaHelpers::FloatLiteral(b, dtype_, 1.0);

    xla::XlaOp alpha = lr * xla::Sqrt(one - beta2_power) / (one - beta1_power);
    m = m * beta1 + grad * (one - beta1);
    v = v * beta2 + xla::Square(grad) * (one - beta2);
    if (use_nesterov_) {
      var = var - alpha * (m * beta1 + grad * (one - beta1)) /
                      (xla::Sqrt(v) + epsilon);
    } else {
      var = var - m * alpha / (xla::Sqrt(v) + epsilon);
    }

    OP_REQUIRES_OK(ctx, ctx->AssignVariable(0, dtype_, var));
    OP_REQUIRES_OK(ctx, ctx->AssignVariable(1, dtype_, m));
    OP_REQUIRES_OK(ctx, ctx->AssignVariable(2, dtype_, v));
  }

 private:
  DataType dtype_;
  bool use_nesterov_;
};
REGISTER_XLA_OP(Name("ResourceApplyAdam")
                    .Device(DEVICE_IPU_XLA_JIT)
                    .TypeConstraint("T", kFloatTypes),
                ResourceApplyAdam);

}  // namespace
}  // namespace tensorflow
