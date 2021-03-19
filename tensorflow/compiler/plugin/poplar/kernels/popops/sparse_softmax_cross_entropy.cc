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
#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/sparse.h"
#include "tensorflow/compiler/plugin/poplar/driver/trace.pb.h"
#include "tensorflow/compiler/plugin/poplar/driver/xla_ipu_common.h"
#include "tensorflow/compiler/plugin/poplar/kernels/custom_kernels_util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/ipu_kernels_common.h"

#include "absl/strings/match.h"
#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/lib/constants.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"

namespace tensorflow {
namespace {

std::pair<xla::XlaOp, xla::XlaOp> CrossEntropyWithLogitsIPU(
    XlaOpKernelContext* ctx, DataType type, xla::PrimitiveType xla_type,
    xla::XlaOp logits) {
  const xla::XlaComputation& max_func = *ctx->GetOrCreateMax(type);

  const int kBatchDim = 0;
  const int kClassDim = 1;

  xla::XlaBuilder* b = ctx->builder();
  // Find the max in each batch, resulting in a tensor of shape [batch]
  auto logits_max =
      xla::Reduce(logits, xla::MinValue(b, xla_type), max_func, {kClassDim});

  // Subtract the max in batch b from every element in batch b.
  // Broadcasts along the batch dimension.
  auto shifted_logits = xla::Sub(logits, logits_max, {kBatchDim});

  // exp(logits - max_logits)
  auto exp_shifted_logits = xla::Exp(shifted_logits);

  // sum_{class} (exp(logits - max_logits))
  const DataType accumulation_type = XlaHelpers::SumAccumulationType(type);
  auto converted =
      XlaHelpers::ConvertElementType(exp_shifted_logits, accumulation_type);
  auto reduce =
      xla::Reduce(converted, XlaHelpers::Zero(b, accumulation_type),
                  *ctx->GetOrCreateAdd(accumulation_type), {kClassDim});
  auto sum_exp = XlaHelpers::ConvertElementType(reduce, type);

  // log(sum(exp(logits - max_logits)))
  auto log_sum_exp = xla::Log(sum_exp);

  // log(sum(exp(logits - max_logits))) - (logits - max_logits))
  auto loss = xla::Neg(xla::Sub(shifted_logits, log_sum_exp, {kBatchDim}));
  auto softmax = xla::Div(exp_shifted_logits, sum_exp, {kBatchDim});

  return {loss, softmax};
}

class SelectScalarFromRowsOp : public IpuOpKernel {
 public:
  explicit SelectScalarFromRowsOp() : IpuOpKernel() {}

  xla::XlaOp GetCustomCall(xla::XlaBuilder* builder,
                           absl::Span<const xla::XlaOp> operands,
                           const xla::Shape& shape) {
    return xla::CustomCall(builder,
                           PoplarOp_Name(PoplarOp::SelectScalarFromRows),
                           operands, shape, attribute_map_.Serialise());
  }

  TF_DISALLOW_COPY_AND_ASSIGN(SelectScalarFromRowsOp);
};

class UpdateScalarInRowsOp : public IpuOpKernel {
 public:
  explicit UpdateScalarInRowsOp() : IpuOpKernel() {}

  xla::XlaOp GetCustomCall(xla::XlaBuilder* builder,
                           absl::Span<const xla::XlaOp> operands,
                           const xla::Shape& shape) {
    return xla::CustomCall(builder, PoplarOp_Name(PoplarOp::UpdateScalarInRows),
                           operands, shape, attribute_map_.Serialise());
  }

  TF_DISALLOW_COPY_AND_ASSIGN(UpdateScalarInRowsOp);
};

class PopopsSparseSoftmaxXentWithLogitsOp : public XlaOpKernel, IpuOpKernel {
 public:
  explicit PopopsSparseSoftmaxXentWithLogitsOp(OpKernelConstruction* ctx)
      : XlaOpKernel(ctx), IpuOpKernel() {}

  void Compile(XlaOpKernelContext* ctx) override {
    VLOG(1) << "Using IPU version of SparseSoftmaxXentWithLogitsOp";

    const TensorShape logits_shape = ctx->InputShape(0);
    const TensorShape labels_shape = ctx->InputShape(1);
    OP_REQUIRES(ctx, TensorShapeUtils::IsMatrix(logits_shape),
                errors::InvalidArgument("logits must be 2-D, but got shape ",
                                        logits_shape.DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(labels_shape),
                errors::InvalidArgument("labels must be 1-D, but got shape ",
                                        labels_shape.DebugString()));
    OP_REQUIRES(ctx, logits_shape.dim_size(0) == labels_shape.dim_size(0),
                errors::InvalidArgument(
                    "logits and labels must have the same first dimension, "
                    "got logits shape ",
                    logits_shape.DebugString(), " and labels shape ",
                    labels_shape.DebugString()));
    OP_REQUIRES(ctx, logits_shape.dim_size(1) > 0,
                errors::InvalidArgument(
                    "Must have at least one class, but got logits shape ",
                    logits_shape.DebugString()));

    int64 batch_size = logits_shape.dim_size(0);

    const DataType logits_type = input_type(0);
    const xla::PrimitiveType xla_logits_type = ctx->input_xla_type(0);

    xla::XlaOp indices = ctx->Input(1);
    xla::XlaBuilder* builder = ctx->builder();

    xla::XlaOp loss_matrix, softmax;
    std::tie(loss_matrix, softmax) = CrossEntropyWithLogitsIPU(
        ctx, logits_type, xla_logits_type, ctx->Input(0));

    xla::StatusOr<xla::Shape> loss_matrix_shape =
        builder->GetShape(loss_matrix);
    OP_REQUIRES(ctx, loss_matrix_shape.ok(), loss_matrix_shape.status());

    xla::Shape loss_shape = xla::ShapeUtil::MakeShape(
        loss_matrix_shape.ValueOrDie().element_type(), {batch_size});

    SelectScalarFromRowsOp select_op;
    xla::XlaOp loss =
        select_op.GetCustomCall(builder, {loss_matrix, indices}, loss_shape);

    xla::StatusOr<xla::Shape> softmax_shape = builder->GetShape(softmax);
    OP_REQUIRES(ctx, softmax_shape.ok(), softmax_shape.status());

    UpdateScalarInRowsOp update_op;
    xla::XlaOp backprop = update_op.GetCustomCall(builder, {softmax, indices},
                                                  softmax_shape.ValueOrDie());

    ctx->SetOutput(0, loss);
    ctx->SetOutput(1, backprop);
  }

  TF_DISALLOW_COPY_AND_ASSIGN(PopopsSparseSoftmaxXentWithLogitsOp);
};

// TODO(T10195)
// REGISTER_XLA_OP(
//     Name("SparseSoftmaxCrossEntropyWithLogits").Device(DEVICE_IPU_XLA_JIT),
//     PopopsSparseSoftmaxXentWithLogitsOp);

}  // namespace
}  // namespace tensorflow
