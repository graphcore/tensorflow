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

class StatefulGradientAccumulate : public XlaOpKernel, IpuOpKernel {
 public:
  explicit StatefulGradientAccumulate(
      OpKernelConstruction* ctx,
      PoplarOp op = PoplarOp::StatefulGradientAccumulate)
      : XlaOpKernel(ctx), IpuOpKernel(), op_(op) {
    int32 num_mini_batches;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("num_mini_batches", &num_mini_batches));
    OP_REQUIRES(
        ctx, num_mini_batches > 0,
        errors::FailedPrecondition("num_mini_batches needs to be at least 1."));
    attribute_map_.AddAttribute("num_mini_batches", num_mini_batches);
  }

  void Compile(XlaOpKernelContext* ctx) override {
    const DataType dtype = output_type(0);
    xla::XlaBuilder* b = ctx->builder();

    auto input = ctx->Input(0);
    auto shape = ctx->InputShape(0);

    xla::Shape xla_shape;
    OP_REQUIRES_OK(ctx, TensorShapeToXLAShape(dtype, shape, &xla_shape));

    xla::XlaOp output = xla::CustomCall(b, PoplarOp_Name(op_), {input},
                                        xla_shape, attribute_map_.Serialise());

    ctx->SetOutput(0, output);
  }

 private:
  const PoplarOp op_;
  TF_DISALLOW_COPY_AND_ASSIGN(StatefulGradientAccumulate);
};

REGISTER_IPU_OP("IpuStatefulGradientAccumulate", StatefulGradientAccumulate);

class StatefulGradientAccumulateWithMomentum : public XlaOpKernel, IpuOpKernel {
 public:
  explicit StatefulGradientAccumulateWithMomentum(OpKernelConstruction* ctx)
      : XlaOpKernel(ctx), IpuOpKernel() {
    int32 num_mini_batches;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("num_mini_batches", &num_mini_batches));
    attribute_map_.AddAttribute("num_mini_batches", num_mini_batches);
  }

  void Compile(XlaOpKernelContext* ctx) override {
    DataType dtype = output_type(0);
    xla::XlaBuilder* b = ctx->builder();

    xla::XlaOp accum;
    TensorShape accum_shape;
    OP_REQUIRES_OK(ctx, ctx->ReadVariableInput(0, dtype, &accum_shape, &accum));

    TensorShape grad_shape = ctx->InputShape(1);

    OP_REQUIRES(ctx, grad_shape.IsSameSize(accum_shape),
                errors::InvalidArgument(
                    "gradient and accum do not have the same shape",
                    grad_shape.DebugString(), " ", accum_shape.DebugString()));

    TensorShape momentum_shape = ctx->InputShape(2);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(momentum_shape),
                errors::InvalidArgument("momentum is not a scalar: ",
                                        momentum_shape.DebugString()));

    xla::XlaOp grad = ctx->Input(1);
    xla::XlaOp momentum = ctx->Input(2);

    xla::Shape xla_shape;
    OP_REQUIRES_OK(ctx, TensorShapeToXLAShape(dtype, accum_shape, &xla_shape));

    xla::XlaOp output = xla::CustomCall(
        b, PoplarOp_Name(PoplarOp::StatefulGradientAccumulateWithMomentum),
        {accum, grad, momentum},
        xla::ShapeUtil::MakeTupleShape({xla_shape, xla_shape}),
        attribute_map_.Serialise());

    accum = xla::GetTupleElement(output, 0);
    grad = xla::GetTupleElement(output, 1);
    ctx->SetOutput(0, grad);
    OP_REQUIRES_OK(ctx, ctx->AssignVariable(0, dtype, accum));
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(StatefulGradientAccumulateWithMomentum);
};
REGISTER_IPU_OP("IpuStatefulGradientAccumulateWithMomentum",
                StatefulGradientAccumulateWithMomentum);

template <bool from_input>
class GradientAccumulatorCreate : public XlaOpKernel, IpuOpKernel {
 public:
  explicit GradientAccumulatorCreate(OpKernelConstruction* ctx)
      : XlaOpKernel(ctx), IpuOpKernel() {
    if (!from_input) {
      OP_REQUIRES_OK(ctx, ctx->GetAttr("shape", &variable_shape_));
    }
  }

  void Compile(XlaOpKernelContext* ctx) override {
    const DataType dtype = output_type(0);
    xla::XlaBuilder* b = ctx->builder();

    std::vector<xla::XlaOp> operands;
    if (from_input) {
      auto variable = ctx->Input(0);
      variable_shape_ = ctx->InputShape(0);
      operands.push_back(variable);
    }

    xla::Shape xla_shape;
    OP_REQUIRES_OK(ctx,
                   TensorShapeToXLAShape(dtype, variable_shape_, &xla_shape));

    xla::XlaOp output =
        xla::CustomCall(b, PoplarOp_Name(PoplarOp::GradientAccumulatorCreate),
                        operands, xla_shape, attribute_map_.Serialise());

    ctx->SetOutput(0, output);
  }

 private:
  TensorShape variable_shape_;

  TF_DISALLOW_COPY_AND_ASSIGN(GradientAccumulatorCreate);
};

REGISTER_IPU_OP("GradientAccumulatorCreate", GradientAccumulatorCreate<true>);

REGISTER_IPU_OP("GradientAccumulatorCreateFromShape",
                GradientAccumulatorCreate<false>)

class GradientAccumulatorAdd : public XlaOpKernel, IpuOpKernel {
 public:
  explicit GradientAccumulatorAdd(OpKernelConstruction* ctx)
      : XlaOpKernel(ctx), IpuOpKernel() {}

  void Compile(XlaOpKernelContext* ctx) override {
    const DataType dtype = output_type(0);
    xla::XlaBuilder* b = ctx->builder();

    auto accumulator = ctx->Input(0);
    auto gradient = ctx->Input(1);
    auto accumulator_shape = ctx->InputShape(0);
    auto gradient_shape = ctx->InputShape(1);

    OP_REQUIRES(ctx, accumulator_shape.IsSameSize(gradient_shape),
                errors::InvalidArgument(
                    "Gradient and the accumulator do not have the same shape",
                    accumulator_shape.DebugString(), " ",
                    gradient_shape.DebugString()));

    xla::Shape xla_shape;
    OP_REQUIRES_OK(ctx,
                   TensorShapeToXLAShape(dtype, gradient_shape, &xla_shape));

    xla::XlaOp output = xla::CustomCall(
        b, PoplarOp_Name(PoplarOp::GradientAccumulatorAdd),
        {accumulator, gradient}, xla_shape, attribute_map_.Serialise());

    ctx->SetOutput(0, output);
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(GradientAccumulatorAdd);
};

REGISTER_IPU_OP("GradientAccumulatorAdd", GradientAccumulatorAdd);

class GradientAccumulatorSink : public XlaOpKernel, IpuOpKernel {
 public:
  explicit GradientAccumulatorSink(
      OpKernelConstruction* ctx,
      PoplarOp op = PoplarOp::GradientAccumulatorSink)
      : XlaOpKernel(ctx), IpuOpKernel(), op_(op) {}

  void Compile(XlaOpKernelContext* ctx) override {
    const DataType dtype = output_type(0);
    xla::XlaBuilder* b = ctx->builder();

    auto input = ctx->Input(0);
    auto shape = ctx->InputShape(0);

    xla::Shape xla_shape;
    OP_REQUIRES_OK(ctx, TensorShapeToXLAShape(dtype, shape, &xla_shape));

    xla::XlaOp output = xla::CustomCall(b, PoplarOp_Name(op_), {input},
                                        xla_shape, attribute_map_.Serialise());

    ctx->SetOutput(0, output);
  }

 private:
  const PoplarOp op_;
  TF_DISALLOW_COPY_AND_ASSIGN(GradientAccumulatorSink);
};

REGISTER_IPU_OP("GradientAccumulatorSink", GradientAccumulatorSink);

}  // namespace tensorflow
