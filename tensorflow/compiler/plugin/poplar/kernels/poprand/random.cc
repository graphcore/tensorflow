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
#include "tensorflow/compiler/plugin/poplar/driver/trace.pb.h"
#include "tensorflow/compiler/plugin/poplar/driver/xla_ipu_common.h"
#include "tensorflow/compiler/plugin/poplar/kernels/custom_kernels_util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/ipu_kernels_common.h"

#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/util/stream_executor_util.h"

#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/literal_util.h"

#include "absl/container/flat_hash_set.h"

using namespace xla::poplarplugin;

namespace tensorflow {

namespace {

class PopopsTruncatedNormalOp : public XlaOpKernel, IpuOpKernel {
 public:
  explicit PopopsTruncatedNormalOp(OpKernelConstruction* ctx)
      : XlaOpKernel(ctx), IpuOpKernel() {}

  ~PopopsTruncatedNormalOp() override{};
  void Compile(XlaOpKernelContext* ctx) override {
    const DataType dtype = output_type(0);

    TensorShape shape;
    OP_REQUIRES_OK(ctx, ctx->ConstantInputAsShape(0, &shape));
    xla::Shape xla_shape;
    OP_REQUIRES_OK(ctx, TensorShapeToXLAShape(dtype, shape, &xla_shape));

    xla::XlaBuilder* b = ctx->builder();

    xla::XlaOp output =
        xla::CustomCall(b, PoplarOp_Name(PoplarOp::TruncatedNormal), {},
                        xla_shape, attribute_map_.Serialise());

    ctx->SetOutput(0, output);
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(PopopsTruncatedNormalOp);
};

class PopopsStatelessRandomGetKeyCounterOp : public XlaOpKernel,
                                             public IpuOpKernel {
 public:
  explicit PopopsStatelessRandomGetKeyCounterOp(OpKernelConstruction* ctx)
      : XlaOpKernel(ctx) {}

  void Compile(XlaOpKernelContext* ctx) override {
    // #TODO: T54602
    // We do not support any RNG algorithms, so in order to ensure compatibility
    // we pass the seed directly to the next Op instead of converting it to a
    // key and counter pair.
    const auto seed = ConvertElementType(
        ConvertElementType(ctx->Input(0), xla::U32), xla::U64);
    ctx->SetOutput(0, seed);
    ctx->SetOutput(1, seed);
  }
};

REGISTER_XLA_OP(Name("StatelessRandomGetKeyCounter").Device(DEVICE_IPU_XLA_JIT),
                PopopsStatelessRandomGetKeyCounterOp);

template <PoplarOp Op>
class PopopsStatelessV2Op : public XlaOpKernel, public IpuOpKernel {
 public:
  explicit PopopsStatelessV2Op(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {}

  void Compile(XlaOpKernelContext* ctx) override {
    // #TODO: T54602
    constexpr int key_input_idx = 1;
    constexpr int minval_input_idx = 4;
    constexpr int maxval_input_idx = 5;

    const DataType dtype = output_type(0);

    TensorShape shape;
    OP_REQUIRES_OK(ctx, ctx->ConstantInputAsShape(0, &shape));
    xla::Shape xla_shape;
    OP_REQUIRES_OK(ctx, TensorShapeToXLAShape(dtype, shape, &xla_shape));

    std::vector<xla::XlaOp> operands = {ctx->Input(key_input_idx)};

    switch (Op) {
      case PoplarOp::StatelessRandomUniform: {
        attribute_map_.AddAttribute("min_val", 0.0F);
        attribute_map_.AddAttribute("max_val", 1.0F);
      } break;
      case PoplarOp::StatelessRandomUniformInt: {
        operands.emplace_back(ctx->Input(minval_input_idx));
        operands.emplace_back(ctx->Input(maxval_input_idx));
      } break;
      default:
        break;
    }

    // Currently we do not support any different RNG state algorithms, except
    // for the default one in Poplar.
    xla::XlaOp output =
        xla::CustomCall(ctx->builder(), PoplarOp_Name(Op), std::move(operands),
                        xla_shape, attribute_map_.Serialise());

    ctx->SetOutput(0, output);
  }
};

REGISTER_XLA_OP(Name("StatelessTruncatedNormalV2")
                    .Device(DEVICE_IPU_XLA_JIT)
                    .CompileTimeConstantInput("shape")
                    .CompileTimeConstantInput("alg")
                    .TypeConstraint("dtype", {DT_FLOAT, DT_HALF}),
                PopopsStatelessV2Op<PoplarOp::StatelessTruncatedNormal>);

REGISTER_XLA_OP(Name("StatelessRandomNormalV2")
                    .Device(DEVICE_IPU_XLA_JIT)
                    .CompileTimeConstantInput("shape")
                    .CompileTimeConstantInput("alg")
                    .TypeConstraint("dtype", {DT_FLOAT, DT_HALF}),
                PopopsStatelessV2Op<PoplarOp::StatelessRandomNormal>);

REGISTER_XLA_OP(Name("StatelessRandomUniformV2")
                    .Device(DEVICE_IPU_XLA_JIT)
                    .CompileTimeConstantInput("shape")
                    .CompileTimeConstantInput("alg")
                    .TypeConstraint("dtype", {DT_FLOAT, DT_HALF}),
                PopopsStatelessV2Op<PoplarOp::StatelessRandomUniform>);

REGISTER_XLA_OP(Name("StatelessRandomUniformIntV2")
                    .Device(DEVICE_IPU_XLA_JIT)
                    .CompileTimeConstantInput("shape")
                    .CompileTimeConstantInput("alg")
                    .TypeConstraint("dtype", {DT_INT32}),
                PopopsStatelessV2Op<PoplarOp::StatelessRandomUniformInt>);

class PopopsStatelessOp : public XlaOpKernel, public IpuOpKernel {
 public:
  explicit PopopsStatelessOp(OpKernelConstruction* ctx, PoplarOp op_type)
      : XlaOpKernel(ctx), op_type_(op_type) {}

  void Compile(XlaOpKernelContext* ctx) override {
    const DataType dtype = output_type(0);

    TensorShape shape;
    OP_REQUIRES_OK(ctx, ctx->ConstantInputAsShape(0, &shape));
    xla::Shape xla_shape;
    OP_REQUIRES_OK(ctx, TensorShapeToXLAShape(dtype, shape, &xla_shape));

    TensorShape seed_shape = ctx->InputShape(1);
    OP_REQUIRES(ctx, seed_shape.dims() == 1 && seed_shape.dim_size(0) == 2,
                errors::InvalidArgument("seed must have shape [2], not ",
                                        seed_shape.DebugString()));

    if (op_type_ == PoplarOp::StatelessRandomUniform) {
      attribute_map_.AddAttribute("min_val", 0.0f);
      attribute_map_.AddAttribute("max_val", 1.0f);
    }

    xla::XlaOp output =
        xla::CustomCall(ctx->builder(), PoplarOp_Name(op_type_),
                        {ctx->Input(1)}, xla_shape, attribute_map_.Serialise());

    ctx->SetOutput(0, output);
  }

 private:
  PoplarOp op_type_;

  TF_DISALLOW_COPY_AND_ASSIGN(PopopsStatelessOp);
};

class PopopsStatelessRandomUniformOp : public PopopsStatelessOp {
 public:
  explicit PopopsStatelessRandomUniformOp(OpKernelConstruction* ctx)
      : PopopsStatelessOp(ctx, PoplarOp::StatelessRandomUniform) {}

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(PopopsStatelessRandomUniformOp);
};

class PopopsStatelessRandomNormalOp : public PopopsStatelessOp {
 public:
  explicit PopopsStatelessRandomNormalOp(OpKernelConstruction* ctx)
      : PopopsStatelessOp(ctx, PoplarOp::StatelessRandomNormal) {}

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(PopopsStatelessRandomNormalOp);
};

class PopopsStatelessTruncatedNormalOp : public PopopsStatelessOp {
 public:
  explicit PopopsStatelessTruncatedNormalOp(OpKernelConstruction* ctx)
      : PopopsStatelessOp(ctx, PoplarOp::StatelessTruncatedNormal) {}

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(PopopsStatelessTruncatedNormalOp);
};

class PopopsStatelessRandomUniformIntOp : public XlaOpKernel,
                                          public IpuOpKernel {
 public:
  explicit PopopsStatelessRandomUniformIntOp(OpKernelConstruction* ctx)
      : XlaOpKernel(ctx) {}

  void Compile(XlaOpKernelContext* ctx) override {
    const DataType dtype = output_type(0);

    TensorShape shape;
    OP_REQUIRES_OK(ctx, ctx->ConstantInputAsShape(0, &shape));
    xla::Shape xla_shape;
    OP_REQUIRES_OK(ctx, TensorShapeToXLAShape(dtype, shape, &xla_shape));

    TensorShape seed_shape = ctx->InputShape(1);
    OP_REQUIRES(ctx, seed_shape.dims() == 1 && seed_shape.dim_size(0) == 2,
                errors::InvalidArgument("seed must have shape [2], not ",
                                        seed_shape.DebugString()));
    TensorShape minval_shape = ctx->InputShape(2);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(minval_shape),
                errors::InvalidArgument("minval must be scalar, got shape ",
                                        minval_shape.DebugString()));
    TensorShape maxval_shape = ctx->InputShape(3);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(maxval_shape),
                errors::InvalidArgument("minval must be scalar, got shape ",
                                        maxval_shape.DebugString()));

    xla::XlaOp output = xla::CustomCall(
        ctx->builder(), PoplarOp_Name(PoplarOp::StatelessRandomUniformInt),
        {ctx->Input(1), ctx->Input(2), ctx->Input(3)}, xla_shape,
        attribute_map_.Serialise());

    ctx->SetOutput(0, output);
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(PopopsStatelessRandomUniformIntOp);
};

REGISTER_XLA_OP(Name("TruncatedNormal")
                    .Device(DEVICE_IPU_XLA_JIT)
                    .CompileTimeConstantInput("shape")
                    .TypeConstraint("dtype", {DT_FLOAT, DT_HALF}),
                PopopsTruncatedNormalOp);

REGISTER_XLA_OP(Name("StatelessTruncatedNormal")
                    .Device(DEVICE_IPU_XLA_JIT)
                    .CompileTimeConstantInput("shape")
                    .TypeConstraint("dtype", {DT_FLOAT, DT_HALF})
                    .TypeConstraint("Tseed", DT_INT32),
                PopopsStatelessTruncatedNormalOp);

REGISTER_XLA_OP(Name("StatelessRandomUniform")
                    .Device(DEVICE_IPU_XLA_JIT)
                    .CompileTimeConstantInput("shape")
                    .TypeConstraint("dtype", {DT_FLOAT, DT_HALF})
                    .TypeConstraint("Tseed", DT_INT32),
                PopopsStatelessRandomUniformOp);

REGISTER_XLA_OP(Name("StatelessRandomUniformInt")
                    .Device(DEVICE_IPU_XLA_JIT)
                    .CompileTimeConstantInput("shape")
                    .TypeConstraint("dtype", {DT_INT32})
                    .TypeConstraint("Tseed", DT_INT32),
                PopopsStatelessRandomUniformIntOp);

REGISTER_XLA_OP(Name("StatelessRandomNormal")
                    .Device(DEVICE_IPU_XLA_JIT)
                    .CompileTimeConstantInput("shape")
                    .TypeConstraint("dtype", {DT_FLOAT, DT_HALF})
                    .TypeConstraint("Tseed", DT_INT32),
                PopopsStatelessRandomNormalOp);

}  // namespace

}  // namespace tensorflow
