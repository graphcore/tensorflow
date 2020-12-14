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
xla::Shape GetSeedShape() { return xla::ShapeUtil::MakeShape(xla::S32, {2}); }

class DropoutOp : public XlaOpKernel, public IpuOpKernel {
 public:
  explicit DropoutOp(OpKernelConstruction* ctx)
      : XlaOpKernel(ctx), IpuOpKernel() {
    float rate;
    float scale;
    std::vector<int64> noise_shape;

    OP_REQUIRES_OK(ctx, ctx->GetAttr("rate", &rate));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("scale", &scale));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("noise_shape", &noise_shape));

    attribute_map_.AddAttribute("rate", rate);
    attribute_map_.AddAttribute("scale", scale);
    // noise_shape is optional and defaults to an empty list.
    if (!noise_shape.empty()) {
      attribute_map_.AddAttribute("noise_shape", noise_shape);
    }
  }

  void Compile(XlaOpKernelContext* ctx) override {
    const DataType dtype = input_type(0);
    xla::XlaBuilder* b = ctx->builder();

    auto input = ctx->Input(0);
    auto shape = ctx->InputShape(0);

    xla::Shape xla_shape;
    OP_REQUIRES_OK(ctx, TensorShapeToXLAShape(dtype, shape, &xla_shape));

    // We are outputting both the output of the operation and the seed used so
    // we can reuse the seed in the backprop pass.
    const xla::Shape output_tuple_shape =
        xla::ShapeUtil::MakeTupleShape({xla_shape, GetSeedShape()});

    xla::XlaOp call_output = xla::CustomCall(
        b, PoplarOp_Name(PoplarOp::Dropout), {input, GetSeed(ctx, b)},
        output_tuple_shape, attribute_map_.Serialise());

    // The actual dropout output.
    xla::XlaOp output = xla::GetTupleElement(call_output, 0);

    // Save the seed used so we can reference it in the backwards pass.
    xla::XlaOp seed = xla::GetTupleElement(call_output, 1);

    ctx->SetOutput(0, output);
    ctx->SetOutput(1, seed);
  }

 protected:
  virtual xla::XlaOp GetSeed(XlaOpKernelContext* ctx,
                             xla::XlaBuilder* builder) {
    const xla::Shape seed_shape = GetSeedShape();
    // Create a seed and hash it with the replica index to make sure each
    // replica has a different seed.
    xla::XlaOp seed = xla::CustomCall(builder, PoplarOp_Name(PoplarOp::Seed),
                                      {}, seed_shape, "");
    xla::XlaOp replica_index =
        xla::CustomCall(builder, PoplarOp_Name(PoplarOp::ReplicationIndex), {},
                        xla::ShapeUtil::MakeShape(xla::S32, {}), "");

    // Hashing function based on
    // tensorflow/core/grappler/graph_analyzer/hash_tools.h
    // seed = seed ^ (replica_index + 0x9E3779B9U + (seed << 6) + (seed >> 2))
    seed = xla::BitcastConvertType(seed, xla::U32);
    replica_index = xla::BitcastConvertType(replica_index, xla::U32);

    xla::XlaOp large_constant = xla::ConstantLiteral(
        builder, xla::LiteralUtil::CreateR0<uint32>(0x9E3779B9U));
    xla::XlaOp constant_six = xla::ConstantLiteral(
        builder, xla::LiteralUtil::CreateR1<uint32>({6U, 6U}));
    xla::XlaOp constant_two = xla::ConstantLiteral(
        builder, xla::LiteralUtil::CreateR1<uint32>({2U, 2U}));

    // seed << 6
    xla::XlaOp shift_left = xla::ShiftLeft(seed, constant_six);
    // seed >> 2
    xla::XlaOp shift_right = xla::ShiftRightLogical(seed, constant_two);

    // Compute the rhs of xor.
    // replica_index + 0x9E3779B9U
    xla::XlaOp rhs = xla::Add(replica_index, large_constant);
    rhs = xla::Broadcast(rhs, seed_shape.dimensions());

    // replica_index + 0x9E3779B9U + (seed << 6)
    rhs = xla::Add(rhs, shift_left);
    // replica_index + 0x9E3779B9U + (seed << 6) + (seed >> 2)
    rhs = xla::Add(rhs, shift_right);

    seed = xla::Xor(seed, rhs);
    return xla::BitcastConvertType(seed, seed_shape.element_type());
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(DropoutOp);
};
REGISTER_IPU_OP("IpuDropout", DropoutOp);

class DropoutWithSeedOp : public DropoutOp {
 public:
  explicit DropoutWithSeedOp(OpKernelConstruction* ctx) : DropoutOp(ctx) {}

 protected:
  xla::XlaOp GetSeed(XlaOpKernelContext* ctx,
                     xla::XlaBuilder* builder) override {
    return ctx->Input(1);
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(DropoutWithSeedOp);
};
REGISTER_IPU_OP("IpuDropoutWithSeed", DropoutWithSeedOp);

}  // namespace tensorflow
