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
#include "tensorflow/compiler/plugin/poplar/driver/tools/tf2xla_util.h"
#include "tensorflow/compiler/plugin/poplar/driver/trace.pb.h"
#include "tensorflow/compiler/plugin/poplar/driver/xla_ipu_common.h"
#include "tensorflow/compiler/plugin/poplar/kernels/custom_kernels_util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/ipu_kernels_common.h"
#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"

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
    xla::XlaOp seed = GetSeed(ctx, b);
    xla::Shape seed_shape = b->GetShape(seed).ConsumeValueOrDie();
    const xla::Shape output_tuple_shape =
        xla::ShapeUtil::MakeTupleShape({xla_shape, seed_shape});

    xla::XlaOp call_output =
        xla::CustomCall(b, PoplarOp_Name(PoplarOp::Dropout), {input, seed},
                        output_tuple_shape, attribute_map_.Serialise());

    // The actual dropout output.
    xla::XlaOp output = xla::GetTupleElement(call_output, 0);

    // Save the seed used so we can reference it in the backwards pass.
    xla::XlaOp seed_output = xla::GetTupleElement(call_output, 1);

    ctx->SetOutput(0, output);
    ctx->SetOutput(1, seed_output);
  }

 protected:
  virtual xla::XlaOp GetSeed(XlaOpKernelContext* ctx,
                             xla::XlaBuilder* builder) {
    const xla::Shape seed_shape = xla::ShapeUtil::MakeShape(xla::S32, {2});
    xla::XlaOp seed = xla::CustomCall(builder, PoplarOp_Name(PoplarOp::Seed),
                                      {}, seed_shape, "");
    return HashSeedWithReplicaIndex(seed);
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
