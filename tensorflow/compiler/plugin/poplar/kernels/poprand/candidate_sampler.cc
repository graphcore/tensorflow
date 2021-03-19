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

#include "tensorflow/compiler/plugin/poplar/driver/tools/xla_util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/ipu_kernels_common.h"
#include "tensorflow/compiler/plugin/poplar/kernels/ops.pb.h"
#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/env.h"

namespace pp = xla::poplarplugin;
namespace tensorflow {

class CandidateSamplerOp : public XlaOpKernel, public IpuOpKernel {
 public:
  explicit CandidateSamplerOp(OpKernelConstruction* ctx, std::string dist)
      : XlaOpKernel(ctx), IpuOpKernel() {
    int64 seed1;
    int64 seed2;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("num_true", &num_true_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("num_sampled", &num_sampled_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("unique", &unique_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("range_max", &range_max_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("seed", &seed1));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("seed2", &seed2));

    // Combine the seeds by adding them
    seed_ = seed1 + seed2;

    attribute_map_.AddAttribute("num_true", num_true_);
    attribute_map_.AddAttribute("unique", unique_);
    attribute_map_.AddAttribute("range_max", range_max_);
    attribute_map_.AddAttribute("dist", dist);
  }

  void Compile(XlaOpKernelContext* ctx) override {
    // Do some checks on the input similar to the CPU OpKernel
    TensorShape true_classes_tensor_shape = ctx->InputShape(0);
    const DataType true_classes_dtype = input_type(0);
    xla::Shape true_classes_xla_shape;
    OP_REQUIRES_OK(ctx, TensorShapeToXLAShape(true_classes_dtype,
                                              true_classes_tensor_shape,
                                              &true_classes_xla_shape));
    OP_REQUIRES(ctx, true_classes_xla_shape.rank() == 2,
                errors::InvalidArgument("true_classes must be a 2D matrix"));

    const uint64 num_cols = true_classes_xla_shape.dimensions(1);
    OP_REQUIRES(
        ctx, num_cols == num_true_,
        errors::InvalidArgument(absl::StrFormat(
            "true_classes must have num_true columns, expected: [%u] was: [%u]",
            num_true_, num_cols)));

    if (unique_) {
      OP_REQUIRES(ctx, num_sampled_ <= range_max_,
                  errors::InvalidArgument(absl::StrFormat(
                      "Cannot generate num_sampled ([%u]) unique samples with"
                      " the given range_max ([%u])",
                      num_sampled_, range_max_)));
    }

    // Define output shapes
    const xla::Shape sampled_candidates_shape =
        xla::ShapeUtil::MakeShape(xla::S64, {num_sampled_});
    // we take in an int64 input but we want an f32 output
    true_classes_xla_shape.set_element_type(xla::F32);
    const xla::Shape sampled_expected_count_shape =
        xla::ShapeUtil::MakeShape(xla::F32, {num_sampled_});
    const xla::Shape output_tuple_shape = xla::ShapeUtil::MakeTupleShape(
        {sampled_candidates_shape, true_classes_xla_shape,
         sampled_expected_count_shape});

    xla::XlaBuilder* b = ctx->builder();
    // Add the seed into the graph. If the seed is not specified (0), create
    // a new seed op that will change every time it's executed. Otherwise,
    // set the given seed as a graph constant.
    // Unlike dropout, we don't need to remember the seed we used
    xla::XlaOp seed;
    const xla::Shape seed_shape = xla::ShapeUtil::MakeShape(xla::S32, {2});
    if (seed_ == 0) {
      seed =
          xla::CustomCall(b, PoplarOp_Name(PoplarOp::Seed), {}, seed_shape, "");
    } else {
      seed = xla::Broadcast(
          xla::ConstantLiteral(b, xla::LiteralUtil::CreateR0<int32>(seed_)),
          seed_shape.dimensions());
    }
    // Ensure seed is different for each replica
    xla::XlaOp replica_seed = pp::HashSeedWithReplicaIndex(seed);

    // Create a custom call for the CandidateSampler PoplarOp
    xla::XlaOp true_classes = ctx->Input(0);
    xla::XlaOp call_output =
        xla::CustomCall(b, PoplarOp_Name(PoplarOp::CandidateSampler),
                        {true_classes, replica_seed}, output_tuple_shape,
                        attribute_map_.Serialise());

    // Set the outputs as GTEs
    xla::XlaOp sampled_candidates = xla::GetTupleElement(call_output, 0);
    xla::XlaOp true_expected_count = xla::GetTupleElement(call_output, 1);
    xla::XlaOp sampled_expected_count = xla::GetTupleElement(call_output, 2);

    ctx->SetOutput(0, sampled_candidates);
    ctx->SetOutput(1, true_expected_count);
    ctx->SetOutput(2, sampled_expected_count);
  }

 private:
  // Macro to disable:
  // 1. Initializing this op with another instance of this op
  // 2. copy assignment (operator=) with another instance of this op
  TF_DISALLOW_COPY_AND_ASSIGN(CandidateSamplerOp);

  int64 num_true_;
  int64 num_sampled_;
  bool unique_;
  int64 range_max_;
  int64 seed_;
  std::string distribution_;
};

class UniformCandidateSamplerOp : public CandidateSamplerOp {
 public:
  explicit UniformCandidateSamplerOp(OpKernelConstruction* ctx)
      : CandidateSamplerOp(ctx, "uniform") {}

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(UniformCandidateSamplerOp);
};
REGISTER_IPU_OP("UniformCandidateSampler", UniformCandidateSamplerOp);

class LogUniformCandidateSamplerOp : public CandidateSamplerOp {
 public:
  explicit LogUniformCandidateSamplerOp(OpKernelConstruction* ctx)
      : CandidateSamplerOp(ctx, "log_uniform") {}

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(LogUniformCandidateSamplerOp);
};
REGISTER_IPU_OP("LogUniformCandidateSampler", LogUniformCandidateSamplerOp);

}  // namespace tensorflow
