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
#include "tensorflow/core/util/tensor_format.h"

namespace tensorflow {

class PopnnCTCLossOpBase : public XlaOpKernel, IpuOpKernel {
 protected:
  explicit PopnnCTCLossOpBase(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
    tensorflow::DataType in_dtype;
    tensorflow::DataType out_dtype;
    int64 blank_index;

    OP_REQUIRES_OK(ctx, ctx->GetAttr("in_dtype", &in_dtype));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("out_dtype", &out_dtype));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("blank_index", &blank_index));

    attribute_map_.AddAttribute("in_dtype", in_dtype);
    attribute_map_.AddAttribute("out_dtype", out_dtype);
    attribute_map_.AddAttribute("blank_index", blank_index);
  }
  virtual PoplarOp OpType() const = 0;

 public:
  ~PopnnCTCLossOpBase() override {}

  void Compile(XlaOpKernelContext* ctx) override {
    // Validate shapes.
    auto data_shape = ctx->InputShape(0);
    auto label_shape = ctx->InputShape(1);

    OP_REQUIRES(ctx, data_shape.dim_size(1) == label_shape.dim_size(0),
                errors::InvalidArgument(absl::StrFormat(
                    "The batch size in the data (dim1: [%u]) and labels (dim0: "
                    "[%u]) tensors must be equal.",
                    data_shape.dim_size(1), label_shape.dim_size(0))));

    int64 max_time = data_shape.dim_size(0);
    int64 batch_size = data_shape.dim_size(1);
    int64 num_classes = data_shape.dim_size(2);

    TensorShape expected_lengths_shape;
    TensorShapeUtils::MakeShape(std::vector<int64>({batch_size}),
                                &expected_lengths_shape);
    OP_REQUIRES(
        ctx, ctx->InputShape(2) == expected_lengths_shape,
        errors::InvalidArgument(absl::StrFormat(
            "The data lengths tensor needs to be of shape [%u].", batch_size)));
    OP_REQUIRES(ctx, ctx->InputShape(3) == expected_lengths_shape,
                errors::InvalidArgument(absl::StrFormat(
                    "The label lengths tensor needs to be of shape [%u].",
                    batch_size)));

    xla::Shape loss_shape =
        xla::ShapeUtil::MakeShape(ctx->output_xla_type(0), {batch_size});
    xla::Shape grad_shape = xla::ShapeUtil::MakeShape(
        ctx->output_xla_type(0), {max_time, batch_size, num_classes});
    xla::Shape output_shape =
        xla::ShapeUtil::MakeTupleShape({loss_shape, grad_shape});
    xla::XlaBuilder& b = *ctx->builder();

    std::vector<xla::XlaOp> args;
    for (int idx = 0; idx < ctx->num_inputs(); idx++) {
      args.push_back(ctx->Input(idx));
    }

    auto op_name = PoplarOp_Name(OpType());

    xla::XlaOp output_tuple = xla::CustomCall(&b, op_name, args, output_shape,
                                              attribute_map_.Serialise());

    xla::XlaOp output_loss = xla::GetTupleElement(output_tuple, 0);
    xla::XlaOp output_grad = xla::GetTupleElement(output_tuple, 1);
    ctx->SetOutput(0, output_loss);
    ctx->SetOutput(1, output_grad);
  }

  TF_DISALLOW_COPY_AND_ASSIGN(PopnnCTCLossOpBase);
};

class PopnnCTCLossWithLogitsOp : public PopnnCTCLossOpBase {
 public:
  explicit PopnnCTCLossWithLogitsOp(OpKernelConstruction* ctx)
      : PopnnCTCLossOpBase(ctx) {}

  ~PopnnCTCLossWithLogitsOp() override{};

  PoplarOp OpType() const override { return PoplarOp::CTCLossWithLogits; }

  TF_DISALLOW_COPY_AND_ASSIGN(PopnnCTCLossWithLogitsOp);
};

REGISTER_IPU_OP("PopnnCTCLossWithLogits", PopnnCTCLossWithLogitsOp);

class PopnnCTCLossWithLogProbsOp : public PopnnCTCLossOpBase {
 public:
  explicit PopnnCTCLossWithLogProbsOp(OpKernelConstruction* ctx)
      : PopnnCTCLossOpBase(ctx) {}

  ~PopnnCTCLossWithLogProbsOp() override{};

  PoplarOp OpType() const override { return PoplarOp::CTCLossWithLogProbs; }

  TF_DISALLOW_COPY_AND_ASSIGN(PopnnCTCLossWithLogProbsOp);
};

REGISTER_IPU_OP("PopnnCTCLossWithLogProbs", PopnnCTCLossWithLogProbsOp);

class PopnnCTCInferenceOpBase : public XlaOpKernel, IpuOpKernel {
  int64 top_paths;

 protected:
  explicit PopnnCTCInferenceOpBase(OpKernelConstruction* ctx)
      : XlaOpKernel(ctx) {
    tensorflow::DataType in_dtype;
    int64 blank_index;
    int64 beam_width;

    OP_REQUIRES_OK(ctx, ctx->GetAttr("in_dtype", &in_dtype));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("blank_index", &blank_index));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("beam_width", &beam_width));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("top_paths", &top_paths));

    attribute_map_.AddAttribute("in_dtype", in_dtype);
    attribute_map_.AddAttribute("beam_width", beam_width);
    attribute_map_.AddAttribute("blank_index", blank_index);
    attribute_map_.AddAttribute("top_paths", top_paths);
  }
  virtual PoplarOp OpType() const = 0;

 public:
  ~PopnnCTCInferenceOpBase() override {}

  void Compile(XlaOpKernelContext* ctx) override {
    // Validate shapes.
    auto data_shape = ctx->InputShape(0);

    int64 max_time = data_shape.dim_size(0);
    int64 batch_size = data_shape.dim_size(1);
    int64 num_classes = data_shape.dim_size(2);

    TensorShape expected_lengths_shape;
    TensorShapeUtils::MakeShape(std::vector<int64>({batch_size}),
                                &expected_lengths_shape);
    OP_REQUIRES(
        ctx, ctx->InputShape(1) == expected_lengths_shape,
        errors::InvalidArgument(absl::StrFormat(
            "The data lengths tensor needs to be of shape [%u].", batch_size)));

    xla::Shape label_probs = xla::ShapeUtil::MakeShape(ctx->output_xla_type(0),
                                                       {batch_size, top_paths});
    xla::Shape label_lengths = xla::ShapeUtil::MakeShape(
        xla::PrimitiveType::S32, {batch_size, top_paths});
    xla::Shape decoded_labels = xla::ShapeUtil::MakeShape(
        xla::PrimitiveType::S32, {batch_size, top_paths, max_time});
    xla::Shape output_shape = xla::ShapeUtil::MakeTupleShape(
        {label_probs, label_lengths, decoded_labels});

    xla::XlaBuilder& b = *ctx->builder();

    std::vector<xla::XlaOp> args;
    for (int idx = 0; idx < ctx->num_inputs(); idx++) {
      args.push_back(ctx->Input(idx));
    }

    auto op_name = PoplarOp_Name(OpType());

    xla::XlaOp output_tuple = xla::CustomCall(&b, op_name, args, output_shape,
                                              attribute_map_.Serialise());

    xla::XlaOp output_label_probs = xla::GetTupleElement(output_tuple, 0);
    xla::XlaOp output_label_lengths = xla::GetTupleElement(output_tuple, 1);
    xla::XlaOp output_decoded_labels = xla::GetTupleElement(output_tuple, 2);
    ctx->SetOutput(0, output_label_probs);
    ctx->SetOutput(1, output_label_lengths);
    ctx->SetOutput(2, output_decoded_labels);
  }

  TF_DISALLOW_COPY_AND_ASSIGN(PopnnCTCInferenceOpBase);
};

class PopnnCTCBeamSearchWithLogitsOp : public PopnnCTCInferenceOpBase {
 public:
  explicit PopnnCTCBeamSearchWithLogitsOp(OpKernelConstruction* ctx)
      : PopnnCTCInferenceOpBase(ctx) {}

  ~PopnnCTCBeamSearchWithLogitsOp() override{};

  PoplarOp OpType() const override { return PoplarOp::CTCBeamSearchWithLogits; }

  TF_DISALLOW_COPY_AND_ASSIGN(PopnnCTCBeamSearchWithLogitsOp);
};

REGISTER_IPU_OP("PopnnCTCBeamSearchWithLogits", PopnnCTCBeamSearchWithLogitsOp);

class PopnnCTCBeamSearchWithLogProbsOp : public PopnnCTCInferenceOpBase {
 public:
  explicit PopnnCTCBeamSearchWithLogProbsOp(OpKernelConstruction* ctx)
      : PopnnCTCInferenceOpBase(ctx) {}

  ~PopnnCTCBeamSearchWithLogProbsOp() override{};

  PoplarOp OpType() const override {
    return PoplarOp::CTCBeamSearchWithLogProbs;
  }

  TF_DISALLOW_COPY_AND_ASSIGN(PopnnCTCBeamSearchWithLogProbsOp);
};

REGISTER_IPU_OP("PopnnCTCBeamSearchWithLogProbs",
                PopnnCTCBeamSearchWithLogProbsOp);

}  // namespace tensorflow
