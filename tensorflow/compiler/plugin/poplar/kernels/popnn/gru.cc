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
#include "tensorflow/core/util/tensor_format.h"

using namespace xla::poplarplugin;

namespace tensorflow {
class PopnnGRULayerOp : public XlaOpKernel, IpuOpKernel {
 public:
  explicit PopnnGRULayerOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("num_channels", &num_channels_));
    attribute_map_.AddAttribute("num_channels", num_channels_);
    OP_REQUIRES_OK(ctx, ctx->GetAttr("is_training", &is_training_));
    attribute_map_.AddAttribute("is_training", is_training_);
    tensorflow::DataType partials_dtype;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("partials_dtype", &partials_dtype));
    attribute_map_.AddAttribute("partials_dtype", partials_dtype);
    OP_REQUIRES_OK(ctx, ctx->GetAttr("reset_after", &reset_after_));
    attribute_map_.AddAttribute("reset_after", reset_after_);
  }

 public:
  ~PopnnGRULayerOp() override{};

  void Compile(XlaOpKernelContext* ctx) override {
    xla::PrimitiveType input_type;
    OP_REQUIRES_OK(ctx,
                   DataTypeToPrimitiveType(ctx->input_type(0), &input_type));

    const TensorShape input_shape = ctx->InputShape(0);
    const auto time_steps = input_shape.dim_size(0);
    const auto batch_size = input_shape.dim_size(1);
    const auto input_size = input_shape.dim_size(2);

    // Validate shapes.
    TensorShape expected_input_state_shape;
    TensorShapeUtils::MakeShape(std::vector<int64>({batch_size, num_channels_}),
                                &expected_input_state_shape);
    OP_REQUIRES(
        ctx, ctx->InputShape(1) == expected_input_state_shape,
        errors::InvalidArgument(absl::StrFormat(
            "The initial hidden state tensor needs to be of shape [%u, %u].",
            batch_size, num_channels_)));

    TensorShape expected_kernel_shape;
    TensorShapeUtils::MakeShape(
        std::vector<int64>({input_size + num_channels_, 3 * num_channels_}),
        &expected_kernel_shape);
    OP_REQUIRES(ctx, ctx->InputShape(2) == expected_kernel_shape,
                errors::InvalidArgument(absl::StrFormat(
                    "The input kernel tensor needs to be of shape [%u, %u].",
                    input_size + num_channels_, 3 * num_channels_)));

    TensorShape expected_biases_shape;
    if (reset_after_) {
      TensorShapeUtils::MakeShape(std::vector<int64>({3, 2, num_channels_}),
                                  &expected_biases_shape);
      OP_REQUIRES(ctx, ctx->InputShape(3) == expected_biases_shape,
                  errors::InvalidArgument(absl::StrFormat(
                      "The biases tensor needs to be of shape [3, 2, %u].",
                      num_channels_)));
    } else {
      TensorShapeUtils::MakeShape(std::vector<int64>({3, num_channels_}),
                                  &expected_biases_shape);
      OP_REQUIRES(ctx, ctx->InputShape(3) == expected_biases_shape,
                  errors::InvalidArgument(absl::StrFormat(
                      "The biases tensor needs to be of shape [3, %u].",
                      num_channels_)));
    }

    xla::Shape output_seq_shape = xla::ShapeUtil::MakeShape(
        input_type, {time_steps, batch_size, num_channels_});
    xla::Shape output_state_shape =
        xla::ShapeUtil::MakeShape(input_type, {batch_size, num_channels_});
    // The 3 in intermidate shape represents the number of gates.
    int num_intermediates = reset_after_ ? 4 : 3;
    xla::Shape intermediates_shape = xla::ShapeUtil::MakeShape(
        input_type, {time_steps, num_intermediates, batch_size, num_channels_});

    std::vector<xla::Shape> output_shapes = {output_seq_shape,
                                             output_state_shape};
    if (is_training_) {
      output_shapes.push_back(intermediates_shape);
    }

    xla::Shape output_tuple_shape =
        xla::ShapeUtil::MakeTupleShape(output_shapes);

    xla::XlaBuilder& b = *ctx->builder();

    std::vector<xla::XlaOp> args;
    for (int idx = 0; idx < ctx->num_inputs(); idx++) {
      args.push_back(ctx->Input(idx));
    }

    xla::XlaOp output_tuple =
        xla::CustomCall(&b, PoplarOp_Name(PoplarOp::GRULayerFwd), args,
                        output_tuple_shape, attribute_map_.Serialise());

    xla::XlaOp output_seq = xla::GetTupleElement(output_tuple, 0);
    xla::XlaOp output_state = xla::GetTupleElement(output_tuple, 1);
    xla::XlaOp intermediates;
    if (is_training_) {
      intermediates = xla::GetTupleElement(output_tuple, 2);
    } else {
      intermediates = xla::Broadcast(XlaHelpers::Zero(&b, ctx->input_type(0)),
                                     intermediates_shape.dimensions());
    }

    ctx->SetOutput(0, output_seq);
    ctx->SetOutput(1, output_state);
    ctx->SetOutput(2, intermediates);
  }

 private:
  bool is_training_;
  int32 num_channels_;
  bool reset_after_;

  TF_DISALLOW_COPY_AND_ASSIGN(PopnnGRULayerOp);
};
REGISTER_IPU_OP("PopnnGRULayer", PopnnGRULayerOp);

class PopnnGRULayerBackpropOp : public XlaOpKernel, IpuOpKernel {
 public:
  explicit PopnnGRULayerBackpropOp(OpKernelConstruction* ctx)
      : XlaOpKernel(ctx) {
    int32 num_channels;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("num_channels", &num_channels));
    attribute_map_.AddAttribute("num_channels", num_channels);
    bool is_training;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("is_training", &is_training));
    attribute_map_.AddAttribute("is_training", is_training);
    tensorflow::DataType partials_dtype;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("partials_dtype", &partials_dtype));
    attribute_map_.AddAttribute("partials_dtype", partials_dtype);
    bool reset_after;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("reset_after", &reset_after));
    attribute_map_.AddAttribute("reset_after", reset_after);
  }

 public:
  ~PopnnGRULayerBackpropOp() override{};

  void Compile(XlaOpKernelContext* ctx) override {
    // Don't need to validate shapes as this is a grad op.
    xla::PrimitiveType input_type;
    OP_REQUIRES_OK(ctx,
                   DataTypeToPrimitiveType(ctx->input_type(0), &input_type));

    xla::Shape input_backprop_shape =
        TensorShapeToXLAShape(input_type, ctx->InputShape(0));
    xla::Shape input_state_backprop_shape =
        TensorShapeToXLAShape(input_type, ctx->InputShape(1));
    xla::Shape kernel_backprop_shape =
        TensorShapeToXLAShape(input_type, ctx->InputShape(2));
    xla::Shape biases_backprop_shape =
        TensorShapeToXLAShape(input_type, ctx->InputShape(3));

    xla::Shape output_tuple_shape = xla::ShapeUtil::MakeTupleShape(
        {input_backprop_shape, input_state_backprop_shape,
         kernel_backprop_shape, biases_backprop_shape});

    xla::XlaBuilder& b = *ctx->builder();

    std::vector<xla::XlaOp> args;
    for (int idx = 0; idx < ctx->num_inputs(); idx++) {
      args.push_back(ctx->Input(idx));
    }

    xla::XlaOp output_tuple =
        xla::CustomCall(&b, PoplarOp_Name(PoplarOp::GRULayerBwd), args,
                        output_tuple_shape, attribute_map_.Serialise());
    xla::XlaOp input_backprop = xla::GetTupleElement(output_tuple, 0);
    xla::XlaOp input_state_backprop = xla::GetTupleElement(output_tuple, 1);
    xla::XlaOp kernel_backprop = xla::GetTupleElement(output_tuple, 2);
    xla::XlaOp biases_backprop = xla::GetTupleElement(output_tuple, 3);

    ctx->SetOutput(0, input_backprop);
    ctx->SetOutput(1, input_state_backprop);
    ctx->SetOutput(2, kernel_backprop);
    ctx->SetOutput(3, biases_backprop);
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(PopnnGRULayerBackpropOp);
};
REGISTER_IPU_OP("PopnnGRULayerBackprop", PopnnGRULayerBackpropOp);
}  // namespace tensorflow
