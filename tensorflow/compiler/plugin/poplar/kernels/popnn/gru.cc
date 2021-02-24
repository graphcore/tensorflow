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

enum class GruType { GRU = 0, DYNAMIC_GRU, AUGRU };

static PoplarOp get_gru_type(GruType gru_type, bool is_fwd) {
  switch (gru_type) {
    case GruType::GRU:
      return is_fwd ? PoplarOp::GRULayerFwd : PoplarOp::GRULayerBwd;
    case GruType::DYNAMIC_GRU:
      return is_fwd ? PoplarOp::DynamicGRULayerFwd
                    : PoplarOp::DynamicGRULayerBwd;
    case GruType::AUGRU:
      return is_fwd ? PoplarOp::AUGRULayerFwd : PoplarOp::AUGRULayerBwd;
  }
}

/*
  This XlaOpKernel is used for both the popnn GRU, and the popnn dynamic GRU
*/
template <GruType gru_type>
class PopnnGRULayerOp : public XlaOpKernel, IpuOpKernel {
 public:
  explicit PopnnGRULayerOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("activation", &activation_));
    attribute_map_.AddAttribute("activation", activation_);

    OP_REQUIRES_OK(
        ctx, ctx->GetAttr("recurrent_activation", &recurrent_activation_));
    attribute_map_.AddAttribute("recurrent_activation", recurrent_activation_);

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

    // Kernel includes a dynamic input, AUGRU includes a att_score parameter
    if (gru_type != GruType::GRU) {
      TensorShape expected_seq_len_shape;
      TensorShapeUtils::MakeShape(std::vector<int64>({batch_size}),
                                  &expected_seq_len_shape);
      OP_REQUIRES(
          ctx, ctx->InputShape(4) == expected_seq_len_shape,
          errors::InvalidArgument(absl::StrFormat(
              "The seq_len tensor needs to be of shape [%u].", batch_size)));
      if (gru_type == GruType::AUGRU) {
        TensorShape expected_att_score_shape;
        TensorShapeUtils::MakeShape(
            std::vector<int64>({time_steps, batch_size}),
            &expected_att_score_shape);
        OP_REQUIRES(
            ctx, ctx->InputShape(5) == expected_att_score_shape,
            errors::InvalidArgument(
                "The attention_score tensor's dimension need to be the "
                "same as the first two dimensions of the input tensor."));
      }
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

    auto op_name = PoplarOp_Name(get_gru_type(gru_type, /*is_fwd=*/true));

    xla::XlaOp output_tuple = xla::CustomCall(
        &b, op_name, args, output_tuple_shape, attribute_map_.Serialise());

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
  std::string activation_;
  std::string recurrent_activation_;

  TF_DISALLOW_COPY_AND_ASSIGN(PopnnGRULayerOp);
};
REGISTER_IPU_OP("PopnnGRULayer", PopnnGRULayerOp<GruType::GRU>);
REGISTER_IPU_OP("PopnnDynamicGRULayer", PopnnGRULayerOp<GruType::DYNAMIC_GRU>);
REGISTER_IPU_OP("PopnnAUGRULayer", PopnnGRULayerOp<GruType::AUGRU>);

/*
  This XlaOpKernel is used for the Grad op for both the popnn GRU, and
  the popnn dynamic GRU
*/
template <GruType gru_type>
class PopnnGRULayerBackpropOp : public XlaOpKernel, IpuOpKernel {
 public:
  explicit PopnnGRULayerBackpropOp(OpKernelConstruction* ctx)
      : XlaOpKernel(ctx) {
    std::string activation;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("activation", &activation));
    attribute_map_.AddAttribute("activation", activation);

    std::string recurrent_activation;
    OP_REQUIRES_OK(ctx,
                   ctx->GetAttr("recurrent_activation", &recurrent_activation));
    attribute_map_.AddAttribute("recurrent_activation", recurrent_activation);

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

    std::vector<xla::Shape> output_shapes = {
        input_backprop_shape, input_state_backprop_shape, kernel_backprop_shape,
        biases_backprop_shape};

    if (gru_type == GruType::AUGRU) {
      xla::Shape attn_backprop_shape =
          TensorShapeToXLAShape(input_type, ctx->InputShape(5));
      output_shapes.push_back(attn_backprop_shape);
    }

    xla::Shape output_tuple_shape =
        xla::ShapeUtil::MakeTupleShape(output_shapes);

    xla::XlaBuilder& b = *ctx->builder();

    std::vector<xla::XlaOp> args;
    for (int idx = 0; idx < ctx->num_inputs(); idx++) {
      args.push_back(ctx->Input(idx));
    }

    auto op_name = PoplarOp_Name(get_gru_type(gru_type, /*is_fwd=*/false));

    xla::XlaOp output_tuple = xla::CustomCall(
        &b, op_name, args, output_tuple_shape, attribute_map_.Serialise());
    xla::XlaOp input_backprop = xla::GetTupleElement(output_tuple, 0);
    xla::XlaOp input_state_backprop = xla::GetTupleElement(output_tuple, 1);
    xla::XlaOp kernel_backprop = xla::GetTupleElement(output_tuple, 2);
    xla::XlaOp biases_backprop = xla::GetTupleElement(output_tuple, 3);

    ctx->SetOutput(0, input_backprop);
    ctx->SetOutput(1, input_state_backprop);
    ctx->SetOutput(2, kernel_backprop);
    ctx->SetOutput(3, biases_backprop);
    if (gru_type == GruType::AUGRU) {
      xla::XlaOp attn_backprop = xla::GetTupleElement(output_tuple, 4);
      ctx->SetOutput(4, attn_backprop);
    }
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(PopnnGRULayerBackpropOp);
};
REGISTER_IPU_OP("PopnnGRULayerBackprop", PopnnGRULayerBackpropOp<GruType::GRU>);
REGISTER_IPU_OP("PopnnDynamicGRULayerBackprop",
                PopnnGRULayerBackpropOp<GruType::DYNAMIC_GRU>);
REGISTER_IPU_OP("PopnnAUGRULayerBackprop",
                PopnnGRULayerBackpropOp<GruType::AUGRU>);
}  // namespace tensorflow
