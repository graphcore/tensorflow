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

#include "tensorflow/compiler/plugin/poplar/driver/xla_ipu_common.h"
#include "tensorflow/compiler/plugin/poplar/kernels/ipu_kernels_common.h"

#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"

using namespace xla::poplarplugin;

namespace tensorflow {

class PopopsMultiSliceOp : public XlaOpKernel, IpuOpKernel {
 public:
  using XlaOpKernel::XlaOpKernel;

  explicit PopopsMultiSliceOp(OpKernelConstruction* ctx)
      : XlaOpKernel(ctx), IpuOpKernel() {
    bool indices_are_sorted;
    OP_REQUIRES_OK(ctx,
                   ctx->GetAttr("indices_are_sorted", &indices_are_sorted));
    attribute_map_.AddAttribute("indices_are_sorted", indices_are_sorted);
  }

  void Compile(XlaOpKernelContext* ctx) override {
    const TensorShape input_shape = ctx->InputShape(0);
    const TensorShape indices_shape = ctx->InputShape(1);

    OP_REQUIRES(ctx, TensorShapeUtils::IsMatrix(input_shape),
                errors::InvalidArgument("input shape must be 2D, but got: ",
                                        input_shape.DebugString()));

    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(indices_shape),
                errors::InvalidArgument("indices shape must be 1D, but got: ",
                                        indices_shape.DebugString()));

    TensorShape output_shape = indices_shape;
    output_shape.AddDim(input_shape.dim_size(1));

    xla::PrimitiveType input_type;
    OP_REQUIRES_OK(ctx,
                   DataTypeToPrimitiveType(ctx->input_type(0), &input_type));

    xla::XlaBuilder& b = *ctx->builder();
    std::vector<xla::XlaOp> args = {ctx->Input(0), ctx->Input(1)};
    xla::Shape xla_output_shape =
        TensorShapeToXLAShape(input_type, output_shape);

    xla::XlaOp call_output =
        xla::CustomCall(&b, PoplarOp_Name(PoplarOp::MultiSlice), args,
                        xla_output_shape, attribute_map_.Serialise());
    ctx->SetOutput(0, call_output);
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(PopopsMultiSliceOp);
};

REGISTER_IPU_OP("IpuMultiSlice", PopopsMultiSliceOp);

class PopopsMultiUpdateOp : public XlaOpKernel, IpuOpKernel {
 public:
  PopopsMultiUpdateOp(OpKernelConstruction* ctx, bool is_update_add = false)
      : XlaOpKernel(ctx), IpuOpKernel(), is_update_add_(is_update_add) {
    bool indices_are_sorted;
    OP_REQUIRES_OK(ctx,
                   ctx->GetAttr("indices_are_sorted", &indices_are_sorted));
    attribute_map_.AddAttribute("indices_are_sorted", indices_are_sorted);
  }

  void Compile(XlaOpKernelContext* ctx) override {
    const TensorShape input_shape = ctx->InputShape(0);
    const TensorShape indices_shape = ctx->InputShape(1);
    const TensorShape updates_shape = ctx->InputShape(2);

    OP_REQUIRES(ctx, TensorShapeUtils::IsMatrix(input_shape),
                errors::InvalidArgument("input shape must be 2D, but got: ",
                                        input_shape.DebugString()));

    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(indices_shape),
                errors::InvalidArgument("indices shape must be 1D, but got: ",
                                        indices_shape.DebugString()));

    OP_REQUIRES(ctx, TensorShapeUtils::IsMatrix(updates_shape),
                errors::InvalidArgument("updates shape must be 2D, but got: ",
                                        updates_shape.DebugString()));
    if (is_update_add_) {
      const TensorShape scale_shape = ctx->InputShape(3);
      OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(scale_shape),
                  errors::InvalidArgument("scale must be a scalar, but got: ",
                                          scale_shape.DebugString()));
    }

    xla::XlaBuilder& b = *ctx->builder();

    xla::PrimitiveType input_type;
    OP_REQUIRES_OK(ctx,
                   DataTypeToPrimitiveType(ctx->input_type(0), &input_type));
    xla::Shape xla_output_shape =
        TensorShapeToXLAShape(input_type, input_shape);
    const auto num_inputs = ctx->num_inputs();
    std::vector<xla::XlaOp> args(num_inputs);
    for (int i = 0; i != num_inputs; ++i) {
      args[i] = ctx->Input(i);
    }

    // Add a trailing 1 dimension to indices.
    args[1] = xla::Reshape(args[1], {indices_shape.dim_size(0), 1});

    xla::XlaOp call_output =
        xla::CustomCall(&b,
                        PoplarOp_Name(is_update_add_ ? PoplarOp::MultiUpdateAdd
                                                     : PoplarOp::MultiUpdate),
                        args, xla_output_shape, attribute_map_.Serialise());
    ctx->SetOutput(0, call_output);
  }

 private:
  const bool is_update_add_;
  TF_DISALLOW_COPY_AND_ASSIGN(PopopsMultiUpdateOp);
};
REGISTER_IPU_OP("IpuMultiUpdate", PopopsMultiUpdateOp);

class PopopsMultiUpdateAddOp : public PopopsMultiUpdateOp {
 public:
  PopopsMultiUpdateAddOp(OpKernelConstruction* ctx)
      : PopopsMultiUpdateOp(ctx, true){};

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(PopopsMultiUpdateAddOp);
};
REGISTER_IPU_OP("IpuMultiUpdateAdd", PopopsMultiUpdateAddOp);
}  // namespace tensorflow
