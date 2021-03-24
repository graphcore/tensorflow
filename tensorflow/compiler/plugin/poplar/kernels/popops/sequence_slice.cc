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

#include "tensorflow/compiler/plugin/poplar/driver/xla_ipu_common.h"
#include "tensorflow/compiler/plugin/poplar/kernels/ipu_kernels_common.h"

#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"

namespace tensorflow {
/*
 * Base SequenceSlice op. Directly wraps poplibs SequenceSlice.
 */
class PopopsSequenceSliceOp : public XlaOpKernel, public IpuOpKernel {
 public:
  using XlaOpKernel::XlaOpKernel;

  explicit PopopsSequenceSliceOp(OpKernelConstruction* ctx)
      : XlaOpKernel(ctx), IpuOpKernel() {
    bool zero_unused = false;

    if (ctx->HasAttr("zero_unused")) {
      OP_REQUIRES_OK(ctx, ctx->GetAttr("zero_unused", &zero_unused));
    }

    attribute_map_.AddAttribute("zero_unused", zero_unused);
  }

  void Compile(XlaOpKernelContext* ctx) override {
    xla::XlaBuilder* b = ctx->builder();
    auto dst = ctx->Input(0);
    auto src = ctx->Input(1);
    auto num_elems = ctx->Input(2);
    auto src_offsets = ctx->Input(3);
    auto dst_offsets = ctx->Input(4);

    xla::Shape dst_shape;
    OP_REQUIRES_OK(ctx, TensorShapeToXLAShape(input_type(0), ctx->InputShape(0),
                                              &dst_shape));

    auto output =
        xla::CustomCall(b, PoplarOp_Name(PoplarOp::SequenceSlice),
                        {dst, src, num_elems, src_offsets, dst_offsets},
                        {dst_shape}, attribute_map_.Serialise());

    ctx->SetOutput(0, output);
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(PopopsSequenceSliceOp);
};
REGISTER_IPU_OP("IpuSequenceSlice", PopopsSequenceSliceOp);

/*
 * Specialization of SequenceSlice for sequence unpacking.
 */
class PopopsSequenceSliceUnpackOp : public PopopsSequenceSliceOp {
 public:
  explicit PopopsSequenceSliceUnpackOp(OpKernelConstruction* ctx)
      : PopopsSequenceSliceOp(ctx) {
    int64 total_elements;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("total_elements", &total_elements));
    attribute_map_.AddAttribute("total_elements", total_elements);
  }

  void Compile(XlaOpKernelContext* ctx) override {
    xla::XlaBuilder* b = ctx->builder();

    auto src = ctx->Input(0);
    auto num_elems = ctx->Input(1);
    auto src_offsets = ctx->Input(2);
    auto dst_offsets = ctx->Input(3);

    // Get the input shape - modified below for output shape.
    xla::Shape src_shape;
    OP_REQUIRES_OK(ctx, TensorShapeToXLAShape(input_type(0), ctx->InputShape(0),
                                              &src_shape));

    // Get the total number of elements/slices.
    const auto total_elements =
        attribute_map_.GetAttributeAsInt("total_elements").ValueOrDie();

    // Outermost dimension of the output is equal to the number of slices.
    // Inner dimensions are equal to input dimensions.
    src_shape.set_dimensions(0, total_elements);

    // Add output op.
    auto output =
        xla::CustomCall(b, PoplarOp_Name(PoplarOp::SequenceSliceUnpack),
                        {src, num_elems, src_offsets, dst_offsets}, {src_shape},
                        attribute_map_.Serialise());

    ctx->SetOutput(0, output);
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(PopopsSequenceSliceUnpackOp);
};

REGISTER_IPU_OP("IpuSequenceSliceUnpack", PopopsSequenceSliceUnpackOp);

}  // namespace tensorflow
