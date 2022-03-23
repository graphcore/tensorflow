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

#include "tensorflow/compiler/plugin/poplar/driver/trace.pb.h"
#include "tensorflow/compiler/plugin/poplar/kernels/ipu_kernels_common.h"
#include "tensorflow/compiler/tf2xla/kernels/gather_op_helpers.h"
#include "tensorflow/compiler/xla/client/lib/comparators.h"

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/types.h"

namespace tensorflow {

class RandomShuffleOp : public XlaOpKernel, public IpuOpKernel {
 public:
  explicit RandomShuffleOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {}

  void Compile(XlaOpKernelContext* ctx) override {
    auto builder = ctx->builder();
    xla::XlaOp input = ctx->Input(0);
    TensorShape input_shape = ctx->InputShape(0);
    const int64 n = input_shape.dim_size(0);
    int64 num_elements = 1;
    for (tensorflow::TensorShapeDim dimension : input_shape) {
      num_elements *= dimension.size;
    }

    if (num_elements <= 1 || n <= 1) {
      // No shuffling is required, so copy input directly to output
      ctx->SetOutput(0, input);
      return;
    }

    xla::XlaOp indices = xla::Iota(builder, xla::S32, n);
    auto keys_shape = xla::ShapeUtil::MakeShape(xla::S32, {n});
    auto keys = xla::RngUniform(
        xla::ConstantR0<int32>(builder, std::numeric_limits<int32>::min()),
        xla::ConstantR0<int32>(builder, std::numeric_limits<int32>::max()),
        keys_shape);
    xla::XlaOp sorted_keys = xla::Sort(
        {keys, indices},
        xla::CreateScalarLtComputation({xla::S32, xla::S32}, builder));
    indices = xla::GetTupleElement(sorted_keys, 1);
    auto keys_tensor_shape = TensorShape({n});
    DataType type = ctx->expected_output_dtype(0);
    xla::XlaOp gather;
    OP_REQUIRES_OK(
        ctx, XlaGather(input, input_shape, indices, keys_tensor_shape,
                       /*axis=*/0, /*indices_are_nd=*/false, type, DT_INT32,
                       builder, &gather));
    ctx->SetOutput(0, gather);
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(RandomShuffleOp);
};

REGISTER_IPU_OP("RandomShuffle", RandomShuffleOp);

}  // namespace tensorflow
