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
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/statusor.h"

#include "absl/container/flat_hash_set.h"

using namespace xla::poplarplugin;

namespace tensorflow {

namespace {
xla::StatusOr<std::vector<int64>> NumpyBroadcast(
    const absl::Span<const int64>& a, const absl::Span<const int64>& b) {
  std::vector<int64> result(std::max(a.size(), b.size()));

  size_t offset = std::min(a.size(), b.size());
  if (a.size() < b.size()) {
    std::copy(b.rbegin(), b.rbegin() + (result.size() - offset),
              result.rbegin());
  } else {
    std::copy(a.rbegin(), a.rbegin() + (result.size() - offset),
              result.rbegin());
  }

  for (size_t i = 0; i < offset; i++) {
    result[i] = std::max(a[i], b[i]);
  }

  return result;
}

xla::StatusOr<xla::Shape> NumpyBroadcast(const xla::Shape& a,
                                         const xla::Shape& b) {
  TF_ASSIGN_OR_RETURN(auto dims,
                      NumpyBroadcast(a.dimensions(), b.dimensions()));

  return xla::ShapeUtil::MakeValidatedShape(a.element_type(), dims);
}
}  // namespace

class PoputilCodeletExpressionOp : public XlaOpKernel, IpuOpKernel {
 public:
  explicit PoputilCodeletExpressionOp(OpKernelConstruction* ctx)
      : XlaOpKernel(ctx), IpuOpKernel() {
    std::string source;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("source", &source));
    attribute_map_.AddAttribute("source", source);
  }

  void Compile(XlaOpKernelContext* ctx) override {
    const DataType dtype = output_type(0);
    xla::XlaBuilder* b = ctx->builder();
    const auto num_inputs = ctx->num_inputs();

    std::vector<xla::XlaOp> inputs(num_inputs);

    xla::PrimitiveType type;
    OP_REQUIRES_OK(ctx, DataTypeToPrimitiveType(dtype, &type));

    xla::Shape out_shape =
        xla::ShapeUtil::MakeShape(type, absl::Span<const int64>{});

    for (int i = 0; i < num_inputs; ++i) {
      inputs[i] = ctx->Input(i);
      auto input_shape = ctx->InputShape(i);
      auto dtype = ctx->input_type(i);
      xla::Shape xla_shape;
      OP_REQUIRES_OK(ctx,
                     TensorShapeToXLAShape(dtype, input_shape, &xla_shape));
      out_shape = NumpyBroadcast(out_shape, xla_shape).ValueOrDie();
    }

    xla::XlaOp output =
        xla::CustomCall(b, PoplarOp_Name(PoplarOp::CodeletExpressionOp), inputs,
                        out_shape, attribute_map_.Serialise());

    ctx->SetOutput(0, output);
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(PoputilCodeletExpressionOp);
};

REGISTER_XLA_OP(Name("CodeletExpressionOp").Device(DEVICE_IPU_XLA_JIT),
                PoputilCodeletExpressionOp);

}  // namespace tensorflow
