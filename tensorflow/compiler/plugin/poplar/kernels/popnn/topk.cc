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

#include "tensorflow/compiler/plugin/poplar/driver/tools/custom_ops/topk.h"
#include "tensorflow/compiler/plugin/poplar/driver/poplar_platform.h"
#include "tensorflow/compiler/plugin/poplar/driver/xla_ipu_common.h"
#include "tensorflow/compiler/plugin/poplar/kernels/ipu_kernels_common.h"
#include "tensorflow/compiler/xla/client/lib/arithmetic.h"

#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/env.h"

#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/lib/sorting.h"
#include "tensorflow/compiler/xla/literal_util.h"

using namespace xla::poplarplugin;

namespace tensorflow {

class TopKOp : public XlaOpKernel, IpuOpKernel {
 public:
  explicit TopKOp(OpKernelConstruction* context)
      : XlaOpKernel(context), IpuOpKernel() {
    OP_REQUIRES_OK(context, context->GetAttr("sorted", &sorted));
  }

  void Compile(XlaOpKernelContext* context) override {
    TensorShape input_shape = context->InputShape(0);

    // The last dimension is expected to be at least K sized.
    int last_dim = input_shape.dims() - 1;
    int last_dim_size = input_shape.dim_size(last_dim);

    // Get the actual "K" number of values to return;
    OP_REQUIRES_OK(context, context->ConstantInputAsIntScalar(1, &num_k));
    OP_REQUIRES(context, num_k >= 0,
                errors::InvalidArgument("Need k >= 0, got ", num_k));

    // Check that the input shape is at least 2D.
    OP_REQUIRES(context, input_shape.dims() >= 1,
                errors::InvalidArgument("input must be >= 1-D, got shape ",
                                        input_shape.DebugString()));

    OP_REQUIRES(
        context, last_dim_size >= num_k,
        errors::InvalidArgument("input must have at least k columns. Had ",
                                last_dim_size, ", needed ", num_k));

    DataType value_type = output_type(0);
    xla::PrimitiveType value_xla_type;
    OP_REQUIRES_OK(context,
                   DataTypeToPrimitiveType(value_type, &value_xla_type));

    DataType index_type = output_type(1);
    xla::PrimitiveType index_xla_type;
    OP_REQUIRES_OK(context,
                   DataTypeToPrimitiveType(index_type, &index_xla_type));

    xla::XlaOp input = context->Input(0);

    // Set the last dimension to be K as poplar will be returning [batches][K]
    // elements which we will reshape into input[:-1][k].
    std::vector<int64_t> dims;

    // This is just a sort operation if "K" is the same size as the dimensions
    // we should be looking at.
    const bool is_just_sort = (last_dim_size == num_k);

    absl::c_for_each(input_shape,
                     [&dims, &is_just_sort, this](TensorShapeDim dim) {
                       dims.push_back(dim.size);
                     });

    dims[dims.size() - 1] = num_k;

    xla::Shape indices = xla::ShapeUtil::MakeShape(index_xla_type, dims);
    xla::Shape values = xla::ShapeUtil::MakeShape(value_xla_type, dims);

    xla::Shape output_tuple_shape =
        xla::ShapeUtil::MakeTupleShape({values, indices});

    xla::XlaOp output_tuple;

    xla::XlaBuilder* builder = context->builder();

    // Add the K and sorted as attributes.
    attribute_map_.AddAttribute("num_k", num_k);

    // Should we returned a sorted array?
    attribute_map_.AddAttribute("sorted", sorted);

    // Fallback on existing TF impl if requested type isn't supported by
    // poplar or if this is just a sort operation (they already perform the
    // k==size lower to sort optimization).
    const bool fallback =
        is_just_sort || !(input_type(0) == DataType::DT_FLOAT ||
                          input_type(0) == DataType::DT_HALF ||
                          input_type(0) == DataType::DT_INT32 ||
                          input_type(0) == DataType::DT_UINT32);

    if (fallback) {
      output_tuple = xla::TopK(input, num_k);
    } else {
      output_tuple =
          xla::CustomCall(builder, PoplarOp_Name(PoplarOp::TopK), {input},
                          output_tuple_shape, attribute_map_.Serialise());
    }

    context->SetOutput(0, xla::GetTupleElement(output_tuple, 0));
    context->SetOutput(1, xla::GetTupleElement(output_tuple, 1));
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(TopKOp);

  int64_t num_k;

  bool sorted;
};

REGISTER_XLA_OP(
    Name("TopKV2").CompileTimeConstantInput("k").Device(DEVICE_IPU_XLA_JIT),
    TopKOp);

REGISTER_XLA_OP(
    Name("TopK").CompileTimeConstantInput("k").Device(DEVICE_IPU_XLA_JIT),
    TopKOp);

}  // namespace tensorflow
