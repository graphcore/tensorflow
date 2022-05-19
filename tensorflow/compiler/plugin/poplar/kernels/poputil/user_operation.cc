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

#include <iostream>

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
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/statusor.h"
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

namespace {

// From kernels/datatsteam/feeds.cc
void XlaShapesFromAttr(OpKernelConstruction* ctx,
                       std::vector<xla::Shape>& result) {
  std::vector<TensorShape> shapes;
  std::vector<tensorflow::DataType> types;
  OP_REQUIRES_OK(ctx, ctx->GetAttr("output_shapes", &shapes));
  OP_REQUIRES_OK(ctx, ctx->GetAttr("output_types", &types));

  for (unsigned i = 0; i < shapes.size(); ++i) {
    xla::PrimitiveType xla_type;
    OP_REQUIRES_OK(ctx, DataTypeToPrimitiveType(types[i], &xla_type));
    result.emplace_back(TensorShapeToXLAShape(xla_type, shapes[i]));
  }
}
}  // namespace

class PoputilUserOpBase : public XlaOpKernel, public IpuOpKernel {
 public:
  explicit PoputilUserOpBase(OpKernelConstruction* context)
      : XlaOpKernel(context), IpuOpKernel() {
    // Extract the library path from the operation.
    OP_REQUIRES_OK(context, context->GetAttr("library_path", &library_path));

    OP_REQUIRES_OK(context, context->GetAttr("op_name", &op_name));

    OP_REQUIRES_OK(context, context->GetAttr("gradient_size", &gradient_size));

    OP_REQUIRES_OK(context,
                   context->GetAttr("partial_derivative_index", &pd_index));

    OP_REQUIRES_OK(context, context->GetAttr("attributes", &attributes));

    XlaShapesFromAttr(context, output_shape);
  }

  virtual void Compile(XlaOpKernelContext* context) override {}

  void CreateCustomCall(XlaOpKernelContext* context) {
    attribute_map_.AddAttribute("op_name", op_name);
    attribute_map_.AddAttribute("library_path", library_path);
    attribute_map_.AddAttribute("gradient_size", gradient_size);
    attribute_map_.AddAttribute("partial_derivative_index", pd_index);
    attribute_map_.AddAttribute("attributes", attributes);

    const size_t num_inputs = context->num_inputs();

    // Create the input tuple.
    std::vector<xla::XlaOp> inputs(num_inputs);
    for (size_t i = 0; i < num_inputs; ++i) {
      inputs[i] = context->Input(i);
    }

    xla::XlaBuilder* builder = context->builder();

    // Transform the output shapes read from the user op into a tuple.
    xla::Shape output_tuple_shape =
        xla::ShapeUtil::MakeTupleShape(output_shape);

    // Call the output with the tuple of inputs and expect a tuple of outputs as
    // defined by the user.
    xla::XlaOp call_output =
        xla::CustomCall(builder, PoplarOp_Name(PoplarOp::UserOp), inputs,
                        output_tuple_shape, attribute_map_.Serialise());

    // Extract each element from the output tuple.
    for (size_t i = 0; i < output_shape.size(); ++i) {
      xla::XlaOp output = xla::GetTupleElement(call_output, i);
      context->SetOutput(i, output);
    }
  }

 protected:
  // The path to the shared library as provided by the user.
  std::string library_path;

  // Path to the name of the user op which will be looked up in the shared
  // library.
  std::string op_name;

  std::vector<xla::Shape> output_shape;

  int64_t gradient_size;

  int64_t pd_index;

  std::string attributes;

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(PoputilUserOpBase);
};

class PoputilUserOp : public PoputilUserOpBase {
 public:
  explicit PoputilUserOp(OpKernelConstruction* context)
      : PoputilUserOpBase(context) {
    OP_REQUIRES_OK(context, context->GetAttr("gp_path", &gp_path));
  }

  void Compile(XlaOpKernelContext* context) final {
    attribute_map_.AddAttribute("gp_path", gp_path);
    attribute_map_.AddAttribute("is_user_read_write", false);
    // Set up all the context information to actually create the custom call.
    CreateCustomCall(context);
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(PoputilUserOp);

  std::string gp_path;
};

class PoputilUserReadWriteOp : public PoputilUserOpBase {
 public:
  explicit PoputilUserReadWriteOp(OpKernelConstruction* context)
      : PoputilUserOpBase(context) {}

  void Compile(XlaOpKernelContext* context) final {
    attribute_map_.AddAttribute("metadata_function", static_cast<int64_t>(0));
    attribute_map_.AddAttribute("allocator_function", static_cast<int64_t>(0));
    attribute_map_.AddAttribute("gp_path", std::string());
    attribute_map_.AddAttribute("is_user_read_write", true);

    // Set up all the context information to actually create the custom call.
    CreateCustomCall(context);
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(PoputilUserReadWriteOp);
};

REGISTER_IPU_OP("IpuUserOp", PoputilUserOp);
REGISTER_IPU_OP("IpuUserReadWriteOp", PoputilUserReadWriteOp);

}  // namespace tensorflow
