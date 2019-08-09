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
#include "tensorflow/compiler/xla/shape_util.h"

#include <iostream>
#include "absl/container/flat_hash_set.h"

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

// This is not the public facing library API but we need to call this one
// because we need access to the OpDef information returned by the last two
// arguments.
Status LoadLibrary(const char* library_filename, void** result,
                   const void** buf, size_t* len);

class PoputilUserOp : public XlaOpKernel, IpuOpKernel {
 private:
  // Helper structure to wrap the parameters/returned values of the LoadLibrary
  // call.
  struct LibraryLoadInfo {
    // System abstract handle to the library handle returned by system dynamic
    // library open call.
    void* handle;

    // Pointer to the list of operations contained within the .so.
    const void* buffer;

    // Size of the above buffer.
    size_t size;
  };

 public:
  explicit PoputilUserOp(OpKernelConstruction* context)
      : XlaOpKernel(context), IpuOpKernel() {
    // Extract the library path from the operation.
    OP_REQUIRES_OK(context, context->GetAttr("library_path", &library_path));

    OP_REQUIRES_OK(context, context->GetAttr("gp_path", &gp_path));

    OP_REQUIRES_OK(context, context->GetAttr("op_name", &op_name));

    OP_REQUIRES_OK(context, context->GetAttr("is_gradient", &is_gradient));
    XlaShapesFromAttr(context, output_shape);
  }

  void Compile(XlaOpKernelContext* context) override {
    LibraryLoadInfo library;

    Status status = ::tensorflow::LoadLibrary(
        library_path.c_str(), &library.handle, &library.buffer, &library.size);

    if (!status.ok()) {
      OP_REQUIRES_OK(context,
                     errors::InvalidArgument(
                         "Couldn't read shared library: " + library_path +
                         " with error:" + status.ToString()));
    }

    int64 fn_ptr = GetSymbolAddressAsInt64(library, op_name);
    if (fn_ptr == 0) {
      OP_REQUIRES_OK(context,
                     errors::InvalidArgument("Couldn't read " + op_name +
                                             " symbol from library"));
    }
    attribute_map_.AddAttribute("operation_fn", fn_ptr);

    attribute_map_.AddAttribute("is_gradient", is_gradient);

    attribute_map_.AddAttribute(
        "elementwise_fn", GetSymbolAddressAsInt64(library, "IsElementWise"));

    attribute_map_.AddAttribute(
        "allocate_input_fn", GetSymbolAddressAsInt64(library, "AllocateInput"));

    attribute_map_.AddAttribute("gp_path", gp_path);

    const auto num_inputs = context->num_inputs();

    // Create the input tuple.
    std::vector<xla::XlaOp> inputs(num_inputs);
    for (int i = 0; i < num_inputs; ++i) {
      inputs[i] = context->Input(i);
    }

    xla::XlaBuilder* builder = context->builder();

    // Transform the output shapes read from the user op into a tuple.
    xla::Shape output_tuple_shape =
        xla::ShapeUtil::MakeTupleShape(output_shape);

    // Call the output with the tuple of inputs and expect a tuple of outputs as
    // defined by the user.
    xla::XlaOp call_output = xla::CustomCall(
        builder,
        GetPoplibsCustomOpTargetString(PoplibsOp::Poputil, PoplibsOp::UserOp),
        inputs, output_tuple_shape, attribute_map_.Serialise());

    // Extract each element from the output tuple.
    for (int i = 0; i < output_shape.size(); ++i) {
      xla::XlaOp output = xla::GetTupleElement(call_output, i);
      context->SetOutput(i, output);
    }
  }

 private:
  int64 GetSymbolAddressAsInt64(const LibraryLoadInfo& library,
                                const std::string& sym_name) {
    // Extract the function from the library. We expect (and require) the user
    // function to be an undecorated 'C' type symbol
    void* function_ptr = nullptr;
    Status status = Env::Default()->GetSymbolFromLibrary(
        library.handle, sym_name.c_str(), &function_ptr);

    if (!status.ok()) {
      return 0l;
    }

    // Convert the pointer to a uint64 (attribute map/json doesn't store
    // pointers).
    return reinterpret_cast<uint64>(function_ptr);
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(PoputilUserOp);

  // The path to the shared library as provided by the user.
  std::string library_path;

  // The (optional) path to any codelets which have been added.
  std::string gp_path;

  // Path to the name of the user op which will be looked up in the shared
  // library.
  std::string op_name;

  std::vector<xla::Shape> output_shape;

  bool is_gradient;
};

REGISTER_XLA_OP(Name("IpuUserOp")
                    .Device(DEVICE_IPU_XLA_JIT)
                    .CompileTimeConstantInput("library_path"),
                PoputilUserOp);

}  // namespace tensorflow
