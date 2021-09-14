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

#define TF_ASSIGN_OR_DEFAULT(lhs, rexpr, default_value)                        \
  TF_ASSIGN_OR_DEFAULT_IMPL(                                                   \
      TF_STATUS_MACROS_CONCAT_NAME(_status_or_value, __COUNTER__), lhs, rexpr, \
      default_value)

#define TF_ASSIGN_OR_DEFAULT_IMPL(statusor, lhs, rexpr, default_value) \
  auto statusor = (rexpr);                                             \
  lhs = (statusor.ok()) ? statusor.ValueOrDie() : (default_value)

using namespace xla::poplarplugin;

namespace tensorflow {

namespace {

static constexpr int32 kApiLevel = 5;

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

xla::StatusOr<void*> GetSymbolAddress(void* handle,
                                      const std::string& sym_name) {
  void* ptr = nullptr;
  TF_RETURN_IF_ERROR(
      Env::Default()->GetSymbolFromLibrary(handle, sym_name.c_str(), &ptr));
  return ptr;
}

xla::StatusOr<uint64> GetSymbolAddressAsInt64(void* handle,
                                              const std::string& sym_name) {
  // Extract the function from the library. We expect (and require) the user
  // function to be an undecorated 'C' type symbol
  // Convert the pointer to a uint64 (attribute map/json doesn't store
  // pointers).
  TF_ASSIGN_OR_RETURN(void* addr, GetSymbolAddress(handle, sym_name));
  return reinterpret_cast<uint64>(addr);
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

  Status LoadLibrary(void** handle, XlaOpKernelContext* context) {
    TF_RETURN_IF_ERROR(
        Env::Default()->LoadDynamicLibrary(library_path.c_str(), handle));

    TF_ASSIGN_OR_DEFAULT(const void* api_level_ptr,
                         GetSymbolAddress(*handle, "custom_op_api_level"),
                         nullptr);

    int32 api_level =
        api_level_ptr ? *reinterpret_cast<const int32*>(api_level_ptr) : 0;

    if (api_level != kApiLevel) {
      return xla::InternalErrorStrCat("Api level of module ", library_path,
                                      ", op name ", op_name, " is ", api_level,
                                      ", expected ", kApiLevel,
                                      ". See section `API Level Versioning` in "
                                      "documentation for more details.");
    }

    // Initialize the symbols which are common to all types of user op.
    TF_ASSIGN_OR_RETURN(int64 fn_ptr,
                        GetSymbolAddressAsInt64(*handle, op_name));

    attribute_map_.AddAttribute("operation_fn", fn_ptr);
    return Status::OK();
  }

  void CreateCustomCall(XlaOpKernelContext* context) {
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

  int64 gradient_size;

  int64 pd_index;

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
    OP_REQUIRES_OK(context, LoadSymbols(context));
    // Set up all the context information to actually create the custom call.
    CreateCustomCall(context);
  }

 private:
  Status LoadSymbols(XlaOpKernelContext* context) {
    void* handle;
    TF_RETURN_IF_ERROR(LoadLibrary(&handle, context));

    // Handle the metadata specific to this type of user op.
    TF_ASSIGN_OR_DEFAULT(int64 metadata_fn_ptr,
                         GetSymbolAddressAsInt64(handle, op_name + "_metadata"),
                         0l);
    TF_ASSIGN_OR_DEFAULT(
        int64 allocator_fn_ptr,
        GetSymbolAddressAsInt64(handle, op_name + "_allocator"), 0l);

    attribute_map_.AddAttribute("metadata_function", metadata_fn_ptr);
    attribute_map_.AddAttribute("allocator_function", allocator_fn_ptr);
    attribute_map_.AddAttribute("gp_path", gp_path);
    attribute_map_.AddAttribute("is_user_read_write", false);

    return Status::OK();
  }

  TF_DISALLOW_COPY_AND_ASSIGN(PoputilUserOp);

  std::string gp_path;
};

class PoputilUserReadWriteOp : public PoputilUserOpBase {
 public:
  explicit PoputilUserReadWriteOp(OpKernelConstruction* context)
      : PoputilUserOpBase(context) {}

  void Compile(XlaOpKernelContext* context) final {
    void* handle;
    OP_REQUIRES_OK(context, LoadLibrary(&handle, context));

    attribute_map_.AddAttribute("metadata_function", static_cast<int64>(0));
    attribute_map_.AddAttribute("allocator_function", static_cast<int64>(0));

    std::string null_string = "";
    attribute_map_.AddAttribute("gp_path", null_string);
    attribute_map_.AddAttribute("is_user_read_write", true);

    // Set up all the context information to actually create the custom call.
    CreateCustomCall(context);
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(PoputilUserReadWriteOp);

  std::string gp_path;
};

REGISTER_IPU_OP("IpuUserOp", PoputilUserOp);
REGISTER_IPU_OP("IpuUserReadWriteOp", PoputilUserReadWriteOp);

}  // namespace tensorflow
