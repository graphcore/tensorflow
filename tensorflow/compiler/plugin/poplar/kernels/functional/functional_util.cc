/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/compiler/plugin/poplar/kernels/functional/functional_util.h"

#include <utility>
#include <vector>

#include "tensorflow/compiler/tf2xla/kernels/tensor_list_utils.h"
#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/xla_compiler.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {
namespace poplarplugin {
XlaCompiler::CompileOptions GetDefaultCompileOptions() {
  XlaCompiler::CompileOptions compile_options;
  compile_options.use_tuple_arg = false;
  compile_options.resolve_compile_time_constants = true;
  compile_options.always_return_tuple = true;
  compile_options.return_updated_values_for_all_resources = true;
  compile_options.is_entry_computation = false;
  compile_options.add_token_input_output = false;
  return compile_options;
}

// Function which tries to get all the arguments to the Op. It optionally tries
// to evaluate any constant inputs to a value so that they can be propagated.
xla::StatusOr<std::vector<XlaCompiler::Argument>> GetXlaArguments(
    XlaOpKernelContext* ctx, const DataTypeVector& input_types,
    int* num_resource_args, bool evaluate_constants) {
  auto builder = ctx->builder();

  std::vector<XlaCompiler::Argument> arguments(input_types.size());
  (*num_resource_args) = 0;
  for (size_t i = 0; i < input_types.size(); ++i) {
    XlaCompiler::Argument& arg = arguments[i];
    DataType type = ctx->input_type(i);

    if (type == DT_RESOURCE) {
      XlaResource* resource;
      TF_RETURN_IF_ERROR(ctx->GetResourceInput(i, &resource));

      arg.initialized = resource->initialized();
      if (!arg.initialized) {
        return errors::Unimplemented("Uninitialized arguments: ", arg.name);
      }
      arg.kind = XlaCompiler::Argument::kResource;
      arg.resource_kind = resource->kind();
      if (arg.resource_kind == XlaResource::kTensorArray) {
        return errors::Unimplemented(
            "Tensor arrays are currently not supported: ", arg.name);
      }

      arg.type = resource->type();
      arg.shape = resource->shape();
      arg.max_array_size = resource->max_array_size();
      for (const auto& gradient : resource->tensor_array_gradients()) {
        arg.tensor_array_gradients.insert(gradient.first);
      }
      arg.name = resource->name();
      VLOG(2) << "Resource " << resource->name()
              << " type: " << DataTypeString(arg.type)
              << " shape: " << arg.HumanString()
              << " initialized: " << arg.initialized;

      (*num_resource_args)++;
    } else {
      arg.type = input_types[i];
      arg.shape = ctx->InputShape(i);
      // Try and replace kParameters with compile-time kConstant.
      const XlaExpression& expression = ctx->InputExpression(i);
      // NOTE: We can not simply check that this is Kind::kConstant because
      // this could be the output of a MetadataOnly op e.g. Size.
      xla::StatusOr<absl::optional<Tensor>> maybe_constant =
          expression.ResolveConstant(ctx->compiler()->client());
      if (evaluate_constants && maybe_constant.ok() &&
          maybe_constant.ValueOrDie().has_value()) {
        arg.kind = XlaCompiler::Argument::kConstant;
        arg.constant_value = std::move(maybe_constant.ValueOrDie().value());
        VLOG(2) << "Constant type: " << DataTypeString(arg.type)
                << " shape: " << arg.HumanString();
      } else {
        arg.kind = XlaCompiler::Argument::kParameter;
        // Use the xla::Shape for the input instead of ctx->InputShape. This
        // is necessary for forwarding shapes of DT_VARIANTs, e.g.
        // TensorLists.
        TF_ASSIGN_OR_RETURN(xla::Shape shape, builder->GetShape(ctx->Input(i)));
        arg.shape = shape;
        if (IsTensorListInput(ctx, i)) {
          // arg.initialized == false means that the element_shape of the list
          // was not available at the time of building the list so an empty list
          // was created instead.
          TF_RETURN_IF_ERROR(
              IsTensorListInitialized(ctx->Input(i), &arg.initialized));
          if (!arg.initialized) {
            return errors::Unimplemented(
                "Uninitialized TensorLists are currently not supported: ",
                arg.name);
          }
        }
        VLOG(2) << "Parameter type: " << DataTypeString(arg.type)
                << " shape: " << arg.HumanString();
      }
    }
  }
  return arguments;
}

xla::StatusOr<std::vector<xla::XlaOp>> GetXlaInputs(
    XlaOpKernelContext* ctx,
    const std::vector<XlaCompiler::Argument>& arguments,
    const std::vector<int>& input_mapping) {
  auto builder = ctx->builder();

  std::vector<xla::XlaOp> inputs(input_mapping.size());
  for (size_t i = 0; i < input_mapping.size(); ++i) {
    const int arg_pos = input_mapping[i];
    switch (arguments[arg_pos].kind) {
      case XlaCompiler::Argument::kResource: {
        XlaResource* resource;
        TF_RETURN_IF_ERROR(ctx->GetResourceInput(arg_pos, &resource));
        TF_RETURN_IF_ERROR(resource->Pack(&inputs[i], builder));
        break;
      }
      case XlaCompiler::Argument::kParameter: {
        inputs[i] = ctx->Input(arg_pos);
        break;
      }
      default: { return errors::InvalidArgument("Invalid argument kind."); }
    }
  }
  return inputs;
}
}  // namespace poplarplugin
}  // namespace tensorflow
