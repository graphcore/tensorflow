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

#include <map>
#include <string>
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
    bool evaluate_constants) {
  std::vector<XlaCompiler::Argument> arguments(input_types.size());
  for (size_t i = 0; i < input_types.size(); ++i) {
    TF_ASSIGN_OR_RETURN(auto arg, GetXlaArgument(ctx, i, evaluate_constants));
    arguments[i] = std::move(arg);
  }

  return arguments;
}

xla::StatusOr<XlaCompiler::Argument> GetXlaArgument(XlaOpKernelContext* ctx,
                                                    size_t input,
                                                    bool evaluate_constant) {
  XlaCompiler::Argument arg;
  DataType type = ctx->input_type(input);

  if (type == DT_RESOURCE) {
    XlaResource* resource;
    TF_RETURN_IF_ERROR(ctx->GetResourceInput(input, &resource));

    arg.initialized = resource->initialized();
    if (!arg.initialized) {
      return errors::Unimplemented("Uninitialized arguments: ", arg.name);
    }
    arg.kind = XlaCompiler::Argument::kResource;
    arg.resource_kind = resource->kind();

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

  } else {
    arg.type = type;
    arg.shape = ctx->InputShape(input);
    // Try and replace kParameters with compile-time kConstant.
    const XlaExpression& expression = ctx->InputExpression(input);
    // NOTE: We can not simply check that this is Kind::kConstant because
    // this could be the output of a MetadataOnly op e.g. Size.
    xla::StatusOr<absl::optional<Tensor>> maybe_constant =
        expression.ResolveConstant(ctx->compiler()->client());
    if (evaluate_constant && maybe_constant.ok() &&
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
      auto* builder = ctx->builder();
      TF_ASSIGN_OR_RETURN(arg.shape, builder->GetShape(ctx->Input(input)));
      arg.type = type;
      VLOG(2) << "Parameter type: " << DataTypeString(arg.type)
              << " shape: " << arg.HumanString();
    }
  }

  return arg;
}

int CountResourceArgs(const std::vector<XlaCompiler::Argument>& arguments) {
  return absl::c_count_if(arguments, [](const XlaCompiler::Argument& arg) {
    return arg.kind == XlaCompiler::Argument::kResource;
  });
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

Status CompileFunction(XlaOpKernelContext* ctx,
                       const XlaCompiler::CompileOptions& options,
                       const NameAttrList& fn_name_attrs,
                       std::vector<XlaCompiler::Argument>& args,
                       XlaCompiler::CompilationResult* result) {
  auto builder = ctx->builder();
  TF_RETURN_IF_ERROR(
      ctx->compiler()->CompileFunction(options, fn_name_attrs, args, result));

  bool has_tensor_array_gradients = false;
  for (const auto& update : result->resource_updates) {
    XlaResource* resource;
    TF_RETURN_IF_ERROR(ctx->GetResourceInput(update.input_index, &resource));
    XlaCompiler::Argument& arg = args[update.input_index];

    // Add any TensorArray gradients touched by the computation to the enclosing
    // graph.
    for (const std::string& grad_source :
         update.tensor_array_gradients_accessed) {
      VLOG(5) << "TensorArray " << resource->name() << " accessed gradient "
              << grad_source;
      XlaResource* gradient;
      TF_RETURN_IF_ERROR(resource->GetOrCreateTensorArrayGradient(
          grad_source, builder, &gradient));
    }

    // Add all of the TensorArray gradients to the argument. For simplicity,
    // we always pass all known gradients.
    for (const auto& gradient : resource->tensor_array_gradients()) {
      arg.tensor_array_gradients.insert(gradient.first);
    }
    if (!resource->tensor_array_gradients().empty()) {
      has_tensor_array_gradients = true;
    }
  }

  if (has_tensor_array_gradients) {
    // Recompile with the new arguments.
    *result = {};
    TF_RETURN_IF_ERROR(
        ctx->compiler()->CompileFunction(options, fn_name_attrs, args, result));
  }

  return Status::OK();
}

FunctionBaseOp::FunctionBaseOp(OpKernelConstruction* ctx)
    : FunctionBaseOp(ctx, /*evaluate constants*/ false) {}
FunctionBaseOp::FunctionBaseOp(OpKernelConstruction* ctx,
                               bool evaluate_constants)
    : XlaOpKernel(ctx), evaluate_constants_(evaluate_constants) {
  OP_REQUIRES_OK(ctx, ctx->GetAttr("to_apply", &to_apply_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr("Tin", &input_types_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr("Tout", &output_types_));
}

void FunctionBaseOp::Compile(XlaOpKernelContext* ctx) {
  auto builder = ctx->builder();
  // First get all the arguments.
  auto arguments_or = GetArguments(ctx);
  OP_REQUIRES_OK(ctx, arguments_or.status());
  std::vector<XlaCompiler::Argument> arguments = arguments_or.ValueOrDie();

  const int num_resource_args = CountResourceArgs(arguments);
  VLOG(2) << "Building function " << ctx->op_kernel().name() << " with "
          << input_types_.size() << " inputs including " << num_resource_args
          << " resources.";

  XlaCompiler::CompileOptions compile_options =
      poplarplugin::GetDefaultCompileOptions();
  compile_options.return_updated_values_for_all_resources = false;

  // Compile the computation.
  XlaCompiler::CompilationResult result;
  OP_REQUIRES_OK(ctx, CompileFunction(ctx, compile_options, *to_apply_,
                                      arguments, &result));

  // Get the non constant XLA arguments.
  auto inputs_or =
      poplarplugin::GetXlaInputs(ctx, arguments, result.input_mapping);
  OP_REQUIRES_OK(ctx, inputs_or.status());
  std::vector<xla::XlaOp> inputs = inputs_or.ValueOrDie();

  auto outputs = xla::Call(builder, *result.computation, inputs);
  // Set the config type of the call.
  OP_REQUIRES_OK(ctx, SetConfig(builder, outputs));

  // Set non resource variable outputs and make sure to set constant outputs
  // as constant.
  int non_const_outputs = 0;
  std::map<int, XlaResource*> output_index_to_resource;
  for (size_t i = 0; i != output_types_.size(); ++i) {
    const XlaCompiler::OutputDescription& output = result.outputs[i];

    if (output.type == DT_RESOURCE) {
      OP_REQUIRES_OK(ctx, ctx->GetResourceInput(output.input_index,
                                                &output_index_to_resource[i]));
    } else if (output.is_tensor_list) {
      ctx->SetTensorListOutput(
          i, xla::GetTupleElement(outputs, non_const_outputs++));
    } else if (output.is_constant) {
      ctx->SetConstantOutput(i, result.outputs[i].constant_value);
    } else {
      ctx->SetOutput(i, xla::GetTupleElement(outputs, non_const_outputs++));
    }
  }

  // Set up the modified resources.
  for (size_t i = 0; i < result.resource_updates.size(); ++i) {
    const XlaCompiler::ResourceUpdate& update = result.resource_updates[i];
    XlaResource* resource;
    OP_REQUIRES_OK(ctx, ctx->GetResourceInput(update.input_index, &resource));
    int pos = non_const_outputs + i;
    OP_REQUIRES(
        ctx, update.modified,
        errors::Internal("Expected the resource output to be modified."));
    OP_REQUIRES_OK(ctx,
                   resource->SetFromPack(
                       arguments[update.input_index].tensor_array_gradients,
                       xla::GetTupleElement(outputs, pos), builder));

    VLOG(2) << "Variable: pos: " << pos << " name: " << resource->name()
            << " modified: " << update.modified
            << " type: " << DataTypeString(update.type)
            << " shape: " << update.shape.DebugString();
  }

  // Set the resource outputs *after* they were updated to make sure to get the
  // latest SSA value.
  for (auto pair : output_index_to_resource) {
    ctx->SetResourceOutput(pair.first, pair.second);
  }
}

xla::StatusOr<std::vector<XlaCompiler::Argument>> FunctionBaseOp::GetArguments(
    XlaOpKernelContext* ctx) const {
  TF_ASSIGN_OR_RETURN(
      auto arguments,
      poplarplugin::GetXlaArguments(ctx, input_types_, evaluate_constants_));
  return arguments;
}

}  // namespace poplarplugin
}  // namespace tensorflow
