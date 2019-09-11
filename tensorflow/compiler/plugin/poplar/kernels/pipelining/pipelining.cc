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
#include "tensorflow/compiler/plugin/poplar/driver/backend_config.pb.h"
#include "tensorflow/compiler/plugin/poplar/kernels/custom_kernels_util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/ipu_kernels_common.h"

#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/xla_compiler.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/op_kernel.h"

using namespace xla::poplarplugin;

namespace tensorflow {
namespace {

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

// Function which tries to get all the arguments to the Op. It tries to evaluate
// any constant inputs to a value so that they can be propagated.
xla::StatusOr<std::vector<XlaCompiler::Argument>> GetXlaArguments(
    XlaOpKernelContext* ctx, const DataTypeVector& input_types,
    int* num_resource_args, int* num_non_constant_args) {
  auto builder = ctx->builder();

  std::vector<XlaCompiler::Argument> arguments(input_types.size());
  (*num_resource_args) = 0;
  (*num_non_constant_args) = 0;
  for (int i = 0; i < input_types.size(); ++i) {
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
      (*num_non_constant_args)++;
    } else {
      // Try and replace kParameters with compile-time kConstant.
      const XlaExpression& expression = ctx->InputExpression(i);
      // NOTE: We can not simply check that this is Kind::kConstant because
      // this could be the output of a MetadataOnly op e.g. Size.
      xla::StatusOr<absl::optional<Tensor>> maybe_constant =
          expression.ResolveConstant(ctx->compiler()->client());
      if (maybe_constant.ok() && maybe_constant.ValueOrDie().has_value()) {
        arg.kind = XlaCompiler::Argument::kConstant;
        arg.type = expression.dtype();
        arg.constant_value = std::move(maybe_constant.ValueOrDie().value());
        arg.shape = expression.GetShape().ValueOrDie();
        VLOG(2) << "Constant type: " << DataTypeString(arg.type)
                << " shape: " << arg.HumanString();
      } else {
        arg.kind = XlaCompiler::Argument::kParameter;
        arg.type = input_types[i];
        // Use the xla::Shape for the input instead of ctx->InputShape. This
        // is necessary for forwarding shapes of DT_VARIANTs, e.g.
        // TensorLists.
        TF_ASSIGN_OR_RETURN(xla::Shape shape, builder->GetShape(ctx->Input(i)));
        arg.shape = shape;
        (*num_non_constant_args)++;
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
    const int num_non_constant_args) {
  auto builder = ctx->builder();

  std::vector<xla::XlaOp> inputs(num_non_constant_args);
  for (int i = 0, next_param_idx = 0; i < arguments.size(); ++i) {
    switch (arguments[i].kind) {
      case XlaCompiler::Argument::kResource: {
        XlaResource* resource;
        TF_RETURN_IF_ERROR(ctx->GetResourceInput(i, &resource));
        TF_RETURN_IF_ERROR(resource->Pack(&inputs[next_param_idx++], builder));
        break;
      }
      case XlaCompiler::Argument::kParameter: {
        inputs[next_param_idx++] = ctx->Input(i);
        break;
      }
      case XlaCompiler::Argument::kConstant: {
        // Do nothing - the constant has been lowered into the computation.
        break;
      }
      default: { return errors::InvalidArgument("Invalid argument kind."); }
    }
  }
  return inputs;
}

class PipelineStageOp : public XlaOpKernel {
 public:
  explicit PipelineStageOp(OpKernelConstruction* ctx, bool is_forward = true)
      : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("to_apply", &to_apply_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("Tin", &input_types_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("Tout", &output_types_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("stage_id", &stage_id_));
    call_config_type_ =
        is_forward ? PoplarBackendConfig::CallConfig::PipelineStage
                   : PoplarBackendConfig::CallConfig::PipelineStageBackward;
  }

  void Compile(XlaOpKernelContext* ctx) override {
    auto builder = ctx->builder();
    // First get all the arguments.
    int num_resource_args = 0;
    int num_non_constant_args = 0;
    auto arguments_or = GetXlaArguments(ctx, input_types_, &num_resource_args,
                                        &num_non_constant_args);
    OP_REQUIRES_OK(ctx, arguments_or.status());
    std::vector<XlaCompiler::Argument> arguments = arguments_or.ValueOrDie();

    VLOG(2) << "Building PipelineStage function with " << input_types_.size()
            << " inputs including " << num_resource_args << " resources.";
    // Compile the computation.
    XlaCompiler::CompilationResult result;
    OP_REQUIRES_OK(
        ctx, ctx->compiler()->CompileFunction(GetDefaultCompileOptions(),
                                              *to_apply_, arguments, &result));

    // Get the non constant XLA arguments.
    auto inputs_or = GetXlaInputs(ctx, arguments, num_non_constant_args);
    OP_REQUIRES_OK(ctx, inputs_or.status());
    std::vector<xla::XlaOp> inputs = inputs_or.ValueOrDie();

    auto outputs = xla::Call(builder, *result.computation, inputs);
    // Set the config type of the call.
    OP_REQUIRES_OK(
        ctx, builder->SetInstructionFrontendAttribute(
                 outputs, FrontendAttributeId_Name(CALL_CONFIG_TYPE),
                 PoplarBackendConfig_CallConfig_Type_Name(call_config_type_)));

    // Set the stage id.
    OP_REQUIRES_OK(ctx,
                   builder->SetInstructionFrontendAttribute(
                       outputs, FrontendAttributeId_Name(PIPELINE_STAGE_ID),
                       std::to_string(stage_id_)));
    // Sets non-variable outputs.
    // Make sure to set constant outputs as constant.
    int computation_output = 0;
    for (int i = 0; i < output_types_.size(); ++i) {
      if (result.outputs[i].is_constant) {
        ctx->SetConstantOutput(i, result.outputs[i].constant_value);
      } else {
        ctx->SetOutput(i, xla::GetTupleElement(outputs, computation_output++));
      }
    }

    // Updates the values of any resource variables modified by the function
    // call.
    for (int i = 0; i < result.resource_updates.size(); ++i) {
      const XlaCompiler::ResourceUpdate& update = result.resource_updates[i];
      XlaResource* resource;
      OP_REQUIRES_OK(ctx, ctx->GetResourceInput(update.input_index, &resource));
      if (update.modified) {
        int pos = computation_output + i;
        OP_REQUIRES_OK(ctx,
                       resource->SetFromPack(
                           arguments[update.input_index].tensor_array_gradients,
                           xla::GetTupleElement(outputs, pos), builder));
      }
      VLOG(2) << "Variable: pos: " << update.input_index
              << " name: " << resource->name()
              << " modified: " << update.modified
              << " type: " << DataTypeString(update.type)
              << " shape: " << update.shape.DebugString();
    }
  }

 private:
  const NameAttrList* to_apply_;
  DataTypeVector input_types_;
  DataTypeVector output_types_;
  PoplarBackendConfig_CallConfig_Type call_config_type_;
  int64 stage_id_;

  TF_DISALLOW_COPY_AND_ASSIGN(PipelineStageOp);
};
REGISTER_IPU_OP("PipelineStage", PipelineStageOp);

class PipelineStageBackwardOp : public PipelineStageOp {
 public:
  explicit PipelineStageBackwardOp(OpKernelConstruction* ctx)
      : PipelineStageOp(ctx, false) {}

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(PipelineStageBackwardOp);
};
REGISTER_IPU_OP("PipelineStageBackward", PipelineStageBackwardOp);

class PipelineOp : public XlaOpKernel {
 public:
  explicit PipelineOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("to_apply", &to_apply_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("Tin", &input_types_));
    DataTypeVector output_types;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("Tout", &output_types));
    OP_REQUIRES(ctx, output_types.size() == 0,
                errors::InvalidArgument(
                    "Expected PipelineStage to have no explicit outputs."));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("repeat_count", &repeat_count_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
    auto builder = ctx->builder();
    // First get all the arguments and compile the computation.
    int num_resource_args = 0;
    int num_non_constant_args = 0;
    auto arguments_or = GetXlaArguments(ctx, input_types_, &num_resource_args,
                                        &num_non_constant_args);
    OP_REQUIRES_OK(ctx, arguments_or.status());
    std::vector<XlaCompiler::Argument> arguments = arguments_or.ValueOrDie();

    VLOG(2) << "Building PipelineStage function with " << input_types_.size()
            << " inputs including " << num_resource_args << " resources.";
    // Compile the computation.
    XlaCompiler::CompilationResult result;
    OP_REQUIRES_OK(
        ctx, ctx->compiler()->CompileFunction(GetDefaultCompileOptions(),
                                              *to_apply_, arguments, &result));

    // Get the non constant XLA arguments.
    auto inputs_or = GetXlaInputs(ctx, arguments, num_non_constant_args);
    OP_REQUIRES_OK(ctx, inputs_or.status());
    std::vector<xla::XlaOp> inputs = inputs_or.ValueOrDie();

    // For pipelines we make sure that the inputs and outputs have the same
    // shape and that the values for every output at index `i` are:
    // 1. the input value `i` if the input is not a resource variable
    // 2. the input value `i` if the input is a resource variable which has not
    //   been modified
    // 3. the modified resource variable corresponding to the value at input `i`
    // To do so we wrap the pipeline in another call, and set up the tuple
    // accordingly.
    xla::XlaComputation wrapped_pipeline;
    {
      std::unique_ptr<xla::XlaBuilder> cb =
          ctx->builder()->CreateSubBuilder("pipeline_wrapper");
      std::vector<xla::XlaOp> inner_inputs(inputs.size());
      std::vector<xla::XlaOp> inner_outputs(inputs.size());
      // First handle cases 1 and 2.
      for (int input_idx = 0; input_idx != inputs.size(); ++input_idx) {
        auto param = xla::Parameter(cb.get(), input_idx,
                                    result.xla_input_shapes[input_idx],
                                    absl::StrCat("input/", input_idx));
        inner_inputs[input_idx] = param;
        inner_outputs[input_idx] = param;
      }
      // Call the computation which is wrapped.
      auto inner_call = xla::Call(cb.get(), *result.computation, inner_inputs);
      // Now go through any resource updates and add necessary GTEs and handle
      // case 3.
      for (int i = 0; i < result.resource_updates.size(); ++i) {
        const XlaCompiler::ResourceUpdate& update = result.resource_updates[i];
        if (update.modified) {
          inner_outputs[update.input_index] =
              xla::GetTupleElement(inner_call, i);
        }
      }
      xla::Tuple(cb.get(), inner_outputs);
      auto comp_or = cb->Build();
      OP_REQUIRES_OK(ctx, comp_or.status());
      wrapped_pipeline = std::move(comp_or.ValueOrDie());
    }
    // Create the actual call.
    auto outputs = xla::Call(builder, wrapped_pipeline, inputs);
    // Set the config type of the call.
    OP_REQUIRES_OK(ctx, builder->SetInstructionFrontendAttribute(
                            outputs, FrontendAttributeId_Name(CALL_CONFIG_TYPE),
                            PoplarBackendConfig_CallConfig_Type_Name(
                                PoplarBackendConfig::CallConfig::Pipeline)));
    // Set the repeat count.
    OP_REQUIRES_OK(ctx,
                   builder->SetInstructionFrontendAttribute(
                       outputs, FrontendAttributeId_Name(PIPELINE_REPEAT_COUNT),
                       std::to_string(repeat_count_)));
    // A pipeline has no explicit outputs, only updates of resource variables.
    for (const XlaCompiler::ResourceUpdate& update : result.resource_updates) {
      XlaResource* resource;
      OP_REQUIRES_OK(ctx, ctx->GetResourceInput(update.input_index, &resource));
      if (update.modified) {
        // Add a GTE from the same input index.
        OP_REQUIRES_OK(
            ctx,
            resource->SetFromPack(
                arguments[update.input_index].tensor_array_gradients,
                xla::GetTupleElement(outputs, update.input_index), builder));
      }
      VLOG(2) << "Variable: pos: " << update.input_index
              << " name: " << resource->name()
              << " modified: " << update.modified
              << " type: " << DataTypeString(update.type)
              << " shape: " << update.shape.DebugString();
    }
  }

 private:
  const NameAttrList* to_apply_;
  DataTypeVector input_types_;
  int64 repeat_count_;

  TF_DISALLOW_COPY_AND_ASSIGN(PipelineOp);
};
REGISTER_IPU_OP("Pipeline", PipelineOp);

}  // namespace
}  // namespace tensorflow
