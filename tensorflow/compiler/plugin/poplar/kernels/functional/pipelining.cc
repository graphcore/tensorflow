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
#include "tensorflow/compiler/plugin/poplar/kernels/functional/functional_util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/functional/rearrange_function_arguments.h"
#include "tensorflow/compiler/plugin/poplar/kernels/ipu_kernels_common.h"
#include "tensorflow/compiler/tf2xla/kernels/tensor_list_utils.h"
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
    auto arguments_or =
        poplarplugin::GetXlaArguments(ctx, input_types_, &num_resource_args);
    OP_REQUIRES_OK(ctx, arguments_or.status());
    std::vector<XlaCompiler::Argument> arguments = arguments_or.ValueOrDie();

    VLOG(2) << "Building PipelineStage (" << ctx->op_kernel().name()
            << ") function with " << input_types_.size() << " inputs including "
            << num_resource_args << " resources.";
    XlaCompiler::CompileOptions compile_options =
        poplarplugin::GetDefaultCompileOptions();
    compile_options.return_updated_values_for_all_resources = false;

    // Compile the computation.
    XlaCompiler::CompilationResult result;
    OP_REQUIRES_OK(ctx, ctx->compiler()->CompileFunction(
                            compile_options, *to_apply_, arguments, &result));

    // Get the non constant XLA arguments.
    auto inputs_or =
        poplarplugin::GetXlaInputs(ctx, arguments, result.input_mapping);
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

    // Set non resource variable outputs and make sure to set constant outputs
    // as constant.
    int non_const_outputs = 0;
    for (size_t i = 0; i != output_types_.size(); ++i) {
      const XlaCompiler::OutputDescription& output = result.outputs[i];

      if (output.is_constant) {
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

class PipelineResourceUpdateOp : public XlaOpKernel {
 public:
  explicit PipelineResourceUpdateOp(OpKernelConstruction* ctx)
      : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("to_apply", &to_apply_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("Tin", &input_types_));
    DataTypeVector output_types;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("Tout", &output_types));
    OP_REQUIRES(ctx, output_types.size() == 0,
                errors::InvalidArgument("Expected PipelineResourceUpdate "
                                        "to have no explicit outputs."));
  }

  void Compile(XlaOpKernelContext* ctx) override {
    auto builder = ctx->builder();
    // First get all the arguments and compile the computation.
    int num_resource_args = 0;
    auto arguments_or =
        poplarplugin::GetXlaArguments(ctx, input_types_, &num_resource_args);
    OP_REQUIRES_OK(ctx, arguments_or.status());
    std::vector<XlaCompiler::Argument> arguments = arguments_or.ValueOrDie();

    VLOG(2) << "Building PipelineResourceUpdate (" << ctx->op_kernel().name()
            << ") function with " << input_types_.size() << " inputs including "
            << num_resource_args << " resources.";

    // Rewrite the PipelineResourceUpdate function such that arguments to
    // PipelineResourceUpdate ops are rearranged and resource variables
    // moved to the back.
    NameAttrList new_to_apply;
    OP_REQUIRES_OK(
        ctx,
        RearrangeFunctionArguments(
            [&ctx](const NameAttrList& function, const FunctionBody** fbody) {
              return ctx->compiler()->FindFunctionBody(function, fbody);
            },
            new_to_apply, *to_apply_, ctx->compiler()->local_flib_def()));

    XlaCompiler::CompileOptions compile_options =
        poplarplugin::GetDefaultCompileOptions();
    compile_options.return_updated_values_for_all_resources = true;

    // Compile the computation.
    XlaCompiler::CompilationResult result;
    OP_REQUIRES_OK(ctx, ctx->compiler()->CompileFunction(
                            compile_options, new_to_apply, arguments, &result));

    // Get the non constant XLA arguments.
    auto inputs_or =
        poplarplugin::GetXlaInputs(ctx, arguments, result.input_mapping);
    OP_REQUIRES_OK(ctx, inputs_or.status());
    std::vector<xla::XlaOp> inputs = inputs_or.ValueOrDie();

    auto outputs = xla::Call(builder, *result.computation, inputs);
    // Set the config type of the call.
    OP_REQUIRES_OK(
        ctx, builder->SetInstructionFrontendAttribute(
                 outputs, FrontendAttributeId_Name(CALL_CONFIG_TYPE),
                 PoplarBackendConfig_CallConfig_Type_Name(
                     PoplarBackendConfig::CallConfig::PipelineResourceUpdate)));

    // We expect the resource update stage to only have resource outputs. This
    // code assumes that the outputs are all of the resource variables in the
    // same order as the inputs. The `return_updated_values_for_all_resources`
    // flag ensures this. Therefore we must update all resources, regardless
    // of if they are updated by the function or not.
    for (size_t i = 0; i < result.resource_updates.size(); ++i) {
      const XlaCompiler::ResourceUpdate& update = result.resource_updates[i];
      XlaResource* resource;
      OP_REQUIRES_OK(ctx, ctx->GetResourceInput(update.input_index, &resource));
      OP_REQUIRES_OK(ctx,
                     resource->SetFromPack(
                         arguments[update.input_index].tensor_array_gradients,
                         xla::GetTupleElement(outputs, i), builder));

      VLOG(2) << "Variable: pos: " << i << " name: " << resource->name()
              << " modified: " << update.modified
              << " type: " << DataTypeString(update.type)
              << " shape: " << update.shape.DebugString();
    }
  }

 private:
  const NameAttrList* to_apply_;
  DataTypeVector input_types_;

  TF_DISALLOW_COPY_AND_ASSIGN(PipelineResourceUpdateOp);
};
REGISTER_IPU_OP("PipelineResourceUpdate", PipelineResourceUpdateOp);

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
    OP_REQUIRES_OK(ctx, ctx->GetAttr("pipeline_depth", &pipeline_depth_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("repeat_count", &repeat_count_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("schedule", &schedule_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("offload_weight_update_variables",
                                     &offload_weight_update_variables_));
    OP_REQUIRES_OK(
        ctx, ctx->GetAttr("pipeline_poplar_config", &pipeline_poplar_config_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
    auto builder = ctx->builder();
    // First get all the arguments and compile the computation.
    int num_resource_args = 0;
    auto arguments_or =
        poplarplugin::GetXlaArguments(ctx, input_types_, &num_resource_args);
    OP_REQUIRES_OK(ctx, arguments_or.status());
    std::vector<XlaCompiler::Argument> arguments = arguments_or.ValueOrDie();

    VLOG(2) << "Building Pipeline (" << ctx->op_kernel().name()
            << ") function with " << input_types_.size() << " inputs including "
            << num_resource_args << " resources.";

    // Rewrite the Pipeline function such that arguments to PipelineStage ops
    // are rearranged and resource variables moved to the back.
    NameAttrList new_to_apply;
    OP_REQUIRES_OK(
        ctx,
        RearrangeFunctionArguments(
            [&ctx](const NameAttrList& function, const FunctionBody** fbody) {
              return ctx->compiler()->FindFunctionBody(function, fbody);
            },
            new_to_apply, *to_apply_, ctx->compiler()->local_flib_def()));

    XlaCompiler::CompileOptions compile_options =
        poplarplugin::GetDefaultCompileOptions();
    compile_options.return_updated_values_for_all_resources = true;

    // Compile the computation.
    XlaCompiler::CompilationResult result;
    OP_REQUIRES_OK(ctx, ctx->compiler()->CompileFunction(
                            compile_options, new_to_apply, arguments, &result));

    // Get the non constant XLA arguments.
    auto inputs_or =
        poplarplugin::GetXlaInputs(ctx, arguments, result.input_mapping);
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
      for (size_t input_idx = 0; input_idx != inputs.size(); ++input_idx) {
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
      for (size_t i = 0; i < result.resource_updates.size(); ++i) {
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
    // Set the pipeline depth.
    OP_REQUIRES_OK(ctx, builder->SetInstructionFrontendAttribute(
                            outputs, FrontendAttributeId_Name(PIPELINE_DEPTH),
                            std::to_string(pipeline_depth_)));
    // Set the repeat count.
    OP_REQUIRES_OK(ctx,
                   builder->SetInstructionFrontendAttribute(
                       outputs, FrontendAttributeId_Name(PIPELINE_REPEAT_COUNT),
                       std::to_string(repeat_count_)));
    // Set the schedule.
    OP_REQUIRES_OK(ctx,
                   builder->SetInstructionFrontendAttribute(
                       outputs, FrontendAttributeId_Name(PIPELINE_SCHEDULE),
                       std::to_string(schedule_)));
    // Set the config field.
    OP_REQUIRES_OK(
        ctx, builder->SetInstructionFrontendAttribute(
                 outputs, FrontendAttributeId_Name(PIPELINE_POPLAR_CONFIG),
                 pipeline_poplar_config_));
    // Set the offload_weight_update_variables flag.
    OP_REQUIRES_OK(
        ctx,
        builder->SetInstructionFrontendAttribute(
            outputs, FrontendAttributeId_Name(PIPELINE_OFFLOAD_WU_VARIABLES),
            std::to_string(offload_weight_update_variables_)));

    // A pipeline has no explicit outputs, only updates of resource variables.
    // We can use the input index to index into the outputs because we have
    // ensured that the inputs and outputs are aligned.
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
  int64 pipeline_depth_;
  int64 repeat_count_;
  int64 schedule_;
  std::string pipeline_poplar_config_;
  bool offload_weight_update_variables_;

  TF_DISALLOW_COPY_AND_ASSIGN(PipelineOp);
};
REGISTER_IPU_OP("Pipeline", PipelineOp);
}  // namespace
}  // namespace tensorflow
