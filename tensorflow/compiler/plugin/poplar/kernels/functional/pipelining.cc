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
#include "tensorflow/compiler/plugin/poplar/driver/tools/attributes_utils.h"
#include "tensorflow/compiler/plugin/poplar/kernels/custom_kernels_util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/functional/functional_util.h"
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

#include "absl/algorithm/algorithm.h"

using namespace xla::poplarplugin;

namespace tensorflow {
namespace {
class PipelineStageOp : public poplarplugin::FunctionBaseOp {
 public:
  explicit PipelineStageOp(OpKernelConstruction* ctx, bool is_forward = true)
      : poplarplugin::FunctionBaseOp(ctx, /*evaluate_constants=*/true) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("stage_id", &stage_id_));
    call_config_type_ =
        is_forward ? PoplarBackendConfig::CallConfig::PipelineStage
                   : PoplarBackendConfig::CallConfig::PipelineStageBackward;
  }

  Status SetConfig(xla::XlaBuilder* builder, xla::XlaOp& operation) override {
    // Set the config type of the call.
    TF_RETURN_IF_ERROR(builder->SetInstructionFrontendAttribute(
        operation, FrontendAttributeId_Name(CALL_CONFIG_TYPE),
        PoplarBackendConfig_CallConfig_Type_Name(call_config_type_)));

    // Set the stage id.
    TF_RETURN_IF_ERROR(builder->SetInstructionFrontendAttribute(
        operation, FrontendAttributeId_Name(PIPELINE_STAGE_ID),
        std::to_string(stage_id_)));
    return Status::OK();
  }

 private:
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

class ResourceUpdateOp : public XlaOpKernel {
 public:
  explicit ResourceUpdateOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("to_apply", &to_apply_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("Tin", &input_types_));
    DataTypeVector output_types;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("Tout", &output_types));
    OP_REQUIRES(ctx, output_types.size() == 0,
                errors::InvalidArgument("Expected ResourceUpdate "
                                        "to have no explicit outputs."));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("offload_weight_update_variables",
                                     &offload_weight_update_variables_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("replicated_optimizer_state_sharding",
                                     &replicated_optimizer_state_sharding_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
    auto builder = ctx->builder();
    // First get all the arguments and compile the computation.
    int num_resource_args = 0;
    auto arguments_or =
        poplarplugin::GetXlaArguments(ctx, input_types_, &num_resource_args);
    OP_REQUIRES_OK(ctx, arguments_or.status());
    std::vector<XlaCompiler::Argument> arguments = arguments_or.ValueOrDie();

    VLOG(2) << "Building ResourceUpdate (" << ctx->op_kernel().name()
            << ") function with " << input_types_.size() << " inputs including "
            << num_resource_args << " resources.";

    XlaCompiler::CompileOptions compile_options =
        poplarplugin::GetDefaultCompileOptions();
    compile_options.return_updated_values_for_all_resources = true;

    // Compile the computation.
    XlaCompiler::CompilationResult result;
    OP_REQUIRES_OK(
        ctx, poplarplugin::CompileFunction(ctx, compile_options, *to_apply_,
                                           arguments, &result));

    // Get the non constant XLA arguments.
    auto inputs_or =
        poplarplugin::GetXlaInputs(ctx, arguments, result.input_mapping);
    OP_REQUIRES_OK(ctx, inputs_or.status());
    std::vector<xla::XlaOp> inputs = inputs_or.ValueOrDie();

    auto outputs = xla::Call(builder, *result.computation, inputs);

    util::SetInstructionFrontEndAttributes(
        ctx, builder, outputs,
        {util::AttrMember(CALL_CONFIG_TYPE,
                          PoplarBackendConfig_CallConfig_Type_Name(
                              PoplarBackendConfig::CallConfig::ResourceUpdate)),
         util::AttrMember(OFFLOAD_WEIGHT_UPDATE_VARIABLES,
                          offload_weight_update_variables_),
         util::AttrMember(PARTITION_OFFLOADED_WEIGHT_UPDATE_VARIABLES,
                          replicated_optimizer_state_sharding_)});

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
  std::string offload_weight_update_variables_;
  std::string replicated_optimizer_state_sharding_;

  TF_DISALLOW_COPY_AND_ASSIGN(ResourceUpdateOp);
};
REGISTER_IPU_OP("ResourceUpdate", ResourceUpdateOp);

// Because CompileFunction doesn't know how many constant parameters have
// been removed resource update indices are wrong. To get round this
// count how many constants there are before the resources so we can
// subtrack these off the returned indices
static int64 FindNumberOfConstantParameters(
    const std::vector<XlaCompiler::Argument>& arguments) {
  bool hit_resource = false;
  return absl::c_count_if(arguments, [&](const XlaCompiler::Argument& arg) {
    hit_resource |= (arg.kind == XlaCompiler::Argument::kResource);
    return (!hit_resource) && (arg.kind == XlaCompiler::Argument::kConstant);
  });
}

static int64 GetIndexWithoutConstants(const int64 orig, const int64 offset) {
  return orig - offset;
}

class PipelineOp : public XlaOpKernel {
  void SetInstructionFrontEndAttributes(
      XlaOpKernelContext* ctx, xla::XlaBuilder* builder,
      const xla::XlaOp& outputs,
      int gradient_accumulation_operand_index) const {
    using AttrMember = util::AttrMember;
    util::AttrMembers attributes = {
        AttrMember(CALL_CONFIG_TYPE,
                   PoplarBackendConfig_CallConfig_Type_Name(
                       PoplarBackendConfig::CallConfig::Pipeline)),
        AttrMember(PIPELINE_BATCH_SERIALIZATION_ITERATIONS,
                   batch_serialization_iterations_),
        AttrMember(PIPELINE_REPEAT_COUNT, repeat_count_),
        AttrMember(PIPELINE_SCHEDULE, schedule_),
        AttrMember(PIPELINE_POPLAR_CONFIG, pipeline_poplar_config_),
        AttrMember(OFFLOAD_ACTIVATIONS, offload_activations_),
        AttrMember(OFFLOAD_GRADIENT_ACCUMULATION_BUFFERS,
                   offload_gradient_accumulation_buffers_),
        AttrMember(PARTITION_VARIABLES, replicated_weight_sharding_),
        AttrMember(OFFLOAD_VARIABLES, offload_weights_),
        AttrMember(RECOMPUTATION_MODE, recomputation_mode_),
        AttrMember("GradientAccumulationOperandIndex",
                   gradient_accumulation_operand_index)};

    util::SetInstructionFrontEndAttributes(ctx, builder, outputs,
                                           std::move(attributes));
  }

  xla::StatusOr<xla::XlaComputation> CreateInnerPipeline(
      XlaOpKernelContext* ctx, const std::vector<xla::XlaOp>& inputs,
      const XlaCompiler::CompilationResult& result,
      const int64 num_constants) const {
    // For pipelines we make sure that the inputs and outputs have the same
    // shape and that the values for every output at index `i` are:
    // 1. the input value `i` if the input is not a resource variable
    // 2. the input value `i` if the input is a resource variable which has not
    //   been modified
    // 3. the modified resource variable corresponding to the value at input `i`
    // To do so we wrap the pipeline in another call, and set up the tuple
    // accordingly.
    std::unique_ptr<xla::XlaBuilder> cb =
        ctx->builder()->CreateSubBuilder("pipeline_wrapper");
    std::vector<xla::XlaOp> inner_inputs(inputs.size());
    std::vector<xla::XlaOp> inner_outputs(inputs.size());
    // First handle cases 1 and 2.
    for (size_t input_idx = 0; input_idx != inputs.size(); ++input_idx) {
      auto param_shape = input_idx != inputs.size() - 1
                             ? result.xla_input_shapes[input_idx]
                             : xla::ShapeUtil::MakeShape(xla::S32, {});
      auto param = xla::Parameter(cb.get(), input_idx, param_shape,
                                  absl::StrCat("input/", input_idx));
      inner_inputs[input_idx] = param;
      inner_outputs[input_idx] = param;
    }

    // First handle cases 1 and 2.
    // Call the computation which is wrapped (but not passing the final
    // operand).
    auto inner_call =
        xla::Call(cb.get(), *result.computation,
                  absl::MakeSpan(inner_inputs.data(), inner_inputs.size() - 1));
    // Now go through any resource updates and add necessary GTEs and handle
    // case 3.
    for (size_t i = 0; i < result.resource_updates.size(); ++i) {
      const XlaCompiler::ResourceUpdate& update = result.resource_updates[i];
      if (update.modified) {
        // This modification relies on the constant inputs not
        // being removed earlier in CompileFunction
        inner_outputs[GetIndexWithoutConstants(update.input_index,
                                               num_constants)] =
            xla::GetTupleElement(inner_call, i);
      }
    }
    CHECK(absl::c_equal(inner_inputs, inner_outputs,
                        [&](const xla::XlaOp& a, const xla::XlaOp& b) {
                          auto a_shape = cb->GetShape(a);
                          auto b_shape = cb->GetShape(b);
                          // if doesn't have a shape let it error further down
                          // the line
                          return !a_shape.ok() || !b_shape.ok() ||
                                 (a_shape.ValueOrDie() == b_shape.ValueOrDie());
                        }));

    xla::Tuple(cb.get(), inner_outputs);
    auto comp_or = cb->Build();
    return comp_or;
  }

  static void UpdateResources(
      const XlaCompiler::CompilationResult& result, XlaOpKernelContext* ctx,
      const xla::XlaOp& outputs,
      const std::vector<XlaCompiler::Argument>& arguments,
      xla::XlaBuilder* builder, const int64 num_constants) {
    // We can use the input index to index into the outputs because we have
    // ensured that the inputs and outputs are aligned.
    for (const XlaCompiler::ResourceUpdate& update : result.resource_updates) {
      XlaResource* resource;
      OP_REQUIRES_OK(ctx, ctx->GetResourceInput(update.input_index, &resource));
      if (update.modified) {
        // Add a GTE from the same input index.
        OP_REQUIRES_OK(
            ctx, resource->SetFromPack(
                     arguments[update.input_index].tensor_array_gradients,
                     xla::GetTupleElement(
                         outputs, GetIndexWithoutConstants(update.input_index,
                                                           num_constants)),
                     builder));
      }
      VLOG(2) << "Variable: pos: " << update.input_index
              << " name: " << resource->name()
              << " modified: " << update.modified
              << " type: " << DataTypeString(update.type)
              << " shape: " << update.shape.DebugString();
    }
  }

 public:
  explicit PipelineOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("to_apply", &to_apply_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("Tin", &input_types_));
    DataTypeVector output_types;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("Tout", &output_types));
    OP_REQUIRES(ctx, output_types.size() == 0,
                errors::InvalidArgument(
                    "Expected PipelineStage to have no explicit outputs."));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("batch_serialization_iterations",
                                     &batch_serialization_iterations_));
    OP_REQUIRES_OK(ctx,
                   ctx->GetAttr("offload_activations", &offload_activations_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("offload_gradient_accumulation_buffers",
                                     &offload_gradient_accumulation_buffers_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("replicated_weight_sharding",
                                     &replicated_weight_sharding_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("offload_weights", &offload_weights_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("repeat_count", &repeat_count_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("schedule", &schedule_));
    OP_REQUIRES_OK(ctx,
                   ctx->GetAttr("recomputation_mode", &recomputation_mode_));
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

    CHECK_EQ(arguments.size(), ctx->num_inputs() - 1);
    // Check all resources are after all other arguments
    CHECK(absl::c_is_partitioned(arguments, [](const XlaCompiler::Argument& a) {
      return (a.kind != XlaCompiler::Argument::kResource);
    }));
    const auto num_constants = FindNumberOfConstantParameters(arguments);

    VLOG(2) << "Building Pipeline (" << ctx->op_kernel().name()
            << ") function with " << input_types_.size() << " inputs including "
            << num_resource_args << " resources.";

    XlaCompiler::CompileOptions compile_options =
        poplarplugin::GetDefaultCompileOptions();
    compile_options.return_updated_values_for_all_resources = true;

    // Compile the computation.
    XlaCompiler::CompilationResult result;
    OP_REQUIRES_OK(
        ctx, poplarplugin::CompileFunction(ctx, compile_options, *to_apply_,
                                           arguments, &result));

    // Get the non constant XLA arguments.
    auto inputs_or =
        poplarplugin::GetXlaInputs(ctx, arguments, result.input_mapping);
    OP_REQUIRES_OK(ctx, inputs_or.status());
    std::vector<xla::XlaOp> inputs = inputs_or.ValueOrDie();

    // Add the gradient accumulation count as the last input. Note that the
    // operand index is the XLA input index, and not the TF input index.
    // These may differ when any of the TF inputs are constants.
    const int gradient_accumulation_operand_index = inputs.size();
    inputs.emplace_back(ctx->Input(ctx->num_inputs() - 1));

    auto wrapped_pipeline =
        CreateInnerPipeline(ctx, inputs, result, num_constants);
    OP_REQUIRES_OK(ctx, wrapped_pipeline.status());

    // Create the actual call.
    auto outputs = xla::Call(builder, wrapped_pipeline.ValueOrDie(), inputs);

    SetInstructionFrontEndAttributes(ctx, builder, outputs,
                                     gradient_accumulation_operand_index);

    // A pipeline has no explicit outputs, only updates of resource variables.
    UpdateResources(result, ctx, outputs, arguments, builder, num_constants);
  }

 private:
  const NameAttrList* to_apply_;
  DataTypeVector input_types_;
  int64 batch_serialization_iterations_;
  int64 repeat_count_;
  int64 schedule_;
  std::string pipeline_poplar_config_;
  std::string offload_activations_;
  std::string offload_gradient_accumulation_buffers_;
  std::string replicated_weight_sharding_;
  std::string offload_weights_;
  std::string recomputation_mode_;

  TF_DISALLOW_COPY_AND_ASSIGN(PipelineOp);
};
REGISTER_IPU_OP("Pipeline", PipelineOp);
}  // namespace
}  // namespace tensorflow
