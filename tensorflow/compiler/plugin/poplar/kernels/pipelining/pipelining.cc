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

class FunctionCompileOp : public XlaOpKernel {
 public:
  explicit FunctionCompileOp(OpKernelConstruction* ctx,
                             PoplarBackendConfig::CallConfig::Type type)
      : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("to_apply", &to_apply));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("Tin", &input_types_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("Tout", &output_types_));
    call_config_type_ = PoplarBackendConfig_CallConfig_Type_Name(type);
  }

  void Compile(XlaOpKernelContext* ctx) override {
    auto builder = ctx->builder();
    // First get all the arguments and compile the computation.
    std::vector<XlaCompiler::Argument> arguments(input_types_.size());
    int num_resource_args = 0;
    int num_non_constant_args = 0;
    for (int i = 0; i < input_types_.size(); ++i) {
      XlaCompiler::Argument& arg = arguments[i];
      DataType type = ctx->input_type(i);

      if (type == DT_RESOURCE) {
        XlaResource* resource;
        OP_REQUIRES_OK(ctx, ctx->GetResourceInput(i, &resource));

        arg.initialized = resource->initialized();
        arg.kind = XlaCompiler::Argument::kResource;
        arg.resource_kind = resource->kind();

        arg.type = resource->type();
        arg.shape = resource->shape();
        OP_REQUIRES(
            ctx, arg.initialized,
            errors::Unimplemented("Uninitialized arguments: ", arg.name));
        arg.max_array_size = resource->max_array_size();
        for (const auto& gradient : resource->tensor_array_gradients()) {
          arg.tensor_array_gradients.insert(gradient.first);
        }
        arg.name = resource->name();
        VLOG(2) << "Resource " << resource->name()
                << " type: " << DataTypeString(arg.type)
                << " shape: " << arg.HumanString()
                << " initialized: " << arg.initialized;

        num_resource_args++;
        num_non_constant_args++;
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
          arg.type = input_types_[i];
          // Use the xla::Shape for the input instead of ctx->InputShape. This
          // is necessary for forwarding shapes of DT_VARIANTs, e.g.
          // TensorLists.
          auto shape_or = builder->GetShape(ctx->Input(i));
          OP_REQUIRES_OK(ctx, shape_or.status());
          arg.shape = shape_or.ValueOrDie();
          num_non_constant_args++;
          VLOG(2) << "Parameter type: " << DataTypeString(arg.type)
                  << " shape: " << arg.HumanString();
        }
      }
    }

    VLOG(2) << "Building function: " << input_types_.size()
            << " inputs including " << num_resource_args << " resources.";
    XlaCompiler::CompileOptions compile_options;
    compile_options.use_tuple_arg = false;
    compile_options.resolve_compile_time_constants = true;
    compile_options.always_return_tuple = true;
    compile_options.return_updated_values_for_all_resources = true;
    compile_options.is_entry_computation = false;
    compile_options.add_token_input_output = false;

    XlaCompiler::CompilationResult result;
    OP_REQUIRES_OK(ctx, ctx->compiler()->CompileFunction(
                            compile_options, *to_apply, arguments, &result));

    // Get the non constant XLA arguments.
    std::vector<xla::XlaOp> inputs(num_non_constant_args);
    for (int i = 0, next_param_idx = 0; i < arguments.size(); ++i) {
      switch (arguments[i].kind) {
        case XlaCompiler::Argument::kResource: {
          XlaResource* resource;
          OP_REQUIRES_OK(ctx, ctx->GetResourceInput(i, &resource));
          OP_REQUIRES_OK(ctx,
                         resource->Pack(&inputs[next_param_idx++], builder));
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
        default: {
          OP_REQUIRES(ctx, false,
                      errors::InvalidArgument("Invalid argument kind."));
        }
      }
    }

    auto outputs = xla::Call(builder, *result.computation, inputs);
    // Set the config type of the call.
    OP_REQUIRES_OK(ctx, builder->SetInstructionFrontendAttribute(
                            outputs, FrontendAttributeId_Name(CALL_CONFIG_TYPE),
                            call_config_type_));
    // Set any extra attributes.
    for (auto key_val_pair : GetExtraFrontendAttributes()) {
      OP_REQUIRES_OK(ctx,
                     builder->SetInstructionFrontendAttribute(
                         outputs, key_val_pair.first, key_val_pair.second));
    }
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

 protected:
  virtual absl::flat_hash_map<std::string, std::string>
  GetExtraFrontendAttributes() {
    return {};
  }

 private:
  const NameAttrList* to_apply;
  DataTypeVector input_types_;
  DataTypeVector output_types_;
  string call_config_type_;

  TF_DISALLOW_COPY_AND_ASSIGN(FunctionCompileOp);
};

class PipelineStageOp : public FunctionCompileOp {
 public:
  explicit PipelineStageOp(OpKernelConstruction* ctx, bool is_forward = true)
      : FunctionCompileOp(
            ctx, is_forward
                     ? PoplarBackendConfig::CallConfig::PipelineStage
                     : PoplarBackendConfig::CallConfig::PipelineStageBackward) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("stage_id", &stage_id_));
  }

 protected:
  absl::flat_hash_map<std::string, std::string> GetExtraFrontendAttributes()
      override {
    return {{FrontendAttributeId_Name(PIPELINE_STAGE_ID),
             std::to_string(stage_id_)}};
  }

 private:
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

class PipelineOp : public FunctionCompileOp {
 public:
  explicit PipelineOp(OpKernelConstruction* ctx)
      : FunctionCompileOp(ctx, PoplarBackendConfig::CallConfig::Pipeline) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("repeat_count", &repeat_count_));
  }

 protected:
  absl::flat_hash_map<std::string, std::string> GetExtraFrontendAttributes()
      override {
    return {{FrontendAttributeId_Name(PIPELINE_REPEAT_COUNT),
             std::to_string(repeat_count_)}};
  }

 private:
  int64 repeat_count_;

  TF_DISALLOW_COPY_AND_ASSIGN(PipelineOp);
};
REGISTER_IPU_OP("Pipeline", PipelineOp);

}  // namespace
}  // namespace tensorflow
