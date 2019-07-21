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
    attribute_map_.AddAttribute("type",
                                PoplarBackendConfig_CallConfig_Type_Name(type));
  }

  void Compile(XlaOpKernelContext* ctx) override {
    auto builder = ctx->builder();
    // First get all the arguments and compile the computation.
    std::vector<XlaCompiler::Argument> arguments(input_types_.size());
    int num_resource_args = 0;
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
      } else {
        arg.kind = XlaCompiler::Argument::kParameter;
        arg.type = input_types_[i];
        // Use the xla::Shape for the input instead of ctx->InputShape. This is
        // necessary for forwarding shapes of DT_VARIANTs, e.g. TensorLists.
        auto shape_or = builder->GetShape(ctx->Input(i));
        OP_REQUIRES_OK(ctx, shape_or.status());
        arg.shape = shape_or.ValueOrDie();
        VLOG(2) << "Arg type: " << DataTypeString(arg.type)
                << " shape: " << arg.HumanString();
      }
    }
    VLOG(2) << "Building function: " << input_types_.size()
            << " inputs including " << num_resource_args << "resources.";
    XlaCompiler::CompileOptions compile_options;
    compile_options.use_tuple_arg = false;
    compile_options.resolve_compile_time_constants = true;
    compile_options.return_updated_values_for_all_resources = true;
    compile_options.is_entry_computation = false;
    compile_options.add_token_input_output = false;

    XlaCompiler::CompilationResult to_apply_func;
    OP_REQUIRES_OK(ctx,
                   ctx->compiler()->CompileFunction(compile_options, *to_apply,
                                                    arguments, &to_apply_func));

    // Get the XLA arguments.
    int num_inputs = to_apply_func.input_mapping.size();
    std::vector<xla::XlaOp> inputs(num_inputs);
    for (int i = 0; i < num_inputs; ++i) {
      if (ctx->input_type(i) == DT_RESOURCE) {
        XlaResource* resource;
        OP_REQUIRES_OK(ctx, ctx->GetResourceInput(i, &resource));
        OP_REQUIRES_OK(ctx, resource->Pack(&inputs[i], builder));
      } else {
        inputs[i] = ctx->Input(i);
      }
    }

    auto outputs = xla::Call(builder, *to_apply_func.computation, inputs);
    // Set the backend config of the call.
    OP_REQUIRES_OK(
        ctx, builder->SetBackendConfig(outputs, attribute_map_.Serialise()));
    // Sets non-variable outputs.
    for (int i = 0; i < output_types_.size(); ++i) {
      xla::XlaOp output_handle = xla::GetTupleElement(outputs, i);
      ctx->SetOutput(i, output_handle);
    }

    // Updates the values of any resource variables modified by the function
    // call.
    for (int i = 0; i < to_apply_func.resource_updates.size(); ++i) {
      const XlaCompiler::ResourceUpdate& update =
          to_apply_func.resource_updates[i];
      XlaResource* resource;
      OP_REQUIRES_OK(ctx, ctx->GetResourceInput(update.input_index, &resource));
      if (update.modified) {
        int pos = to_apply_func.outputs.size() + i;
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
  const NameAttrList* to_apply;
  DataTypeVector input_types_;
  DataTypeVector output_types_;
  IPUCustomKernelsUtil::AttributeMap attribute_map_;

  TF_DISALLOW_COPY_AND_ASSIGN(FunctionCompileOp);
};

class PipelineStageOp : public FunctionCompileOp {
 public:
  explicit PipelineStageOp(OpKernelConstruction* ctx)
      : FunctionCompileOp(ctx, PoplarBackendConfig::CallConfig::PipelineStage) {
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(PipelineStageOp);
};
REGISTER_IPU_OP("PipelineStage", PipelineStageOp);

class PipelineOp : public FunctionCompileOp {
 public:
  explicit PipelineOp(OpKernelConstruction* ctx)
      : FunctionCompileOp(ctx, PoplarBackendConfig::CallConfig::Pipeline) {}

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(PipelineOp);
};
REGISTER_IPU_OP("Pipeline", PipelineOp);

}  // namespace
}  // namespace tensorflow
