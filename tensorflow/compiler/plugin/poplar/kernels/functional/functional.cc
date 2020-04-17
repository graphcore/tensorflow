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

namespace pp = xla::poplarplugin;

namespace tensorflow {

class FunctionOp : public XlaOpKernel {
 public:
  explicit FunctionOp(OpKernelConstruction* ctx, bool is_forward = true)
      : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("to_apply", &to_apply_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("Tin", &input_types_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("Tout", &output_types_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
    auto builder = ctx->builder();
    // First get all the arguments.
    int num_resource_args = 0;
    // Note that we explicitly don't evaluate constants as we want them to be
    // inputs to the function (otherwise they will be lowered and the functions
    // can become unique).
    auto arguments_or = poplarplugin::GetXlaArguments(
        ctx, input_types_, &num_resource_args, /*evalute_constants*/ false);
    OP_REQUIRES_OK(ctx, arguments_or.status());
    std::vector<XlaCompiler::Argument> arguments = arguments_or.ValueOrDie();

    VLOG(2) << "Building function " << ctx->op_kernel().name() << " with "
            << input_types_.size() << " inputs including " << num_resource_args
            << " resources.";
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
    compile_options.return_updated_values_for_all_resources = false;

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
                 outputs, pp::FrontendAttributeId_Name(pp::CALL_CONFIG_TYPE),
                 pp::PoplarBackendConfig_CallConfig_Type_Name(
                     pp::PoplarBackendConfig::CallConfig::Function)));

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

  TF_DISALLOW_COPY_AND_ASSIGN(FunctionOp);
};
REGISTER_IPU_OP("Function", FunctionOp);

}  // namespace tensorflow
