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
class FunctionOp : public poplarplugin::FunctionBaseOp {
 public:
  explicit FunctionOp(OpKernelConstruction* ctx)
      : poplarplugin::FunctionBaseOp(ctx, /*evaluate_constants=*/false) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("unique_sharding", &unique_sharding_));
    OP_REQUIRES_OK(ctx,
                   ctx->GetAttr("keep_input_layouts", &keep_input_layouts_));
  }

 protected:
  Status SetConfig(xla::XlaBuilder* builder, xla::XlaOp& operation) override {
    TF_RETURN_IF_ERROR(builder->SetInstructionFrontendAttribute(
        operation, pp::FrontendAttributeId_Name(pp::CALL_CONFIG_TYPE),
        pp::PoplarBackendConfig_CallConfig_Type_Name(
            pp::PoplarBackendConfig::CallConfig::Function)));
    TF_RETURN_IF_ERROR(builder->SetInstructionFrontendAttribute(
        operation, pp::FrontendAttributeId_Name(pp::UNIQUE_SHARDING),
        std::to_string(unique_sharding_)));
    TF_RETURN_IF_ERROR(builder->SetInstructionFrontendAttribute(
        operation, pp::FrontendAttributeId_Name(pp::KEEP_INPUT_LAYOUTS),
        std::to_string(keep_input_layouts_)));
    return Status::OK();
  }

 private:
  bool unique_sharding_;
  bool keep_input_layouts_;
  TF_DISALLOW_COPY_AND_ASSIGN(FunctionOp);
};
REGISTER_IPU_OP("Function", FunctionOp);

class MultiConvOp : public poplarplugin::FunctionBaseOp {
 public:
  explicit MultiConvOp(OpKernelConstruction* ctx)
      : poplarplugin::FunctionBaseOp(ctx, /*evaluate_constants=*/false) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("option_flags", &option_flags_));
  }

 protected:
  Status SetConfig(xla::XlaBuilder* builder, xla::XlaOp& operation) override {
    TF_RETURN_IF_ERROR(builder->SetInstructionFrontendAttribute(
        operation, pp::FrontendAttributeId_Name(pp::CALL_CONFIG_TYPE),
        pp::PoplarBackendConfig_CallConfig_Type_Name(
            pp::PoplarBackendConfig::CallConfig::MultiConv)));
    TF_RETURN_IF_ERROR(builder->SetInstructionFrontendAttribute(
        operation, pp::FrontendAttributeId_Name(pp::OPTION_FLAGS),
        option_flags_));
    return Status::OK();
  }

 private:
  std::string option_flags_;
  TF_DISALLOW_COPY_AND_ASSIGN(MultiConvOp);
};
REGISTER_IPU_OP("MultiConv", MultiConvOp);

}  // namespace tensorflow
