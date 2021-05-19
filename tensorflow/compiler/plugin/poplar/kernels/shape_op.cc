/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

// XLA-specific Shape Ops.

#include "tensorflow/compiler/tf2xla/kernels/shape_util.h"
#include "tensorflow/compiler/tf2xla/kernels/tensor_list_utils.h"
#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/lib/constants.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/kernel_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"

#include "tensorflow/compiler/plugin/poplar/kernels/ipu_kernels_common.h"

namespace tensorflow {
namespace {

class ZerosLikeOp : public XlaOpKernel, public IpuOpKernel {
 public:
  explicit ZerosLikeOp(OpKernelConstruction* ctx)
      : XlaOpKernel(ctx), IpuOpKernel() {}

  void Compile(XlaOpKernelContext* ctx) override {
    if (IsTensorListInput(ctx, 0)) {
      bool is_initialized;
      OP_REQUIRES_OK(ctx,
                     IsTensorListInitialized(ctx->Input(0), &is_initialized));
      if (is_initialized) {
        // Input 0 is a initialised tensorlist
        return HandleTensorList(ctx);
      }
    }

    if (input_type(0) == DT_VARIANT) {
      HandleZerosLikeVariant(ctx);
    } else {
      HandleArrayInput(ctx);
    }
  }

 private:
  void HandleZerosLikeVariant(XlaOpKernelContext* ctx) {
    CHECK(input_type(0) == DT_VARIANT);
    // This operation should never actually be materialised in the backend, so
    // we give it a dummy name.
    ctx->SetOutput(0, xla::CustomCall(ctx->builder(), "__ZerosLikeVariant", {},
                                      xla::ShapeUtil::MakeOpaqueShape(),
                                      attribute_map_.Serialise()));
  }

  void HandleTensorList(XlaOpKernelContext* ctx) {
    xla::XlaOp list = ctx->Input(0);
    auto list_shape_or = ctx->builder()->GetShape(list);
    OP_REQUIRES_OK(ctx, list_shape_or.status());
    xla::XlaOp new_list;
    OP_REQUIRES_OK(ctx,
                   CreateZerosTensorListWithShape(
                       ctx->builder(), list_shape_or.ValueOrDie(), &new_list));

    xla::XlaOp push_index;
    OP_REQUIRES_OK(ctx, GetTensorListPushIndex(list, &push_index));

    xla::XlaOp result;
    OP_REQUIRES_OK(ctx, SetTensorListPushIndex(new_list, push_index, &result));
    ctx->SetTensorListOutput(0, result);
  }

  void HandleArrayInput(XlaOpKernelContext* ctx) {
    CHECK(input_type(0) != DT_VARIANT);

    const TensorShape input_shape = ctx->InputShape(0);

    auto zero = XlaHelpers::Zero(ctx->builder(), input_type(0));
    ctx->SetOutput(0, xla::Broadcast(zero, input_shape.dim_sizes()));
  }
};

REGISTER_XLA_OP(
    Name("ZerosLike").Device(DEVICE_IPU_XLA_JIT).AllowVariantTypes(),
    ZerosLikeOp);
REGISTER_XLA_OP(Name("ZerosLike").Device(DEVICE_XLA_IPU).AllowVariantTypes(),
                ZerosLikeOp);

}  // namespace
}  // namespace tensorflow
