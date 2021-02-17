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

#include "tensorflow/compiler/plugin/poplar/driver/xla_ipu_common.h"

#include "tensorflow/compiler/jit/xla_device.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {

class DeviceSync : public OpKernel {
 public:
  explicit DeviceSync(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    XlaDevice* xla_device =
        dynamic_cast<XlaDevice*>(ctx->device()->UnderlyingDevice());
    OP_REQUIRES(ctx, xla_device,
                errors::Internal("Cannot synchronize a non-XLA device \"",
                                 ctx->device()->name(), "\"."));
    OP_REQUIRES_OK(ctx, xla_device->Sync());
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(DeviceSync);
};

REGISTER_KERNEL_BUILDER(Name("DeviceSync").Device(DEVICE_XLA_IPU), DeviceSync);

}  // namespace tensorflow
